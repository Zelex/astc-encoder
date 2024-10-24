#include <immintrin.h> // For SSE intrinsics
#include <math.h>
#include <stdlib.h>
#include <string.h> // For memcpy

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h> // For NEON intrinsics
#endif
#include "astcenc.h"
#include "astcenc_internal.h"
#include "astcenc_internal_entry.h"
#include "astcenc_mathlib.h"
#include "astcenc_vecmathlib.h"
#include "astcenccli_internal.h"

struct Int128
{
private:
	union
	{
		struct
		{
			uint64_t lo;
			uint64_t hi;
		};
		uint8_t bytes[16];
	};

public:
	Int128() : lo(0), hi(0)
	{
	}

	explicit Int128(const uint8_t* data)
	{
		memcpy(bytes, data, 16);
	}

	void to_bytes(uint8_t* data) const
	{
		memcpy(data, bytes, 16);
	}

	uint8_t get_byte(int index) const
	{
		return bytes[index];
	}

	Int128 shift_left(int shift) const
	{
		Int128 result;
		if (shift >= 128)
		{
			return result; // Returns zero
		}
		if (shift == 0)
		{
			return *this;
		}
		if (shift < 64)
		{
			result.hi = (hi << shift) | (lo >> (64 - shift));
			result.lo = lo << shift;
		}
		else
		{
			result.hi = lo << (shift - 64);
			result.lo = 0;
		}
		return result;
	}

	Int128 bitwise_and(const Int128& other) const
	{
		Int128 result;
		result.lo = lo & other.lo;
		result.hi = hi & other.hi;
		return result;
	}

	Int128 bitwise_or(const Int128& other) const
	{
		Int128 result;
		result.lo = lo | other.lo;
		result.hi = hi | other.hi;
		return result;
	}

	bool is_equal(const Int128& other) const
	{
		return lo == other.lo && hi == other.hi;
	}

	static Int128 from_int(long long val)
	{
		Int128 result;
		result.lo = val;
		result.hi = val < 0 ? -1LL : 0;
		return result;
	}

	Int128 subtract(const Int128& other) const
	{
		Int128 result;
		result.lo = lo - other.lo;
		result.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return result;
	}

	Int128 bitwise_not() const
	{
		Int128 result;
		result.lo = ~lo;
		result.hi = ~hi;
		return result;
	}

	uint64_t get_uint64(int index) const
	{
		return index == 0 ? lo : hi;
	}

	std::string to_string() const
	{
		char buffer[33];
		snprintf(buffer, sizeof(buffer), "%016llx%016llx", hi, lo);
		return std::string(buffer);
	}
};

#define MAX_MTF_SIZE (256 + 64 + 16 + 1)
// #define MAX_MTF_SIZE (1024 + 256 + 64 + 16 + 1)
#define CACHE_SIZE (0x10000) // Should be a power of 2 for efficient modulo operation
#define BEST_CANDIDATES_COUNT (16)
#define MAX_THREADS (128)
#define MODE_MASK (0x7FF)
#define MAX_BLOCKS_PER_ITEM (8192)
#define MEASURE_RATE calculate_bit_cost_zip

struct Histo
{
	int h[256];
	int size;
};

struct Mtf
{
	Int128 list[MAX_MTF_SIZE];
	int size;
	int max_size;
};

struct CachedBlock
{
	Int128 encoded;
	uint8_t decoded[6 * 6 * 6 * 4 * 4]; // Max size for both U8 and float types
	bool valid;
};

static inline float log2_fast(float val)
{
	union
	{
		float val;
		int x;
	} u = {val};
	float log_2 = (float)(((u.x >> 23) & 255) - 128);
	u.x &= ~(255 << 23);
	u.x += 127 << 23;
	log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val - 0.65871759316667f;
	return log_2;
}

static uint32_t hash_128(const Int128& value)
{
	// FNV-1a inspired constants
	const uint32_t PRIME = 0x01000193;
	const uint32_t SEED = 0x811C9DC5;

	uint32_t hash = SEED;

	// Process 4 bytes at a time with better mixing
	for (int i = 0; i < 16; i += 4)
	{
		uint32_t chunk = value.get_byte(i) | (value.get_byte(i + 1) << 8) | (value.get_byte(i + 2) << 16) | (value.get_byte(i + 3) << 24);

		// Mix the chunk
		chunk *= 0xcc9e2d51;
		chunk = (chunk << 15) | (chunk >> 17);
		chunk *= 0x1b873593;

		// Mix it into the hash
		hash ^= chunk;
		hash = (hash << 13) | (hash >> 19);
		hash = hash * 5 + 0xe6546b64;
	}

	// Final avalanche mix
	hash ^= hash >> 16;
	hash *= 0x85ebca6b;
	hash ^= hash >> 13;
	hash *= 0xc2b2ae35;
	hash ^= hash >> 16;

	return hash;
}

static void histo_reset(Histo* h)
{
	for (int i = 0; i < 256; i++)
		h->h[i] = 0;
	h->size = 0;
}

static void histo_update(Histo* h, const Int128& value, const Int128& mask)
{
	for (int i = 0; i < 8; i++)
	{
		uint8_t m = mask.get_byte(i);
		if (m)
		{
			uint8_t byte = value.get_byte(i);
			h->h[byte]++;
			h->size++;
		}
	}
}

static float histo_cost(Histo* h, const Int128& value, const Int128& mask)
{
	float tlb = (float)h->size;
	float cost = 1.0f;
	int count = 0;

	for (int i = 0; i < 16; i++)
	{
		if (mask.get_byte(i))
		{
			int c = h->h[value.get_byte(i)] + 1;
			tlb += 1;
			cost *= tlb / c;
			count++;
		}
	}

	return count > 0 ? log2_fast(cost) : 0.0f;
}

static void mtf_init(Mtf* mtf, int max_size)
{
	mtf->size = 0;
	mtf->max_size = max_size > MAX_MTF_SIZE ? MAX_MTF_SIZE : max_size;
}

static int mtf_search(Mtf* mtf, const Int128& value, const Int128& mask)
{
	// Pre-compute the masked value once
	Int128 masked_value = value.bitwise_and(mask);

	int i = 0;
	for (; i + 3 < mtf->size; i += 4)
	{
		// Check if any of these 4 entries match
		bool match0 = mtf->list[i + 0].bitwise_and(mask).is_equal(masked_value);
		bool match1 = mtf->list[i + 1].bitwise_and(mask).is_equal(masked_value);
		bool match2 = mtf->list[i + 2].bitwise_and(mask).is_equal(masked_value);
		bool match3 = mtf->list[i + 3].bitwise_and(mask).is_equal(masked_value);

		// Return the first matching position
		if (match0)
			return i;
		if (match1)
			return i + 1;
		if (match2)
			return i + 2;
		if (match3)
			return i + 3;
	}

	// Handle remaining entries
	for (; i < mtf->size; i++)
	{
		if (mtf->list[i].bitwise_and(mask).is_equal(masked_value))
			return i;
	}

	return -1;
}

static int mtf_encode(Mtf* mtf, const Int128& value, const Int128& mask)
{
	int pos = mtf_search(mtf, value, mask);

	if (pos == -1)
	{
		if (mtf->size < mtf->max_size)
			mtf->size++;
		pos = mtf->size - 1;
	}

	for (int i = pos; i > 0; i--)
		mtf->list[i] = mtf->list[i - 1];
	mtf->list[0] = value;

	return pos;
}

static inline float calculate_match_cost(int mtf_value, Int128 literal_value, Int128 mask, Mtf* mtf)
{
	// Constants for modeling LZMA compression characteristics
	const float MATCH_LEN_BITS = 2.0f;       // Base bits for match length encoding
	const float MATCH_DIST_BITS = 2.5f;      // Base bits for match distance encoding
	const float RECENT_DIST_DISCOUNT = 0.5f; // Discount for recently used distances
	const float ALIGN_BITS_COST = 0.5f;      // Cost for match position alignment bits
	const int NUM_REP_DISTS = 4;             // Number of recent distances to track
	const float REP_MATCH_DISCOUNT = 0.3f;   // Additional discount for repeated matches
	const float LONG_REP_DISCOUNT = 0.4f;    // Extra discount for long repeated matches

	// Calculate match length
	int match_length = 0;
	for (int i = 0; i < 16; i++)
	{
		if (mask.get_byte(i))
			match_length++;
	}

	// Base match length cost (varies with length)
	float len_cost = MATCH_LEN_BITS;
	if (match_length > 8)
	{
		len_cost += log2_fast(match_length - 7.f);
	}

	// Match distance cost
	float dist_cost = MATCH_DIST_BITS;
	if (mtf_value >= NUM_REP_DISTS)
	{
		// Normal match distance encoding
		dist_cost += log2_fast(mtf_value + 1.f);
		// Add alignment bits cost
		dist_cost += ALIGN_BITS_COST;
	}
	else
	{
		// Recent distance match (cheaper to encode)
		dist_cost *= RECENT_DIST_DISCOUNT;
	}

	// Check for repeated matches with proper masking
	bool is_rep_match = false;
	if (mtf_value > 0 && mtf->size > 0)
	{
		Int128 current_masked = literal_value.bitwise_and(mask);
		Int128 prev_masked = mtf->list[mtf_value - 1].bitwise_and(mask);
		is_rep_match = current_masked.is_equal(prev_masked);

		if (is_rep_match)
		{
			// Apply repeat match discount
			dist_cost *= REP_MATCH_DISCOUNT;
			// Additional discount for long repeated matches
			if (match_length > 4)
			{
				dist_cost *= LONG_REP_DISCOUNT;
			}
		}
	}

	return len_cost + dist_cost;
}

static float calculate_bit_cost_zip(int mtf_value_1, int mtf_value_2, const Int128& literal_value, Mtf* mtf_1, Mtf* mtf_2, const Int128& mask_1, const Int128& mask_2, Histo* histogram)
{
	// Constants for modeling ZIP compression characteristics
	const float MATCH_OVERHEAD = 3.0f;   // Bytes needed to encode match length/distance
	const float LITERAL_OVERHEAD = 0.1f; // Small overhead for literal runs
	const float MTF_DISCOUNT = 0.85f;    // Discount factor for recently used values
	const float REPEAT_DISCOUNT = 0.7f;  // Additional discount for repeated matches
	const int MAX_MATCH_BENEFIT = 16;    // Maximum benefit from a single match

	float cost = 0.0f;

	// Handle literal encoding when no MTF matches found
	if (mtf_value_1 == -1 && mtf_value_2 == -1)
	{
		// Use histogram-based cost for literal bytes
		cost = histo_cost(histogram, literal_value, mask_1.bitwise_or(mask_2));

		// Add literal overhead
		int literal_bytes = 0;
		for (int i = 0; i < 16; i++)
		{
			if (mask_1.get_byte(i) || mask_2.get_byte(i))
			{
				literal_bytes++;
			}
		}
		cost += LITERAL_OVERHEAD * literal_bytes;
		return cost;
	}

	// Calculate costs for matched portions
	/*
	if (mtf_value_1 == -1)
	{
	    float literal_cost = histo_cost(histogram, literal_value, mask_1);
	    int literal_bytes = 0;
	    for (int i = 0; i < 16; i++)
	    {
	        if (mask_1.get_byte(i))
	            literal_bytes++;
	    }
	    cost += literal_cost + LITERAL_OVERHEAD * literal_bytes;
	}
	else
	*/
	if (mtf_value_1 != -1)
	{
		// Base cost for match encoding
		float match_cost = MATCH_OVERHEAD;

		// Calculate match length (number of bytes matched)
		int match_length = 0;
		for (int i = 0; i < 16; i++)
		{
			if (mask_1.get_byte(i))
				match_length++;
		}

		// Adjust cost based on match position and length
		float position_cost = log2_fast(mtf_value_1 + 1.0f);

		// Apply MTF discount for recent matches
		if (mtf_value_1 < 4)
		{
			position_cost *= MTF_DISCOUNT;
		}

		// Check for repeated matches
		bool is_repeat = false;
		if (mtf_value_1 > 0 && mtf_1->size > 0)
		{
			Int128 current_masked = literal_value.bitwise_and(mask_1);
			Int128 prev_masked = mtf_1->list[mtf_value_1 - 1].bitwise_and(mask_1);
			is_repeat = current_masked.is_equal(prev_masked);
			if (is_repeat)
			{
				position_cost *= REPEAT_DISCOUNT;
			}
		}

		// Cap the benefit from long matches
		float match_benefit = (float)astc::min(match_length, MAX_MATCH_BENEFIT);
		cost += match_cost + position_cost - match_benefit;
	}

	// Handle remaining literal portion if only partial match
	if (mtf_value_2 == -1)
	{
		float literal_cost = histo_cost(histogram, literal_value, mask_2);
		int literal_bytes = 0;
		for (int i = 0; i < 16; i++)
		{
			if (mask_2.get_byte(i))
				literal_bytes++;
		}
		cost += literal_cost + LITERAL_OVERHEAD * literal_bytes;
	}
	else
	{
		// Handle second match similarly to first match
		float match_cost = MATCH_OVERHEAD;
		int match_length = 0;
		for (int i = 0; i < 16; i++)
		{
			if (mask_2.get_byte(i))
				match_length++;
		}

		float position_cost = log2_fast(mtf_value_2 + 1.0f);
		if (mtf_value_2 < 4)
		{
			position_cost *= MTF_DISCOUNT;
		}

		bool is_repeat = false;
		if (mtf_value_2 > 0 && mtf_2->size > 0)
		{
			Int128 current_masked = literal_value.bitwise_and(mask_2);
			Int128 prev_masked = mtf_2->list[mtf_value_2 - 1].bitwise_and(mask_2);
			is_repeat = current_masked.is_equal(prev_masked);
			if (is_repeat)
			{
				position_cost *= REPEAT_DISCOUNT;
			}
		}

		float match_benefit = (float)astc::min(match_length, MAX_MATCH_BENEFIT);
		cost += match_cost + position_cost - match_benefit;
	}

	return cost;
}

static float calculate_bit_cost_lzma(int mtf_value_1, int mtf_value_2, const Int128& literal_value, Mtf* mtf_1, Mtf* mtf_2, const Int128& mask_1, const Int128& mask_2, Histo* histogram)
{
	const float LITERAL_CONTEXT_BITS = 0.5f;        // Cost for literal context modeling
	const float CONSECUTIVE_MATCH_DISCOUNT = 0.8f;  // Cheaper to encode consecutive matches
	const float LITERAL_RUN_DISCOUNT = 0.9f;        // Cheaper to encode runs of literals
	const float LITERAL_AFTER_MATCH_PENALTY = 1.2f; // More expensive to switch to literals

	float cost = 0.0f;

	if (mtf_value_1 == -1 && mtf_value_2 == -1)
	{
		// Both literals - might be more efficient as a run
		float literal_cost = histo_cost(histogram, literal_value, mask_1.bitwise_or(mask_2));
		uint8_t prev_byte = 0;

		for (int i = 0; i < 16; i++)
		{
			if (mask_1.get_byte(i) || mask_2.get_byte(i))
			{
				cost += LITERAL_CONTEXT_BITS * (1.0f - log2_fast(histogram->h[prev_byte] + 1.f) / 8.0f);
				prev_byte = literal_value.get_byte(i);
			}
		}

		cost += literal_cost * LITERAL_RUN_DISCOUNT;
	}
	else if (mtf_value_1 != -1 && mtf_value_2 != -1)
	{
		// Both matches - second match might be cheaper
		cost += calculate_match_cost(mtf_value_1, literal_value, mask_1, mtf_1);
		cost += calculate_match_cost(mtf_value_2, literal_value, mask_2, mtf_2) * CONSECUTIVE_MATCH_DISCOUNT;
	}
	else if (mtf_value_1 == -1)
	{
		// Literal followed by match
		float literal_cost = histo_cost(histogram, literal_value, mask_1);
		uint8_t prev_byte = 0;

		for (int i = 0; i < 16; i++)
		{
			if (mask_1.get_byte(i))
			{
				cost += LITERAL_CONTEXT_BITS * (1.0f - log2_fast(histogram->h[prev_byte] + 1.f) / 8.0f);
				prev_byte = literal_value.get_byte(i);
			}
		}
		cost += literal_cost;

		// Match after literals
		cost += calculate_match_cost(mtf_value_2, literal_value, mask_2, mtf_2);
	}
	else
	{
		// Match followed by literal
		cost += calculate_match_cost(mtf_value_1, literal_value, mask_1, mtf_1);

		float literal_cost = histo_cost(histogram, literal_value, mask_2);
		uint8_t prev_byte = 0;
		for (int i = 0; i < 16; i++)
		{
			if (mask_2.get_byte(i))
			{
				cost += LITERAL_CONTEXT_BITS * (1.0f - log2_fast(histogram->h[prev_byte] + 1.f) / 8.0f);
				prev_byte = literal_value.get_byte(i);
			}
		}
		cost += literal_cost * LITERAL_AFTER_MATCH_PENALTY;
	}

	return cost;
}

static inline float calculate_ssd_weighted(const uint8_t* img1, const uint8_t* img2, int total, const float* weights, const vfloat4& channel_weights)
{
	vfloat4 sum0 = vfloat4::zero(), sum1 = vfloat4::zero(), sum2 = vfloat4::zero(), sum3 = vfloat4::zero();

	int i;
	for (i = 0; i < total - 15; i += 16)
	{
		vfloat4 diff = int_to_float(vint4(img1 + i)) - int_to_float(vint4(img2 + i));
		sum0 += diff * diff * weights[i >> 2];

		diff = int_to_float(vint4(img1 + i + 4)) - int_to_float(vint4(img2 + i + 4));
		sum1 += diff * diff * weights[(i >> 2) + 1];

		diff = int_to_float(vint4(img1 + i + 8)) - int_to_float(vint4(img2 + i + 8));
		sum2 += diff * diff * weights[(i >> 2) + 2];

		diff = int_to_float(vint4(img1 + i + 12)) - int_to_float(vint4(img2 + i + 12));
		sum3 += diff * diff * weights[(i >> 2) + 3];
	}

	for (; i < total; i += 4)
	{
		vfloat4 diff = int_to_float(vint4(img1 + i)) - int_to_float(vint4(img2 + i));
		sum0 += diff * diff * weights[i >> 2];
	}

	return dot_s((sum0 + sum1 + sum2 + sum3), channel_weights);
}

static inline float calculate_mrsse_weighted(const float* img1, const float* img2, int total, const float* weights, const vfloat4& channel_weights)
{
	vfloat4 sum0 = vfloat4::zero(), sum1 = vfloat4::zero(), sum2 = vfloat4::zero(), sum3 = vfloat4::zero();

	int i;
	for (i = 0; i < total - 15; i += 16)
	{
		vfloat4 diff = vfloat4(img1 + i) - vfloat4(img2 + i);
		sum0 += diff * diff * weights[i >> 2];

		diff = vfloat4(img1 + i + 4) - vfloat4(img2 + i + 4);
		sum1 += diff * diff * weights[(i >> 2) + 1];

		diff = vfloat4(img1 + i + 8) - vfloat4(img2 + i + 8);
		sum2 += diff * diff * weights[(i >> 2) + 2];

		diff = vfloat4(img1 + i + 12) - vfloat4(img2 + i + 12);
		sum3 += diff * diff * weights[(i >> 2) + 3];
	}

	for (; i < total; i += 4)
	{
		vfloat4 diff = vfloat4(img1 + i) - vfloat4(img2 + i);
		sum0 += diff * diff * weights[i >> 2];
	}

	// Combine all sums, apply channel weights, and scale by 256.0f
	return dot_s((sum0 + sum1 + sum2 + sum3), channel_weights) * 256.0f;
}

static void astc_decompress_block(const block_size_descriptor& bsd, const uint8_t* block_ptr, uint8_t* output, int block_width, int block_height, int block_depth, int block_type)
{
	image_block blk{};
	blk.texel_count = static_cast<uint8_t>(block_width * block_height * block_depth);
	blk.data_min = vfloat4::zero();
	blk.data_max = vfloat4(1.0f, 1.0f, 1.0f, 1.0f);
	blk.grayscale = false;
	blk.xpos = 0;
	blk.ypos = 0;
	blk.zpos = 0;

	symbolic_compressed_block scb;
	physical_to_symbolic(bsd, block_ptr, scb);

	astcenc_profile profile = block_type == ASTCENC_TYPE_U8 ? ASTCENC_PRF_LDR : ASTCENC_PRF_HDR;
	decompress_symbolic_block(profile, bsd, 0, 0, 0, scb, blk);

	// Convert the decompressed data to the output format
	for (int i = 0; i < blk.texel_count; i++)
	{
		vfloat4 color = vfloat4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);
		if (block_type == ASTCENC_TYPE_U8)
		{
			// Convert to 8-bit UNORM
			vint4 colori = float_to_int_rtn(color * 255.0f);
			output[i * 4 + 0] = static_cast<uint8_t>(colori.lane<0>());
			output[i * 4 + 1] = static_cast<uint8_t>(colori.lane<1>());
			output[i * 4 + 2] = static_cast<uint8_t>(colori.lane<2>());
			output[i * 4 + 3] = static_cast<uint8_t>(colori.lane<3>());
		}
		else
		{
			// Store as 32-bit float
			float* output_f = reinterpret_cast<float*>(output);
			output_f[i * 4 + 0] = color.lane<0>();
			output_f[i * 4 + 1] = color.lane<1>();
			output_f[i * 4 + 2] = color.lane<2>();
			output_f[i * 4 + 3] = color.lane<3>();
		}
	}
}

void process_and_recalculate_astc_block(const block_size_descriptor& bsd, uint8_t* block_ptr, int block_width, int block_height, int block_depth, int block_type, const uint8_t* original_decoded, const uint8_t* original_block_ptr)
{
	// Extract block information
	symbolic_compressed_block scb;
	physical_to_symbolic(bsd, block_ptr, scb);

	if (scb.block_type == SYM_BTYPE_CONST_U16 || scb.block_type == SYM_BTYPE_CONST_F16)
	{
		return;
	}

	// Get block mode and partition info
	block_mode mode = bsd.get_block_mode(scb.block_mode);
	const auto& pi = bsd.get_partition_info(scb.partition_count, scb.partition_index);
	const auto& di = bsd.get_decimation_info(mode.decimation_mode);

	bool is_dual_plane = mode.is_dual_plane != 0;
	unsigned int plane2_component = scb.plane2_component;

	// Initialize endpoints
	endpoints ep;
	ep.partition_count = scb.partition_count;

	// Initialize rgbs and rgbo vectors
	vfloat4 rgbs_vectors[BLOCK_MAX_PARTITIONS];
	vfloat4 rgbo_vectors[BLOCK_MAX_PARTITIONS];

	// Create an image_block from the pre-decoded data
	image_block blk;
	blk.texel_count = block_width * block_height * block_depth;
	for (int i = 0; i < blk.texel_count; i++)
	{
		if (block_type == ASTCENC_TYPE_U8)
		{
			blk.data_r[i] = original_decoded[i * 4 + 0] / 255.0f;
			blk.data_g[i] = original_decoded[i * 4 + 1] / 255.0f;
			blk.data_b[i] = original_decoded[i * 4 + 2] / 255.0f;
			blk.data_a[i] = original_decoded[i * 4 + 3] / 255.0f;
		}
		else
		{
			const float* original_decoded_f = reinterpret_cast<const float*>(original_decoded);
			blk.data_r[i] = original_decoded_f[i * 4 + 0];
			blk.data_g[i] = original_decoded_f[i * 4 + 1];
			blk.data_b[i] = original_decoded_f[i * 4 + 2];
			blk.data_a[i] = original_decoded_f[i * 4 + 3];
		}
	}

	// Set up other image_block properties
	blk.xpos = 0;
	blk.ypos = 0;
	blk.zpos = 0;
	blk.data_min = vfloat4::zero();
	blk.data_max = vfloat4(1.0f);
	blk.grayscale = false;

	// Compute channel means and set rgb_lns and alpha_lns flags
	vfloat4 ch_sum = vfloat4::zero();
	for (int i = 0; i < blk.texel_count; i++)
		ch_sum += blk.texel(i);
	blk.data_mean = ch_sum / static_cast<float>(blk.texel_count);
	blk.rgb_lns[0] = blk.data_max.lane<0>() > 1.0f || blk.data_max.lane<1>() > 1.0f || blk.data_max.lane<2>() > 1.0f;
	blk.alpha_lns[0] = blk.data_max.lane<3>() > 1.0f;

	// Set channel weights (assuming equal weights for simplicity)
	blk.channel_weight = vfloat4(1.0f);

	// Extract endpoints from the original block
	symbolic_compressed_block original_scb;
	physical_to_symbolic(bsd, original_block_ptr, original_scb);

	for (unsigned int i = 0; i < scb.partition_count; i++)
	{
		vint4 color0, color1;
		bool rgb_hdr, alpha_hdr;

		// Determine the appropriate astcenc_profile based on block_type
		astcenc_profile decode_mode = (block_type == ASTCENC_TYPE_U8) ? ASTCENC_PRF_LDR : ASTCENC_PRF_HDR;

		unpack_color_endpoints(decode_mode, original_scb.color_formats[i], original_scb.color_values[i], rgb_hdr, alpha_hdr, color0, color1);

		ep.endpt0[i] = int_to_float(color0) * (1.0f / 255.0f);
		ep.endpt1[i] = int_to_float(color1) * (1.0f / 255.0f);
	}

	if (is_dual_plane)
	{
		recompute_ideal_colors_2planes(blk, bsd, di, scb.weights, scb.weights + WEIGHTS_PLANE2_OFFSET, ep, rgbs_vectors[0], rgbo_vectors[0], plane2_component);
	}
	else
	{
		recompute_ideal_colors_1plane(blk, pi, di, scb.weights, ep, rgbs_vectors, rgbo_vectors);
	}

	// Update endpoints in the symbolic compressed block
	for (unsigned int i = 0; i < scb.partition_count; i++)
	{
		vfloat4 endpt0 = ep.endpt0[i];
		vfloat4 endpt1 = ep.endpt1[i];

		// Quantize endpoints
		vfloat4 color0 = endpt0 * 65535.0f;
		vfloat4 color1 = endpt1 * 65535.0f;

		quant_method quant_level = static_cast<quant_method>(mode.quant_mode);

		// Note/TODO: Pack Color endpoints doesn't support RGB_DELTA or
		// RGBA_DELTA
		uint8_t fmt = scb.color_formats[i];
		if (quant_level > QUANT_6 && (fmt == FMT_RGB || fmt == FMT_RGBA))
		{
			// Store quantized endpoints
			scb.color_formats[i] = pack_color_endpoints(color0, color1, rgbs_vectors[i], rgbo_vectors[i], fmt, scb.color_values[i], quant_level);
		}
	}

	// Compress to a physical block
	symbolic_to_physical(bsd, scb, block_ptr);
}

static void recompute_ideal_weights_1plane(const image_block& blk, const partition_info& pi, const decimation_info& di, uint8_t* weights, const endpoints& ep)
{
	int texels_per_block = blk.texel_count;
	promise(texels_per_block > 0);

	// Initialize weight accumulators for each stored weight
	std::vector<float> weight_accum(di.weight_count, 0.0f);
	std::vector<float> weight_contrib_sum(di.weight_count, 0.0f);

	// For each texel
	for (int i = 0; i < texels_per_block; i++)
	{
		int partition = pi.partition_of_texel[i];

		// Get the ideal weight for this texel using the current endpoints
		vfloat4 color0 = ep.endpt0[partition];
		vfloat4 color1 = ep.endpt1[partition];
		vfloat4 color_diff = color1 - color0;

		// Get the actual color we're trying to match
		vfloat4 color = vfloat4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);

		// Compute ideal weight using dot product method
		float weight_i = 0.0f;
		float len2 = dot_s(color_diff, color_diff * blk.channel_weight);
		if (len2 > 1e-10f)
		{
			weight_i = dot_s(color - color0, color_diff * blk.channel_weight) / len2;

			// Apply the same weight adjustment as the core encoder
			if (weight_i > 0.5f)
			{
				weight_i = (weight_i - 0.5f) * 1.2f + 0.5f;
			}

			weight_i = astc::clamp(weight_i, 0.0f, 1.0f);
		}

		// For each weight that contributes to this texel
		for (int j = 0; j < di.texel_weight_count[i]; j++)
		{
			int weight_idx = di.texel_weights_tr[j][i];
			float contrib = di.texel_weight_contribs_float_tr[j][weight_idx];

			// Apply channel weighting to match core encoder
			float channel_scale = hadd_s(blk.channel_weight) * 0.25f;
			weight_accum[weight_idx] += weight_i * contrib * channel_scale;
			weight_contrib_sum[weight_idx] += contrib * channel_scale;
		}
	}

	// Compute final weights in 0-64 range using accumulated contributions
	for (int i = 0; i < di.weight_count; i++)
	{
		float weight_avg = weight_accum[i] / weight_contrib_sum[i];

		if (weight_avg > 0.5f)
		{
			weight_avg = 0.5f + (weight_avg - 0.5f) * 1.2f;
		}

		weights[i] = static_cast<uint8_t>(astc::clamp(weight_avg * 64.0f + 0.5f, 0.0f, 64.0f));
	}
}

static void recompute_ideal_weights_2planes(const image_block& blk, const block_size_descriptor& bsd, const decimation_info& di, uint8_t* weights_plane1, uint8_t* weights_plane2, const endpoints& ep, int plane2_component)
{
	int texels_per_block = blk.texel_count;
	promise(texels_per_block > 0);

	// Initialize weight accumulators for each stored weight
	std::vector<float> weight_accum1(di.weight_count, 0.0f);
	std::vector<float> weight_accum2(di.weight_count, 0.0f);
	std::vector<float> weight_contrib_sum(di.weight_count, 0.0f);

	// Create plane masks
	vfloat4 plane1_mask = vfloat4(1.0f);
	plane1_mask.set_lane<0>(plane2_component == 0 ? 0.0f : 1.0f);
	plane1_mask.set_lane<1>(plane2_component == 1 ? 0.0f : 1.0f);
	plane1_mask.set_lane<2>(plane2_component == 2 ? 0.0f : 1.0f);
	plane1_mask.set_lane<3>(plane2_component == 3 ? 0.0f : 1.0f);

	// For each texel
	for (int i = 0; i < texels_per_block; i++)
	{
		vfloat4 color = vfloat4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);

		// Get the ideal weights for this texel
		vfloat4 color0 = ep.endpt0[0];
		vfloat4 color1 = ep.endpt1[0];

		// Compute ideal weight for plane 1
		float weight1_i = 0.0f;
		vfloat4 color_diff1 = (color1 - color0) * plane1_mask;
		float len2_1 = dot_s(color_diff1, color_diff1 * blk.channel_weight);
		if (len2_1 > 1e-10f)
		{
			weight1_i = dot_s((color - color0) * plane1_mask, color_diff1 * blk.channel_weight) / len2_1;
			if (weight1_i > 0.5f)
			{
				weight1_i = (weight1_i - 0.5f) * 1.2f + 0.5f;
			}
			weight1_i = astc::clamp(weight1_i, 0.0f, 1.0f);
		}

		// Compute ideal weight for plane 2
		float weight2_i = 0.0f;
		float color0_p2 = color0.lane<0>() * (plane2_component == 0) + color0.lane<1>() * (plane2_component == 1) + color0.lane<2>() * (plane2_component == 2) + color0.lane<3>() * (plane2_component == 3);
		float color1_p2 = color1.lane<0>() * (plane2_component == 0) + color1.lane<1>() * (plane2_component == 1) + color1.lane<2>() * (plane2_component == 2) + color1.lane<3>() * (plane2_component == 3);
		float color_p2 = color.lane<0>() * (plane2_component == 0) + color.lane<1>() * (plane2_component == 1) + color.lane<2>() * (plane2_component == 2) + color.lane<3>() * (plane2_component == 3);

		float diff_p2 = color1_p2 - color0_p2;
		if (fabsf(diff_p2) > 1e-10f)
		{
			weight2_i = (color_p2 - color0_p2) / diff_p2;
			if (weight2_i > 0.5f)
			{
				weight2_i = (weight2_i - 0.5f) * 1.2f + 0.5f;
			}
			weight2_i = astc::clamp(weight2_i, 0.0f, 1.0f);
		}

		// For each weight that contributes to this texel
		for (int j = 0; j < di.texel_weight_count[i]; j++)
		{
			int weight_idx = di.texel_weights_tr[j][i];
			float contrib = di.texel_weight_contribs_float_tr[j][weight_idx];

			float channel_scale = hadd_s(blk.channel_weight) * 0.25f;
			weight_accum1[weight_idx] += weight1_i * contrib * channel_scale;
			weight_accum2[weight_idx] += weight2_i * contrib * channel_scale;
			weight_contrib_sum[weight_idx] += contrib * channel_scale;
		}
	}

	// Compute final weights in 0-64 range
	for (int i = 0; i < di.weight_count; i++)
	{
		float weight_avg1 = weight_accum1[i] / weight_contrib_sum[i];
		float weight_avg2 = weight_accum2[i] / weight_contrib_sum[i];

		if (weight_avg1 > 0.5f)
		{
			weight_avg1 = 0.5f + (weight_avg1 - 0.5f) * 1.2f;
		}
		if (weight_avg2 > 0.5f)
		{
			weight_avg2 = 0.5f + (weight_avg2 - 0.5f) * 1.2f;
		}

		weights_plane1[i] = static_cast<uint8_t>(astc::clamp(weight_avg1 * 64.0f + 0.5f, 0.0f, 64.0f));
		weights_plane2[i] = static_cast<uint8_t>(astc::clamp(weight_avg2 * 64.0f + 0.5f, 0.0f, 64.0f));
	}
}

static void optimize_weights_for_endpoints(const block_size_descriptor& bsd, const uint8_t* block_ptr, uint8_t* output_block, const uint8_t* original_decoded, int block_width, int block_height, int block_depth, int block_type, const float* gradients, const vfloat4& channel_weights)
{
	// Extract block information
	symbolic_compressed_block scb;
	physical_to_symbolic(bsd, block_ptr, scb);

	// Skip constant color blocks
	if (scb.block_type == SYM_BTYPE_CONST_U16 || scb.block_type == SYM_BTYPE_CONST_F16)
	{
		memcpy(output_block, block_ptr, 16);
		return;
	}

	// Create image block from original decoded data
	image_block blk;
	blk.texel_count = block_width * block_height * block_depth;
	for (int i = 0; i < blk.texel_count; i++)
	{
		if (block_type == ASTCENC_TYPE_U8)
		{
			blk.data_r[i] = original_decoded[i * 4 + 0] / 255.0f;
			blk.data_g[i] = original_decoded[i * 4 + 1] / 255.0f;
			blk.data_b[i] = original_decoded[i * 4 + 2] / 255.0f;
			blk.data_a[i] = original_decoded[i * 4 + 3] / 255.0f;
		}
		else
		{
			const float* original_decoded_f = reinterpret_cast<const float*>(original_decoded);
			blk.data_r[i] = original_decoded_f[i * 4 + 0];
			blk.data_g[i] = original_decoded_f[i * 4 + 1];
			blk.data_b[i] = original_decoded_f[i * 4 + 2];
			blk.data_a[i] = original_decoded_f[i * 4 + 3];
		}
	}

	// Set up block properties
	blk.xpos = 0;
	blk.ypos = 0;
	blk.zpos = 0;
	blk.data_min = vfloat4::zero();
	blk.data_max = vfloat4(1.0f);
	blk.grayscale = false;
	blk.channel_weight = channel_weights;

	// Get block mode and partition info
	block_mode mode = bsd.get_block_mode(scb.block_mode);
	const auto& pi = bsd.get_partition_info(scb.partition_count, scb.partition_index);
	const auto& di = bsd.get_decimation_info(mode.decimation_mode);

	// Initialize endpoints
	endpoints ep;
	ep.partition_count = scb.partition_count;

	// Extract endpoints from the input block
	astcenc_profile decode_mode = (block_type == ASTCENC_TYPE_U8) ? ASTCENC_PRF_LDR : ASTCENC_PRF_HDR;
	for (unsigned int i = 0; i < scb.partition_count; i++)
	{
		vint4 color0, color1;
		bool rgb_hdr, alpha_hdr;
		unpack_color_endpoints(decode_mode, scb.color_formats[i], scb.color_values[i], rgb_hdr, alpha_hdr, color0, color1);

		ep.endpt0[i] = int_to_float(color0) * (1.0f / 65535.0f);
		ep.endpt1[i] = int_to_float(color1) * (1.0f / 65535.0f);
	}

	// Recompute weights
	if (mode.is_dual_plane)
	{
		recompute_ideal_weights_2planes(blk, bsd, di, scb.weights, scb.weights + WEIGHTS_PLANE2_OFFSET, ep, scb.plane2_component);
	}
	else
	{
		recompute_ideal_weights_1plane(blk, pi, di, scb.weights, ep);
	}

	// Check if weights actually improved the error
	uint8_t temp_block[16];
	symbolic_to_physical(bsd, scb, temp_block);
	memcpy(output_block, temp_block, 16);
}

int get_weight_bits(uint8_t* data, int block_width, int block_height, int block_depth)
{
	uint16_t mode = data[0] | (data[1] << 8);

	if ((mode & 0x1ff) == 0x1fc)
		return 0; // void-extent
	if ((mode & 0x00f) == 0)
		return 0; // Reserved

	uint8_t b01 = (mode >> 0) & 3;
	uint8_t b23 = (mode >> 2) & 3;
	uint8_t p0 = (mode >> 4) & 1;
	uint8_t b56 = (mode >> 5) & 3;
	uint8_t b78 = (mode >> 7) & 3;
	uint8_t P = (mode >> 9) & 1;
	uint8_t Dp = (mode >> 10) & 1;
	uint8_t b9_10 = (mode >> 9) & 3;
	uint8_t p12;

	int W, H, D;
	if (block_depth <= 1)
	{
		// 2D
		D = 1;
		if ((mode & 0x1c3) == 0x1c3)
			return 0; // Reserved*
		if (b01 == 0)
		{
			p12 = b23;
			switch (b78)
			{
			case 0:
				W = 12;
				H = 2 + b56;
				break;
			case 1:
				W = 2 + b56;
				H = 12;
				break;
			case 2:
				W = 6 + b56;
				H = 6 + b9_10;
				Dp = 0;
				P = 0;
				break;
			case 3:
				if (b56 == 0)
				{
					W = 6;
					H = 10;
				}
				else if (b56 == 1)
				{
					W = 10;
					H = 6;
				}
				else
				{
					/* NOTREACHED */
					// assert(0);
					return 0;
				}
				break;
			}
		}
		else
		{
			p12 = b01;
			switch (b23)
			{
			case 0:
				W = 4 + b78;
				H = 2 + b56;
				break;
			case 1:
				W = 8 + b78;
				H = 2 + b56;
				break;
			case 2:
				W = 2 + b56;
				H = 8 + b78;
				break;
			case 3:
				if (b78 & 2)
				{
					W = 2 + (b78 & 1);
					H = 6 + b56;
				}
				else
				{
					W = 2 + b56;
					H = 2 + (b78 & 1);
				}
				break;
			}
		}
	}
	else
	{
		// 3D
		if ((mode & 0x1e3) == 0x1e3)
			return 0; // Reserved*
		if (b01 != 0)
		{
			p12 = b01;
			W = 2 + b56;
			H = 2 + b78;
			D = 2 + b23;
		}
		else
		{
			p12 = b23;
			switch (b78)
			{
			case 0:
				W = 6;
				H = 2 + b9_10;
				D = 2 + b56;
				break;
			case 1:
				W = 2 + b56;
				H = 6;
				D = 2 + b9_10;
				break;
			case 2:
				W = 2 + b56;
				H = 2 + b9_10;
				D = 6;
				break;
			case 3:
				switch (b56)
				{
				case 0:
					W = 6;
					H = 2;
					D = 2;
					break;
				case 1:
					W = 2;
					H = 6;
					D = 2;
					break;
				case 2:
					W = 2;
					H = 2;
					D = 6;
					break;
				case 3:
					/* NOTREACHED*/
					assert(0);
					return 0;
				}
				break;
			}
		}
	}
	// error cases
	if (W > block_width)
		return 0;
	if (H > block_height)
		return 0;
	if (D > block_depth)
		return 0;

	uint8_t p = (p12 << 1) | p0;
	int trits = 0, quints = 0, bits = 0;
	if (!P)
	{
		int t[8] = {-1, -1, 0, 1, 0, 0, 1, 0};
		int q[8] = {-1, -1, 0, 0, 0, 1, 0, 0};
		int b[8] = {-1, -1, 1, 0, 2, 0, 1, 3};
		trits = t[p];
		quints = q[p];
		bits = b[p];
	}
	else
	{
		int t[8] = {-1, -1, 0, 1, 0, 0, 1, 0};
		int q[8] = {-1, -1, 1, 0, 0, 1, 0, 0};
		int b[8] = {-1, -1, 1, 2, 4, 2, 3, 5};
		trits = t[p];
		quints = q[p];
		bits = b[p];
	}

	int num_weights = W * H * D;
	if (Dp)
		num_weights *= 2;

	if (num_weights > 64)
		return 0;

	int weight_bits = (num_weights * 8 * trits + 4) / 5 + (num_weights * 7 * quints + 2) / 3 + num_weights * bits;

	// error cases
	if (weight_bits < 24 || weight_bits > 96)
		return 0;

	return (uint8_t)weight_bits;
}

#if 0
extern int hack_bits_for_weights;
void test_weight_bits(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth)
{
    block_size_descriptor *bsd = (block_size_descriptor*) malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    for (size_t i=0; i < data_len; i += 16) 
    {
        uint8_t *block = data + i;
        uint8_t decoded[6*6*6*4];
        astc_decompress_block(*bsd, block, decoded, block_width, block_height, block_depth, ASTCENC_TYPE_U8);
        int bits = get_weight_bits(block, block_width, block_height, block_depth);
	    if (bits != hack_bits_for_weights) 
        {
		    printf("Internal error: decoded weight bits count didn't match\n");
	    }
    }
    free(bsd);
}
#endif

struct WorkItem
{
	size_t start_block;
	size_t end_block;
	bool is_forward;
};

static inline void calculate_masks(int weight_bits, Int128& weights_mask, Int128& endpoints_mask)
{
	Int128 one = Int128::from_int(1);
	weights_mask = one.shift_left(weight_bits).subtract(one).shift_left(128 - weight_bits);
	endpoints_mask = weights_mask.bitwise_not();
}

struct BitsAndWeightBits
{
	Int128 bits;
	int weight_bits;
};

static inline BitsAndWeightBits get_bits_and_weight_bits(const uint8_t* block, const uint8_t* weight_bits_tbl)
{
	BitsAndWeightBits result;
	result.bits = Int128(block);
	result.weight_bits = weight_bits_tbl[(block[0] | (block[1] << 8)) & MODE_MASK];
	return result;
}

struct RDError
{
	float mse_error;
	float rate_error;
};

static void dual_mtf_pass(uint8_t* data, uint8_t* ref1, uint8_t* ref2, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, block_size_descriptor* bsd, uint8_t* all_original_decoded, float* all_gradients, vfloat4 channel_weights)
{
	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	const int num_threads = astc::min(MAX_THREADS, (int)std::thread::hardware_concurrency());
	RDError* error_buffer = (RDError*)calloc(blocks_x * blocks_y * blocks_z, sizeof(RDError));

	// Initialize weight bits table
	uint8_t* weight_bits_tbl = (uint8_t*)malloc(2048);
	for (size_t i = 0; i < 2048; ++i)
	{
		uint8_t block[16];
		block[0] = (i & 255);
		block[1] = ((i >> 8) & 255);
		weight_bits_tbl[i] = get_weight_bits(&block[0], block_width, block_height, block_depth);
	}

	std::queue<WorkItem> work_queue;
	std::mutex queue_mutex;
	std::condition_variable cv;
	bool all_work_done = false;

	auto thread_function = [&](int thread_id)
	{
		CachedBlock* block_cache = (CachedBlock*)calloc(CACHE_SIZE, sizeof(CachedBlock));
		Mtf mtf_weights;
		Mtf mtf_endpoints;
		Histo histogram;

		mtf_init(&mtf_weights, MAX_MTF_SIZE);
		mtf_init(&mtf_endpoints, MAX_MTF_SIZE);
		histo_reset(&histogram);

		// Use shared error_buffer instead of thread-specific one
		auto propagate_error = [&](long long x, long long y, long long z, float mse_diff, float rate_diff, bool is_forward)
		{
			// Calculate the work group boundary
			long long current_block_idx = z * blocks_x * blocks_y + y * blocks_x + x;
			long long work_group_start = (current_block_idx / MAX_BLOCKS_PER_ITEM) * MAX_BLOCKS_PER_ITEM;
			long long work_group_end = astc::min(work_group_start + MAX_BLOCKS_PER_ITEM, (long long)num_blocks);

			// Don't propagate if target block would be outside current work group
			if (x < 0 || x >= blocks_x || y < 0 || y >= blocks_y || z < 0 || z >= blocks_z)
				return;

			long long block_idx = z * blocks_x * blocks_y + y * blocks_x + x;
			if (block_idx < work_group_start || block_idx >= work_group_end)
				return;

			// Error filter coefficients (similar to Floyd-Steinberg)
			const float error_weights[4] = {7.0f / 16.0f, 3.0f / 16.0f, 5.0f / 16.0f, 1.0f / 16.0f};

			if (is_forward)
			{
				// Forward pass error diffusion pattern
				error_buffer[block_idx].mse_error += mse_diff * error_weights[0];
				error_buffer[block_idx].rate_error += rate_diff * error_weights[0];

				if (x + 1 < blocks_x && block_idx + 1 < work_group_end)
				{
					error_buffer[block_idx + 1].mse_error += mse_diff * error_weights[1];
					error_buffer[block_idx + 1].rate_error += rate_diff * error_weights[1];
				}
				if (y + 1 < blocks_y && block_idx + blocks_x < work_group_end)
				{
					error_buffer[block_idx + blocks_x].mse_error += mse_diff * error_weights[2];
					error_buffer[block_idx + blocks_x].rate_error += rate_diff * error_weights[2];
				}
				if (x + 1 < blocks_x && y + 1 < blocks_y && block_idx + blocks_x + 1 < work_group_end)
				{
					error_buffer[block_idx + blocks_x + 1].mse_error += mse_diff * error_weights[3];
					error_buffer[block_idx + blocks_x + 1].rate_error += rate_diff * error_weights[3];
				}
			}
			else
			{
				// Backward pass error diffusion pattern
				error_buffer[block_idx].mse_error += mse_diff * error_weights[0];
				error_buffer[block_idx].rate_error += rate_diff * error_weights[0];

				if (x > 0 && block_idx - 1 >= work_group_start)
				{
					error_buffer[block_idx - 1].mse_error += mse_diff * error_weights[1];
					error_buffer[block_idx - 1].rate_error += rate_diff * error_weights[1];
				}
				if (y > 0 && block_idx - blocks_x >= work_group_start)
				{
					error_buffer[block_idx - blocks_x].mse_error += mse_diff * error_weights[2];
					error_buffer[block_idx - blocks_x].rate_error += rate_diff * error_weights[2];
				}
				if (x > 0 && y > 0 && block_idx - blocks_x - 1 >= work_group_start)
				{
					error_buffer[block_idx - blocks_x - 1].mse_error += mse_diff * error_weights[3];
					error_buffer[block_idx - blocks_x - 1].rate_error += rate_diff * error_weights[3];
				}
			}
		};

		auto process_block = [&](size_t block_index, bool is_forward)
		{
			size_t z = block_index / (blocks_x * blocks_y);
			size_t rem = block_index % (blocks_x * blocks_y);
			size_t y = rem / blocks_x;
			size_t x = rem % blocks_x;

			uint8_t* current_block = data + block_index * block_size;
			BitsAndWeightBits current = get_bits_and_weight_bits(current_block, weight_bits_tbl);
			Int128 current_bits = current.bits;
			int current_weight_bits = current.weight_bits;
			Int128 best_match = current_bits;

			Int128 weights_mask, endpoints_mask;
			calculate_masks(current_weight_bits, weights_mask, endpoints_mask);

			uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

			// Function to get or compute MSE for a candidate
			auto get_or_compute_mse = [&](const Int128& candidate_bits) -> float
			{
				uint32_t hash = hash_128(candidate_bits) & (CACHE_SIZE - 1); // Modulo CACHE_SIZE

				// Check if the candidate is in the cache
				if (block_cache[hash].valid && block_cache[hash].encoded.is_equal(candidate_bits))
				{
					// Compute MSE using cached decoded data
					if (block_type == ASTCENC_TYPE_U8)
					{
						return calculate_ssd_weighted(original_decoded, block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}
					return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
				}

				// If not in cache, compute and cache the result
				uint8_t temp_block[16];
				candidate_bits.to_bytes(temp_block);

				// Decode and compute MSE
				astc_decompress_block(*bsd, temp_block, block_cache[hash].decoded, block_width, block_height, block_depth, block_type);
				block_cache[hash].encoded = candidate_bits;
				block_cache[hash].valid = true;

				if (block_type == ASTCENC_TYPE_U8)
				{
					return calculate_ssd_weighted(original_decoded, block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
				}
				return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
			};

			// Decode the original block to compute initial MSE
			float original_mse = get_or_compute_mse(current_bits);

			int mtf_weights_pos = mtf_search(&mtf_weights, current_bits, weights_mask);
			int mtf_endpoints_pos = mtf_search(&mtf_endpoints, current_bits, endpoints_mask);

			// Before computing best_rd_cost, get the propagated error
			RDError propagated = error_buffer[block_index];
			float adjusted_mse = original_mse + propagated.mse_error;
			float adjusted_rate = MEASURE_RATE(mtf_weights_pos, mtf_endpoints_pos, current_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
			adjusted_rate += propagated.rate_error;

			float best_rd_cost = adjusted_mse + lambda * adjusted_rate;

			struct Candidate
			{
				Int128 bits;
				float rd_cost;
				int mtf_position;
				int mode;
				int weight_bits;
			};
			Candidate best_weights[BEST_CANDIDATES_COUNT];
			Candidate best_endpoints[BEST_CANDIDATES_COUNT];
			int endpoints_count = 0;
			int weights_count = 0;

			auto add_candidate = [&](Candidate* candidates, int& count, const Int128& bits, float rd_cost, int mtf_position)
			{
				if (count < BEST_CANDIDATES_COUNT || rd_cost < candidates[BEST_CANDIDATES_COUNT - 1].rd_cost)
				{
					int insert_pos = count < BEST_CANDIDATES_COUNT ? count : BEST_CANDIDATES_COUNT - 1;

					// Find the position to insert
					while (insert_pos > 0 && rd_cost < candidates[insert_pos - 1].rd_cost)
					{
						candidates[insert_pos] = candidates[insert_pos - 1];
						insert_pos--;
					}

					int mode = (bits.get_byte(0) | (bits.get_byte(1) << 8)) & MODE_MASK;
					int weight_bits = get_bits_and_weight_bits((uint8_t*)&bits, weight_bits_tbl).weight_bits;
					candidates[insert_pos] = {bits, rd_cost, mtf_position, mode, weight_bits};

					if (count < BEST_CANDIDATES_COUNT)
						count++;
				}
			};

			// Add the current block to the candidates
			add_candidate(best_weights, weights_count, current_bits, original_mse + lambda * MEASURE_RATE(mtf_weights_pos, mtf_endpoints_pos, current_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram), mtf_weights_pos);
			add_candidate(best_endpoints, endpoints_count, current_bits, original_mse + lambda * MEASURE_RATE(mtf_weights_pos, mtf_endpoints_pos, current_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram), mtf_endpoints_pos);

			// Replace the ref1 and ref2 bit extraction with the helper function
			BitsAndWeightBits ref1_wb = get_bits_and_weight_bits(ref1 + block_index * block_size, weight_bits_tbl);
			Int128 ref1_bits = ref1_wb.bits;
			int ref1_weight_bits = ref1_wb.weight_bits;
			Int128 ref1_weight_mask, ref1_endpoint_mask;
			calculate_masks(ref1_weight_bits, ref1_weight_mask, ref1_endpoint_mask);
			int mtf_weights_pos_ref1 = mtf_search(&mtf_weights, ref1_bits, ref1_weight_mask);
			int mtf_endpoints_pos_ref1 = mtf_search(&mtf_endpoints, ref1_bits, ref1_endpoint_mask);
			float ref1_mse = get_or_compute_mse(ref1_bits);
			add_candidate(best_weights, weights_count, ref1_bits, ref1_mse + lambda * MEASURE_RATE(mtf_weights_pos_ref1, mtf_endpoints_pos_ref1, ref1_bits, &mtf_weights, &mtf_endpoints, ref1_weight_mask, ref1_endpoint_mask, &histogram), mtf_weights_pos_ref1);
			add_candidate(best_endpoints, endpoints_count, ref1_bits, ref1_mse + lambda * MEASURE_RATE(mtf_weights_pos_ref1, mtf_endpoints_pos_ref1, ref1_bits, &mtf_weights, &mtf_endpoints, ref1_weight_mask, ref1_endpoint_mask, &histogram), mtf_endpoints_pos_ref1);

			// Add ref2
			BitsAndWeightBits ref2_wb = get_bits_and_weight_bits(ref2 + block_index * block_size, weight_bits_tbl);
			Int128 ref2_bits = ref2_wb.bits;
			int ref2_weight_bits = ref2_wb.weight_bits;
			Int128 ref2_weight_mask, ref2_endpoint_mask;
			calculate_masks(ref2_weight_bits, ref2_weight_mask, ref2_endpoint_mask);
			int mtf_weights_pos_ref2 = mtf_search(&mtf_weights, ref2_bits, ref2_weight_mask);
			int mtf_endpoints_pos_ref2 = mtf_search(&mtf_endpoints, ref2_bits, ref2_endpoint_mask);
			float ref2_mse = get_or_compute_mse(ref2_bits);
			add_candidate(best_weights, weights_count, ref2_bits, ref2_mse + lambda * MEASURE_RATE(mtf_weights_pos_ref2, mtf_endpoints_pos_ref2, ref2_bits, &mtf_weights, &mtf_endpoints, ref2_weight_mask, ref2_endpoint_mask, &histogram), mtf_weights_pos_ref2);
			add_candidate(best_endpoints, endpoints_count, ref2_bits, ref2_mse + lambda * MEASURE_RATE(mtf_weights_pos_ref2, mtf_endpoints_pos_ref2, ref2_bits, &mtf_weights, &mtf_endpoints, ref2_weight_mask, ref2_endpoint_mask, &histogram), mtf_endpoints_pos_ref2);

			// Find best endpoint candidates
			for (int k = 0; k < mtf_endpoints.size; k++)
			{
				Int128 candidate_endpoints = mtf_endpoints.list[k];
				int endpoints_weight_bits = get_bits_and_weight_bits((uint8_t*)&candidate_endpoints, weight_bits_tbl).weight_bits;

				Int128 weights_mask, endpoints_mask;
				calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);

				float mse = get_or_compute_mse(candidate_endpoints);

				// Find the corresponding weight position
				int weight_pos = mtf_search(&mtf_weights, candidate_endpoints, weights_mask);

				float bit_cost = MEASURE_RATE(weight_pos, k, candidate_endpoints, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
				float rd_cost = mse + lambda * bit_cost;

				// Insert into best_endpoints if it's one of the best candidates
				add_candidate(best_endpoints, endpoints_count, candidate_endpoints, rd_cost, k);

#if 0 // Worse results... TODO/Fixme
				uint8_t temp_block[16];
				optimize_weights_for_endpoints(*bsd, (uint8_t*)&candidate_endpoints, temp_block, original_decoded, block_width, block_height, block_depth, block_type, all_gradients, channel_weights);

				float mse2 = get_or_compute_mse(Int128(temp_block));
				int weight_pos2 = mtf_search(&mtf_weights, Int128(temp_block), weights_mask);
				float bit_cost2 = MEASURE_RATE(weight_pos2, k, Int128(temp_block), &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
				float rd_cost2 = mse2 + lambda * bit_cost2;

				add_candidate(best_endpoints, endpoints_count, Int128(temp_block), rd_cost2, k);
#endif
			}

			// Find best weight candidates
			for (int k = 0; k < mtf_weights.size; k++)
			{
				Int128 candidate_weights = mtf_weights.list[k];
				int weights_weight_bits = get_bits_and_weight_bits((uint8_t*)&candidate_weights, weight_bits_tbl).weight_bits;

				Int128 weights_mask, endpoints_mask;
				calculate_masks(weights_weight_bits, weights_mask, endpoints_mask);
				Int128 temp_bits = candidate_weights.bitwise_and(weights_mask);

				// Try every endpoint candidate that matches in weight bits
				for (int m = 0; m < endpoints_count; m++)
				{
					int endpoint_weight_bits = best_endpoints[m].weight_bits;
					if (weights_weight_bits == endpoint_weight_bits)
					{
						Int128 combined_bits = temp_bits.bitwise_or(best_endpoints[m].bits.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(combined_bits);
						float bit_cost = MEASURE_RATE(k, best_endpoints[m].mtf_position, combined_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
						float rd_cost = mse + lambda * bit_cost;

						// Insert into best_weights if it's one of the best candidates
						add_candidate(best_weights, weights_count, candidate_weights, rd_cost, k);
					}
				}
			}

			// Search through combinations of best candidates
			for (int i = 0; i < weights_count; i++)
			{
				for (int j = 0; j < endpoints_count; j++)
				{
					Int128 candidate_endpoints = best_endpoints[j].bits;
					int endpoints_weight_bits = best_endpoints[j].weight_bits;
					Int128 weights_mask, endpoints_mask;
					calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);

					// Try candidate_endpoints as-is first
					float mse = get_or_compute_mse(candidate_endpoints);
					float rd_cost = mse + lambda * MEASURE_RATE(mtf_search(&mtf_weights, candidate_endpoints, weights_mask), best_endpoints[j].mtf_position, candidate_endpoints, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
					if (rd_cost < best_rd_cost)
					{
						best_match = candidate_endpoints;
						best_rd_cost = rd_cost;
					}

					// Then try the candidate_endpoints with candidate weights weights

					Int128 temp_bits = best_weights[i].bits.bitwise_and(weights_mask).bitwise_or(best_endpoints[j].bits.bitwise_and(endpoints_mask));
					mse = get_or_compute_mse(temp_bits);
					rd_cost = mse + lambda * MEASURE_RATE(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
					if (rd_cost < best_rd_cost)
					{
						best_match = temp_bits;
						best_rd_cost = rd_cost;
					}

					// now do the same thing for weights
					Int128 candidate_weights = best_weights[i].bits;
					int weights_weight_bits = best_weights[i].weight_bits;
					calculate_masks(weights_weight_bits, weights_mask, endpoints_mask);

					temp_bits = candidate_weights.bitwise_and(weights_mask).bitwise_or(best_endpoints[j].bits.bitwise_and(endpoints_mask));
					mse = get_or_compute_mse(temp_bits);
					rd_cost = mse + lambda * MEASURE_RATE(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
					if (rd_cost < best_rd_cost)
					{
						best_match = temp_bits;
						best_rd_cost = rd_cost;
					}
				}
			}

			if (!best_match.is_equal(current_bits))
				best_match.to_bytes(current_block);

			// Recalculate masks for the best match
			BitsAndWeightBits best_match_wb = get_bits_and_weight_bits(current_block, weight_bits_tbl);
			int best_weight_bits = best_match_wb.weight_bits;
			Int128 best_weights_mask, best_endpoints_mask;
			calculate_masks(best_weight_bits, best_weights_mask, best_endpoints_mask);

			// Update histogram with literal mask
			int best_mtf_weights_pos = mtf_search(&mtf_weights, best_match, best_weights_mask);
			int best_mtf_endpoints_pos = mtf_search(&mtf_endpoints, best_match, best_endpoints_mask);
			Int128 literal_mask = Int128::from_int(0);
			if (best_mtf_weights_pos == -1)
				literal_mask = literal_mask.bitwise_or(best_weights_mask);
			if (best_mtf_endpoints_pos == -1)
				literal_mask = literal_mask.bitwise_or(best_endpoints_mask);
			histo_update(&histogram, best_match, literal_mask);
			mtf_encode(&mtf_weights, best_match, best_weights_mask);
			mtf_encode(&mtf_endpoints, best_match, best_endpoints_mask);

			// After finding the best match, propagate the error
			float final_mse = get_or_compute_mse(best_match);
			float final_rate = MEASURE_RATE(mtf_search(&mtf_weights, best_match, best_weights_mask), mtf_search(&mtf_endpoints, best_match, best_endpoints_mask), best_match, &mtf_weights, &mtf_endpoints, best_weights_mask, best_endpoints_mask, &histogram);

			float mse_diff = final_mse - adjusted_mse;
			float rate_diff = final_rate - adjusted_rate;

			propagate_error(x, y, z, mse_diff, rate_diff, is_forward);
		};

		while (true)
		{
			WorkItem work_item;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cv.wait(lock, [&]() { return !work_queue.empty() || all_work_done; });
				if (all_work_done && work_queue.empty())
				{
					free(block_cache);
					return;
				}
				if (!work_queue.empty())
				{
					work_item = work_queue.front();
					work_queue.pop();
				}
			}

			// Reset MTF structures for each work item
			mtf_init(&mtf_weights, MAX_MTF_SIZE);
			mtf_init(&mtf_endpoints, MAX_MTF_SIZE);

			// Process the work item
			if (work_item.is_forward)
				for (size_t i = work_item.start_block; i < work_item.end_block; i++)
					process_block(i, true);
			else
				for (size_t i = work_item.end_block; i-- > work_item.start_block;)
					process_block(i, false);
		}
	};

	// Function to run a single pass (backward or forward)
	auto run_pass = [&](bool is_forward)
	{
		// Fill the work queue
		for (size_t start_block = 0; start_block < num_blocks; start_block += MAX_BLOCKS_PER_ITEM)
		{
			size_t end_block = astc::min(start_block + MAX_BLOCKS_PER_ITEM, num_blocks);
			work_queue.push({start_block, end_block, is_forward});
		}

		std::vector<std::thread> threads;

		// Start threads
		for (int i = 0; i < num_threads; ++i)
			threads.emplace_back(thread_function, i);

		// Wait for all threads to finish
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			all_work_done = true;
		}
		cv.notify_all();

		for (auto& thread : threads)
			thread.join();

		// Reset for next pass
		all_work_done = false;
		work_queue = std::queue<WorkItem>();
	};

	// Run backward pass
	run_pass(false);

	// Run forward pass
	run_pass(true);

	free(error_buffer);
	free(weight_bits_tbl);
}

template <typename T>
void reconstruct_image(T* all_original_decoded, int width, int height, int depth, int block_width, int block_height, int block_depth, T* output_image)
{
	int blocks_x = (width + block_width - 1) / block_width;
	int blocks_y = (height + block_height - 1) / block_height;
	int blocks_z = (depth + block_depth - 1) / block_depth;
	int channels = 4;

	for (int z = 0; z < blocks_z; z++)
	{
		for (int y = 0; y < blocks_y; y++)
		{
			for (int x = 0; x < blocks_x; x++)
			{
				int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
				T* block_data = all_original_decoded + block_index * block_width * block_height * block_depth * channels;

				for (int bz = 0; bz < block_depth; bz++)
				{
					for (int by = 0; by < block_height; by++)
					{
						for (int bx = 0; bx < block_width; bx++)
						{
							int image_x = x * block_width + bx;
							int image_y = y * block_height + by;
							int image_z = z * block_depth + bz;

							if (image_x < width && image_y < height && image_z < depth)
							{
								int image_index = (image_z * height * width + image_y * width + image_x) * channels;
								int block_pixel_index = (bz * block_height * block_width + by * block_width + bx) * channels;

								for (int c = 0; c < channels; c++)
								{
									output_image[image_index + c] = block_data[block_pixel_index + c];
								}
							}
						}
					}
				}
			}
		}
	}
}

#define MAX_KERNEL_SIZE 33

// Generate 1D Gaussian kernel
static void generate_gaussian_kernel(float sigma, float* kernel, int* kernel_radius)
{
	*kernel_radius = (int)ceil(3.0f * sigma);
	if (*kernel_radius > MAX_KERNEL_SIZE / 2)
		*kernel_radius = MAX_KERNEL_SIZE / 2;

	float sum = 0.0f;
	for (int x = -(*kernel_radius); x <= *kernel_radius; x++)
	{
		float value = expf(-(x * x) / (2.0f * sigma * sigma));
		kernel[x + *kernel_radius] = value;
		sum += value;
	}

	// Normalize kernel
	for (int i = 0; i < 2 * (*kernel_radius) + 1; i++)
		kernel[i] /= sum;
}

// Apply 1D convolution for 3D images
template <typename T>
static void apply_1d_convolution_3d(const T* input, T* output, int width, int height, int depth, int channels, const float* kernel, int kernel_radius, int direction)
{
	for (int z = 0; z < depth; z++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float sum[4] = {0};
				for (int k = -kernel_radius; k <= kernel_radius; k++)
				{
					int sx = direction == 0 ? x + k : x;
					int sy = direction == 1 ? y + k : y;
					int sz = direction == 2 ? z + k : z;
					if (sx >= 0 && sx < width && sy >= 0 && sy < height && sz >= 0 && sz < depth)
					{
						const T* pixel = input + (sz * height * width + sy * width + sx) * channels;
						float kvalue = kernel[k + kernel_radius];
						for (int c = 0; c < channels; c++)
							sum[c] += pixel[c] * kvalue;
					}
				}
				T* out_pixel = output + (z * height * width + y * width + x) * channels;
				for (int c = 0; c < channels; c++)
					if (std::is_same_v<T, uint8_t>)
						out_pixel[c] = (uint8_t)(sum[c] + 0.5f);
					else
						out_pixel[c] = (T)sum[c];
			}
		}
	}
}

// Separable Gaussian blur for 3D images
template <typename T>
void gaussian_blur_3d(const T* input, T* output, int width, int height, int depth, int channels, float sigma)
{
	float kernel[MAX_KERNEL_SIZE];
	int kernel_radius;
	generate_gaussian_kernel(sigma, kernel, &kernel_radius);

	T* temp1 = (T*)malloc(width * height * depth * channels * sizeof(T));
	T* temp2 = (T*)malloc(width * height * depth * channels * sizeof(T));

	// X direction pass
	apply_1d_convolution_3d(input, temp1, width, height, depth, channels, kernel, kernel_radius, 0);

	// Y direction pass
	apply_1d_convolution_3d(temp1, temp2, width, height, depth, channels, kernel, kernel_radius, 1);

	// Z direction pass (only if depth > 1)
	if (depth > 1)
	{
		apply_1d_convolution_3d(temp2, output, width, height, depth, channels, kernel, kernel_radius, 2);
	}
	else
	{
		memcpy(output, temp2, width * height * depth * channels * sizeof(T));
	}

	free(temp1);
	free(temp2);
}

template <typename T>
void high_pass_filter_squared_blurred(const T* input, float* output, int width, int height, int depth, float sigma_highpass, float sigma_blur, const vfloat4& channel_weights)
{
	size_t pixel_count = width * height * depth;
	size_t image_size = pixel_count * 4;
	T* blurred = (T*)malloc(image_size * sizeof(T));
	float* squared_diff = (float*)malloc(pixel_count * sizeof(float));

	// Apply initial Gaussian blur for high-pass filter
	gaussian_blur_3d(input, blurred, width, height, depth, 4, sigma_highpass);

	// Calculate squared differences (combined across channels, using channel weights)
	for (size_t i = 0; i < pixel_count; i++)
	{
		float diff_r = (float)input[i * 4 + 0] - (float)blurred[i * 4 + 0];
		float diff_g = (float)input[i * 4 + 1] - (float)blurred[i * 4 + 1];
		float diff_b = (float)input[i * 4 + 2] - (float)blurred[i * 4 + 2];
		float diff_a = (float)input[i * 4 + 3] - (float)blurred[i * 4 + 3];

		float diff_sum = 0;
		diff_sum += diff_r * diff_r; // * channel_weights.lane<0>();
		diff_sum += diff_g * diff_g; // * channel_weights.lane<1>();
		diff_sum += diff_b * diff_b; // * channel_weights.lane<2>();
		diff_sum += diff_a * diff_a; // * channel_weights.lane<3>();
		squared_diff[i] = diff_sum;
	}

	// Apply second Gaussian blur to the squared differences
	gaussian_blur_3d(squared_diff, output, width, height, depth, 1, sigma_blur);

	// Map x |-> C1/(C2 + sqrt(x))
	float C1 = 256.0f;
	float C2 = 1.0f;
	float activity_scalar = 3.0f;
	for (size_t i = 0; i < pixel_count; i++)
	{
		output[i] = C1 / (C2 + activity_scalar * astc::sqrt(output[i]));
		// output[i] = astc::max(output[i], 1.f);
	}

	free(blurred);
	free(squared_diff);
}

void optimize_for_lz(uint8_t* data, uint8_t* exhaustive_data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, vfloat4 channel_weights)
{
	// nothing to do if lambda is 0
	if (lambda <= 0)
	{
		memcpy(data, exhaustive_data, data_len);
		return;
	}

	channel_weights = channel_weights * 0.25f;

	// Map lambda from [10, 40] to ...
	float lambda_0 = 0.0f;
	float lambda_5 = 0.125f;
	float lambda_10 = 0.225f;
	float lambda_40 = 0.725f;
	if (MEASURE_RATE == calculate_bit_cost_lzma)
	{
		lambda_0 = 0.0f;
		lambda_5 = 0.175f;
		lambda_10 = 0.3f;
		lambda_40 = 0.9f;
	}

	if (lambda <= 0.0f)
	{
		lambda = 0.0f;
	}
	else if (lambda <= 5.0f)
	{
		lambda = lambda_0 + (lambda / 5.0f) * (lambda_5 - lambda_0);
	}
	else if (lambda <= 10.0f)
	{
		lambda = lambda_5 + ((lambda - 5.0f) / 5.0f) * (lambda_10 - lambda_5);
	}
	else
	{
		lambda = lambda_10 + (lambda - 10.0f) * (lambda_40 - lambda_10) / (40.0f - 10.0f);
	}

	if (lambda <= 0)
		lambda = 0;

	// float lambda_scale = (block_width*block_height*block_depth) / 16.f;
	// lambda *= lambda_scale;

	// Initialize block_size_descriptor once
	block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
	init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 43, *bsd);

	// Calculate gradient magnitudes and adjusted lambdas for all blocks
	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	size_t decoded_block_size = block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
	uint8_t* all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);

	// Preserve original blocks
	uint8_t* original_blocks = (uint8_t*)malloc(data_len);
	memcpy(original_blocks, data, data_len);

	for (size_t i = 0; i < num_blocks; i++)
	{
		uint8_t* original_block = exhaustive_data + i * block_size;
		uint8_t* decoded_block = all_original_decoded + i * decoded_block_size;
		astc_decompress_block(*bsd, original_block, decoded_block, block_width, block_height, block_depth, block_type);
	}

	// Calculate the full image dimensions
	int width = blocks_x * block_width;
	int height = blocks_y * block_height;
	int depth = blocks_z * block_depth;

	// Allocate memory for the reconstructed image
	size_t image_size = width * height * depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
	uint8_t* reconstructed_image = (uint8_t*)malloc(image_size);
	float* high_pass_image = (float*)malloc(width * height * depth * sizeof(float)); // Single channel

	if (block_type == ASTCENC_TYPE_U8)
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image(all_original_decoded, width, height, depth, block_width, block_height, block_depth, reconstructed_image);

		// Apply high-pass filter with squared differences and additional blur
		high_pass_filter_squared_blurred(reconstructed_image, high_pass_image, width, height, depth, 2.2f, 1.25f, channel_weights);
	}
	else
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image((float*)all_original_decoded, width, height, depth, block_width, block_height, block_depth, (float*)reconstructed_image);

		// Apply high-pass filter with squared differences and additional blur
		high_pass_filter_squared_blurred((float*)reconstructed_image, high_pass_image, width, height, depth, 2.2f, 1.25f, channel_weights);
	}

	dual_mtf_pass(data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, high_pass_image, channel_weights);
	dual_mtf_pass(data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, high_pass_image, channel_weights);

	// Clean up
	free(bsd);
	free(all_original_decoded);
	free(original_blocks);
	free(reconstructed_image);
	free(high_pass_image);
}
