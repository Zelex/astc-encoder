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

class Int128
{
private:
#if defined(_MSC_VER) && defined(_M_X64)
	__m128i value;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
	__int128 value;
#else
#error "No 128-bit integer type available for this platform"
#endif

public:
	Int128() : value{}
	{
	}

	explicit Int128(const uint8_t* bytes)
	{
#if defined(_MSC_VER) && defined(_M_X64)
		value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes));
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		memcpy(&value, bytes, 16);
#endif
	}

	void to_bytes(uint8_t* bytes) const
	{
#if defined(_MSC_VER) && defined(_M_X64)
		_mm_storeu_si128(reinterpret_cast<__m128i*>(bytes), value);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		memcpy(bytes, &value, 16);
#endif
	}

	uint8_t get_byte(int index) const
	{
#if defined(_MSC_VER) && defined(_M_X64)
		return reinterpret_cast<const uint8_t*>(&value)[index];
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		return (value >> (index * 8)) & 0xFF;
#endif
	}

	Int128 shift_left(int shift) const
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		if (shift >= 128)
		{
			result.value = _mm_setzero_si128();
		}
		else if (shift == 0)
		{
			result.value = value;
		}
		else if (shift < 64)
		{
			__m128i lo = _mm_slli_epi64(value, shift);
			__m128i hi = _mm_slli_epi64(_mm_srli_si128(value, 8), shift);
			__m128i v_cross = _mm_srli_epi64(value, 64 - shift);
			v_cross = _mm_and_si128(v_cross, _mm_set_epi64x(0, -1));
			hi = _mm_or_si128(hi, v_cross);
			result.value = _mm_or_si128(_mm_slli_si128(hi, 8), lo);
		}
		else
		{
			__m128i hi = _mm_slli_epi64(value, shift - 64 + 1);
			result.value = _mm_slli_si128(hi, 8);
		}
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = value << shift;
#endif
		return result;
	}

	Int128 bitwise_and(const Int128& other) const
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		result.value = _mm_and_si128(value, other.value);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = value & other.value;
#endif
		return result;
	}

	Int128 bitwise_or(const Int128& other) const
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		result.value = _mm_or_si128(value, other.value);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = value | other.value;
#endif
		return result;
	}

	bool is_equal(const Int128& other) const
	{
#if defined(_MSC_VER) && defined(_M_X64)
		return _mm_movemask_epi8(_mm_cmpeq_epi8(value, other.value)) == 0xFFFF;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		return value == other.value;
#endif
	}

	static Int128 from_int(long long val)
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		result.value = _mm_set_epi64x(0, val);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = static_cast<__int128>(val);
#endif
		return result;
	}

	Int128 subtract(const Int128& other) const
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		__m128i borrow = _mm_setzero_si128();
		result.value = _mm_sub_epi64(value, other.value);
		borrow = _mm_srli_epi64(_mm_cmpgt_epi64(other.value, value), 63);
		__m128i high_result = _mm_sub_epi64(_mm_srli_si128(value, 8), _mm_srli_si128(other.value, 8));
		high_result = _mm_sub_epi64(high_result, borrow);
		result.value = _mm_or_si128(result.value, _mm_slli_si128(high_result, 8));
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = value - other.value;
#endif
		return result;
	}

	Int128 bitwise_not() const
	{
		Int128 result;
#if defined(_MSC_VER) && defined(_M_X64)
		result.value = _mm_xor_si128(value, _mm_set1_epi32(-1));
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		result.value = ~value;
#endif
		return result;
	}

	uint64_t get_uint64(int index) const
	{
#if defined(_MSC_VER) && defined(_M_X64)
		return reinterpret_cast<const uint64_t*>(&value)[index];
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
		return (index == 0) ? (uint64_t)value : (uint64_t)(value >> 64);
#endif
	}

	std::string to_string() const
	{
		char buffer[33];
		snprintf(buffer, sizeof(buffer), "%016llx%016llx", get_uint64(1), get_uint64(0));
		return std::string(buffer);
	}
};

// #define MAX_MTF_SIZE (256 + 64 + 16 + 1)
#define MAX_MTF_SIZE (1024 + 256 + 64 + 16 + 1)
#define CACHE_SIZE (4096) // Should be a power of 2 for efficient modulo operation
#define BEST_CANDIDATES_COUNT (16)
#define MAX_THREADS (128)
#define MODE_MASK (0x7FF)

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
	uint32_t hash = 0;
	for (int i = 0; i < 4; i++)
	{
		hash ^= value.get_byte(i * 4) | (value.get_byte(i * 4 + 1) << 8) | (value.get_byte(i * 4 + 2) << 16) | (value.get_byte(i * 4 + 3) << 24);
	}
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
	Int128 masked_value = value.bitwise_and(mask);
	for (int i = 0; i < mtf->size; i++)
		if (mtf->list[i].bitwise_and(mask).is_equal(masked_value))
			return i;
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

static float calculate_bit_cost(int mtf_value, const Int128& literal_value, Mtf* mtf, const Int128& mask, Histo* histogram)
{
	Int128 masked_literal = literal_value.bitwise_and(mask);
	if (mtf_value == -1)
		return histo_cost(histogram, masked_literal, mask);
	return log2_fast(mtf_value + 1.f);
}

static float calculate_bit_cost_2(int mtf_value_1, int mtf_value_2, const Int128& literal_value, Mtf* mtf_1, Mtf* mtf_2, const Int128& mask_1, const Int128& mask_2, Histo* histogram)
{
	if (mtf_value_1 == -1 && mtf_value_2 == -1)
	{
		return histo_cost(histogram, literal_value, mask_1.bitwise_or(mask_2));
	}
	else if (mtf_value_1 == -1)
	{
		return histo_cost(histogram, literal_value, mask_1) + log2_fast(mtf_value_2 + 1.f);
	}
	else if (mtf_value_2 == -1)
	{
		return histo_cost(histogram, literal_value, mask_2) + log2_fast(mtf_value_1 + 1.f);
	}
	return log2_fast(mtf_value_1 + 1.f) + log2_fast(mtf_value_2 + 1.f);
}

template <typename T1, typename T2>
static inline float calculate_ssd_weighted(const T1* img1, const T2* img2, int total, const float* weights, const vfloat4& channel_weights)
{
	vfloat4 sum = vfloat4::zero();
	for (int i = 0; i < total; i += 4)
	{
		float weight = weights[i >> 2];
		vfloat4 diff = vfloat4((float)img1[i] - (float)img2[i], (float)img1[i + 1] - (float)img2[i + 1], (float)img1[i + 2] - (float)img2[i + 2], (float)img1[i + 3] - (float)img2[i + 3]);
		sum += diff * diff * weight;
	}
	return dot_s(sum, channel_weights);
}

template <typename T1, typename T2>
static inline float calculate_mrsse_weighted(const T1* img1, const T2* img2, int total, const float* weights, const vfloat4& channel_weights)
{
	vfloat4 sum = vfloat4::zero();
	for (int i = 0; i < total; i += 4)
	{
		float weight = weights[i >> 2];
		vfloat4 diff = vfloat4((float)img1[i] - (float)img2[i], (float)img1[i + 1] - (float)img2[i + 1], (float)img1[i + 2] - (float)img2[i + 2], (float)img1[i + 3] - (float)img2[i + 3]);
		sum += diff * diff * weight;
	}
	return dot_s(sum, channel_weights) * 256.0f;
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

static void dual_mtf_pass(uint8_t* data, uint8_t* ref1, uint8_t* ref2, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, block_size_descriptor* bsd, uint8_t* all_original_decoded, float* all_gradients, vfloat4 channel_weights)
{
	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	const int max_blocks_per_item = 8192;
	const int num_threads = astc::min(MAX_THREADS, (int)std::thread::hardware_concurrency());

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

	auto thread_function = [&]()
	{
		uint8_t modified_decoded[6 * 6 * 6 * 4 * 4];
		CachedBlock* block_cache = (CachedBlock*)calloc(CACHE_SIZE, sizeof(CachedBlock));
		Mtf mtf_weights;
		Mtf mtf_endpoints;
		Histo histogram;

		mtf_init(&mtf_weights, MAX_MTF_SIZE);
		mtf_init(&mtf_endpoints, MAX_MTF_SIZE);
		histo_reset(&histogram);

		// Helper function to process a single block
		auto process_block = [&](size_t block_index, bool is_forward)
		{
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
					else
					{
						return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}
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
				else
				{
					return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
				}
			};

			// Decode the original block to compute initial MSE
			astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
			float original_mse;
			if (block_type == ASTCENC_TYPE_U8)
			{
				original_mse = calculate_ssd_weighted(original_decoded, modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
			}
			else
			{
				original_mse = calculate_mrsse_weighted((float*)original_decoded, (float*)modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
			}

			int mtf_weights_pos = mtf_search(&mtf_weights, current_bits, weights_mask);
			int mtf_endpoints_pos = mtf_search(&mtf_endpoints, current_bits, endpoints_mask);
			float best_rd_cost = original_mse + lambda * calculate_bit_cost_2(mtf_weights_pos, mtf_endpoints_pos, current_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);

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
			add_candidate(best_weights, weights_count, current_bits, original_mse + lambda * calculate_bit_cost_2(mtf_weights_pos, mtf_weights_pos, current_bits, &mtf_weights, &mtf_weights, weights_mask, endpoints_mask, &histogram), mtf_weights_pos);
			add_candidate(best_endpoints, endpoints_count, current_bits, original_mse + lambda * calculate_bit_cost_2(mtf_endpoints_pos, mtf_endpoints_pos, current_bits, &mtf_endpoints, &mtf_endpoints, weights_mask, endpoints_mask, &histogram), mtf_endpoints_pos);

			// Replace the ref1 and ref2 bit extraction with the helper function
			BitsAndWeightBits ref1_wb = get_bits_and_weight_bits(ref1 + block_index * block_size, weight_bits_tbl);
			Int128 ref1_bits = ref1_wb.bits;
			int ref1_weight_bits = ref1_wb.weight_bits;
			Int128 ref1_weight_mask, ref1_endpoint_mask;
			calculate_masks(ref1_weight_bits, ref1_weight_mask, ref1_endpoint_mask);
			int mtf_weights_pos_ref1 = mtf_search(&mtf_weights, ref1_bits, ref1_weight_mask);
			int mtf_endpoints_pos_ref1 = mtf_search(&mtf_endpoints, ref1_bits, ref1_endpoint_mask);
			float ref1_mse = get_or_compute_mse(ref1_bits);
			add_candidate(best_weights, weights_count, ref1_bits, ref1_mse + lambda * calculate_bit_cost_2(mtf_weights_pos_ref1, mtf_weights_pos_ref1, ref1_bits, &mtf_weights, &mtf_weights, ref1_weight_mask, ref1_endpoint_mask, &histogram), mtf_weights_pos_ref1);
			add_candidate(best_endpoints, endpoints_count, ref1_bits, ref1_mse + lambda * calculate_bit_cost_2(mtf_endpoints_pos_ref1, mtf_endpoints_pos_ref1, ref1_bits, &mtf_endpoints, &mtf_endpoints, ref1_weight_mask, ref1_endpoint_mask, &histogram), mtf_endpoints_pos_ref1);

			// Add ref2
			BitsAndWeightBits ref2_wb = get_bits_and_weight_bits(ref2 + block_index * block_size, weight_bits_tbl);
			Int128 ref2_bits = ref2_wb.bits;
			int ref2_weight_bits = ref2_wb.weight_bits;
			Int128 ref2_weight_mask, ref2_endpoint_mask;
			calculate_masks(ref2_weight_bits, ref2_weight_mask, ref2_endpoint_mask);
			int mtf_weights_pos_ref2 = mtf_search(&mtf_weights, ref2_bits, ref2_weight_mask);
			int mtf_endpoints_pos_ref2 = mtf_search(&mtf_endpoints, ref2_bits, ref2_endpoint_mask);
			float ref2_mse = get_or_compute_mse(ref2_bits);
			add_candidate(best_weights, weights_count, ref2_bits, ref2_mse + lambda * calculate_bit_cost_2(mtf_weights_pos_ref2, mtf_weights_pos_ref2, ref2_bits, &mtf_weights, &mtf_weights, ref2_weight_mask, ref2_endpoint_mask, &histogram), mtf_weights_pos_ref2);
			add_candidate(best_endpoints, endpoints_count, ref2_bits, ref2_mse + lambda * calculate_bit_cost_2(mtf_endpoints_pos_ref2, mtf_endpoints_pos_ref2, ref2_bits, &mtf_endpoints, &mtf_endpoints, ref2_weight_mask, ref2_endpoint_mask, &histogram), mtf_endpoints_pos_ref2);

			// Find best endpoint candidates
			for (int k = 0; k < mtf_endpoints.size; k++)
			{
				Int128 candidate_endpoints = mtf_endpoints.list[k];
				int endpoints_weight_bits = get_bits_and_weight_bits((uint8_t*)&candidate_endpoints, weight_bits_tbl).weight_bits;

				Int128 weights_mask, endpoints_mask;
				calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);
				float mse = get_or_compute_mse(candidate_endpoints);
				float bit_cost = calculate_bit_cost_2(k, k, candidate_endpoints, &mtf_endpoints, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
				float rd_cost = mse + lambda * bit_cost;

				// Insert into best_endpoints if it's one of the best candidates
				add_candidate(best_endpoints, endpoints_count, candidate_endpoints, rd_cost, k);
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
						uint8_t temp_block[16];
						combined_bits.to_bytes(temp_block);
						astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

						float mse;
						if (block_type == ASTCENC_TYPE_U8)
						{
							mse = calculate_ssd_weighted(original_decoded, modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
						}
						else
						{
							mse = calculate_mrsse_weighted((float*)original_decoded, (float*)modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
						}

						float bit_cost = calculate_bit_cost_2(k, best_endpoints[m].mtf_position, combined_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);
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

					Int128 temp_bits = best_weights[i].bits.bitwise_and(weights_mask).bitwise_or(best_endpoints[j].bits.bitwise_and(endpoints_mask));
					uint8_t temp_block[16];
					temp_bits.to_bytes(temp_block);

					astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

					float mse;
					if (block_type == ASTCENC_TYPE_U8)
					{
						mse = calculate_ssd_weighted(original_decoded, modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}
					else
					{
						mse = calculate_mrsse_weighted((float*)original_decoded, (float*)modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}

					float rd_cost = mse + lambda * calculate_bit_cost_2(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);

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
					temp_bits.to_bytes(temp_block);

					astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

					if (block_type == ASTCENC_TYPE_U8)
					{
						mse = calculate_ssd_weighted(original_decoded, modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}
					else
					{
						mse = calculate_mrsse_weighted((float*)original_decoded, (float*)modified_decoded, block_width * block_height * block_depth * 4, all_gradients, channel_weights);
					}

					rd_cost = mse + lambda * calculate_bit_cost_2(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, &mtf_weights, &mtf_endpoints, weights_mask, endpoints_mask, &histogram);

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

			histo_update(&histogram, best_match, Int128::from_int(0).bitwise_not());
			mtf_encode(&mtf_weights, best_match, best_weights_mask);
			mtf_encode(&mtf_endpoints, best_match, best_endpoints_mask);
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
		for (size_t start_block = 0; start_block < num_blocks; start_block += max_blocks_per_item)
		{
			size_t end_block = astc::min(start_block + max_blocks_per_item, num_blocks);
			work_queue.push({start_block, end_block, is_forward});
		}

		std::vector<std::thread> threads;

		// Start threads
		for (int i = 0; i < num_threads; ++i)
			threads.emplace_back(thread_function);

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
		diff_sum += diff_r * diff_r * channel_weights.lane<0>();
		diff_sum += diff_g * diff_g * channel_weights.lane<1>();
		diff_sum += diff_b * diff_b * channel_weights.lane<2>();
		diff_sum += diff_a * diff_a * channel_weights.lane<3>();
		squared_diff[i] = diff_sum;
	}

	// Apply second Gaussian blur to the squared differences
	gaussian_blur_3d(squared_diff, output, width, height, depth, 1, sigma_blur);

	// Map x |-> C1/(C2 + sqrt(x))
	float C1 = 256.0f;
	float C2 = 1.0f;
	float activity_scalar = 3.0f * 2;
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
	float lambda_5 = 0.175f;
	float lambda_10 = 0.275f;
	float lambda_40 = 0.725f;

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
