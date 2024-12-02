/*
This file implements a rate-distortion optimization algorithm for ASTC compressed textures.
The algorithm aims to improve compression by exploiting redundancy between neighboring blocks
while maintaining visual quality.

The core approach uses Move-To-Front (MTF) coding and error diffusion to optimize both the
weights and endpoints of ASTC blocks. It processes the image in both forward and backward
passes, maintaining separate MTF lists for weights and endpoints. For each block, it:

1. Analyzes the high-frequency content using a multi-scale approach to determine visually
   important areas that need higher quality encoding
2. Searches for similar blocks in the MTF lists that could provide better rate-distortion
   tradeoffs
3. Propagates quantization errors to neighboring blocks using an error diffusion filter
4. Updates the MTF lists and statistics to adapt to local texture patterns

The algorithm uses a lambda parameter to control the rate-distortion tradeoff - higher values
favor better compression while lower values preserve more quality. It also supports different
effort levels that control how exhaustively it searches for optimizations.

The implementation is multi-threaded and includes optimizations like SIMD processing, block
caching, and efficient bit manipulation to maintain good performance even at high effort
levels.
*/
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

#include "ThirdParty/rad_tm_stub.h"

// Maximum size for Move-To-Front (MTF) lists
#define MAX_MTF_SIZE (256 + 64 + 16 + 3)
// #define MAX_MTF_SIZE (1024 + 256 + 64 + 16 + 1)
#define MTF_ENDPOINTS_SIZE (256 + 64 + 16 + 3)
#define MTF_WEIGHTS_SIZE (16 + 3)
// Cache size for block decompression results, should be power of 2
#define CACHE_SIZE (0x10000)
// Number of best candidates to consider for each block
#define BEST_CANDIDATES_COUNT (8)
// Maximum number of threads to use
#define MAX_THREADS (128)
// Mode mask for block types
#define MODE_MASK (0x7FF)
// Maximum number of blocks to process per worker thread item
#define MAX_BLOCKS_PER_ITEM (8192)

// 128-bit integer structure for handling ASTC block data
// Used to manipulate compressed ASTC blocks which are 128 bits (16 bytes) in size
struct Int128
{
	union
	{
		uint64_t uint64[2]; // Access as two 64-bit integers
		uint32_t uint32[4]; // Access as four 32-bit integers
		uint8_t bytes[16];  // Access as 16 bytes
	};

	// Default constructor initializes to zero
	Int128() : uint64()
	{
	}

	// Constructor from raw byte data
	explicit Int128(const uint8_t* data)
	{
		memcpy(bytes, data, 16);
	}

	// Left shift operation with bounds checking
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
			result.uint64[1] = (uint64[1] << shift) | (uint64[0] >> (64 - shift));
			result.uint64[0] = uint64[0] << shift;
		}
		else
		{
			result.uint64[1] = uint64[0] << (shift - 64);
			result.uint64[0] = 0;
		}
		return result;
	}

	// Bitwise AND operation
	Int128 bitwise_and(const Int128& other) const
	{
		Int128 result;
		result.uint64[0] = uint64[0] & other.uint64[0];
		result.uint64[1] = uint64[1] & other.uint64[1];
		return result;
	}

	// Bitwise OR operation
	Int128 bitwise_or(const Int128& other) const
	{
		Int128 result;
		result.uint64[0] = uint64[0] | other.uint64[0];
		result.uint64[1] = uint64[1] | other.uint64[1];
		return result;
	}

	// Equality check
	bool is_equal(const Int128& other) const
	{
		return uint64[0] == other.uint64[0] && uint64[1] == other.uint64[1];
	}

	// Constructor from integer value
	static Int128 from_int(long long val)
	{
		Int128 result;
		result.uint64[0] = val;
		result.uint64[1] = val < 0 ? -1LL : 0;
		return result;
	}

	// Subtraction operation
	Int128 subtract(const Int128& other) const
	{
		Int128 result;
		result.uint64[0] = uint64[0] - other.uint64[0];
		result.uint64[1] = uint64[1] - other.uint64[1] - (uint64[0] < other.uint64[0] ? 1 : 0);
		return result;
	}

	// Bitwise NOT operation
	Int128 bitwise_not() const
	{
		Int128 result;
		result.uint64[0] = ~uint64[0];
		result.uint64[1] = ~uint64[1];
		return result;
	}

	// Convert to string representation (for debugging)
	std::string to_string() const
	{
		char buffer[33];
		snprintf(buffer, sizeof(buffer), "%016llx%016llx", uint64[1], uint64[0]);
		return std::string(buffer);
	}
};

// Histogram structure for tracking byte frequencies in compressed data
// Used for entropy coding and compression ratio estimation
struct Histo
{
	int h[256]; // Frequency count for each byte
	int size;   // Total count
};

// Move-To-Front (MTF) list structure for maintaining recently used values
// Helps exploit temporal locality in the compressed data
struct Mtf
{
	Int128 list[MAX_MTF_SIZE]; // List of recently used values
	int size;                  // Current size
	int max_size;              // Maximum size
};

// Cache entry for storing decoded block results
// Reduces redundant decompression operations
struct CachedBlock
{
	Int128 encoded;                     // Encoded block data
	uint8_t decoded[6 * 6 * 6 * 4 * 4]; // Max size for both U8 and float types
	bool valid;                         // Validity flag
};

/*
static void jo_write_tga(const char *filename, void *rgba, int width, int height, int numChannels, int bit_depth) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) {
        return;
    }
    // Header
    fwrite("\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00", 12, 1, fp);
    fwrite(&width, 2, 1, fp);
    fwrite(&height, 2, 1, fp);
    int bpc = (numChannels * bit_depth) | 0x2000; // N bits per pixel
    int byte_depth = (bit_depth+7) / 8;
    fwrite(&bpc, 2, 1, fp);
    // Swap RGBA to BGRA if using 3 or more channels
    int remap[4] = {numChannels >= 3 ? 2*byte_depth : 0, 1*byte_depth, numChannels >= 3 ? 0 : 2*byte_depth, 3*byte_depth};
    char *s = (char *)rgba;
    for(int i = 0; i < width*height; ++i) {
        for(int j = 0; j < numChannels; ++j) {
            fwrite(s + remap[j], byte_depth, 1, fp);
        }
        s += numChannels*byte_depth;
    }
    fclose(fp);
}
*/

// Fast log2 approximation function
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

// Hash function for 128-bit values
// Used for cache lookups and hash table operations
static uint32_t hash_128(const Int128& value)
{
	// FNV-1a inspired constants for 64-bit operations
	const uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
	const uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;

	// Get the two 64-bit words directly
	uint64_t h1 = value.uint64[0];
	uint64_t h2 = value.uint64[1];

	// Mix the first word
	h1 *= PRIME64_1;
	h1 = (h1 << 31) | (h1 >> 33);
	h1 *= PRIME64_2;

	// Mix the second word
	h2 *= PRIME64_2;
	h2 = (h2 << 29) | (h2 >> 35);
	h2 *= PRIME64_1;

	// Combine the results
	uint32_t result = (uint32_t)(h1 ^ h2);

	// Final avalanche
	result ^= result >> 15;
	result *= 0x85ebca6b;
	result ^= result >> 13;
	result *= 0xc2b2ae35;
	result ^= result >> 16;

	return result;
}

// Reset histogram counts to zero
static void histo_reset(Histo* h)
{
	memset(h->h, 0, sizeof(h->h));
	h->size = 0;
}

// Update histogram with byte frequencies from masked value
static void histo_update(Histo* h, const Int128& value, const Int128& mask)
{
	// Iterate over each byte in the 128-bit value
	for (int i = 0; i < 16; i++)
	{
		uint8_t m = mask.bytes[i];
		if (m) // Only count bytes where the mask is non-zero
		{
			uint8_t byte = value.bytes[i];
			h->h[byte]++; // Increment the count for this byte
			h->size++;    // Increment the total count
		}
	}
}

// Calculate entropy-based cost for encoding a value with given mask
static float histo_cost(Histo* h, Int128 value, Int128 mask)
{
	// Return 0 if mask is all zeros (nothing to encode)
	if (mask.uint64[0] == 0 && mask.uint64[1] == 0)
		return 0.0f;

	float tlb = (float)h->size + 1.0f; // Total bytes plus 1 for Laplace smoothing
	float cost1 = 1.0f;
	float cost2 = 1.0f;
	float cost3 = 1.0f;
	float cost4 = 1.0f;

	// Process 4 bytes at a time for efficiency
	for (int i = 0; i < 16; i += 4)
	{
		uint32_t m = mask.uint32[i >> 2];

		if (m)
		{
			if (m & 0xFF)
				cost1 *= tlb / (h->h[value.bytes[i]] + 1.0f);
			if (m & 0xFF00)
				cost2 *= tlb / (h->h[value.bytes[i + 1]] + 1.0f);
			if (m & 0xFF0000)
				cost3 *= tlb / (h->h[value.bytes[i + 2]] + 1.0f);
			if (m & 0xFF000000)
				cost4 *= tlb / (h->h[value.bytes[i + 3]] + 1.0f);
		}
	}

	// Combine the costs and take the log2
	return log2_fast((cost1 * cost2) * (cost3 * cost4));
}

// Initialize Move-To-Front list with given maximum size
static void mtf_init(Mtf* mtf, int max_size)
{
	mtf->size = 0;
	mtf->max_size = max_size > MAX_MTF_SIZE ? MAX_MTF_SIZE : max_size;
}

// Search for a masked value in the MTF list
// Returns position if found, -1 if not found
static int mtf_search(Mtf* mtf, const Int128& value, const Int128& mask)
{
	// Pre-compute the masked value once for efficiency
	Int128 masked_value = value.bitwise_and(mask);

	// Search in groups of 4 for better performance
	int i = 0;
	for (; i + 3 < mtf->size; i += 4)
	{
		// Check if any of these 4 entries match
		bool match0 = mtf->list[i + 0].bitwise_and(mask).is_equal(masked_value);
		if (match0)
			return i;
		bool match1 = mtf->list[i + 1].bitwise_and(mask).is_equal(masked_value);
		if (match1)
			return i + 1;
		bool match2 = mtf->list[i + 2].bitwise_and(mask).is_equal(masked_value);
		if (match2)
			return i + 2;
		bool match3 = mtf->list[i + 3].bitwise_and(mask).is_equal(masked_value);
		if (match3)
			return i + 3;
	}

	// Handle remaining entries
	for (; i < mtf->size; i++)
	{
		if (mtf->list[i].bitwise_and(mask).is_equal(masked_value))
			return i;
	}

	return -1; // Not found
}

// Encode a value using Move-To-Front coding
// Returns the position where the value was found or inserted
static int mtf_encode(Mtf* mtf, const Int128& value, const Int128& mask)
{
	// Search for the value in the list
	int pos = mtf_search(mtf, value, mask);

	if (pos == -1)
	{
		// If not found, insert at the end
		if (mtf->size < mtf->max_size)
			mtf->size++;
		pos = mtf->size - 1;
	}

	// Move the found value to the front of the list
	for (int i = pos; i > 0; i--)
		mtf->list[i] = mtf->list[i - 1];
	mtf->list[0] = value;

	return pos;
}

// Calculate bit cost for encoding using MTF and literal values
static float calculate_bit_cost_simple(int mtf_value_1,             // Position in first MTF list (or -1 if not found)
                                       int mtf_value_2,             // Position in second MTF list (or -1 if not found)
                                       const Int128& literal_value, // Actual value to encode
                                       const Int128& mask_1,        // Mask for first MTF list
                                       const Int128& mask_2,        // Mask for second MTF list
                                       Histo* histogram             // Histogram for entropy coding
)
{
	// Case 1: Both parts need literal encoding
	if (mtf_value_1 == -1 && mtf_value_2 == -1)
		return histo_cost(histogram, literal_value, mask_1.bitwise_or(mask_2));

	// Case 2: first part needs literal encoding
	if (mtf_value_1 == -1)
		return histo_cost(histogram, literal_value, mask_1) + log2_fast(mtf_value_2 + 1.0f);

	// Case 3: second part needs literal encoding
	if (mtf_value_2 == -1)
		return log2_fast(mtf_value_1 + 1.0f) + histo_cost(histogram, literal_value, mask_2);

	// Case 4: Both parts can use MTF encoding
	return log2_fast(mtf_value_1 + 1.0f) + log2_fast(mtf_value_2 + 1.0f);
}

// Calculate Sum of Squared Differences (SSD) with weights for U8 image data
static inline float calculate_ssd_weighted(const uint8_t* img1,           // first image data
                                           const uint8_t* img2,           // second image data
                                           int total,                     // total number of pixels * 4 (RGBA)
                                           const float* weights,          // Per-pixel weights
                                           const vfloat4& channel_weights // Per-channel importance weights
)
{
	// Initialize accumulator vectors for SIMD processing
	vfloat4 sum0 = vfloat4::zero(), sum1 = vfloat4::zero();
	vfloat4 sum2 = vfloat4::zero(), sum3 = vfloat4::zero();

	// Process 16 pixels (64 bytes) at a time using SIMD
	int i;
	for (i = 0; i < total - 15; i += 16)
	{
		// Load and compute differences for 4 pixels at a time
		vfloat4 diff0 = int_to_float(vint4(img1 + i)) - int_to_float(vint4(img2 + i));
		vfloat4 diff1 = int_to_float(vint4(img1 + i + 4)) - int_to_float(vint4(img2 + i + 4));
		vfloat4 diff2 = int_to_float(vint4(img1 + i + 8)) - int_to_float(vint4(img2 + i + 8));
		vfloat4 diff3 = int_to_float(vint4(img1 + i + 12)) - int_to_float(vint4(img2 + i + 12));

		// Square differences and multiply by weights
		sum0 += diff0 * diff0 * vfloat4(weights + i);
		sum1 += diff1 * diff1 * vfloat4(weights + i + 4);
		sum2 += diff2 * diff2 * vfloat4(weights + i + 8);
		sum3 += diff3 * diff3 * vfloat4(weights + i + 12);
	}

	// Process remaining pixels
	for (; i < total; i += 4)
	{
		vfloat4 diff = int_to_float(vint4(img1 + i)) - int_to_float(vint4(img2 + i));
		sum0 += diff * diff * weights[i];
	}

	// Combine all sums, apply channel weights
	return dot_s(((sum0 + sum1) + (sum2 + sum3)), channel_weights);
}

// Calculate Mean Relative Squared Signal Error (MRSSE) with weights for float image data
static inline float calculate_mrsse_weighted(const float* img1,             // first image data
                                             const float* img2,             // second image data
                                             int total,                     // total number of pixels * 4 (RGBA)
                                             const float* weights,          // Per-pixel weights
                                             const vfloat4& channel_weights // Per-channel importance weights
)
{
	// Initialize accumulator vectors for SIMD processing
	vfloat4 sum0 = vfloat4::zero(), sum1 = vfloat4::zero();
	vfloat4 sum2 = vfloat4::zero(), sum3 = vfloat4::zero();

	// Process 16 pixels (64 bytes) at a time using SIMD
	int i;
	for (i = 0; i < total - 15; i += 16)
	{
		// Load and compute differences for 4 pixels at a time
		vfloat4 diff0 = vfloat4(img1 + i) - vfloat4(img2 + i);
		vfloat4 diff1 = vfloat4(img1 + i + 4) - vfloat4(img2 + i + 4);
		vfloat4 diff2 = vfloat4(img1 + i + 8) - vfloat4(img2 + i + 8);
		vfloat4 diff3 = vfloat4(img1 + i + 12) - vfloat4(img2 + i + 12);

		// Square differences and multiply by weights
		sum0 += diff0 * diff0 * vfloat4(weights + i);
		sum1 += diff1 * diff1 * vfloat4(weights + i + 4);
		sum2 += diff2 * diff2 * vfloat4(weights + i + 8);
		sum3 += diff3 * diff3 * vfloat4(weights + i + 12);
	}

	// Process remaining pixels
	for (; i < total; i += 4)
	{
		vfloat4 diff = vfloat4(img1 + i) - vfloat4(img2 + i);
		sum0 += diff * diff * weights[i];
	}

	// Combine all sums, apply channel weights, and scale by 256.0f
	return dot_s(((sum0 + sum1) + (sum2 + sum3)), channel_weights) * 256.0f;
}

// Decompress an ASTC block to either U8 or float output
static void astc_decompress_block(const block_size_descriptor& bsd, // Block size descriptor
                                  const uint8_t* block_ptr,         // Block data pointer
                                  uint8_t* output,                  // Output buffer
                                  int block_width,                  // Block dimensions
                                  int block_height,
                                  int block_depth,
                                  int block_type // Block type (ASTCENC_TYPE_U8 or ASTCENC_TYPE_F32)
)
{
	// Initialize image block structure
	image_block blk{};
	blk.texel_count = static_cast<uint8_t>(block_width * block_height * block_depth);
	blk.data_min = vfloat4::zero();
	blk.data_max = vfloat4(1.0f, 1.0f, 1.0f, 1.0f);
	blk.grayscale = false;
	blk.xpos = 0;
	blk.ypos = 0;
	blk.zpos = 0;

	// Convert physical block to symbolic representation
	symbolic_compressed_block scb;
	physical_to_symbolic(bsd, block_ptr, scb);

	// Determine profile based on block type
	astcenc_profile profile = block_type == ASTCENC_TYPE_U8 ? ASTCENC_PRF_LDR : ASTCENC_PRF_HDR;

	// Decompress symbolic block
	decompress_symbolic_block(profile, bsd, 0, 0, 0, scb, blk);

	// Convert the decompressed data to the output format
	if (block_type == ASTCENC_TYPE_U8)
	{
		// convert to 8-bit RGBA
		for (int i = 0; i < blk.texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Load RGBA components
			vfloat r = vfloat(blk.data_r + i);
			vfloat g = vfloat(blk.data_g + i);
			vfloat b = vfloat(blk.data_b + i);
			vfloat a = vfloat(blk.data_a + i);

			// Clamp to [0,1] and convert to 8-bit
			vint ri = float_to_int_rtn(min(r, 1.0f) * 255.0f);
			vint gi = float_to_int_rtn(min(g, 1.0f) * 255.0f);
			vint bi = float_to_int_rtn(min(b, 1.0f) * 255.0f);
			vint ai = float_to_int_rtn(min(a, 1.0f) * 255.0f);

			// Interleave and store the results
			vint rgbai = interleave_rgba8(ri, gi, bi, ai);
			unsigned int used_texels = astc::min(blk.texel_count - i, ASTCENC_SIMD_WIDTH);
			store_lanes_masked(output + i * 4, rgbai, vint::lane_id() < vint(used_texels));
		}
	}
	else
	{
		// Store as 32-bit float RGBA
		float* output_f = reinterpret_cast<float*>(output);
		for (int i = 0; i < blk.texel_count; i++)
		{
			output_f[i * 4 + 0] = blk.data_r[i];
			output_f[i * 4 + 1] = blk.data_g[i];
			output_f[i * 4 + 2] = blk.data_b[i];
			output_f[i * 4 + 3] = blk.data_a[i];
		}
	}
}

// Get the number of bits used for weights in an ASTC block
int get_weight_bits(uint8_t* data, int block_width, int block_height, int block_depth)
{
	// Extract block mode from first two bytes
	uint16_t mode = data[0] | (data[1] << 8);

	// Check for special block types
	if ((mode & 0x1ff) == 0x1fc)
		return 0; // void-extent block
	if ((mode & 0x00f) == 0)
		return 0; // Reserved block mode

	// Extract individual bits from the mode
	uint8_t b01 = (mode >> 0) & 3;   // Bits 0-1
	uint8_t b23 = (mode >> 2) & 3;   // Bits 2-3
	uint8_t p0 = (mode >> 4) & 1;    // Bit 4
	uint8_t b56 = (mode >> 5) & 3;   // Bits 5-6
	uint8_t b78 = (mode >> 7) & 3;   // Bits 7-8
	uint8_t P = (mode >> 9) & 1;     // Bit 9
	uint8_t Dp = (mode >> 10) & 1;   // Bit 10
	uint8_t b9_10 = (mode >> 9) & 3; // Bits 9-10
	uint8_t p12;

	// Variables for block dimensions
	int W = 0, H = 0, D = 0;

	// Handle 2D blocks
	if (block_depth <= 1)
	{
		// 2D
		D = 1;
		if ((mode & 0x1c3) == 0x1c3)
			return 0; // Reserved*

		if (b01 == 0)
		{
			p12 = b23;
			// Determine block dimensions based on mode bits
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
			// Alternative block dimensions calculations
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
		// Handle 3D blocks
		if ((mode & 0x1e3) == 0x1e3)
			return 0; // Reserved*
		if (b01 != 0)
		{
			p12 = b01;
			// 3D block dimensions
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
					return 0; // Invalid mode
				}
				break;
			}
		}
	}

	// Validate block dimensions
	if (W > block_width || H > block_height || D > block_depth)
		return 0;

	// Calculate weight bits based on encoding parameters
	uint8_t p = (p12 << 1) | p0;
	int trits = 0, quints = 0, bits = 0;

	if (!P)
	{
		// Non-packed mode weight bit patterns
		int t[8] = {-1, -1, 0, 1, 0, 0, 1, 0}; // trit patterns
		int q[8] = {-1, -1, 0, 0, 0, 1, 0, 0}; // quint patterns
		int b[8] = {-1, -1, 1, 0, 2, 0, 1, 3}; // bit patterns
		trits = t[p];
		quints = q[p];
		bits = b[p];
	}
	else
	{
		// Packed mode weight bit patterns
		int t[8] = {-1, -1, 0, 1, 0, 0, 1, 0};
		int q[8] = {-1, -1, 1, 0, 0, 1, 0, 0};
		int b[8] = {-1, -1, 1, 2, 4, 2, 3, 5};
		trits = t[p];
		quints = q[p];
		bits = b[p];
	}

	// Calculate total number of weights
	int num_weights = W * H * D;
	if (Dp)
		num_weights *= 2; // Dual plane mode doubles weights

	// Check weight count limit
	if (num_weights > 64)
		return 0;

	// Calculate total weight bits
	int weight_bits = (num_weights * 8 * trits + 4) / 5 +  // Trit bits
	                  (num_weights * 7 * quints + 2) / 3 + // Quint bits
	                  num_weights * bits;                  // Plain bits

	// Check weight bit range
	if (weight_bits < 24 || weight_bits > 96)
		return 0;

	return (uint8_t)weight_bits;
}

#if 0 // Debug/test code (disabled)
extern int hack_bits_for_weights;
void test_weight_bits(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth)
{
	// Allocate and initialize block size descriptor
    block_size_descriptor *bsd = (block_size_descriptor*) malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

	// Test each block
    for (size_t i=0; i < data_len; i += 16) 
    {
        uint8_t *block = data + i;
        uint8_t decoded[6*6*6*4]; // buffer for decoded block
        astc_decompress_block(*bsd, block, decoded, block_width, block_height, block_depth, ASTCENC_TYPE_U8);
        int bits = get_weight_bits(block, block_width, block_height, block_depth);
	    if (bits != hack_bits_for_weights) 
		    printf("Internal error: decoded weight bits count didn't match\n");
    }
    free(bsd);
}
#endif

// Structure to define a work item for thread processing
struct WorkItem
{
	size_t start_block; // Starting block index
	size_t end_block;   // Ending block index ( exclusive )
	bool is_forward;    // Forward or backward pass
};

// Calculate masks for weights and endpoints based on weight bits
static inline void calculate_masks(int weight_bits, Int128& weights_mask, Int128& endpoints_mask)
{
	Int128 one = Int128::from_int(1);
	// Create weight mask by shifting 1s into position
	weights_mask = one.shift_left(weight_bits).subtract(one).shift_left(128 - weight_bits);
	// Endpoints mask is complement of weights mask
	endpoints_mask = weights_mask.bitwise_not();
}

// Get weight bits from a block using pre-computed lookup table
static inline int get_weight_bits(const uint8_t* block, const uint8_t* weight_bits_tbl)
{
	return weight_bits_tbl[(block[0] | (block[1] << 8)) & MODE_MASK];
}

// Structure to track rate-distortion errors
struct RDError
{
	float mse_error;  // Mean squared error component
	float rate_error; // Rate (compression) error component
};

static inline uint32_t xorshift32(uint32_t& state)
{
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	return state;
}

// Performs forward and backwards optimization passes over blocks
static void dual_mtf_pass(int thread_count, // Number of threads to use
                          bool silentmode,  // Supress progress output
                          uint8_t* data,    // Input/output compressed data
                          uint8_t* ref1,    // Pointer to reference 1 data
                          uint8_t* ref2,    // Pointer to reference 2 data
                          size_t data_len,  // Length of input data
                          int blocks_x,     // Block dimensions
                          int blocks_y,
                          int blocks_z,                  // Block dimensions
                          int block_width,               // Block size
                          int block_height,              // Block dimensions
                          int block_depth,               // Block dimensions
                          int block_type,                // ASTC block type
                          float lambda,                  // Lambda parameter for rate-distortion optimization
                          block_size_descriptor* bsd,    // Block size descriptor
                          uint8_t* all_original_decoded, // Pointer to original decoded data
                          float* all_gradients,          // Per-pixel weights
                          vfloat4 channel_weights,       // Channel importance weights
                          float effort                   // Optimization effort
)
{
	// Uses multiple threads to process blocks
	// Each thread maintains its own MTF lists and histogram
	// Processes blocks in chunks, applying rate-distortion optimization

	tmFunction(0, 0); // Timing/profiling marker

	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	// Limit thread count to hardware, MAX_THREADS, and requested amount
	const int num_threads = astc::min((int)thread_count, MAX_THREADS, (int)std::thread::hardware_concurrency());
	// Allocate error buffer for all blocks
	RDError* error_buffer = (RDError*)calloc(blocks_x * blocks_y * blocks_z, sizeof(RDError));

	// Add progress tracking
	std::atomic<size_t> blocks_processed{0};
	size_t total_blocks = num_blocks;
	if (effort >= 8)
		total_blocks *= 2; // include backwards pass

	// Lanmbda function to print progress bar
	auto print_progress = [&blocks_processed, total_blocks, silentmode]()
	{
		// tmFunction(0, 0);
		static int last_percentage = -1;
		float progress = static_cast<float>(blocks_processed) / total_blocks;
		int current_percentage = static_cast<int>(progress * 100.0f);

		// Only update if percentage has changed
		if (current_percentage == last_percentage)
			return;

		last_percentage = current_percentage;

		// Draw progress bar if not in silent mode
		if (!silentmode)
		{
			const int bar_width = 50;
			int pos = static_cast<int>(bar_width * progress);
			printf("\rOptimizing: [");
			for (int i = 0; i < bar_width; ++i)
			{
				if (i < pos)
					printf("=");
				else if (i == pos)
					printf(">");
				else
					printf(" ");
			}
			printf("] %3d%%", static_cast<int>(progress * 100.0f));
			fflush(stdout);
		}
	};

	// Initialize weight bits table
	uint8_t* weight_bits_tbl = (uint8_t*)malloc(2048);
	for (size_t i = 0; i < 2048; ++i)
	{
		uint8_t block[16];
		block[0] = (i & 255);
		block[1] = ((i >> 8) & 255);
		weight_bits_tbl[i] = (uint8_t)get_weight_bits(block, block_width, block_height, block_depth);
	}

	// Thread synchronization primitives
	std::queue<WorkItem> work_queue;
	std::mutex queue_mutex;
	std::condition_variable cv;
	bool all_work_done = false;

	// Main thread worker function
	auto thread_function = [&](int thread_id)
	{
		tmProfileThread(0, 0, 0); // Thread profiling marker
		tmZone(0, 0, "thread_function");

		// Allocate thread-local resources
		CachedBlock* block_cache = (CachedBlock*)calloc(CACHE_SIZE, sizeof(CachedBlock));
		Mtf mtf_weights;   // MTF list for weights
		Mtf mtf_endpoints; // MTF list for endpoints
		Histo histogram;   // Histogram for encoding statistics

		// Function to initialize MTF and histogram with random blocks
		auto seed_structures = [&](Mtf& mtf_w, Mtf& mtf_e, Histo& hist, uint32_t seed, size_t start_block, size_t end_block)
		{
			uint32_t rng_state = seed + thread_id;

			// Number of random blocks to sample
			const int num_samples = 64;

			// Reset structures
			mtf_init(&mtf_w, mtf_w.max_size);
			mtf_init(&mtf_e, mtf_e.max_size);
			histo_reset(&hist);

			// Calculate range size
			size_t range_size = end_block - start_block;
			if (range_size == 0)
				return;

			// Sample random blocks from within the work item range
			for (int i = 0; i < num_samples; i++)
			{
				// Generate random block index within the work item range
				size_t block_idx = start_block + (xorshift32(rng_state) % range_size);

				// Get block data
				Int128 block_bits(data + block_idx * 16);
				int block_weight_bits = get_weight_bits(block_bits.bytes, weight_bits_tbl);

				// Calculate masks
				Int128 weights_mask, endpoints_mask;
				calculate_masks(block_weight_bits, weights_mask, endpoints_mask);

				// Update structures
				mtf_encode(&mtf_w, block_bits, weights_mask);
				mtf_encode(&mtf_e, block_bits, endpoints_mask);
				histo_update(&hist, block_bits, weights_mask.bitwise_or(endpoints_mask));
			}
		};

		// Error propagation function for rate-distortion optimization
		auto propagate_error = [&](long long x, long long y, long long z, float mse_diff, float rate_diff, bool is_forward)
		{
			// Calculate the work group boundary
			long long current_block_idx = z * blocks_x * blocks_y + y * blocks_x + x;
			long long work_group_start = (current_block_idx / MAX_BLOCKS_PER_ITEM) * MAX_BLOCKS_PER_ITEM;
			long long work_group_end = astc::min(work_group_start + MAX_BLOCKS_PER_ITEM, (long long)num_blocks);

			// Don't propagate if target block would be outside current work group
			if (x < 0 || x >= blocks_x || y < 0 || y >= blocks_y || z < 0 || z >= blocks_z)
				return;

			// For 2D slices (block_depth == 1), don't propagate errors across Z boundaries
			if (block_depth == 1)
			{
				long long target_z = z;
				long long current_z = current_block_idx / (blocks_x * blocks_y);
				if (target_z != current_z)
					return;
			}

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

		// Process individual blocks within the thread
		auto process_block = [&](size_t block_index, bool is_forward)
		{
			// Calculate block coordinates
			size_t z = block_index / (blocks_x * blocks_y);
			size_t rem = block_index % (blocks_x * blocks_y);
			size_t y = rem / blocks_x;
			size_t x = rem % blocks_x;

			// Get current block data and compute its weight bits
			uint8_t* current_block = data + block_index * block_size;
			Int128 current_bits(current_block);
			int current_weight_bits = get_weight_bits(current_bits.bytes, weight_bits_tbl);
			Int128 best_match = current_bits;

			// Don't process blocks with no weight bits, accept void-extent as is
			if (current_weight_bits == 0)
			{
				histo_update(&histogram, current_bits, Int128::from_int(-1));
				mtf_encode(&mtf_weights, current_bits, Int128::from_int(0));
				mtf_encode(&mtf_endpoints, current_bits, Int128::from_int(-1));
				return;
			}

			// Get pointers to original decoded data and gradients for this block
			uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));
			const float* gradients = all_gradients + block_index * (block_width * block_height * block_depth * 4);

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
						return calculate_ssd_weighted(original_decoded, block_cache[hash].decoded, block_width * block_height * block_depth * 4, gradients, channel_weights);
					}
					return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, gradients, channel_weights);
				}

				// If not in cache, compute and cache the result

				// Decode and compute MSE
				astc_decompress_block(*bsd, candidate_bits.bytes, block_cache[hash].decoded, block_width, block_height, block_depth, block_type);
				block_cache[hash].encoded = candidate_bits;
				block_cache[hash].valid = true;

				if (block_type == ASTCENC_TYPE_U8)
				{
					return calculate_ssd_weighted(original_decoded, block_cache[hash].decoded, block_width * block_height * block_depth * 4, gradients, channel_weights);
				}
				return calculate_mrsse_weighted((float*)original_decoded, (float*)block_cache[hash].decoded, block_width * block_height * block_depth * 4, gradients, channel_weights);
			};

			// Decode the original block to compute initial MSE
			float original_mse = get_or_compute_mse(current_bits);

			// Calculate masks for weights and endpoints
			Int128 current_weights_mask, current_endpoints_mask;
			calculate_masks(current_weight_bits, current_weights_mask, current_endpoints_mask);
			int mtf_weights_pos = mtf_search(&mtf_weights, current_bits, current_weights_mask);
			int mtf_endpoints_pos = mtf_search(&mtf_endpoints, current_bits, current_endpoints_mask);

			// Before computing best_rd_cost, get the propagated error
			RDError propagated = error_buffer[block_index];
			float adjusted_mse = original_mse + propagated.mse_error;
			float adjusted_rate = calculate_bit_cost_simple(mtf_weights_pos, mtf_endpoints_pos, current_bits, current_weights_mask, current_endpoints_mask, &histogram);
			adjusted_rate += propagated.rate_error;

			// Calculate initial best RD cost
			float best_rd_cost = adjusted_mse + lambda * adjusted_rate;

			// Candidate structure for storing best candidates
			struct Candidate
			{
				Int128 bits;      // ASTC block data
				float rd_cost;    // Rate-distortion cost
				int mtf_position; // MTF position
				int mode;         // Mode
				int weight_bits;  // Weight bits
			};
			Candidate best_weights[BEST_CANDIDATES_COUNT];
			Candidate best_endpoints[BEST_CANDIDATES_COUNT];
			int endpoints_count = 0;
			int weights_count = 0;

			// Function to add a new candidate to the best candidates list
			auto add_candidate = [&](Candidate* candidates, int& count, const Int128& bits, float rd_cost, int mtf_position)
			{
				// Check if candidate should be added
				if (count < BEST_CANDIDATES_COUNT || rd_cost < candidates[BEST_CANDIDATES_COUNT - 1].rd_cost)
				{
					int insert_pos = count < BEST_CANDIDATES_COUNT ? count : BEST_CANDIDATES_COUNT - 1;

					// Find the position to insert
					while (insert_pos > 0 && rd_cost < candidates[insert_pos - 1].rd_cost)
					{
						candidates[insert_pos] = candidates[insert_pos - 1];
						insert_pos--;
					}

					// Extract mode and weight bits from the candidate bits
					int mode = (bits.bytes[0] | (bits.bytes[1] << 8)) & MODE_MASK;
					int weight_bits = get_weight_bits(bits.bytes, weight_bits_tbl);

					// Insert the candidate into the list
					candidates[insert_pos] = {bits, rd_cost, mtf_position, mode, weight_bits};

					// Increment count if not at capacity
					if (count < BEST_CANDIDATES_COUNT)
						count++;
				}
			};

			// Add the current block to the candidates
			float current_rate = calculate_bit_cost_simple(mtf_weights_pos, mtf_endpoints_pos, current_bits, current_weights_mask, current_endpoints_mask, &histogram);
			float current_rd_cost = original_mse + lambda * current_rate;
			add_candidate(best_weights, weights_count, current_bits, current_rd_cost, mtf_weights_pos);
			add_candidate(best_endpoints, endpoints_count, current_bits, current_rd_cost, mtf_endpoints_pos);

			// Replace the ref1 and ref2 bit extraction with the helper function
			Int128 ref1_bits(ref1 + block_index * block_size);
			int ref1_weight_bits = get_weight_bits(ref1_bits.bytes, weight_bits_tbl);
			Int128 ref1_weight_mask, ref1_endpoint_mask;
			calculate_masks(ref1_weight_bits, ref1_weight_mask, ref1_endpoint_mask);
			int mtf_weights_pos_ref1 = mtf_search(&mtf_weights, ref1_bits, ref1_weight_mask);
			int mtf_endpoints_pos_ref1 = mtf_search(&mtf_endpoints, ref1_bits, ref1_endpoint_mask);
			float ref1_mse = get_or_compute_mse(ref1_bits);
			float ref1_rate = calculate_bit_cost_simple(mtf_weights_pos_ref1, mtf_endpoints_pos_ref1, ref1_bits, ref1_weight_mask, ref1_endpoint_mask, &histogram);
			float ref1_rd_cost = ref1_mse + lambda * ref1_rate;
			add_candidate(best_weights, weights_count, ref1_bits, ref1_rd_cost, mtf_weights_pos_ref1);
			add_candidate(best_endpoints, endpoints_count, ref1_bits, ref1_rd_cost, mtf_endpoints_pos_ref1);

			// Add ref2
			Int128 ref2_bits(ref2 + block_index * block_size);
			int ref2_weight_bits = get_weight_bits(ref2_bits.bytes, weight_bits_tbl);
			Int128 ref2_weight_mask, ref2_endpoint_mask;
			calculate_masks(ref2_weight_bits, ref2_weight_mask, ref2_endpoint_mask);
			int mtf_weights_pos_ref2 = mtf_search(&mtf_weights, ref2_bits, ref2_weight_mask);
			int mtf_endpoints_pos_ref2 = mtf_search(&mtf_endpoints, ref2_bits, ref2_endpoint_mask);
			float ref2_mse = get_or_compute_mse(ref2_bits);
			float ref2_rate = calculate_bit_cost_simple(mtf_weights_pos_ref2, mtf_endpoints_pos_ref2, ref2_bits, ref2_weight_mask, ref2_endpoint_mask, &histogram);
			float ref2_rd_cost = ref2_mse + lambda * ref2_rate;
			add_candidate(best_weights, weights_count, ref2_bits, ref2_rd_cost, mtf_weights_pos_ref2);
			add_candidate(best_endpoints, endpoints_count, ref2_bits, ref2_rd_cost, mtf_endpoints_pos_ref2);

			// Find best endpoint candidates
			for (int k = 0; k < mtf_endpoints.size; k++)
			{
				Int128 candidate_endpoints = mtf_endpoints.list[k];
				int endpoints_weight_bits = get_weight_bits(candidate_endpoints.bytes, weight_bits_tbl);

				Int128 weights_mask, endpoints_mask;
				calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);

				float mse = get_or_compute_mse(candidate_endpoints);

				// Find the corresponding weight position
				int weight_pos = mtf_search(&mtf_weights, candidate_endpoints, weights_mask);

				float bit_cost = calculate_bit_cost_simple(weight_pos, k, candidate_endpoints, weights_mask, endpoints_mask, &histogram);
				float rd_cost = mse + lambda * bit_cost;

				// Insert into best_endpoints if it's one of the best candidates
				add_candidate(best_endpoints, endpoints_count, candidate_endpoints, rd_cost, k);
			}

			// Find best weight candidates
			for (int k = 0; k < mtf_weights.size; k++)
			{
				Int128 candidate_weights = mtf_weights.list[k];
				int weights_weight_bits = get_weight_bits(candidate_weights.bytes, weight_bits_tbl);

				Int128 weights_mask, endpoints_mask;
				calculate_masks(weights_weight_bits, weights_mask, endpoints_mask);
				Int128 temp_bits = candidate_weights.bitwise_and(weights_mask);

				// Try every endpoint candidate that matches in weight bits
				for (int m = 0; m < endpoints_count; m++)
				{
					int endpoint_weight_bits = best_endpoints[m].weight_bits;
					if (weights_weight_bits == endpoint_weight_bits && weights_weight_bits != 0)
					{
						Int128 combined_bits = temp_bits.bitwise_or(best_endpoints[m].bits.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(combined_bits);
						float bit_cost = calculate_bit_cost_simple(k, best_endpoints[m].mtf_position, combined_bits, weights_mask, endpoints_mask, &histogram);
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
					Int128 candidate_weights = best_weights[i].bits;
					int endpoints_weight_bits = best_endpoints[j].weight_bits;
					int weights_weight_bits = best_weights[i].weight_bits;
					Int128 weights_mask, endpoints_mask;
					calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);

					// Try candidate_endpoints as-is first
					float rd_cost = best_endpoints[j].rd_cost;
					if (rd_cost < best_rd_cost)
					{
						best_match = candidate_endpoints;
						best_rd_cost = rd_cost;
					}

					// Then try the candidate_endpoints with candidate weights weights
					if (endpoints_weight_bits != 0)
					{
						Int128 temp_bits = candidate_weights.bitwise_and(weights_mask).bitwise_or(candidate_endpoints.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(temp_bits);
						rd_cost = mse + lambda * calculate_bit_cost_simple(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, weights_mask, endpoints_mask, &histogram);
						if (rd_cost < best_rd_cost)
						{
							best_match = temp_bits;
							best_rd_cost = rd_cost;
						}
					}

					// now do the same thing for weights
					if (weights_weight_bits != 0)
					{
						calculate_masks(weights_weight_bits, weights_mask, endpoints_mask);

						Int128 temp_bits = candidate_weights.bitwise_and(weights_mask).bitwise_or(candidate_endpoints.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(temp_bits);
						rd_cost = mse + lambda * calculate_bit_cost_simple(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, weights_mask, endpoints_mask, &histogram);
						if (rd_cost < best_rd_cost)
						{
							best_match = temp_bits;
							best_rd_cost = rd_cost;
						}
					}
				}
			}

			// If we found a better match, update the current block
			if (!best_match.is_equal(current_bits))
				memcpy(current_block, best_match.bytes, 16);

			// Recalculate masks for the best match
			int best_weight_bits = get_weight_bits(current_block, weight_bits_tbl);
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

			// Update statistics
			histo_update(&histogram, best_match, literal_mask);
			mtf_encode(&mtf_weights, best_match, best_weights_mask);
			mtf_encode(&mtf_endpoints, best_match, best_endpoints_mask);

			// After finding the best match, propagate the error
			float final_mse = get_or_compute_mse(best_match);
			float final_rate = calculate_bit_cost_simple(mtf_search(&mtf_weights, best_match, best_weights_mask), mtf_search(&mtf_endpoints, best_match, best_endpoints_mask), best_match, best_weights_mask, best_endpoints_mask, &histogram);

			// Calculate differences for error propagation
			float mse_diff = final_mse - adjusted_mse;
			float rate_diff = final_rate - adjusted_rate;

			// Propagate the error to neighboring blocks
			propagate_error(x, y, z, mse_diff, rate_diff, is_forward);

			// Update progress after processing each block
			blocks_processed++;
			if (thread_id == 0)
			{ // Only thread 0 prints progress
				print_progress();
			}
		};

		// Main thread loop to process work items from the queue
		while (true)
		{
			WorkItem work_item = {};
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cv.wait(lock, [&]() { return !work_queue.empty() || all_work_done; });
				// Exit if all work is done and the queue is empty
				if (all_work_done && work_queue.empty())
				{
					free(block_cache);
					return;
				}
				// Get next work item if available
				if (!work_queue.empty())
				{
					work_item = work_queue.front();
					work_queue.pop();
				}
			}

			// Reset MTF structures for each work item
			if (effort >= 9)
			{
				// Maximum effort: use largest MTF sizes
				mtf_init(&mtf_weights, MAX_MTF_SIZE);
				mtf_init(&mtf_endpoints, MAX_MTF_SIZE);
			}
			else if (effort >= 5)
			{
				// Medium effort: use medium MTF sizes
				mtf_init(&mtf_weights, MTF_WEIGHTS_SIZE * 4);
				mtf_init(&mtf_endpoints, MTF_ENDPOINTS_SIZE);
			}
			else
			{
				// Low effort: use small MTF sizes
				mtf_init(&mtf_weights, MTF_WEIGHTS_SIZE);
				mtf_init(&mtf_endpoints, MTF_ENDPOINTS_SIZE);
			}
			histo_reset(&histogram);

			// Seed the structures with random blocks
			// Use work_item index as part of the seed for variety between chunks
			seed_structures(mtf_weights, mtf_endpoints, histogram, (unsigned)work_item.start_block, work_item.start_block, work_item.end_block);

			// Process the work item
			if (work_item.is_forward)
				// Forward pass: process blocks from start to end
				for (size_t i = work_item.start_block; i < work_item.end_block; i++)
					process_block(i, true);
			else
				// Backward pass: process blocks from end to start
				for (size_t i = work_item.end_block; i-- > work_item.start_block;)
					process_block(i, false);
		}
	};

	// Function to run a single pass (backward or forward)
	auto run_pass = [&](bool is_forward)
	{
		tmZone(0, 0, "run_pass");

		// Fill the work queue with block ranges
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

		// Wait for all threads to finish
		for (auto& thread : threads)
			thread.join();

		// Reset for next pass
		all_work_done = false;
		work_queue = std::queue<WorkItem>();
	};

	// Print start message if not in silent mode
	if (!silentmode)
		printf("Starting optimization pass...\n");

	// Run backward pass
	if (effort >= 8)
		run_pass(false);

	// Run forward pass
	run_pass(true);

	// Print completion message if not in silent mode
	if (!silentmode)
		printf("\nOptimization complete!\n");

	// Free allocated memory
	free(error_buffer);
	free(weight_bits_tbl);
}

// Reconstruct the full image from the decoded blocks
template <typename T>
void reconstruct_image(T* all_original_decoded, int width, int height, int depth, int block_width, int block_height, int block_depth, T* output_image)
{
	tmFunction(0, 0); // Profiling marker

	// Calculate the number of blocks in each dimension
	int blocks_x = (width + block_width - 1) / block_width;
	int blocks_y = (height + block_height - 1) / block_height;
	int blocks_z = (depth + block_depth - 1) / block_depth;
	int channels = 4; // Assuming 4 channels (RGBA)

	// Reconstruct image from blocks
	for (int z = 0; z < blocks_z; z++)
	{
		for (int y = 0; y < blocks_y; y++)
		{
			for (int x = 0; x < blocks_x; x++)
			{
				// Calculate the index of the current block
				int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
				T* block_data = all_original_decoded + block_index * block_width * block_height * block_depth * channels;

				// Copy block pioxels to output image
				for (int bz = 0; bz < block_depth; bz++)
				{
					for (int by = 0; by < block_height; by++)
					{
						for (int bx = 0; bx < block_width; bx++)
						{
							// Calculate image and block pixel coordinatges
							int image_x = x * block_width + bx;
							int image_y = y * block_height + by;
							int image_z = z * block_depth + bz;

							// Only copy pixels within the image bounds
							if (image_x < width && image_y < height && image_z < depth)
							{
								int image_index = (image_z * height * width + image_y * width + image_x) * channels;
								int block_pixel_index = (bz * block_height * block_width + by * block_width + bx) * channels;

								// Copy all channels
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
	tmFunction(0, 0);

	// Calculate kernel radius based on sigma (3 sigma rule)
	*kernel_radius = (int)ceil(3.0f * sigma);
	if (*kernel_radius > MAX_KERNEL_SIZE / 2)
		*kernel_radius = MAX_KERNEL_SIZE / 2;

	// Generate kernel values
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
	tmFunction(0, 0);

	for (int z = 0; z < depth; z++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float sum[4] = {0};
				float kernel_sum = 0.0f; // Track the sum of kernel weights used

				for (int k = -kernel_radius; k <= kernel_radius; k++)
				{
					int sx = direction == 0 ? x + k : x;
					int sy = direction == 1 ? y + k : y;
					int sz = direction == 2 ? z + k : z;

					if (sx >= 0 && sx < width && sy >= 0 && sy < height && sz >= 0 && sz < depth)
					{
						const T* pixel = input + (sz * height * width + sy * width + sx) * channels;
						float kvalue = kernel[k + kernel_radius];
						kernel_sum += kvalue;
						for (int c = 0; c < channels; c++)
							sum[c] += pixel[c] * kvalue;
					}
				}

				// Normalize by the actual sum of kernel weights used
				if (kernel_sum > 0.0f)
				{
					for (int c = 0; c < channels; c++)
						sum[c] /= kernel_sum;
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
void gaussian_blur_3d(const T* input, T* output, int width, int height, int depth, int channels, float sigma, int block_depth)
{
	tmFunction(0, 0);

	float kernel[MAX_KERNEL_SIZE];
	int kernel_radius;
	generate_gaussian_kernel(sigma, kernel, &kernel_radius);

	T* temp1 = (T*)malloc(width * height * depth * channels * sizeof(T));
	T* temp2 = (T*)malloc(width * height * depth * channels * sizeof(T));

	// X direction pass
	apply_1d_convolution_3d(input, temp1, width, height, depth, channels, kernel, kernel_radius, 0);

	// Y direction pass
	apply_1d_convolution_3d(temp1, temp2, width, height, depth, channels, kernel, kernel_radius, 1);

	// Z direction pass (only if depth > 1 AND block_depth > 1)
	if (depth > 1 && block_depth > 1)
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
void high_pass_filter_squared_blurred(const T* input, float* output, int width, int height, int depth, float sigma_highpass, float sigma_blur, const vfloat4& channel_weights, int block_depth)
{
	tmFunction(0, 0);

	(void)channel_weights;
	size_t pixel_count = width * height * depth;
	size_t image_size = pixel_count * 4;
	T* blurred = (T*)malloc(image_size * sizeof(T));
	float* squared_diff = (float*)malloc(pixel_count * sizeof(float));

	// Apply initial Gaussian blur for high-pass filter
	gaussian_blur_3d(input, blurred, width, height, depth, 4, sigma_highpass, block_depth);

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
	gaussian_blur_3d(squared_diff, output, width, height, depth, 1, sigma_blur, block_depth);

	// Map x |-> C1/(C2 + sqrt(x))
	float C1 = 256.0f;
	float C2 = 1.0f;
	float activity_scalar = 4.0f;

	// Create debug output buffer
	// uint8_t* debug_output = (uint8_t*)malloc(pixel_count);
	// float max_val = 0;

	// First pass to find maximum value
	for (size_t i = 0; i < pixel_count; i++)
	{
		float val = C1 / (C2 + activity_scalar * astc::sqrt(output[i]));
		// max_val = astc::max(max_val, val);
		output[i] = val;
	}

	// Second pass to normalize and convert to 8-bit
	// for (size_t i = 0; i < pixel_count; i++)
	//{
	// float normalized = output[i] / max_val;
	// debug_output[i] = (uint8_t)(normalized * 255.0f);
	//}

	// Save debug output as TGA
	// For 3D textures, save first slice only
	// if (depth > 1)
	//{
	// jo_write_tga("activity_mask_slice0.tga", debug_output, width, height, 1, 8);
	//}
	// else
	//{
	// jo_write_tga("activity_mask.tga", debug_output, width, height, 1, 8);
	//}

	// free(debug_output);
	free(blurred);
	free(squared_diff);
}

static void high_pass_to_block_gradients(const float* high_pass_image, float* block_gradients, int width, int height, int depth, int block_width, int block_height, int block_depth)
{
	tmFunction(0, 0);

	int blocks_x = (width + block_width - 1) / block_width;
	int blocks_y = (height + block_height - 1) / block_height;
	int blocks_z = (depth + block_depth - 1) / block_depth;

	// For each block (for 2D slices, each z represents a separate slice)
	for (int z = 0; z < blocks_z; z++)
	{
		for (int y = 0; y < blocks_y; y++)
		{
			for (int x = 0; x < blocks_x; x++)
			{
				int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
				float* block_data = block_gradients + block_index * block_width * block_height * block_depth * 4;

				// For each texel in the block
				for (int bz = 0; bz < block_depth; bz++)
				{
					for (int by = 0; by < block_height; by++)
					{
						for (int bx = 0; bx < block_width; bx++)
						{
							int image_x = x * block_width + bx;
							int image_y = y * block_height + by;
							int image_z = z * block_depth + bz;

							// Handle edge cases where block extends beyond image bounds
							if (image_x < width && image_y < height && image_z < depth)
							{
								int image_index = (image_z * height * width + image_y * width + image_x);
								int block_texel_index = (bz * block_height * block_width + by * block_width + bx) * 4;

								// Copy high-pass value to all 4 channels of block gradients
								float gradient = high_pass_image[image_index];
								block_data[block_texel_index + 0] = gradient;
								block_data[block_texel_index + 1] = gradient;
								block_data[block_texel_index + 2] = gradient;
								block_data[block_texel_index + 3] = gradient;
							}
							else
							{
								// For texels outside image bounds, use zero gradient
								int block_texel_index = (bz * block_height * block_width + by * block_width + bx) * 4;
								block_data[block_texel_index + 0] = 0.0f;
								block_data[block_texel_index + 1] = 0.0f;
								block_data[block_texel_index + 2] = 0.0f;
								block_data[block_texel_index + 3] = 0.0f;
							}
						}
					}
				}
			}
		}
	}
}

static inline float get_swizzled_value(const float components[4], const astcenc_swz swz)
{
	switch (swz)
	{
	case ASTCENC_SWZ_R:
	case ASTCENC_SWZ_G:
	case ASTCENC_SWZ_B:
	case ASTCENC_SWZ_A:
		return components[swz];
	case ASTCENC_SWZ_0:
		return 0.0f;
	case ASTCENC_SWZ_1:
		return 1.0f;
	case ASTCENC_SWZ_Z:
		// Reconstruct Z from X and Y assuming unit vector
		float x = components[0] * 2.0f - 1.0f;
		float y = components[1] * 2.0f - 1.0f;
		float z = 1.0f - x * x - y * y;
		return z <= 0.0f ? 0.5f : (astc::sqrt(z) * 0.5f + 0.5f);
	}
	return 0.0f;
}

// Main optimization function for ASTC Lz optimization
astcenc_error astcenc_optimize_for_lz(astcenc_image* image_uncomp_in, int image_uncomp_in_component_count, bool image_uncomp_in_is_hdr, const astcenc_swizzle* swizzle, uint8_t* data, uint8_t* exhaustive_data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float channel_weights[4], int thread_count, bool silentmode, float lambda, float effort)
{
	// Main optimization logic
	// Uses dual_mtf_pass to perform forward and backward passes
	// over the compressed data, attempting to optimize each block
	// while maintaining good compression ratio.

	tmFunction(0, 0);

	// nothing to do if lambda is 0
	if (lambda <= 0)
	{
		memcpy(data, exhaustive_data, data_len);
		return ASTCENC_SUCCESS;
	}

	for (int i = 0; i < 4; ++i)
		channel_weights[i] = channel_weights[i] * 0.25f;

	vfloat4 channel_weights_vec = vfloat4(channel_weights[0], channel_weights[1], channel_weights[2], channel_weights[3]);

	// Map lambda from [10, 40] to ...
	float lambda_0 = 0.0f;
	float lambda_5 = 2.5f;
	float lambda_10 = 4.f;
	float lambda_40 = 9.5f;

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

	float lambda_scale = astc::sqrt((block_width * block_height * block_depth) / 16.f);
	lambda *= lambda_scale;

	// Initialize block_size_descriptor once
	block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
	init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 43, *bsd);

	// Calculate dimensions
	int width = blocks_x * block_width;
	int height = blocks_y * block_height;
	int depth = blocks_z * block_depth;

	// Calculate block data sizes
	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	size_t decoded_block_size = 0;
	switch (block_type)
	{
	case ASTCENC_TYPE_U8:
		decoded_block_size = block_width * block_height * block_depth * 4;
		break;
	case ASTCENC_TYPE_F16:
	case ASTCENC_TYPE_F32:
		decoded_block_size = block_width * block_height * block_depth * 4 * 4; // Always use float (4 bytes per component)
		break;
	}

	// Allocate memory for block-organized uncompressed data
	uint8_t* all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);

	// Convert image_uncomp_in data into block-organized format
	for (int z = 0; z < blocks_z; z++)
	{
		for (int y = 0; y < blocks_y; y++)
		{
			for (int x = 0; x < blocks_x; x++)
			{
				int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
				uint8_t* block_data = all_original_decoded + block_index * decoded_block_size;

				// Copy pixels from image_uncomp_in to block data
				for (int bz = 0; bz < block_depth; bz++)
				{
					for (int by = 0; by < block_height; by++)
					{
						for (int bx = 0; bx < block_width; bx++)
						{
							// Clamp coordinates to image bounds
							int image_x = astc::min<int>(x * block_width + bx, image_uncomp_in->dim_x - 1);
							int image_y = astc::min<int>(y * block_height + by, image_uncomp_in->dim_y - 1);
							int image_z = astc::min<int>(z * block_depth + bz, image_uncomp_in->dim_z - 1);

							int image_index = (image_z * image_uncomp_in->dim_y * image_uncomp_in->dim_x + image_y * image_uncomp_in->dim_x + image_x) * image_uncomp_in_component_count;
							int block_pixel_index = (bz * block_height * block_width + by * block_width + bx) * 4;

							// Load components as floats first (for consistent swizzling)
							float components[4] = {0.0f, 0.0f, 0.0f, 1.0f}; // Default values
							for (int c = 0; c < image_uncomp_in_component_count; c++)
							{
								if (image_uncomp_in_is_hdr)
								{
									if (image_uncomp_in->data_type == ASTCENC_TYPE_F16)
									{
										uint16_t f16_val = ((uint16_t*)image_uncomp_in->data[0])[image_index + c];
										components[c] = float16_to_float(f16_val);
									}
									else
									{
										components[c] = ((float*)image_uncomp_in->data[0])[image_index + c];
									}
								}
								else
								{
									uint8_t u8_val = ((uint8_t*)image_uncomp_in->data[0])[image_index + c];
									components[c] = (float)u8_val / 255.0f;
								}
							}

							// Apply swizzle and store
							float swizzled[4];
							swizzled[0] = get_swizzled_value(components, swizzle->r);
							swizzled[1] = get_swizzled_value(components, swizzle->g);
							swizzled[2] = get_swizzled_value(components, swizzle->b);
							swizzled[3] = get_swizzled_value(components, swizzle->a);

							if (block_type == ASTCENC_TYPE_U8)
							{
								uint8_t* block_data_u8 = (uint8_t*)block_data;
								block_data_u8[block_pixel_index + 0] = (uint8_t)(swizzled[0] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 1] = (uint8_t)(swizzled[1] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 2] = (uint8_t)(swizzled[2] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 3] = (uint8_t)(swizzled[3] * 255.0f + 0.5f);
							}
							else
							{
								float* block_data_f32 = (float*)block_data;
								block_data_f32[block_pixel_index + 0] = swizzled[0];
								block_data_f32[block_pixel_index + 1] = swizzled[1];
								block_data_f32[block_pixel_index + 2] = swizzled[2];
								block_data_f32[block_pixel_index + 3] = swizzled[3];
							}
						}
					}
				}
			}
		}
	}

	// Preserve original blocks
	uint8_t* original_blocks = (uint8_t*)malloc(data_len);
	memcpy(original_blocks, data, data_len);

	// Allocate memory for the reconstructed image
	size_t image_size = width * height * depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
	uint8_t* reconstructed_image = (uint8_t*)malloc(image_size);
	float* high_pass_image = (float*)malloc(width * height * depth * sizeof(float)); // Single channel
	float* block_gradients = (float*)malloc(num_blocks * block_width * block_height * block_depth * 4 * sizeof(float));

	if (block_type == ASTCENC_TYPE_U8)
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image(all_original_decoded, width, height, depth, block_width, block_height, block_depth, reconstructed_image);

		// Apply high-pass filter with squared differences and additional blur
		high_pass_filter_squared_blurred(reconstructed_image, high_pass_image, width, height, depth, 2.2f, 1.25f, channel_weights_vec, block_depth);
	}
	else
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image((float*)all_original_decoded, width, height, depth, block_width, block_height, block_depth, (float*)reconstructed_image);

		// Apply high-pass filter with squared differences and additional blur
		high_pass_filter_squared_blurred((float*)reconstructed_image, high_pass_image, width, height, depth, 2.2f, 1.25f, channel_weights_vec, block_depth);
	}

	// Convert high-pass filtered image back to block gradients
	high_pass_to_block_gradients(high_pass_image, block_gradients, width, height, depth, block_width, block_height, block_depth);

	dual_mtf_pass(thread_count, silentmode, data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_gradients, channel_weights_vec, effort);
	if (effort >= 5)
	{
		dual_mtf_pass(thread_count, silentmode, data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_gradients, channel_weights_vec, effort);
	}

	// Clean up
	free(bsd);
	free(all_original_decoded);
	free(original_blocks);
	free(reconstructed_image);
	free(high_pass_image);
	free(block_gradients);

	return ASTCENC_SUCCESS;
}
