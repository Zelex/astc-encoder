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
#include <cmath>
#include <cstdlib>
#include <cstring> // For memcpy

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>

// Add platform-specific aligned allocation macros
#if defined(_MSC_VER)
	// Windows/MSVC
	#include <malloc.h>
	#define ASTC_ALIGNED_MALLOC(size, align) _aligned_malloc(size, align)
	#define ASTC_ALIGNED_FREE(ptr) _aligned_free(ptr)
#elif defined(__APPLE__) || defined(__linux__)
	// macOS, iOS, or Linux with C11 support
	#define ASTC_ALIGNED_MALLOC(size, align) aligned_alloc(align, (size + align - 1) & ~(align - 1))
	#define ASTC_ALIGNED_FREE(ptr) free(ptr)
#else
	// Fallback for other platforms - you might want to add more platform-specific cases
	#include <cstdlib>
	static inline void* aligned_alloc_fallback(size_t align, size_t size) 
	{
		void* ptr = nullptr;
		if (posix_memalign(&ptr, align, size) != 0) 
			return nullptr;
		return ptr;
	}
	#define ASTC_ALIGNED_MALLOC(size, align) aligned_alloc_fallback(align, size)
	#define ASTC_ALIGNED_FREE(ptr) free(ptr)
#endif

#include "astcenc.h"
#include "astcenc_internal.h"
#include "astcenc_internal_entry.h"
#include "astcenc_mathlib.h"
#include "astcenc_vecmathlib.h"

#include "ThirdParty/rad_tm_stub.h"

// Maximum size for Move-To-Front (MTF) lists
static constexpr unsigned int MAX_MTF_SIZE{1024 + 256 + 64 + 16 + 3};
static constexpr unsigned int MTF_ENDPOINTS_SIZE{256 + 64 + 16 + 3};
static constexpr unsigned int MTF_WEIGHTS_SIZE{16 + 3};
// Cache size for block decompression results, should be power of 2
static constexpr unsigned int CACHE_SIZE{0x10000};
// Number of best candidates to consider for each block
static constexpr unsigned int BEST_CANDIDATES_COUNT{8};
// Maximum number of threads to use
static constexpr unsigned int MAX_THREADS{128};
// Mode mask for block types
static constexpr unsigned int MODE_MASK{0x7FF};
// Maximum number of blocks to process per worker thread item
static constexpr unsigned int MAX_BLOCKS_PER_ITEM{8192};

// 128-bit integer structure for handling ASTC block data
// Used to manipulate compressed ASTC blocks which are 128 bits (16 bytes) in size
struct int128 
{
	union 
	{
		uint64_t uint64[2]; // Access as two 64-bit integers
		uint32_t uint32[4]; // Access as four 32-bit integers
		uint8_t bytes[16];  // Access as 16 bytes
	};

	// Default constructor initializes to zero
	int128() : uint64() 
	{
	}

	// Constructor from raw byte data
	explicit int128(
		const uint8_t *data
	) {
		memcpy(bytes, data, 16);
	}

	// Left shift operation with bounds checking
	int128 shift_left(
		int shift
	) const {
		int128 result;
		if (shift >= 128) 
			return result; // Returns zero
		if (shift == 0) 
			return *this;
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
	int128 bitwise_and(
		const int128 &other
	) const {
		int128 result;
		result.uint64[0] = uint64[0] & other.uint64[0];
		result.uint64[1] = uint64[1] & other.uint64[1];
		return result;
	}

	// Bitwise OR operation
	int128 bitwise_or(
		const int128 &other
	) const {
		int128 result;
		result.uint64[0] = uint64[0] | other.uint64[0];
		result.uint64[1] = uint64[1] | other.uint64[1];
		return result;
	}

	// Equality check
	bool is_equal(
		const int128 &other
	) const {
		return uint64[0] == other.uint64[0] && uint64[1] == other.uint64[1];
	}

	// Constructor from integer value
	static int128 from_int(
		long long val
	) {
		int128 result;
		result.uint64[0] = val;
		result.uint64[1] = val < 0 ? -1LL : 0;
		return result;
	}

	// Bitwise NOT operation
	int128 bitwise_not(
	) const {
		int128 result;
		result.uint64[0] = ~uint64[0];
		result.uint64[1] = ~uint64[1];
		return result;
	}
};

// Histogram structure for tracking byte frequencies in compressed data
// Used for entropy coding and compression ratio estimation
struct histogram 
{
	int h[256]; // Frequency count for each byte
	int size;   // Total count
};

// Move-To-Front (MTF) list structure for maintaining recently used values
// Helps exploit temporal locality in the compressed data
struct mtf_list 
{
	int128 list[MAX_MTF_SIZE]; // List of recently used values
	int size;                  // Current size
	int max_size;              // Maximum size
};

// Cache entry for storing decoded block results
// Reduces redundant decompression operations
struct block_cache_entry 
{
	int128 encoded;                     // Encoded block data
	uint8_t decoded[6 * 6 * 6 * 4 * 4]; // Max size for both U8 and float types
	bool valid;                         // Validity flag
};

/*
static void jo_write_tga(
	const char *filename, 
	void *rgba, 
	int width, 
	int height, 
	int numChannels, 
	int bit_depth
) {
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
	for(int i = 0; i < width*height; ++i) 
	{ 
		for(int j = 0; j < numChannels; ++j) 
		{
			fwrite(s + remap[j], byte_depth, 1, fp);
		}
		s += numChannels*byte_depth;
	}
	fclose(fp);
}
*/

// Fast log2 approximation function
// Note: Could use float_as_int and int_as_float to avoid the union, but the compiler 
// mixes it up with the SIMD versions and it won't work.
static inline float log2_fast(
	float val
) {
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
static uint32_t hash_128(
	const int128 &value
) {
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
static void histo_reset(
	histogram *h
) {
	memset(h->h, 0, sizeof(h->h));
	h->size = 0;
}

// Update histogram with byte frequencies from masked value
static void histo_update(
	histogram *h, 
	const int128 &value, 
	const int128 &mask
) {
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
static float histo_cost(
	histogram *h, 
	int128 value, 
	int128 mask
) {
	// Return 0 if mask is all zeros (nothing to encode)
	if (mask.uint64[0] == 0 && mask.uint64[1] == 0)
		return 0.0f;

	float tlb = (float)h->size + 1.0f; // Total bytes plus 1 for Laplace smoothing
	float cost = 1.0f;

	// Process each byte individually to avoid endianness assumptions
	for (int i = 0; i < 16; i++) 
	{
		// Only process bytes where mask is non-zero
		if (mask.bytes[i]) 
		{
			cost *= tlb / (h->h[value.bytes[i]] + 1.0f);
		}
	}

	// Return log2 of final cost
	return log2_fast(cost);
}

// Initialize Move-To-Front list with given maximum size
static void mtf_init(
	mtf_list *mtf, 
	int max_size
) {
	mtf->size = 0;
	mtf->max_size = max_size > MAX_MTF_SIZE ? MAX_MTF_SIZE : max_size;
}

// Search for a masked value in the MTF list
// Returns position if found, -1 if not found
static int mtf_search(
	mtf_list *mtf, 
	const int128 &value, 
	const int128 &mask
) {
	// Pre-compute the masked value once for efficiency
	int128 masked_value = value.bitwise_and(mask);

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
static int mtf_encode(
	mtf_list *mtf, 
	const int128 &value, 
	const int128 &mask
) {
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
// Returns the total bit cost combining MTF positions and literal encoding costs
static float calculate_bit_cost_simple(
	int mtf_value_1,             // MTF position for first part (-1 if not found)
	int mtf_value_2,             // MTF position for second part (-1 if not found)
	const int128 &literal_value, // Full literal value to encode
	const int128 &mask_1,        // Mask for first part
	const int128 &mask_2,        // Mask for second part
	histogram *histogram
) // Histogram for literal encoding costs
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
static inline float calculate_ssd_weighted(
	const uint8_t* img1,           // first image data
	const uint8_t* img2,           // second image data
	int texel_count,               // total number of texels
	const float* weights,          // Per-texel weights (one float per texel)
	const vfloat4& channel_weights // Per-channel importance weights
) {
	vfloat4 sum = vfloat4::zero();

	for(int i = 0; i < texel_count; i++)
	{
		vfloat4 diff = int_to_float(vint4(img1 + i*4) - vint4(img2 + i*4));
		haccumulate(sum, diff * diff * vfloat4::load1(weights + i));
	}

	return dot_s(sum, channel_weights);
}

// Calculate Mean Relative Sum of Squared Errors (MRSSE) with weights for float image data
static inline float calculate_mrsse_weighted(
	const float* img1,             // first image data
	const float* img2,             // second image data
	int texel_count,               // total number of texels
	const float* weights,          // Per-texel weights (one float per texel)
	const vfloat4& channel_weights // Per-channel importance weights
) {
	vfloat4 sum = vfloat4::zero();
	for(int i = 0; i < texel_count; i++)
	{
		vfloat4 diff = vfloat4(img1 + i*4) - vfloat4(img2 + i*4);
		haccumulate(sum, diff * diff * vfloat4::load1(weights + i));
	}
	return dot_s(sum, channel_weights) * 256.0f;
}

// Decompress an ASTC block to either U8 or float output
static void astc_decompress_block(
	const block_size_descriptor &bsd, // Block size descriptor
	const uint8_t *block_ptr,         // Block data pointer
	uint8_t *output,                  // Output buffer
	int block_width,                  // Block dimensions
	int block_height,
	int block_depth,
	int block_type // Block type (ASTCENC_TYPE_U8 or ASTCENC_TYPE_F32)
) {
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
		float *output_f = reinterpret_cast<float *>(output);
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
int get_weight_bits(
	uint8_t *data, 
	int block_width, 
	int block_height, 
	int block_depth
) {
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
		} else {
			p12 = b01;
			// Alternative block dimensions calculations
			switch (b23) {
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
	} else {
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

// Structure to define a work item for thread processing
struct work_item 
{
	size_t start_block; // Starting block index
	size_t end_block;   // Ending block index ( exclusive )
	bool is_forward;    // Forward or backward pass
};

// Calculate masks for weights and endpoints based on weight bits
static inline void calculate_masks(
	int weight_bits, 
	int128 &weights_mask, 
	int128 &endpoints_mask
) {
	// Create weight mask by shifting 1s into position
	weights_mask = int128::from_int(-1).shift_left(128 - weight_bits);
	// Endpoints mask is complement of weights mask
	endpoints_mask = weights_mask.bitwise_not();
}

// Get weight bits from a block using pre-computed lookup table
static inline int get_weight_bits(
	const uint8_t *block, 
	const uint8_t *weight_bits_tbl
) {
	return weight_bits_tbl[(block[0] | (block[1] << 8)) & MODE_MASK];
}

// Structure to track rate-distortion errors
struct rd_error 
{
	float mse_error;  // Mean squared error component
	float rate_error; // Rate (compression) error component
};

static inline uint32_t xorshift32(
	uint32_t &state
) {
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	return state;
}

// Performs forward and backwards optimization passes over blocks
static void dual_mtf_pass(
	int thread_count, // Number of threads to use
	bool silentmode,  // Supress progress output
	uint8_t *data,    // Input/output compressed data
	uint8_t *ref1,    // Pointer to reference 1 data
	uint8_t *ref2,    // Pointer to reference 2 data
	size_t data_len,  // Length of input data
	int blocks_x,     // Block dimensions
	int blocks_y,
	int blocks_z,                  // Block dimensions
	int block_width,               // Block size
	int block_height,              // Block dimensions
	int block_depth,               // Block dimensions
	int block_type,                // ASTC block type
	float lambda,                  // Lambda parameter for rate-distortion optimization
	block_size_descriptor *bsd,    // Block size descriptor
	uint8_t *all_original_decoded, // Pointer to original decoded data
	float *per_texel_weights,      // Per-texel weights for activity
	vfloat4 channel_weights,       // Channel importance weights
	float effort                   // Optimization effort
) {
	// Uses multiple threads to process blocks
	// Each thread maintains its own MTF lists and histogram
	// Processes blocks in chunks, applying rate-distortion optimization

	tmFunction(0, 0); // Timing/profiling marker

	const int block_size = 16;
	size_t num_blocks = data_len / block_size;
	// Limit thread count to hardware, MAX_THREADS, and requested amount
	const int num_threads = astc::min<int>((int)thread_count, MAX_THREADS, (int)std::thread::hardware_concurrency());
	// Allocate error buffer for all blocks
	rd_error *error_buffer = new rd_error[blocks_x * blocks_y * blocks_z]();

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
	uint8_t *weight_bits_tbl = new uint8_t[2048];
	for (size_t i = 0; i < 2048; ++i) 
	{
		uint8_t block[16];
		block[0] = (i & 255);
		block[1] = ((i >> 8) & 255);
		weight_bits_tbl[i] = (uint8_t)get_weight_bits(block, block_width, block_height, block_depth);
	}

	// Thread synchronization primitives
	std::queue<work_item> work_queue;
	std::mutex queue_mutex;
	std::condition_variable cv;
	bool all_work_done = false;

	// Main thread worker function
	auto thread_function = [&](int thread_id) 
	{
		tmProfileThread(0, 0, 0); // Thread profiling marker
		tmZone(0, 0, "thread_function");

		// Allocate thread-local resources
		block_cache_entry *block_cache = new block_cache_entry[CACHE_SIZE]();
		mtf_list mtf_weights;   // MTF list for weights
		mtf_list mtf_endpoints; // MTF list for endpoints
		histogram hist;         // Histogram for encoding statistics

		// Function to initialize MTF and histogram with random blocks
		auto seed_structures = [&](mtf_list &mtf_w, mtf_list &mtf_e, histogram &hist, uint32_t seed, size_t start_block, size_t end_block) 
		{
			// This needs to be consistently initialized, and we also need the splitting into jobs to
			// be consistent, to avoid this random sampling introducing non-determinism between runs
			uint32_t rng_state = seed;

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
				int128 block_bits(data + block_idx * 16);
				int block_weight_bits = get_weight_bits(block_bits.bytes, weight_bits_tbl);

				// Calculate masks
				int128 weights_mask, endpoints_mask;
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
			size_t texels_per_block = block_width * block_height * block_depth;

			// Get current block data and compute its weight bits
			uint8_t *current_block = data + block_index * block_size;
			int128 current_bits(current_block);
			int current_weight_bits = get_weight_bits(current_bits.bytes, weight_bits_tbl);
			int128 best_match = current_bits;

			// Don't process blocks with no weight bits, accept void-extent as is
			if (current_weight_bits == 0) 
			{
				histo_update(&hist, current_bits, int128::from_int(-1));
				mtf_encode(&mtf_weights, current_bits, int128::from_int(0));
				mtf_encode(&mtf_endpoints, current_bits, int128::from_int(-1));
				return;
			}

			// Get pointers to original decoded data and gradients for this block
			uint8_t *original_decoded = all_original_decoded + block_index * (texels_per_block * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : sizeof(float)));
			const float *block_texel_weights = per_texel_weights + block_index * texels_per_block;

			// Function to get or compute MSE for a candidate
			auto get_or_compute_mse = [&](const int128& candidate_bits) -> float 
			{
				uint32_t hash = hash_128(candidate_bits) & (CACHE_SIZE - 1); // Modulo CACHE_SIZE
				int block_texel_count = block_width * block_height * block_depth;

				// If our current cached block doesn't match, decode first
				if (!block_cache[hash].valid || !block_cache[hash].encoded.is_equal(candidate_bits))
				{
					astc_decompress_block(*bsd, candidate_bits.bytes, block_cache[hash].decoded, block_width, block_height, block_depth, block_type);
					block_cache[hash].encoded = candidate_bits;
					block_cache[hash].valid = true;
				}

				// Compute error using cached decoded data
				if (block_type == ASTCENC_TYPE_U8) 
				{
					return calculate_ssd_weighted(original_decoded, block_cache[hash].decoded, block_texel_count, block_texel_weights, channel_weights);
				}
				else
				{
					return calculate_mrsse_weighted((float *)original_decoded, (float *)block_cache[hash].decoded, block_texel_count, block_texel_weights, channel_weights);
				}
			};

			// Decode the original block to compute initial MSE
			float original_mse = get_or_compute_mse(current_bits);

			// Calculate masks for weights and endpoints
			int128 current_weights_mask, current_endpoints_mask;
			calculate_masks(current_weight_bits, current_weights_mask, current_endpoints_mask);
			int mtf_weights_pos = mtf_search(&mtf_weights, current_bits, current_weights_mask);
			int mtf_endpoints_pos = mtf_search(&mtf_endpoints, current_bits, current_endpoints_mask);

			// Before computing best_rd_cost, get the propagated error
			rd_error propagated = error_buffer[block_index];
			float adjusted_mse = original_mse + propagated.mse_error;
			float adjusted_rate = calculate_bit_cost_simple(mtf_weights_pos, mtf_endpoints_pos, current_bits, current_weights_mask, current_endpoints_mask, &hist);
			adjusted_rate += propagated.rate_error;

			// Calculate initial best RD cost
			float best_rd_cost = adjusted_mse + lambda * adjusted_rate;

			// Candidate structure for storing best candidates
			struct candidate 
			{
				int128 bits;      // ASTC block data
				float rd_cost;    // Rate-distortion cost
				int mtf_position; // MTF position
				int mode;         // Mode
				int weight_bits;  // Weight bits
			};
			candidate best_weights[BEST_CANDIDATES_COUNT];
			candidate best_endpoints[BEST_CANDIDATES_COUNT];
			int endpoints_count = 0;
			int weights_count = 0;

			// Function to add a new candidate to the best candidates list
			auto add_candidate = [&](candidate *candidates, int &count, const int128 &bits, float rd_cost, int mtf_position) 
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
			float current_rate = calculate_bit_cost_simple(mtf_weights_pos, mtf_endpoints_pos, current_bits, current_weights_mask, current_endpoints_mask, &hist);
			float current_rd_cost = original_mse + lambda * current_rate;
			add_candidate(best_weights, weights_count, current_bits, current_rd_cost, mtf_weights_pos);
			add_candidate(best_endpoints, endpoints_count, current_bits, current_rd_cost, mtf_endpoints_pos);

			// Replace the ref1 and ref2 bit extraction with the helper function
			int128 ref1_bits(ref1 + block_index * block_size);
			int ref1_weight_bits = get_weight_bits(ref1_bits.bytes, weight_bits_tbl);
			int128 ref1_weight_mask, ref1_endpoint_mask;
			calculate_masks(ref1_weight_bits, ref1_weight_mask, ref1_endpoint_mask);
			int mtf_weights_pos_ref1 = mtf_search(&mtf_weights, ref1_bits, ref1_weight_mask);
			int mtf_endpoints_pos_ref1 = mtf_search(&mtf_endpoints, ref1_bits, ref1_endpoint_mask);
			float ref1_mse = get_or_compute_mse(ref1_bits);
			float ref1_rate = calculate_bit_cost_simple(mtf_weights_pos_ref1, mtf_endpoints_pos_ref1, ref1_bits, ref1_weight_mask, ref1_endpoint_mask, &hist);
			float ref1_rd_cost = ref1_mse + lambda * ref1_rate;
			add_candidate(best_weights, weights_count, ref1_bits, ref1_rd_cost, mtf_weights_pos_ref1);
			add_candidate(best_endpoints, endpoints_count, ref1_bits, ref1_rd_cost, mtf_endpoints_pos_ref1);

			// Add ref2
			int128 ref2_bits(ref2 + block_index * block_size);
			int ref2_weight_bits = get_weight_bits(ref2_bits.bytes, weight_bits_tbl);
			int128 ref2_weight_mask, ref2_endpoint_mask;
			calculate_masks(ref2_weight_bits, ref2_weight_mask, ref2_endpoint_mask);
			int mtf_weights_pos_ref2 = mtf_search(&mtf_weights, ref2_bits, ref2_weight_mask);
			int mtf_endpoints_pos_ref2 = mtf_search(&mtf_endpoints, ref2_bits, ref2_endpoint_mask);
			float ref2_mse = get_or_compute_mse(ref2_bits);
			float ref2_rate = calculate_bit_cost_simple(mtf_weights_pos_ref2, mtf_endpoints_pos_ref2, ref2_bits, ref2_weight_mask, ref2_endpoint_mask, &hist);
			float ref2_rd_cost = ref2_mse + lambda * ref2_rate;
			add_candidate(best_weights, weights_count, ref2_bits, ref2_rd_cost, mtf_weights_pos_ref2);
			add_candidate(best_endpoints, endpoints_count, ref2_bits, ref2_rd_cost, mtf_endpoints_pos_ref2);

			// Find best endpoint candidates
			for (int k = 0; k < mtf_endpoints.size; k++) 
			{
				int128 candidate_endpoints = mtf_endpoints.list[k];
				int endpoints_weight_bits = get_weight_bits(candidate_endpoints.bytes, weight_bits_tbl);

				int128 weights_mask, endpoints_mask;
				calculate_masks(endpoints_weight_bits, weights_mask, endpoints_mask);

				float mse = get_or_compute_mse(candidate_endpoints);

				// Find the corresponding weight position
				int weight_pos = mtf_search(&mtf_weights, candidate_endpoints, weights_mask);

				float bit_cost = calculate_bit_cost_simple(weight_pos, k, candidate_endpoints, weights_mask, endpoints_mask, &hist);
				float rd_cost = mse + lambda * bit_cost;

				// Insert into best_endpoints if it's one of the best candidates
				add_candidate(best_endpoints, endpoints_count, candidate_endpoints, rd_cost, k);
			}

			// Find best weight candidates
			for (int k = 0; k < mtf_weights.size; k++) 
			{
				int128 candidate_weights = mtf_weights.list[k];
				int weights_weight_bits = get_weight_bits(candidate_weights.bytes, weight_bits_tbl);

				int128 weights_mask, endpoints_mask;
				calculate_masks(weights_weight_bits, weights_mask, endpoints_mask);
				int128 temp_bits = candidate_weights.bitwise_and(weights_mask);

				// Try every endpoint candidate that matches in weight bits
				for (int m = 0; m < endpoints_count; m++) 
				{
					int endpoint_weight_bits = best_endpoints[m].weight_bits;
					if (weights_weight_bits == endpoint_weight_bits && weights_weight_bits != 0) 
					{
						int128 combined_bits = temp_bits.bitwise_or(best_endpoints[m].bits.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(combined_bits);
						float bit_cost = calculate_bit_cost_simple(k, best_endpoints[m].mtf_position, combined_bits, weights_mask, endpoints_mask, &hist);
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
					int128 candidate_endpoints = best_endpoints[j].bits;
					int128 candidate_weights = best_weights[i].bits;
					int endpoints_weight_bits = best_endpoints[j].weight_bits;
					int weights_weight_bits = best_weights[i].weight_bits;
					int128 weights_mask, endpoints_mask;
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
						int128 temp_bits = candidate_weights.bitwise_and(weights_mask).bitwise_or(candidate_endpoints.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(temp_bits);
						rd_cost = mse + lambda * calculate_bit_cost_simple(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, weights_mask, endpoints_mask, &hist);
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

						int128 temp_bits = candidate_weights.bitwise_and(weights_mask).bitwise_or(candidate_endpoints.bitwise_and(endpoints_mask));
						float mse = get_or_compute_mse(temp_bits);
						rd_cost = mse + lambda * calculate_bit_cost_simple(best_weights[i].mtf_position, best_endpoints[j].mtf_position, temp_bits, weights_mask, endpoints_mask, &hist);
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
			int128 best_weights_mask, best_endpoints_mask;
			calculate_masks(best_weight_bits, best_weights_mask, best_endpoints_mask);

			// Update histogram with literal mask
			int best_mtf_weights_pos = mtf_search(&mtf_weights, best_match, best_weights_mask);
			int best_mtf_endpoints_pos = mtf_search(&mtf_endpoints, best_match, best_endpoints_mask);
			int128 literal_mask = int128::from_int(0);
			if (best_mtf_weights_pos == -1)
				literal_mask = literal_mask.bitwise_or(best_weights_mask);
			if (best_mtf_endpoints_pos == -1)
				literal_mask = literal_mask.bitwise_or(best_endpoints_mask);

			// Update statistics
			histo_update(&hist, best_match, literal_mask);
			mtf_encode(&mtf_weights, best_match, best_weights_mask);
			mtf_encode(&mtf_endpoints, best_match, best_endpoints_mask);

			// After finding the best match, propagate the error
			float final_mse = get_or_compute_mse(best_match);
			float final_rate = calculate_bit_cost_simple(mtf_search(&mtf_weights, best_match, best_weights_mask), mtf_search(&mtf_endpoints, best_match, best_endpoints_mask), best_match, best_weights_mask, best_endpoints_mask, &hist);

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
			work_item work_item = {};
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cv.wait(lock, [&]() { return !work_queue.empty() || all_work_done; });
				// Exit if all work is done and the queue is empty
				if (all_work_done && work_queue.empty()) 
				{
					delete[] block_cache;
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
			histo_reset(&hist);

			// Seed the structures with random blocks
			// Use work_item index as part of the seed for variety between chunks
			seed_structures(mtf_weights, mtf_endpoints, hist, (unsigned)work_item.start_block, work_item.start_block, work_item.end_block);

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
		for (auto &thread : threads)
			thread.join();

		// Reset for next pass
		all_work_done = false;
		work_queue = std::queue<work_item>();
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
	delete[] error_buffer;
	delete[] weight_bits_tbl;
}

// Reconstruct the full image from the decoded blocks
template <typename T> void reconstruct_image(
	T *all_original_decoded, 
	int width, 
	int height, 
	int depth, 
	int block_width, 
	int block_height, 
	int block_depth, 
	T *output_image
) {
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
				T *block_data = all_original_decoded + block_index * block_width * block_height * block_depth * channels;

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

static constexpr int MAX_KERNEL_SIZE { 33 };

// Generate 1D Gaussian kernel
static void generate_gaussian_kernel(
	float sigma, 
	float *kernel, 
	int *kernel_radius
) {
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
static void apply_1d_convolution_3d(
	const T *input, 
	T *output, 
	int width, 
	int height, 
	int depth, 
	int channels, 
	const float *kernel, 
	int kernel_radius, 
	int direction
) {
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
						const T *pixel = input + (sz * height * width + sy * width + sx) * channels;
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

				T *out_pixel = output + (z * height * width + y * width + x) * channels;
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
template <typename T> static void gaussian_blur_3d(
	const T *input, 
	T *output, 
	int width, 
	int height, 
	int depth, 
	int channels, 
	float sigma, 
	int block_depth
) {
	tmFunction(0, 0);

	float kernel[MAX_KERNEL_SIZE];
	int kernel_radius;
	generate_gaussian_kernel(sigma, kernel, &kernel_radius);

	T *temp1 = new T[width * height * depth * channels];
	T *temp2 = new T[width * height * depth * channels];

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

	delete[] temp1;
	delete[] temp2;
}

template <typename T> static void compute_activity_map(
	const T *input, 
	float *output, 
	int width, 
	int height, 
	int depth, 
	float sigma_highpass, 
	float sigma_blur, 
	const vfloat4 &channel_weights, 
	int block_depth,
	int thread_count,
	bool silentmode
) {
	tmFunction(0, 0);

	auto start_time = std::chrono::high_resolution_clock::now();

	// Use fixed tile dimensions based on image type
	int tile_size_x, tile_size_y, tile_size_z;
	if (depth <= 1) {
		// 2D image - use 256x256 tiles
		tile_size_x = tile_size_y = 256;
		tile_size_z = 1;
	} else {
		// 3D image - use 32x32x32 tiles
		tile_size_x = tile_size_y = tile_size_z = 32;
	}

	// Calculate number of tiles needed in each dimension
	int tiles_x = (width + tile_size_x - 1) / tile_size_x;
	int tiles_y = (height + tile_size_y - 1) / tile_size_y;
	int tiles_z = (depth + tile_size_z - 1) / tile_size_z;
	int total_tiles = tiles_x * tiles_y * tiles_z;

	if (!silentmode) 
	{
		printf("Processing %dx%dx%d image in %dx%dx%d tiles (%d total) using %d threads\n",
			width, height, depth, tiles_x, tiles_y, tiles_z, total_tiles, thread_count);
	}

	// Mutex and atomic counter for thread synchronization
	std::mutex queue_mutex;
	std::atomic<int> tiles_processed{0};
	std::mutex print_mutex;

	// Queue of tile coordinates to process
	std::queue<std::tuple<int, int, int>> work_queue;

	// Fill work queue with tile coordinates
	for (int tz = 0; tz < tiles_z; tz++) 
	{
		for (int ty = 0; ty < tiles_y; ty++) 
		{
			for (int tx = 0; tx < tiles_x; tx++) 
			{
				work_queue.push({tx, ty, tz});
			}
		}
	}

	// For timing individual thread performance
	std::vector<double> thread_times(thread_count, 0.0);

	// Worker thread function
	auto process_tiles = [&](int thread_id) 
	{
		auto thread_start = std::chrono::high_resolution_clock::now();

		// Allocate thread-local buffers
		T *tile_input = new T[tile_size_x * tile_size_y * tile_size_z * 4];
		T *tile_blurred = new T[tile_size_x * tile_size_y * tile_size_z * 4];
		float *tile_squared_diff = new float[tile_size_x * tile_size_y * tile_size_z];
		float *tile_output = new float[tile_size_x * tile_size_y * tile_size_z];

		int tiles_by_this_thread = 0;

		while (true) 
		{
			// Get next tile coordinates from queue
			std::tuple<int, int, int> tile_coord;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				if (work_queue.empty()) 
				{
					// Record thread timing before cleanup
					auto thread_end = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> thread_duration = thread_end - thread_start;
					thread_times[thread_id] = thread_duration.count();

					// Clean up and exit if no more work
					delete[] tile_input;
					delete[] tile_blurred;
					delete[] tile_squared_diff;
					delete[] tile_output;
					return;
				}
				tile_coord = work_queue.front();
				work_queue.pop();
			}

			int tx = std::get<0>(tile_coord);
			int ty = std::get<1>(tile_coord);
			int tz = std::get<2>(tile_coord);

			// Calculate tile bounds
			int x_start = tx * tile_size_x;
			int y_start = ty * tile_size_y;
			int z_start = tz * tile_size_z;
			int x_end = astc::min(x_start + tile_size_x, width);
			int y_end = astc::min(y_start + tile_size_y, height);
			int z_end = astc::min(z_start + tile_size_z, depth);
			int tile_width = x_end - x_start;
			int tile_height = y_end - y_start;
			int tile_depth = z_end - z_start;

			// Copy input data to tile buffer
			for (int z = 0; z < tile_depth; z++) 
			{
				for (int y = 0; y < tile_height; y++) 
				{
					for (int x = 0; x < tile_width; x++) 
					{
						int src_idx = ((z + z_start) * height * width + (y + y_start) * width + (x + x_start)) * 4;
						int dst_idx = (z * tile_height * tile_width + y * tile_width + x) * 4;
						for (int c = 0; c < 4; c++) 
						{
							tile_input[dst_idx + c] = input[src_idx + c];
						}
					}
				}
			}

			// Process the tile
			gaussian_blur_3d(tile_input, tile_blurred, tile_width, tile_height, tile_depth, 4, sigma_highpass, block_depth);

			// Calculate squared differences
			size_t tile_pixels = tile_width * tile_height * tile_depth;
			for (size_t i = 0; i < tile_pixels; i++) 
			{
				float diff_r = (float)tile_input[i * 4 + 0] - (float)tile_blurred[i * 4 + 0];
				float diff_g = (float)tile_input[i * 4 + 1] - (float)tile_blurred[i * 4 + 1];
				float diff_b = (float)tile_input[i * 4 + 2] - (float)tile_blurred[i * 4 + 2];
				float diff_a = (float)tile_input[i * 4 + 3] - (float)tile_blurred[i * 4 + 3];

				float diff_sum = 0;
				(void)channel_weights;
				diff_sum += diff_r * diff_r;// * channel_weights.lane<0>();
				diff_sum += diff_g * diff_g;// * channel_weights.lane<1>();
				diff_sum += diff_b * diff_b;// * channel_weights.lane<2>();
				diff_sum += diff_a * diff_a;// * channel_weights.lane<3>();
				tile_squared_diff[i] = diff_sum;
			}

			// Apply second Gaussian blur
			gaussian_blur_3d(tile_squared_diff, tile_output, tile_width, tile_height, tile_depth, 1, sigma_blur, block_depth);

			// Map activity values and copy to output
			float C1 = 256.0f;
			float C2 = 1.0f;
			float activity_scalar = 4.0f;

			for (int z = 0; z < tile_depth; z++) 
			{
				for (int y =0; y < tile_height; y++) 
				{
					for (int x =0; x < tile_width; x++) 
					{
						int src_idx = z * tile_height * tile_width + y * tile_width + x;
						int dst_idx = (z + z_start) * height * width + (y + y_start) * width + (x + x_start);
						float val = C1 / (C2 + activity_scalar * astc::sqrt(tile_output[src_idx]));
						output[dst_idx] = val;
					}
				}
			}

			// Update progress
			int processed = ++tiles_processed;
			tiles_by_this_thread++;

			// Print progress every 5% or when specifically useful
			if (!silentmode && processed % astc::max(1, total_tiles / 20) == 0) 
			{
				auto current_time = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed = current_time - start_time;
				double progress = static_cast<double>(processed) / total_tiles;
				double estimated_total = elapsed.count() / progress;
				double remaining = estimated_total - elapsed.count();

				std::lock_guard<std::mutex> print_lock(print_mutex);
				printf("\rProgress: %.1f%% (%d/%d tiles) - %.1fs elapsed, %.1fs remaining",
					progress * 100.0, processed, total_tiles, 
					elapsed.count(), remaining);
				fflush(stdout);
			}
		}
	};

	// Launch worker threads
	std::vector<std::thread> threads;
	for (int i = 0; i < thread_count; i++) 
	{
		threads.emplace_back(process_tiles, i);
	}

	// Wait for all threads to complete
	for (auto& thread : threads) 
	{
		thread.join();
	}

	if (!silentmode) 
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> total_duration = end_time - start_time;

		// Print final statistics
		printf("\n\nActivity map computation complete:\n");
		printf("Total time: %.3f seconds\n", total_duration.count());
		printf("Average time per tile: %.3f ms\n", 
			(total_duration.count() * 1000.0) / total_tiles);
		
		// Print thread statistics
		double min_thread_time = *std::min_element(thread_times.begin(), thread_times.end());
		double max_thread_time = *std::max_element(thread_times.begin(), thread_times.end());
		double avg_thread_time = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / thread_count;
		
		printf("Thread timing - min: %.3fs, max: %.3fs, avg: %.3fs\n", 
			min_thread_time, max_thread_time, avg_thread_time);
		printf("Thread time variance: %.1f%%\n", 
			((max_thread_time - min_thread_time) / avg_thread_time) * 100.0);
	}
}

static void convert_activity_map_to_block_texel_weights(
	const float *activity_map,
	float *block_texel_weights,
	int width, 
	int height, 
	int depth, 
	int block_width, 
	int block_height, 
	int block_depth
) {
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
				int block_index = (z * blocks_y + y) * blocks_x + x;
				float *block_data = block_texel_weights + block_index * block_width * block_height * block_depth;

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
							int block_texel_index = (bz * block_height + by) * block_width + bx;

							// Handle edge cases where block extends beyond image bounds
							if (image_x < width && image_y < height && image_z < depth) 
							{
								int image_index = (image_z * height + image_y) * width + image_x;

								float activity_weight = activity_map[image_index];
								block_data[block_texel_index] = activity_weight;
							} 
							else 
							{
								// For texels outside image bounds, use zero weight
								block_data[block_texel_index] = 0.0f;
							}
						}
					}
				}
			}
		}
	}
}

static inline float get_swizzled_value(
	const float components[4], 
	const astcenc_swz swz
) {
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
astcenc_error astcenc_optimize_for_lz(
	astcenc_image * image_uncomp_in, 
	int image_uncomp_in_component_count, 
	bool image_uncomp_in_is_hdr, 
	const astcenc_swizzle *swizzle, 
	uint8_t *data, 
	uint8_t *exhaustive_data, 
	size_t data_len, 
	int blocks_x, 
	int blocks_y, 
	int blocks_z, 
	int block_width, 
	int block_height, 
	int block_depth, 
	int block_type, 
	float channel_weights[4], 
	int thread_count, 
	bool silentmode, 
	float lambda, 
	float effort
) {
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
	block_size_descriptor *bsd = reinterpret_cast<block_size_descriptor*>(
		ASTC_ALIGNED_MALLOC(sizeof(block_size_descriptor), ASTCENC_VECALIGN));
	if (!bsd) 
	{
		// Handle allocation failure
		return ASTCENC_ERR_OUT_OF_MEM;
	}
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
	uint8_t *all_original_decoded = new uint8_t[num_blocks * decoded_block_size];

	// Convert image_uncomp_in data into block-organized format
	for (int z = 0; z < blocks_z; z++) 
	{
		for (int y = 0; y < blocks_y; y++) 
		{
			for (int x = 0; x < blocks_x; x++) 
			{
				int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
				uint8_t *block_data = all_original_decoded + block_index * decoded_block_size;

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
										uint16_t f16_val = ((uint16_t *)image_uncomp_in->data[0])[image_index + c];
										components[c] = float16_to_float(f16_val);
									} 
									else 
									{
										components[c] = ((float *)image_uncomp_in->data[0])[image_index + c];
									}
								} 
								else 
								{
									uint8_t u8_val = ((uint8_t *)image_uncomp_in->data[0])[image_index + c];
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
								uint8_t *block_data_u8 = (uint8_t *)block_data;
								block_data_u8[block_pixel_index + 0] = (uint8_t)(swizzled[0] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 1] = (uint8_t)(swizzled[1] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 2] = (uint8_t)(swizzled[2] * 255.0f + 0.5f);
								block_data_u8[block_pixel_index + 3] = (uint8_t)(swizzled[3] * 255.0f + 0.5f);
							} 
							else 
							{
								float *block_data_f32 = (float *)block_data;
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
	uint8_t *original_blocks = new uint8_t[data_len];
	memcpy(original_blocks, data, data_len);

	// Allocate memory for the reconstructed image
	size_t image_size = width * height * depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
	uint8_t *reconstructed_image = new uint8_t[image_size];
	float *activity_map = new float[width * height * depth]; // Single channel
	float *block_texel_weights = new float[num_blocks * block_width * block_height * block_depth]; // One value per texel

	if (block_type == ASTCENC_TYPE_U8) 
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image(all_original_decoded, width, height, depth, block_width, block_height, block_depth, reconstructed_image);

		// Compute activity map
		compute_activity_map(reconstructed_image, activity_map, width, height, depth, 2.2f, 1.25f, channel_weights_vec, block_depth, thread_count, silentmode);
	} 
	else 
	{
		// Reconstruct the image from all_original_decoded
		reconstruct_image((float *)all_original_decoded, width, height, depth, block_width, block_height, block_depth, (float *)reconstructed_image);

		// Apply high-pass filter with squared differences and additional blur
		compute_activity_map((float *)reconstructed_image, activity_map, width, height, depth, 2.2f, 1.25f, channel_weights_vec, block_depth, thread_count, silentmode);
	}

	// We want per-texel weights for the encoder, not the raw activity map
	convert_activity_map_to_block_texel_weights(activity_map, block_texel_weights, width, height, depth, block_width, block_height, block_depth);
	delete[] activity_map;

	dual_mtf_pass(thread_count, silentmode, data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_texel_weights, channel_weights_vec, effort);
	if (effort >= 5) 
	{
		dual_mtf_pass(thread_count, silentmode, data, original_blocks, exhaustive_data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_texel_weights, channel_weights_vec, effort);
	}

	// Clean up
	ASTC_ALIGNED_FREE(bsd);
	delete[] all_original_decoded;
	delete[] original_blocks;
	delete[] reconstructed_image;
	delete[] block_texel_weights;

	return ASTCENC_SUCCESS;
}

