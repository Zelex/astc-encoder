#include <stdlib.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"

// cross platform popcountll
#if defined(_MSC_VER)
#include <intrin.h>
#define popcountll _mm_popcnt_u64
#else
#include <x86intrin.h>
#define popcountll __builtin_popcountll
#endif

struct astc_block_t {
    unsigned char data[16];
};

struct unique_bits_t { 
    long long bits;
    int count;
};

static __forceinline int bitfield_extract(uint64_t bitfield, int start, int count) {
    return (int)((bitfield >> start) & ((1ull << count) - 1));
}

static int compare_long_long(const void *a, const void *b) {
    long long l1 = *((long long*)a);
    long long l2 = *((long long*)b);
    if (l1 < l2) return -1;
    if (l1 > l2) return 1;
    return 0;
}

// Sort descending based on count
static int compare_unique_bits(const void *a, const void *b) {
    unique_bits_t *ub1 = (unique_bits_t*)a;
    unique_bits_t *ub2 = (unique_bits_t*)b;
    if (ub1->count < ub2->count) return 1;
    if (ub1->count > ub2->count) return -1;
    return 0;
}

static inline float calculate_mse(const uint8_t* img1, const uint8_t* img2, int total) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        int diff = img1[i] - img2[i];
        sum += diff * diff;
    }
    return (float)sum / total;
}

void astc_decompress_block(
    const block_size_descriptor& bsd,
    const uint8_t* block_ptr,
    uint8_t* output,
    int block_width,
    int block_height,
    int block_depth,
    int block_type)
{
    image_block blk {};
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
    for (int i = 0; i < blk.texel_count; i++) {
        vfloat4 color = vfloat4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);
        if (block_type == ASTCENC_TYPE_U8) {
            // Convert to 8-bit UNORM
            vint4 colori = float_to_int_rtn(color * 255.0f);
            output[i * 4 + 0] = static_cast<uint8_t>(colori.lane<0>());
            output[i * 4 + 1] = static_cast<uint8_t>(colori.lane<1>());
            output[i * 4 + 2] = static_cast<uint8_t>(colori.lane<2>());
            output[i * 4 + 3] = static_cast<uint8_t>(colori.lane<3>());
        } else {
            // Store as 32-bit float
            float* output_f = reinterpret_cast<float*>(output);
            output_f[i * 4 + 0] = color.lane<0>();
            output_f[i * 4 + 1] = color.lane<1>();
            output_f[i * 4 + 2] = color.lane<2>();
            output_f[i * 4 + 3] = color.lane<3>();
        }
    }
}

/**
 * @brief Optimize compressed data for better LZ compression.
 *
 * @param data          The compressed image data.
 * @param data_len      The length of the compressed data.
 * @param block_width   The width of each compressed block.
 * @param block_height  The height of each compressed block.
 * @param block_depth   The depth of each compressed block.
 * @param block_type    The type of each compressed block.
 */
void optimize_for_lz(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth, int block_type) {
    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    const int block_size = 16;
    const long long INDEX_MASK = 0x0000FFFFFFFFFFF;
    const int BITS_OFFSET = 8;
    const float MSE_THRESHOLD = 8;

    // Gather up all the block indices (second 8-bytes). Sort them by frequency.
    size_t num_blocks = data_len / block_size;
    long long *bits = new long long[num_blocks];
    unique_bits_t *unique_bits = new unique_bits_t[num_blocks];
    size_t num_unique_bits = 0;

    // Count the frequency of each bit pattern
    for (size_t i = 0; i < num_blocks; i++) { 
        bits[i] = *((long long*)(data + i * block_size + BITS_OFFSET)) & INDEX_MASK;
    }

    qsort(bits, num_blocks, sizeof(long long), compare_long_long);

    for (size_t i = 0; i < num_blocks; i++) {
        if (i > 0 && bits[i] == bits[i - 1]) {
            unique_bits[num_unique_bits - 1].count++;
        } else {
            unique_bits[num_unique_bits].bits = bits[i];
            unique_bits[num_unique_bits].count = 1;
            num_unique_bits++;
        }
    }

    // Sort the unique bit patterns by frequency using qsort
    qsort(unique_bits, num_unique_bits, sizeof(unique_bits_t), compare_unique_bits);

    size_t N = (64 < num_unique_bits) ? 64 : num_unique_bits; 

    // Print the unique bit patterns and their frequencies
    //for (size_t i = 0; i < N; i++) {
        //printf("Bit pattern: %llx, Frequency: %d\n", unique_bits[i].bits, unique_bits[i].count);
    //}

	uint8_t original_decoded[16 * 16 * 4] = { 0 };
    uint8_t modified_decoded[16 * 16 * 4] = { 0 };

    // In the original data, now find the bits that most resemble the unique bits in the first N spots, and replace them
    for (size_t i = 0; i < num_blocks; i++) {
		uint8_t *current_block = data + i * block_size;
        long long current_bits = *((long long*)(data + i * block_size + BITS_OFFSET));
        long long masked_current_bits = current_bits & INDEX_MASK;
        size_t best_match = masked_current_bits;
        float best_mse = FLT_MAX;

		// When calling astc_decompress_block, pass the bsd
		astc_decompress_block(*bsd, current_block, original_decoded, block_width, block_height, block_depth, block_type);

        for (size_t j = 0; j < N; j++) {
			// Create a temporary modified block
			uint8_t temp_block[16];  // Assuming 16-byte ASTC blocks
			memcpy(temp_block, current_block, 16);
			uint64_t *temp_bits = (uint64_t*)(temp_block + BITS_OFFSET);
			*temp_bits = (*temp_bits & ~INDEX_MASK) | (unique_bits[j].bits & INDEX_MASK);

			astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

			float mse = calculate_mse(original_decoded, modified_decoded, block_width*block_height*4);
			if (mse < best_mse && mse < MSE_THRESHOLD) {
				best_match = unique_bits[j].bits;
				best_mse = mse;
			}
        }

        if (best_match != masked_current_bits) {
            // Replace only the index bits with the best matching frequent pattern
            long long new_bits = (current_bits & ~INDEX_MASK) | (best_match & INDEX_MASK);
            *((long long*)(data + i * block_size + BITS_OFFSET)) = new_bits;
        }
    }

    // Clean up
    delete[] bits;
    delete[] unique_bits;
    free(bsd);
}