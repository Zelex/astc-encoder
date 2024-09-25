#include <stdlib.h>
#include "astcenc.h"
#include "astcenccli_internal.h"

// cross platform popcountll
#if defined(_MSC_VER)
#include <intrin.h>
#define popcountll _mm_popcnt_u64
#else
#include <x86intrin.h>
#define popcountll __builtin_popcountll
#endif

struct unique_bits_t { 
    long long bits;
    int count;
};

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

/**
 * @brief Optimize compressed data for better LZ compression.
 *
 * @param data          The compressed image data.
 * @param data_len      The length of the compressed data.
 * @param block_size    The size of each compressed block (typically 16 bytes for ASTC).
 */
void optimize_for_lz(
    uint8_t* data,
    size_t data_len,
    size_t block_size
) {
    // Gather up all the block indices (second 8-bytes). Sort them by frequency.
    size_t num_blocks = data_len / block_size;
    long long *bits = new long long[num_blocks];
    unique_bits_t *unique_bits = new unique_bits_t[num_blocks];
    size_t num_unique_bits = 0;

    // Count the frequency of each bit pattern
    for (size_t i = 0; i < num_blocks; i++) { // Changed to size_t
        bits[i] = *((long long*)(data + i * block_size + 8));
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

    // In the original data, now find the bits that most resemble the unique bits in the first N spots, and replace them
    size_t N = (256 < num_unique_bits) ? 256 : num_unique_bits; 
    for (size_t i = 0; i < num_blocks; i++) {
        long long current_bits = *((long long*)(data + i * block_size + 8));
        size_t best_match = -1;
        long long best_match_diff = 16; // Maximum possible difference. 64 is best of 256, 16 is best of 256 otherwise keep

        for (size_t j = 0; j < N; j++) {
            long long diff = popcountll(current_bits ^ unique_bits[j].bits);
            if (diff < best_match_diff) {
                best_match = j;
                best_match_diff = diff;
            }
        }

        if (best_match != -1) {
            // Replace the bits with the best matching frequent pattern
            *((long long*)(data + i * block_size + 8)) = unique_bits[best_match].bits;
        }
    }

    // Clean up
    delete[] bits;
    delete[] unique_bits;
}