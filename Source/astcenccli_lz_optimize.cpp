#include "astcenc.h"
#include "astcenccli_internal.h"

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
}