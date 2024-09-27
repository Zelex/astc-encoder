#include <stdlib.h>
#include <math.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"

#define MTF_SIZE 1024

typedef struct {
    long long list[MTF_SIZE];
    int size;
    int literal_histogram[256]; // Single histogram for all bytes
} MTF_LL;

static inline float log2_fast(float val) {
    union { float val; int x; } u = { val };
    float log_2 = (float)(((u.x >> 23) & 255) - 128);              
    u.x   &= ~(255 << 23);
    u.x   += 127 << 23;
    log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f; 
    return log_2;
} 

static void mtf_ll_init(MTF_LL* mtf) {
    mtf->size = 0;
    memset(mtf->literal_histogram, 0, sizeof(mtf->literal_histogram)); // Initialize histogram
}

// Search for a value in the MTF list
static int mtf_ll_search(MTF_LL* mtf, long long value) {
    for (int i = 0; i < mtf->size; i++) {
        if (mtf->list[i] == value) {
            return i;
        }
    }
    return -1;
}

static int mtf_ll_encode(MTF_LL* mtf, long long value) {
    int pos = mtf_ll_search(mtf, value);
    
    if (pos == -1) {
        // Value not found, add it to the front
        if (mtf->size < MTF_SIZE) {
            mtf->size++;
        }
        pos = mtf->size - 1;
    }

    // Move the value to the front
    for (int i = pos; i > 0; i--) {
        mtf->list[i] = mtf->list[i - 1];
    }
    mtf->list[0] = value;
    
    return pos;
}

static bool mtf_ll_contains(MTF_LL* mtf, long long value) {
    return mtf_ll_search(mtf, value) != -1;
}

static int mtf_ll_peek_position(MTF_LL* mtf, long long value) {
    int pos = mtf_ll_search(mtf, value);
    if (pos != -1) {
        return pos;
    }
    return mtf->size; // Return size if not found (which would be its position if added)
}

static void mtf_ll_update_histogram(MTF_LL* mtf, long long value) {
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (value >> (i * 8)) & 0xFF;
        mtf->literal_histogram[byte]++;
    }
}

static float calculate_bit_cost(int mtf_value, bool is_literal, long long literal_value, MTF_LL* mtf) {
    if (is_literal) {
        float total_entropy = 0.0f;
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (literal_value >> (i * 8)) & 0xFF;
            float prob = (float)mtf->literal_histogram[byte] / (MTF_SIZE * 8);
            total_entropy += -log2_fast(prob);
        }
        return 1.0f + total_entropy; // flag bit + entropy-coded value
    } else {
        return log2_fast((float)(mtf_value + 1)); // Cost for an MTF value
    }
}

template<typename T>
static inline float calculate_mse(const T* img1, const T* img2, int total) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float diff = (float)img1[i] - (float)img2[i];
        sum += diff * diff;
    }
    return sum / total;
}

template<typename T>
static float calculate_gradient_magnitude_2d(const T* img, int width, int height) {
    float total_magnitude = 0.0f;
    int channels = 4; // Assuming RGBA

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                float gx = (float)img[((y * width + x + 1) * channels) + c] - 
                           (float)img[((y * width + x - 1) * channels) + c];
                float gy = (float)img[(((y + 1) * width + x) * channels) + c] - 
                           (float)img[(((y - 1) * width + x) * channels) + c];
                total_magnitude += sqrtf(gx * gx + gy * gy);
            }
        }
    }

    return total_magnitude / (width * height * channels);
}

template<typename T>
static float calculate_gradient_magnitude_3d(const T* img, int width, int height, int depth) {
    float total_magnitude = 0.0f;
    int channels = 4; // Assuming RGBA

    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < channels; c++) {
                    float gx = (float)img[(((z * height + y) * width + x + 1) * channels) + c] - 
                               (float)img[(((z * height + y) * width + x - 1) * channels) + c];
                    float gy = (float)img[(((z * height + y + 1) * width + x) * channels) + c] - 
                               (float)img[(((z * height + y - 1) * width + x) * channels) + c];
                    float gz = (float)img[((((z + 1) * height + y) * width + x) * channels) + c] - 
                               (float)img[((((z - 1) * height + y) * width + x) * channels) + c];
                    total_magnitude += sqrtf(gx * gx + gy * gy + gz * gz);
                }
            }
        }
    }

    return total_magnitude / (width * height * depth * channels);
}

template<typename T>
static float calculate_gradient_magnitude(const T* img, int width, int height, int depth) {
    if (depth == 1) {
        return calculate_gradient_magnitude_2d(img, width, height);
    } else {
        return calculate_gradient_magnitude_3d(img, width, height, depth);
    }
}

static void astc_decompress_block(
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
 * @param lambda        The rate-distortion trade-off parameter.
 */
void optimize_for_lz(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth, int block_type, float lambda) {
    if (lambda <= 0.0f) {
        lambda = 0.1f;
    }

    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    const int block_size = 16;
    const long long INDEX_MASK = 0xFFFFFFFFFFFFFFFFull;
    const int BITS_OFFSET = 8;
    const float BASE_MSE_THRESHOLD = 1.0f;
    const float MAX_MSE_THRESHOLD = 128.0f;
    const float GRADIENT_SCALE = 8.5f;
    const float TEMPERATURE = 0.0f;

    size_t num_blocks = data_len / block_size;

    MTF_LL global_mtf;
    mtf_ll_init(&global_mtf);

    uint8_t *original_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
    uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);

    // In the original data, now search the MTF list for potential replacements
    for (size_t i = 0; i < num_blocks; i++) {
        uint8_t *current_block = data + i * block_size;
        long long current_bits = *((long long*)(data + i * block_size + BITS_OFFSET));
        long long masked_current_bits = current_bits & INDEX_MASK;
        size_t best_match = masked_current_bits;
        float best_rd_cost = FLT_MAX;

        astc_decompress_block(*bsd, current_block, original_decoded, block_width, block_height, block_depth, block_type);

        float original_bit_cost = calculate_bit_cost(mtf_ll_peek_position(&global_mtf, masked_current_bits), 
                                                     !mtf_ll_contains(&global_mtf, masked_current_bits),
                                                     masked_current_bits, &global_mtf);

        // Calculate gradient magnitude of the original block
        float gradient_magnitude;
        if (block_type == ASTCENC_TYPE_U8) {
            gradient_magnitude = calculate_gradient_magnitude(original_decoded, block_width, block_height, block_depth);
        } else {
            gradient_magnitude = calculate_gradient_magnitude((float*)original_decoded, block_width, block_height, block_depth);
        }
        
        // Adjust MSE_THRESHOLD based on gradient magnitude
        float normalized_gradient = fminf(gradient_magnitude / 255.0f, 1.0f);
        float gradient_factor = expf(GRADIENT_SCALE * normalized_gradient) - 1.0f;
        float adjusted_mse_threshold = BASE_MSE_THRESHOLD + gradient_factor * (MAX_MSE_THRESHOLD - BASE_MSE_THRESHOLD);
        adjusted_mse_threshold = fminf(adjusted_mse_threshold, MAX_MSE_THRESHOLD);

        // Search through the MTF list
        for (int j = 0; j < global_mtf.size; j++) {
            long long candidate_bits = global_mtf.list[j];
            
            // Create a temporary modified block
            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            uint64_t *temp_bits = (uint64_t*)(temp_block + BITS_OFFSET);
            *temp_bits = (*temp_bits & ~INDEX_MASK) | (candidate_bits & INDEX_MASK);

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
				mse = calculate_mse(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
            } else {
				mse = calculate_mse((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
            }

            // Calculate bit cost for the modified block
            float modified_bit_cost = calculate_bit_cost(j, false, candidate_bits, &global_mtf);

            // Calculate rate-distortion cost
            float rd_cost = mse + lambda * (modified_bit_cost - original_bit_cost);

            if (rd_cost < best_rd_cost && mse < adjusted_mse_threshold) {
                best_match = candidate_bits;
                best_rd_cost = rd_cost;
            }
        }

        if (best_match != masked_current_bits) {
            // Replace only the index bits with the best matching pattern from MTF
            long long new_bits = (current_bits & ~INDEX_MASK) | (best_match & INDEX_MASK);
            *((long long*)(data + i * block_size + BITS_OFFSET)) = new_bits;
        }

        // Update the global MTF with the chosen bits
        mtf_ll_encode(&global_mtf, best_match & INDEX_MASK);

        // Update the literal histogram with the chosen bits
        mtf_ll_update_histogram(&global_mtf, best_match & INDEX_MASK);
    }

    // Clean up
    free(bsd);
    free(original_decoded);
    free(modified_decoded);
}