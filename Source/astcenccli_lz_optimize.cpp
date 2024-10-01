#include <stdlib.h>
#include <math.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"

//#define MTF_SIZE 1024
#define MTF_SIZE (256+64+16+1)

static const float BASE_GRADIENT_SCALE = 10.0f;
static const float GRADIENT_POW = 1.0f;
static const float EDGE_WEIGHT = 2.0f;
static const float CORNER_WEIGHT = 1.0f;

#define ERROR_FN calculate_mse

typedef struct {
    long long list[MTF_SIZE];
    int size;
    int literal_histogram[256]; // Single histogram for all bytes
    int literal_histogram_size;
} MTF_LL;

static inline float log2_fast(float val) {
    union { float val; int x; } u = { val };
    float log_2 = (float)(((u.x >> 23) & 255) - 128);              
    u.x   &= ~(255 << 23);
    u.x   += 127 << 23;
    log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f; 
    return log_2;
} 

static void mtf_ll_reset_histogram(MTF_LL* mtf) {
    for(int i = 0; i < 256; i++) {
        mtf->literal_histogram[i] = 1;
    }
    mtf->literal_histogram_size = 256;
}

static void mtf_ll_init(MTF_LL* mtf) {
    mtf->size = 0;
    mtf_ll_reset_histogram(mtf);
}

// Search for a value in the MTF list
static int mtf_ll_search(MTF_LL* mtf, long long value, long long mask) {
    value &= mask;
    for (int i = 0; i < mtf->size; i++) {
        if ((mtf->list[i] & mask) == value) {
            return i;
        }
    }
    return -1;
}

static int mtf_ll_encode(MTF_LL* mtf, long long value, long long mask) {
    int pos = mtf_ll_search(mtf, value, mask);
    
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

static bool mtf_ll_contains(MTF_LL* mtf, long long value, long long mask) {
    return mtf_ll_search(mtf, value, mask) != -1;
}

static int mtf_ll_peek_position(MTF_LL* mtf, long long value, long long mask) {
    int pos = mtf_ll_search(mtf, value, mask);
    if (pos != -1) {
        return pos;
    }
    return mtf->size; // Return size if not found (which would be its position if added)
}

static void mtf_ll_update_histogram(MTF_LL* mtf, long long value, long long mask) {
    for (int i = 0; i < 8; i++) {
        uint8_t m = (mask >> (i * 8)) & 0xFF;
        if(m) {
            uint8_t byte = (value >> (i * 8)) & 0xFF;
            mtf->literal_histogram[byte]++;
            mtf->literal_histogram_size++;
        }
    }
}

static float calculate_bit_cost(int mtf_value, bool is_literal, long long literal_value, MTF_LL* mtf, long long mask) {
    literal_value &= mask;
    if (is_literal) {
        float total_entropy = 0.0f;
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (literal_value >> (i * 8)) & 0xFF;
            float prob = (float)mtf->literal_histogram[byte] / (mtf->literal_histogram_size + 1);
            total_entropy -= log2_fast(prob);
        }
        return 15.f + total_entropy; // flag bit + entropy-coded value
    } else {
        return 10.f + log2_fast((float)(mtf_value + 32)); // Cost for an MTF value
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

// Calculate Sum of Absolute Differences Error
template<typename T>
static inline float calculate_sad(const T* img1, const T* img2, int total) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float diff = (float)img1[i] - (float)img2[i];
        sum += diff < 0 ? -diff : diff;
    }
    return sum / total;
}

template<typename T>
static float calculate_gradient_magnitude_2d(const T* img, int width, int height, int channels) {
    float total_magnitude = 0.0f;

    for (int c = 0; c < channels; c++) {
        // Horizontal gradients
        for (int y = 0; y < height; y++) {
            float gx = (float)img[(y * width + 3) * channels + c] - (float)img[(y * width) * channels + c];
            total_magnitude += gx * gx * EDGE_WEIGHT;
            
            for (int x = 1; x < width - 1; x++) {
                gx = (float)img[(y * width + x + 1) * channels + c] - (float)img[(y * width + x - 1) * channels + c];
                total_magnitude += gx * gx;
            }
        }

        // Vertical gradients
        for (int x = 0; x < width; x++) {
            float gy = (float)img[((height - 1) * width + x) * channels + c] - (float)img[x * channels + c];
            total_magnitude += gy * gy * EDGE_WEIGHT;
            
            for (int y = 1; y < height - 1; y++) {
                gy = (float)img[((y + 1) * width + x) * channels + c] - (float)img[((y - 1) * width + x) * channels + c];
                total_magnitude += gy * gy;
            }
        }

        // Diagonal gradients (for corners)
        float gd1 = (float)img[((height - 1) * width + width - 1) * channels + c] - (float)img[0];
        float gd2 = (float)img[((height - 1) * width) * channels + c] - (float)img[width - 1];
        total_magnitude += (gd1 * gd1 + gd2 * gd2) * CORNER_WEIGHT;
    }

    return sqrtf(total_magnitude) / (width * height * channels);
}

template<typename T>
static float calculate_gradient_magnitude_3d(const T* img, int width, int height, int depth, int channels) {
    float total_magnitude = 0.0f;

    for (int c = 0; c < channels; c++) {
        for (int z = 0; z < depth; z++) {
            // XY plane gradients
            total_magnitude += calculate_gradient_magnitude_2d(img + z * width * height * channels, width, height, channels);
        }

        // Z-direction gradients
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float gz = (float)img[((depth - 1) * height * width + y * width + x) * channels + c] - 
                           (float)img[(y * width + x) * channels + c];
                total_magnitude += gz * gz * EDGE_WEIGHT;

                for (int z = 1; z < depth - 1; z++) {
                    gz = (float)img[(((z + 1) * height * width + y * width + x) * channels) + c] - 
                         (float)img[(((z - 1) * height * width + y * width + x) * channels) + c];
                    total_magnitude += gz * gz;
                }
            }
        }
    }

    return sqrtf(total_magnitude) / (width * height * depth * channels);
}

template<typename T>
static float calculate_gradient_magnitude(const T* img, int width, int height, int depth) {
    int channels = 4; // Assuming RGBA
    if (depth == 1) {
        return calculate_gradient_magnitude_2d(img, width, height, channels);
    } else {
        return calculate_gradient_magnitude_3d(img, width, height, depth, channels);
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

int get_weight_bits(uint8_t *data, int block_width, int block_height, int block_depth)
{
    uint16_t mode = data[0] | (data[1]<<8);

    if ((mode & 0x1ff) == 0x1fc)
        return 0; // void-extent
    if ((mode & 0x00f) == 0    )
        return 0; // Reserved

    uint8_t b01 = (mode >>  0) & 3;
    uint8_t b23 = (mode >>  2) & 3;
    uint8_t p0  = (mode >>  4) & 1;
    uint8_t b56 = (mode >>  5) & 3;
    uint8_t b78 = (mode >>  7) & 3;
    uint8_t P   = (mode >>  9) & 1;
    uint8_t Dp  = (mode >> 10) & 1;
    uint8_t b9_10 = (mode >> 9) & 3;
    uint8_t p12;

    int W,H,D;
    if (block_depth <= 1) {
        // 2D
        D = 1;
        if ((mode & 0x1c3) == 0x1c3)
            return 0; // Reserved*
        if (b01 == 0) {
            p12 = b23;
            switch (b78) {
                case 0:
                    W = 12;
                    H =  2 + b56;
                    break;
                case 1:
                    W =  2 + b56;
                    H = 12;
                    break;
                case 2:
                    W = 6 + b56;
                    H = 6 + b9_10;
                    Dp = 0;
                    P = 0;
                    break;
                case 3:
                    if (b56 == 0) {
                        W = 6;
                        H = 10;
                    } else if (b56 == 1) {
                        W = 10;
                        H = 6;
                    } else {
                        /* NOTREACHED */
                        assert(0);
                        return 0;
                    }
                    break;
            }
        } else {
            p12 = b01;
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
                    if (b78 & 2) {
                        W = 2 + (b78 & 1);
                        H = 6 + b56      ;
                    } else {
                        W = 2 + b56      ;
                        H = 2 + (b78 & 1);
                    }
                    break;
            }
        }
    } else {
        // 3D
        if ((mode & 0x1e3) == 0x1e3)
            return 0; // Reserved*
        if (b01 != 0) {
            p12 = b01;
            W = 2 + b56;
            H = 2 + b78;
            D = 2 + b23;
        } else {
            p12 = b23;
            switch (b78) {
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
                    switch (b56) {
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
    if (W > block_width ) return 0;
    if (H > block_height) return 0;
    if (D > block_depth ) return 0;

    uint8_t p = (p12 << 1) | p0;
    int trits=0,quints=0,bits=0;
    if (!P) {
        int t[8] = { -1,-1, 0,1,0,0,1,0 };
        int q[8] = { -1,-1, 0,0,0,1,0,0 };
        int b[8] = { -1,-1, 1,0,2,0,1,3 };
        trits  = t[p];
        quints = q[p];
        bits   = b[p];
    } else {
        int t[8] = { -1,-1, 0,1,0,0,1,0 };
        int q[8] = { -1,-1, 1,0,0,1,0,0 };
        int b[8] = { -1,-1, 1,2,4,2,3,5 };
        trits  = t[p];
        quints = q[p];
        bits   = b[p];
    }

    int num_weights = W * H * D;
    if (Dp)
        num_weights *= 2;

    if (num_weights > 64)
        return 0;

    int weight_bits =   (num_weights * 8 * trits  + 4)/5
                      + (num_weights * 7 * quints + 2)/3
                      + num_weights * bits;

    // error cases
    if (weight_bits < 24 || weight_bits > 96)
        return 0;

    return (uint8_t) weight_bits;
}

#if 0
extern int hack_bits_for_weights;
void test_weight_bits(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth)
{
    block_size_descriptor *bsd = (block_size_descriptor*) malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    for (size_t i=0; i < data_len; i += 16) {
        uint8_t *block = data + i;
        uint8_t decoded[6*6*6*4];
        astc_decompress_block(*bsd, block, decoded, block_width, block_height, block_depth, ASTCENC_TYPE_U8);
        int bits = get_weight_bits(block, block_width, block_height, block_depth);
	    if (bits != hack_bits_for_weights) {
		    printf("Internal error: decoded weight bits count didn't match\n");
	    }
    }
    free(bsd);
}
#endif

static void mtf_pass(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth, int block_type, float lambda, int BITS_OFFSET, long long INDEX_MASK, block_size_descriptor* bsd, uint8_t* all_original_decoded) {
    uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;

    MTF_LL mtf;
    mtf_ll_init(&mtf);

    // Helper function to process a single block
    auto process_block = [&](size_t block_index) {
        uint8_t *current_block = data + block_index * block_size;
        long long current_bits = *((long long*)(current_block + BITS_OFFSET));
        size_t best_match = current_bits;
        float best_rd_cost = FLT_MAX;

        uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

        float original_bit_cost = calculate_bit_cost(mtf_ll_peek_position(&mtf, current_bits, INDEX_MASK), 
                                                     !mtf_ll_contains(&mtf, current_bits, INDEX_MASK),
                                                     current_bits, &mtf, INDEX_MASK);

        // Calculate gradient magnitude and adjust lambda
        float gradient_magnitude;
        float max_possible_gradient;
        if (block_type == ASTCENC_TYPE_U8) {
            gradient_magnitude = calculate_gradient_magnitude(original_decoded, block_width, block_height, block_depth);
            max_possible_gradient = 255.0f * sqrtf(3.0f);
        } else {
            gradient_magnitude = calculate_gradient_magnitude((float*)original_decoded, block_width, block_height, block_depth);
            max_possible_gradient = 1.0f * sqrtf(3.0f);
        }
        
        float normalized_gradient = gradient_magnitude / max_possible_gradient;
        float gradient_scale = BASE_GRADIENT_SCALE * lambda; 
        float gradient_factor = powf(normalized_gradient, GRADIENT_POW) * gradient_scale; 
        float adjusted_lambda = lambda * (1.0f + gradient_factor * 1000.f);

        // Decode the original block to compute initial MSE
        astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
        float original_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            original_mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
        } else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
        }

        best_rd_cost = original_mse + adjusted_lambda * original_bit_cost;

        // Search through the MTF list
        for (int k = 0; k < mtf.size; k++) {
            long long candidate_bits = mtf.list[k];
            
            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            uint64_t *temp_bits = (uint64_t*)(temp_block + BITS_OFFSET);
            *temp_bits = (*temp_bits & ~INDEX_MASK) | (candidate_bits & INDEX_MASK);

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
                mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
            } else {
                mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
            }

            float modified_bit_cost = calculate_bit_cost(k, false, candidate_bits, &mtf, INDEX_MASK);
            float rd_cost = mse + adjusted_lambda * modified_bit_cost;

            if (rd_cost < best_rd_cost) {
                best_match = candidate_bits;
                best_rd_cost = rd_cost;
            }
        }

        if (best_match != current_bits) {
            long long new_bits = (current_bits & ~INDEX_MASK) | (best_match & INDEX_MASK);
            *((long long*)(current_block + BITS_OFFSET)) = new_bits;
        }

        // Update the literal histogram with the chosen bits
        if (!mtf_ll_contains(&mtf, best_match, INDEX_MASK)) {
            mtf_ll_update_histogram(&mtf, best_match, INDEX_MASK);
        }

        // Update the MTF with the chosen bits
        mtf_ll_encode(&mtf, best_match, INDEX_MASK);
    };

    // Forward pass
    for (size_t i = 0; i < num_blocks; i++) {
        if (i % 8192 == 0 && i > 0) {
            //mtf_ll_reset_histogram(&mtf);
        }
        process_block(i);
    }

    // Reset MTF for backward pass
    mtf_ll_init(&mtf);

    // Backward pass
    for (size_t i = num_blocks; i-- > 0;) {
        if (i % 8192 == 8191 && i < num_blocks - 1) {
            //mtf_ll_reset_histogram(&mtf);
        }
        process_block(i);
    }

    // Clean up
    free(modified_decoded);
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
        lambda = 0.015f;
    }

    // Create a copy of the original data to decode for later passes
    uint8_t *original_data = (uint8_t*)malloc(data_len);
    memcpy(original_data, data, data_len);

    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    // Decode all original blocks once
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    size_t decoded_block_size = block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
    uint8_t *all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);

    for (size_t i = 0; i < num_blocks; i++) {
        uint8_t *original_block = original_data + i * block_size;
        uint8_t *decoded_block = all_original_decoded + i * decoded_block_size;
        astc_decompress_block(*bsd, original_block, decoded_block, block_width, block_height, block_depth, block_type);
    }

    free(original_data);

    // MTF passes...
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 8, 0xFFFFFFFFFFFF0000ull, bsd, all_original_decoded);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 2, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded);

    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 8, 0xFFFFFFFFFFFF0000ull, bsd, all_original_decoded);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 2, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded);

    // Clean up
    free(bsd);
    free(all_original_decoded);
}