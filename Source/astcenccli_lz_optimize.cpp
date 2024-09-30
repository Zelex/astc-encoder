#include <stdlib.h>
#include <math.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"

//#define MTF_SIZE 1024
#define MTF_SIZE (256+64+16+1)

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

static void mtf_ll_update_histogram(MTF_LL* mtf, long long value) {
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (value >> (i * 8)) & 0xFF;
        mtf->literal_histogram[byte]++;
    }
}

static float calculate_bit_cost(int mtf_value, bool is_literal, long long literal_value, MTF_LL* mtf, long long mask) {
    literal_value &= mask;
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

static void mtf_pass(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth, int block_type, float lambda, int BITS_OFFSET, long long INDEX_MASK) {
    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    // Initialize weight bits table -- block size matters in determining that certain ones are errors
    /*
    #define NUM_BLOCK_MODES 2048
    uint8_t *weight_bits = (uint8_t *) malloc(NUM_BLOCK_MODES);
    for (size_t i = 0; i < NUM_BLOCK_MODES; ++i) {
        uint8_t block[16];
        block[0] = (i & 255);
        block[1] = ((i>>8) & 255);
        weight_bits[i] = get_weight_bits(&block[0], block_width, block_height, block_depth);
    }

    #define GET_WEIGHT_BITS(block)  weight_bits[((uint16_t*)block)[0] & 0x7ff]
    */

    const int block_size = 16;

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
		//const long long INDEX_MASK = 0xFFFFFFFFFFFF0000ull;
        //const unsigned long long INDEX_MASK = (1ull<<GET_WEIGHT_BITS(current_block))-1;
        size_t best_match = current_bits;
        float best_rd_cost = FLT_MAX;

        astc_decompress_block(*bsd, current_block, original_decoded, block_width, block_height, block_depth, block_type);

        float original_bit_cost = calculate_bit_cost(mtf_ll_peek_position(&global_mtf, current_bits, INDEX_MASK), 
                                                     !mtf_ll_contains(&global_mtf, current_bits, INDEX_MASK),
                                                     current_bits, &global_mtf, INDEX_MASK);


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
            float modified_bit_cost = calculate_bit_cost(j, false, candidate_bits, &global_mtf, INDEX_MASK);

            // Calculate rate-distortion cost
            float rd_cost = mse + lambda * (modified_bit_cost - original_bit_cost);

            if (rd_cost < best_rd_cost && mse < adjusted_mse_threshold) {
                best_match = candidate_bits;
                best_rd_cost = rd_cost;
            }
        }

        if (best_match != current_bits) {
            // Replace only the index bits with the best matching pattern from MTF
            long long new_bits = (current_bits & ~INDEX_MASK) | (best_match & INDEX_MASK);
            *((long long*)(data + i * block_size + BITS_OFFSET)) = new_bits;
        }

        // Update the global MTF with the chosen bits
        mtf_ll_encode(&global_mtf, best_match, INDEX_MASK);

        // Update the literal histogram with the chosen bits
        mtf_ll_update_histogram(&global_mtf, best_match);
    }

    // Clean up
    //free(weight_bits);
    free(bsd);
    free(original_decoded);
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
        lambda = 0.1f;
    }

    // First pass, weights
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 8, 0xFFFFFFFFFFFF0000ull);

    // Second pass, color endpoints
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, 0xFFFFFFFFFFFFFFFFull);

    // Third pass, more color endpoints
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 2, 0xFFFFFFFFFFFFFFFFull);
}