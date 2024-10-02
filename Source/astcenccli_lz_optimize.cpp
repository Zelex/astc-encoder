#include <stdlib.h>
#include <math.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"

#if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
    typedef __m128i int128_t;

    // Helper functions for MSVC
    static inline uint8_t get_byte(const int128_t& value, int index) {
        return ((uint8_t*)&value)[index];
    }

    static inline uint64_t get_uint64(const int128_t& value, int index) {
        return ((uint64_t*)&value)[index];
    }

    static inline int128_t shift_left(const int128_t value, int shift) {
        if (shift >= 128) {
            return _mm_setzero_si128();
        } else if (shift == 0) {
            return value;
        } else if (shift < 64) {
            __m128i lo = _mm_slli_epi64(value, shift);
            __m128i hi = _mm_slli_epi64(_mm_srli_si128(value, 8), shift);
            __m128i v_cross = _mm_srli_epi64(value, 64 - shift);
            v_cross = _mm_and_si128(v_cross, _mm_set_epi64x(0, -1));
            hi = _mm_or_si128(hi, v_cross);
            return _mm_or_si128(_mm_slli_si128(hi, 8), lo);
        } else {
            // For shifts >= 64, we need to move the lower bits to the upper half
            __m128i hi = _mm_slli_epi64(value, shift - 64 + 1);
            return _mm_slli_si128(hi, 8);
        }
    }

    static inline int128_t bitwise_and(const int128_t& a, const int128_t& b) {
        return _mm_and_si128(a, b);
    }

    static inline int128_t bitwise_or(const int128_t& a, const int128_t& b) {
        return _mm_or_si128(a, b);
    }

    static inline bool is_equal(const int128_t& a, const int128_t& b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
    }

    static inline int128_t create_from_int(long long value) {
        return _mm_set_epi64x(0, value);
    }

    static inline int128_t subtract(const int128_t& a, const int128_t& b) {
        __m128i borrow = _mm_setzero_si128();
        __m128i result = _mm_sub_epi64(a, b);
        borrow = _mm_srli_epi64(_mm_cmpgt_epi64(b, a), 63);
        __m128i high_result = _mm_sub_epi64(_mm_srli_si128(a, 8), _mm_srli_si128(b, 8));
        high_result = _mm_sub_epi64(high_result, borrow);
        return _mm_or_si128(result, _mm_slli_si128(high_result, 8));
    }

    static inline int128_t bitwise_not(const int128_t& a) {
        return _mm_xor_si128(a, _mm_set1_epi32(-1));
    }

    static inline char *to_string(const int128_t& value) {
        static char buffer[257] = { 0 };
        sprintf(buffer, "%016llx%016llx", get_uint64(value, 1), get_uint64(value, 0));
        return buffer;
    }

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
    typedef __int128 int128_t;

    // Helper functions for GCC
    static inline uint8_t get_byte(const int128_t& value, int index) {
        return (value >> (index * 8)) & 0xFF;
    }

    static inline int128_t shift_left(const int128_t& value, int shift) {
        return value << shift;
    }

    static inline int128_t bitwise_and(const int128_t& a, const int128_t& b) {
        return a & b;
    }

    static inline int128_t bitwise_or(const int128_t& a, const int128_t& b) {
        return a | b;
    }

    static inline bool is_equal(const int128_t& a, const int128_t& b) {
        return a == b;
    }

    static inline int128_t create_from_int(long long value) {
        return (int128_t)value;
    }

    static inline int128_t subtract(const int128_t& a, const int128_t& b) {
        return a - b;
    }

    static inline int128_t bitwise_not(const int128_t& a) {
        return ~a;
    }

#else
    #error "No 128-bit integer type available for this platform"
#endif

//#define MTF_SIZE 1024
#define MTF_SIZE (256+64+16+1)

static const float BASE_GRADIENT_SCALE = 10.0f;
static const float GRADIENT_POW = 3.0f;
static const float EDGE_WEIGHT = 2.0f;
static const float CORNER_WEIGHT = 1.0f;
static const float VARIANCE_THRESHOLD = 0.001f;
static const float SIGMOID_CENTER = 0.1f;
static const float SIGMOID_STEEPNESS = 20.0f;

#define ERROR_FN calculate_mse

typedef struct {
    int h[256];
    int size;
} histo_t;

typedef struct {
    int128_t list[MTF_SIZE];
    int size;
    histo_t histogram;
} MTF_LL;

typedef struct {
    float gradient_magnitude;
    float adjusted_lambda;
} block_info_t;

template<typename T> static inline T max(T a, T b) { return a > b ? a : b; }
template<typename T> static inline T min(T a, T b) { return a < b ? a : b; }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-SIGMOID_STEEPNESS * (x - SIGMOID_CENTER))); }

static inline float log2_fast(float val) {
    union { float val; int x; } u = { val };
    float log_2 = (float)(((u.x >> 23) & 255) - 128);              
    u.x   &= ~(255 << 23);
    u.x   += 127 << 23;
    log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f; 
    return log_2;
} 

static void histo_reset(histo_t *h) {
    for(int i = 0; i < 256; i++) {
        h->h[i] = 0;
    }
    h->size = 0;
}

static void histo_update(histo_t *h, int128_t value, int128_t mask) {
    for (int i = 0; i < 8; i++) {
        uint8_t m = get_byte(mask, i);
        if(m) {
            uint8_t byte = get_byte(value, i);
            h->h[byte]++;
            h->size++;
        }
    }
}

static float histo_cost(histo_t *h, int128_t value, int128_t mask) {
    float tlb = (float)h->size;
    int c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 16; i++) {
        if (get_byte(mask, i)) {
            c[i] = h->h[get_byte(value, i)] + 1;
            tlb += 1;
        }
    }
    float cost = 0;
    for (int i = 0; i < 16; i++) {
        if (c[i] > 0) {
            cost += log2_fast(tlb / c[i]);
        }
    }
    return cost;
}

static void mtf_ll_init(MTF_LL* mtf) {
    mtf->size = 0;
    histo_reset(&mtf->histogram);
}

// Search for a value in the MTF list
static int mtf_ll_search(MTF_LL* mtf, int128_t value, int128_t mask) {
    value = bitwise_and(value, mask);
    for (int i = 0; i < mtf->size; i++) {
        if (is_equal(bitwise_and(mtf->list[i], mask), value)) {
            return i;
        }
    }
    return -1;
}

static int mtf_ll_encode(MTF_LL* mtf, int128_t value, int128_t mask) {
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

static bool mtf_ll_contains(MTF_LL* mtf, int128_t value, int128_t mask) {
    return mtf_ll_search(mtf, value, mask) != -1;
}

static int mtf_ll_peek_position(MTF_LL* mtf, int128_t value, int128_t mask) {
    int pos = mtf_ll_search(mtf, value, mask);
    if (pos != -1) {
        return pos;
    }
    return mtf->size; // Return size if not found (which would be its position if added)
}

static float calculate_bit_cost(int mtf_value, int128_t literal_value, MTF_LL* mtf, int128_t mask) {
    literal_value = bitwise_and(literal_value, mask);
    if (mtf_value == mtf->size) {
        return 8.f + histo_cost(&mtf->histogram, literal_value, mask);
    } else {
        return 10.f + log2_fast((float)(mtf_value + 32)); // Cost for an MTF value
    }
}

template<typename T>
static inline float calculate_mse(const T* img1, const T* img2, const float* gradients, int total) {
#if 0
    // Just normal MSE
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float diff = (float)img1[i] - (float)img2[i];
        sum += diff * diff;
    }
    return sum / total;
#else
    float sum = 0.0;
    float gradient_sum = 0.0;
    for (int i = 0; i < total; i++) {
        float diff = (float)img1[i] - (float)img2[i];
        float weighted_diff = diff * diff * gradients[i/4];
        sum += weighted_diff;
        gradient_sum += gradients[i/4];
    }
    return sum / gradient_sum;
#endif
}

// Calculate Sum of Absolute Differences Error
template<typename T>
static inline float calculate_sad(const T* img1, const T* img2, const float* gradients, int total) {
    float sum = 0.0;
    float gradient_sum = 0.0;
    for (int i = 0; i < total; i++) {
        float diff = ((float)img1[i] - (float)img2[i]) * gradients[i/4];
        sum += diff < 0 ? -diff : diff;
        gradient_sum += gradients[i/4];
    }
    return sum / gradient_sum;
}

template<typename T>
static float calculate_local_variance(const T* img, int width, int height, int channels, int x, int y, int window_size) {
    float mean = 0.0f;
    float mean_sq = 0.0f;
    int count = 0;

    for (int dy = -window_size/2; dy <= window_size/2; dy++) {
        for (int dx = -window_size/2; dx <= window_size/2; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float val = (float)img[(ny * width + nx) * channels] / 255.0f;
                mean += val;
                mean_sq += val * val;
                count++;
            }
        }
    }

    mean /= count;
    mean_sq /= count;
    return mean_sq - mean * mean;
}

template<typename T>
static float calculate_gradient_magnitude_2d(const T* img, int width, int height, int channels) {
    float total_magnitude = 0.0f;
    int window_size = 5;  // Size of local window for variance calculation

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float local_variance = calculate_local_variance(img, width, height, channels, x, y, window_size);
            
            if (local_variance > VARIANCE_THRESHOLD) {
                float gx = 0.0f, gy = 0.0f;
                for (int c = 0; c < channels; c++) {
                    if (x > 0 && x < width - 1) {
                        gx += fabsf((float)img[(y * width + x + 1) * channels + c] - (float)img[(y * width + x - 1) * channels + c]);
                    }
                    if (y > 0 && y < height - 1) {
                        gy += fabsf((float)img[((y + 1) * width + x) * channels + c] - (float)img[((y - 1) * width + x) * channels + c]);
                    }
                }
                gx /= (255.0f * channels);
                gy /= (255.0f * channels);
                
                float gradient = sqrtf(gx * gx + gy * gy);
                total_magnitude += sigmoid(gradient);
            }
        }
    }

    return total_magnitude / (width * height);
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
                        //assert(0);
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

template<typename T>
static void calculate_per_pixel_gradients(const T* img, int width, int height, int depth, int channels, float* gradients) {
    const int total_pixels = width * height * depth;
    const float norm_factor = 1.0f / (255.0f * channels);

    // Pre-compute edge weights
    const float edge_weights[4] = {EDGE_WEIGHT, 1.0f, 1.0f, CORNER_WEIGHT};

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float gx = 0.0f, gy = 0.0f, gz = 0.0f;
                int index = (z * height * width + y * width + x) * channels;

                // Determine edge type (corner, edge, or center)
                int edge_type = (x == 0 || x == width - 1) + (y == 0 || y == height - 1) + (z == 0 || z == depth - 1);

                for (int c = 0; c < channels; c++) {
                    // X direction
                    if (x < width - 1) {
                        gx += fabsf((float)img[index + channels + c] - (float)img[index + c]);
                    } else if (x > 0) {
                        gx += fabsf((float)img[index + c] - (float)img[index - channels + c]);
                    }

                    // Y direction
                    if (y < height - 1) {
                        gy += fabsf((float)img[index + width * channels + c] - (float)img[index + c]);
                    } else if (y > 0) {
                        gy += fabsf((float)img[index + c] - (float)img[index - width * channels + c]);
                    }

                    // Z direction
                    if (depth > 1) {
                        if (z < depth - 1) {
                            gz += fabsf((float)img[index + width * height * channels + c] - (float)img[index + c]);
                        } else if (z > 0) {
                            gz += fabsf((float)img[index + c] - (float)img[index - width * height * channels + c]);
                        }
                    }
                }

                float gradient = sqrtf(gx * gx + gy * gy + gz * gz) * norm_factor;
                gradients[z * height * width + y * width + x] = (1.0f - sigmoid(gradient)) * edge_weights[edge_type];
            }
        }
    }
}

static void mtf_pass(uint8_t* data, size_t data_len, int block_width, int block_height, int block_depth, int block_type, float lambda, int BITS_OFFSET, const int128_t INDEX_MASK, block_size_descriptor* bsd, uint8_t* all_original_decoded, block_info_t* block_info, float* all_gradients) {
    uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;

    MTF_LL mtf;
    mtf_ll_init(&mtf);


    // Initialize weight bits table -- block size matters in determining that certain ones are errors
#define NUM_BLOCK_MODES 2048
    uint8_t* weight_bits = (uint8_t*)malloc(NUM_BLOCK_MODES);
    for (size_t i = 0; i < NUM_BLOCK_MODES; ++i) {
        uint8_t block[16];
        block[0] = (i & 255);
        block[1] = ((i >> 8) & 255);
        weight_bits[i] = get_weight_bits(&block[0], block_width, block_height, block_depth);
    }

#define GET_WEIGHT_BITS(block)  weight_bits[*(uint16_t*)(block) & 0x7ff]

    // Helper function to process a single block
    auto process_block = [&](size_t block_index) {
        uint8_t *current_block = data + block_index * block_size;
        int128_t current_bits = *((int128_t*)(current_block + BITS_OFFSET));
        int128_t best_match = current_bits;

        // Generate mask from weights_bits
        int128_t WEIGHTS_MASK = INDEX_MASK;
        if (is_equal(WEIGHTS_MASK, create_from_int(0))) {
            int WEIGHT_BITS = weight_bits[((uint16_t*)current_block)[0] & 0x7ff];
            int128_t one = create_from_int(1);
            int128_t mask = shift_left(one, WEIGHT_BITS);
            mask = subtract(mask, one);
            WEIGHTS_MASK = shift_left(mask, 128 - WEIGHT_BITS);
        } else if (is_equal(WEIGHTS_MASK, create_from_int(1))) {
            // Create a mask with all the OTHER bits set 
            int WEIGHT_BITS = weight_bits[((uint16_t*)current_block)[0] & 0x7ff];
            int128_t one = create_from_int(1);
            int128_t mask = shift_left(one, WEIGHT_BITS);
            mask = subtract(mask, one);
            WEIGHTS_MASK = shift_left(mask, 128 - WEIGHT_BITS);
            WEIGHTS_MASK = bitwise_not(WEIGHTS_MASK);
            // Turn off the first 17 bits (Mode + CEM) (Doesn't work??)
            //WEIGHTS_MASK = bitwise_and(WEIGHTS_MASK, bitwise_not(create_from_int(0x1FFFF)));
        }

        uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

        float original_bit_cost = calculate_bit_cost(mtf_ll_peek_position(&mtf, current_bits, WEIGHTS_MASK), current_bits, &mtf, WEIGHTS_MASK);

        // Decode the original block to compute initial MSE
        astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
        float original_mse;
        float *gradients = all_gradients + block_index * block_width * block_height * block_depth;
        if (block_type == ASTCENC_TYPE_U8) {
            original_mse = ERROR_FN(original_decoded, modified_decoded, gradients, block_width*block_height*block_depth*4);
        } else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, gradients, block_width*block_height*block_depth*4);
        }

        float adjusted_lambda = block_info[block_index].adjusted_lambda;
        float best_rd_cost = original_mse + adjusted_lambda * original_bit_cost;

        // Search through the MTF list
        for (int k = 0; k < mtf.size; k++) {
            int128_t candidate_bits = mtf.list[k];
            
            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            int128_t* temp_bits = (int128_t*)(temp_block + BITS_OFFSET);
            *temp_bits = bitwise_or(
                bitwise_and(*temp_bits, bitwise_not(WEIGHTS_MASK)),
                bitwise_and(candidate_bits, WEIGHTS_MASK)
            );

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
                mse = ERROR_FN(original_decoded, modified_decoded, gradients, block_width*block_height*block_depth*4);
            } else {
                mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, gradients, block_width*block_height*block_depth*4);
            }

            float modified_bit_cost = calculate_bit_cost(k, candidate_bits, &mtf, WEIGHTS_MASK);
            float rd_cost = mse + adjusted_lambda * modified_bit_cost;

            if (rd_cost < best_rd_cost) {
                best_match = candidate_bits;
                best_rd_cost = rd_cost;
            }
        }

        if (!is_equal(best_match, current_bits)) {
            int128_t new_bits = bitwise_or(
                bitwise_and(current_bits, bitwise_not(WEIGHTS_MASK)),
                bitwise_and(best_match, WEIGHTS_MASK)
            );
            *((int128_t*)(current_block + BITS_OFFSET)) = new_bits;
        }

        // Update the literal histogram with the chosen bits
        if (!mtf_ll_contains(&mtf, best_match, WEIGHTS_MASK)) {
            histo_update(&mtf.histogram, best_match, WEIGHTS_MASK);
        }

        // Update the MTF with the chosen bits
        mtf_ll_encode(&mtf, best_match, WEIGHTS_MASK);
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
    free(weight_bits);
}

void print_adjusted_lambda_ascii(const block_info_t* block_info, int blocks_width, int blocks_height) {
    char* ascii_image = (char*)malloc(blocks_width * blocks_height + blocks_height + 1);
    
    float min_lambda = FLT_MAX;
    float max_lambda = -FLT_MAX;

    // Find min and max lambda
    for (int i = 0; i < blocks_width * blocks_height; i++) {
        min_lambda = min(min_lambda, block_info[i].adjusted_lambda);
        max_lambda = max(max_lambda, block_info[i].adjusted_lambda);
    }

    // Create ASCII representation
    for (int y = 0; y < blocks_height; y++) {
        for (int x = 0; x < blocks_width; x++) {
            int block_index = y * blocks_width + x;
            float normalized_lambda = (block_info[block_index].adjusted_lambda - min_lambda) / (max_lambda - min_lambda);
            
            // Map normalized lambda to ASCII characters
            char ascii_char;
            if (normalized_lambda < 0.2) ascii_char = '.';
            else if (normalized_lambda < 0.4) ascii_char = ':';
            else if (normalized_lambda < 0.6) ascii_char = 'o';
            else if (normalized_lambda < 0.8) ascii_char = 'O';
            else ascii_char = '#';
            
            ascii_image[y * (blocks_width + 1) + x] = ascii_char;
        }
        ascii_image[y * (blocks_width + 1) + blocks_width] = '\n';
    }
    ascii_image[blocks_height * (blocks_width + 1)] = '\0';

    // Print ASCII representation
    printf("Adjusted Lambda ASCII Representation:\n");
    printf("%s\n", ascii_image);

    free(ascii_image);
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
        lambda = 1.0f;
    }

    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    // Calculate gradient magnitudes and adjusted lambdas for all blocks
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    size_t decoded_block_size = block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
    uint8_t *all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);
    float *all_gradients = (float*)malloc(num_blocks * block_width * block_height * block_depth * sizeof(float));

    for (size_t i = 0; i < num_blocks; i++) {
        uint8_t *original_block = data + i * block_size;
        uint8_t *decoded_block = all_original_decoded + i * decoded_block_size;
        float *gradients = all_gradients + i * block_width * block_height * block_depth;
        
        astc_decompress_block(*bsd, original_block, decoded_block, block_width, block_height, block_depth, block_type);
        
        if (block_type == ASTCENC_TYPE_U8) {
            calculate_per_pixel_gradients(decoded_block, block_width, block_height, block_depth, 4, gradients);
        } else {
            calculate_per_pixel_gradients((float*)decoded_block, block_width, block_height, block_depth, 4, gradients);
        }
    }

    block_info_t* block_info = (block_info_t*)malloc(num_blocks * sizeof(block_info_t));
    for (size_t i = 0; i < num_blocks; i++) {
        float *gradients = all_gradients + i * block_width * block_height * block_depth;
        float avg_gradient = 0.0f;
        for (int j = 0; j < block_width * block_height * block_depth; j++) {
            avg_gradient += gradients[j];
        }
        avg_gradient /= (block_width * block_height * block_depth);

        float gradient_factor = powf(1 - avg_gradient, GRADIENT_POW) * BASE_GRADIENT_SCALE;
        float adjusted_lambda = lambda * gradient_factor;
        //printf("%f\n", adjusted_lambda);

        block_info[i].gradient_magnitude = avg_gradient;
        block_info[i].adjusted_lambda = adjusted_lambda;
    }

    //print_adjusted_lambda_ascii(block_info, 768/block_width, 512/block_height);

    // MTF passes...
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, create_from_int(0), bsd, all_original_decoded, block_info, all_gradients);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, create_from_int(1), bsd, all_original_decoded, block_info, all_gradients);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, create_from_int(0), bsd, all_original_decoded, block_info, all_gradients);
    mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, create_from_int(1), bsd, all_original_decoded, block_info, all_gradients);

    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 8, 0xFFFFFFFFFFFF0000ull, bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 7, 0xFFFFFFull, bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 2, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded, block_info, all_gradients);

    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 8, 0xFFFFFFFFFFFF0000ull, bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 0, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, block_width, block_height, block_depth, block_type, lambda, 2, 0xFFFFFFFFFFFFFFFFull, bsd, all_original_decoded, block_info, all_gradients);

    // Clean up
    free(bsd);
    free(all_original_decoded);
    free(all_gradients);
}