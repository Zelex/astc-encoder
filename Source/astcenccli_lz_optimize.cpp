#include <stdlib.h>
#include <math.h>
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"
#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

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

//#define MAX_MTF_SIZE (256+64+16+1)
#define MAX_MTF_SIZE (1024+256+64+16+1)

static const float BASE_GRADIENT_SCALE = 10.0f;
static const float GRADIENT_POW = 3.0f;
static const float EDGE_WEIGHT = 2.0f;
static const float CORNER_WEIGHT = 1.0f;
static const float SIGMOID_CENTER = 0.1f;
static const float SIGMOID_STEEPNESS = 20.0f;

#define ERROR_FN calculate_mse

typedef struct {
    int h[256];
    int size;
} histo_t;

typedef struct {
    int128_t list[MAX_MTF_SIZE];
    int size;
    int max_size;
    histo_t histogram;
} MTF_LL;

typedef struct {
    float adjusted_lambda_weights;
    float adjusted_lambda_endpoints;
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

static inline void histo_update(histo_t *h, int128_t value, int128_t mask) {
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

static void mtf_ll_init(MTF_LL* mtf, int max_size) {
    mtf->size = 0;
    mtf->max_size = max_size > MAX_MTF_SIZE ? MAX_MTF_SIZE : max_size;
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
        if (mtf->size < mtf->max_size) {
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
        return 1.f + histo_cost(&mtf->histogram, literal_value, mask);
    } else {
        return 10.f + log2_fast((float)(mtf_value + 32)); // Cost for an MTF value
    }
}

template<typename T>
static inline float calculate_mse(const T* img1, const T* img2, int total) {
    float sum = 0.0;
    const float weights[4] = {0.299f, 0.587f, 0.114f, 1.0f}; // R, G, B, A weights
    for (int i = 0; i < total; i += 4) {
        for (int c = 0; c < 4; c++) {
            float diff = (float)img1[i + c] - (float)img2[i + c];
            sum += weights[c] * diff * diff;
        }
    }
    return sum / (total / 4);
}

template<typename T>
static inline float calculate_nmse(const T* x, const T* y, int n) {
    double sum_squared_diff = 0;
    double sum_x2 = 0;

    for (int i = 0; i < n; i++) {
        double diff = x[i] - y[i];
        sum_squared_diff += diff * diff;
        sum_x2 += x[i] * x[i];
    }

    if (sum_x2 == 0) {
        return (sum_squared_diff == 0) ? 0 : INFINITY;
    }

    return (float)(sum_squared_diff / sum_x2);
}

template<typename T>
static inline float calculate_ssim(const T* x, const T* y, int n) {
    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }

    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double var_x = (sum_x2 / n) - (mean_x * mean_x);
    double var_y = (sum_y2 / n) - (mean_y * mean_y);
    double cov_xy = (sum_xy / n) - (mean_x * mean_y);

    double numerator = (2 * mean_x * mean_y + C1) * (2 * cov_xy + C2);
    double denominator = (mean_x * mean_x + mean_y * mean_y + C1) * (var_x + var_y + C2);
    return (float)(numerator / denominator);
}

// https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf
template<typename T>
static inline float calculate_ssim2(const T* x, const T* y, int n) {
    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    const double C1 = 0.01 * 0.01;  // For pixel values in [0,1]
    const double C2 = 0.03 * 0.03;  // For pixel values in [0,1]
    const double epsilon = DBL_EPSILON;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }

    double mean_x = sum_x / n;
    double mean_y = sum_y / n;

    // Use two-pass algorithm for improved numerical stability in variance calculation
    double var_x = 0, var_y = 0, cov_xy = 0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }
    var_x /= (n - 1);
    var_y /= (n - 1);
    cov_xy /= (n - 1);

    // Avoid division by zero
    double l = (2 * mean_x * mean_y + C1) / (mean_x * mean_x + mean_y * mean_y + C1);
    
    double std_x = sqrt(var_x);
    double std_y = sqrt(var_y);
    double c = (2 * std_x * std_y + C2) / (var_x + var_y + C2);
    
    double s;
    if (std_x * std_y < epsilon) {
        s = 1.0;  // If both standard deviations are very close to zero, assume perfect structural similarity
    } else {
        s = (cov_xy + C2/2) / (std_x * std_y + C2/2);
    }

    double d1 = fmax(0, fmin(1, 1 - l));  // Clamp to [0, 1]
    double d2 = fmax(0, fmin(1, 1 - c * s));  // Clamp to [0, 1]

    return sqrtf((float)(d1 + d2));// *sqrtf((float)mse);
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

void process_and_recalculate_astc_block(
    const block_size_descriptor& bsd,
    uint8_t* block_ptr,
    int block_width,
    int block_height,
    int block_depth,
    int block_type,
    const uint8_t* original_decoded,
    const uint8_t* original_block_ptr)
{
    // Extract block information
    symbolic_compressed_block scb;
    physical_to_symbolic(bsd, block_ptr, scb);

    if (scb.block_type == SYM_BTYPE_CONST_U16) {
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
    for (int i = 0; i < blk.texel_count; i++) {
        if (block_type == ASTCENC_TYPE_U8) {
            blk.data_r[i] = original_decoded[i * 4 + 0] / 255.0f;
            blk.data_g[i] = original_decoded[i * 4 + 1] / 255.0f;
            blk.data_b[i] = original_decoded[i * 4 + 2] / 255.0f;
            blk.data_a[i] = original_decoded[i * 4 + 3] / 255.0f;
        } else {
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
    for (int i = 0; i < blk.texel_count; i++) {
        ch_sum += blk.texel(i);
    }
    blk.data_mean = ch_sum / static_cast<float>(blk.texel_count);
    blk.rgb_lns[0] = blk.data_max.lane<0>() > 1.0f || blk.data_max.lane<1>() > 1.0f || blk.data_max.lane<2>() > 1.0f;
    blk.alpha_lns[0] = blk.data_max.lane<3>() > 1.0f;

    // Set channel weights (assuming equal weights for simplicity)
    blk.channel_weight = vfloat4(1.0f);

    // Extract endpoints from the original block
    symbolic_compressed_block original_scb;
    physical_to_symbolic(bsd, original_block_ptr, original_scb);

    for (unsigned int i = 0; i < scb.partition_count; i++) {
        vint4 color0, color1;
        bool rgb_hdr, alpha_hdr;
        
        // Determine the appropriate astcenc_profile based on block_type
        astcenc_profile decode_mode = (block_type == ASTCENC_TYPE_U8) ? ASTCENC_PRF_LDR : ASTCENC_PRF_HDR;

        unpack_color_endpoints(
            decode_mode,
            original_scb.color_formats[i],
            original_scb.color_values[i],
            rgb_hdr,
            alpha_hdr,
            color0,
            color1
        );

        ep.endpt0[i] = int_to_float(color0) * (1.0f / 255.0f);
        ep.endpt1[i] = int_to_float(color1) * (1.0f / 255.0f);
    }

    if (is_dual_plane)
    {
        recompute_ideal_colors_2planes(
            blk, bsd, di,
            scb.weights, scb.weights + WEIGHTS_PLANE2_OFFSET, 
            ep, rgbs_vectors[0], rgbo_vectors[0], plane2_component
        );
    }
    else
    {
        recompute_ideal_colors_1plane(
            blk, pi, di, scb.weights,
            ep, rgbs_vectors, rgbo_vectors
        );
    }

    // Update endpoints in the symbolic compressed block
    for (unsigned int i = 0; i < scb.partition_count; i++) {
        vfloat4 endpt0 = ep.endpt0[i];
        vfloat4 endpt1 = ep.endpt1[i];

        // Quantize endpoints
        vfloat4 color0 = endpt0 * 65535.0f;
        vfloat4 color1 = endpt1 * 65535.0f;

        quant_method quant_level = static_cast<quant_method>(mode.quant_mode);

        // Note/TODO: Pack Color endpoints doesn't support RGB_DELTA or RGBA_DELTA
		uint8_t fmt = scb.color_formats[i];
        if (quant_level > QUANT_6 && (fmt == FMT_RGB || fmt == FMT_RGBA)) {
            // Store quantized endpoints
            scb.color_formats[i] = pack_color_endpoints(
                color0,
                color1,
                rgbs_vectors[i],
                rgbo_vectors[i],
                fmt,
                scb.color_values[i],
                quant_level
            );
        }
    }

    // Compress to a physical block
    symbolic_to_physical(bsd, scb, block_ptr);
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

static inline uint32_t xy_to_morton(uint32_t x, uint32_t y) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

static inline uint32_t xyz_to_morton(uint32_t x, uint32_t y, uint32_t z) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y <<  8)) & 0x0300F00F;
    y = (y | (y <<  4)) & 0x030C30C3;
    y = (y | (y <<  2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z <<  8)) & 0x0300F00F;
    z = (z | (z <<  4)) & 0x030C30C3;
    z = (z | (z <<  2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
}

static void mtf_pass(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, const int128_t INDEX_MASK, block_size_descriptor* bsd, uint8_t* all_original_decoded, block_info_t* block_info, float* all_gradients) {
    uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;

    MTF_LL mtf;


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
        int128_t current_bits = *((int128_t*)(current_block));
        int128_t best_match = current_bits;

        int mode = ((uint16_t*)current_block)[0] & 0x7ff;
        int WEIGHT_BITS = weight_bits[mode];
        
        int128_t WEIGHTS_MASK = INDEX_MASK;
        bool is_weights_search = is_equal(WEIGHTS_MASK, create_from_int(0));

        // Generate mask from weights_bits
        if (is_weights_search) {
            // edge case, constant color blocks
            if (WEIGHT_BITS == 0) {
                return;
            }
            int128_t one = create_from_int(1);
			WEIGHTS_MASK = shift_left(subtract(shift_left(one, WEIGHT_BITS), one), 128 - WEIGHT_BITS);
        } else {
            // Create a mask with all the OTHER bits set 
            int128_t one = create_from_int(1);
			WEIGHTS_MASK = shift_left(subtract(shift_left(one, WEIGHT_BITS), one), 128 - WEIGHT_BITS);
            WEIGHTS_MASK = bitwise_not(WEIGHTS_MASK);
            // Turn off the first 17 bits (Mode + CEM) (Doesn't work??)
            //WEIGHTS_MASK = bitwise_and(WEIGHTS_MASK, bitwise_not(create_from_int(0x1FFFF)));
        }

        uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

        float original_bit_cost = calculate_bit_cost(mtf_ll_peek_position(&mtf, current_bits, WEIGHTS_MASK), current_bits, &mtf, WEIGHTS_MASK);

        // Decode the original block to compute initial MSE
        astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
        float original_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            original_mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
        } else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
        }

        float adjusted_lambda = is_weights_search ? block_info[block_index].adjusted_lambda_weights : block_info[block_index].adjusted_lambda_endpoints;
        float best_rd_cost = original_mse + adjusted_lambda * original_bit_cost;

        // Search through the MTF list
        for (int k = 0; k < mtf.size; k++) {
            int128_t candidate_bits = mtf.list[k];
            int candidate_mode = ((uint16_t*)&candidate_bits)[0] & 0x7ff;
            int candidate_weight_bits = weight_bits[candidate_mode];
            if (is_weights_search) {
				if(candidate_weight_bits != WEIGHT_BITS) {
					continue;
				}
            } else {
				if(candidate_weight_bits < WEIGHT_BITS) {
					continue;
				}
            }
            
            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            int128_t* temp_bits = (int128_t*)(temp_block);
            *temp_bits = bitwise_or(
                bitwise_and(*temp_bits, bitwise_not(WEIGHTS_MASK)),
                bitwise_and(candidate_bits, WEIGHTS_MASK)
            );

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
                mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
            } else {
                mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
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
            *((int128_t*)(current_block)) = new_bits;
        }
		histo_update(&mtf.histogram, best_match, bitwise_not(create_from_int(0)));
		mtf_ll_encode(&mtf, best_match, WEIGHTS_MASK);
    };

    // Note: slightly better to do backwards first. due to tie breakers, you want the forward pass to always win.

    int mtf_size = MAX_MTF_SIZE;

    // Morton order pass
    mtf_ll_init(&mtf, mtf_size);
    if(block_depth > 1) {
        for (int z = 0; z < blocks_z; z++) {
            for (int y = 0; y < blocks_y; y++) {
                for (int x = 0; x < blocks_x; x++) {
                    uint32_t morton = xyz_to_morton(x, y, z);
                    size_t block_index = morton;
                    if (block_index < num_blocks) {
                        process_block(block_index);
                    }
                }
            }
        }
    } else {
        for (int y = 0; y < blocks_y; y++) {
            for (int x = 0; x < blocks_x; x++) {
                uint32_t morton = xy_to_morton(x, y);
                size_t block_index = morton;
                if (block_index < num_blocks) {
                    process_block(block_index);
                }
            }
        }
    }

    // Backward pass
    mtf_ll_init(&mtf, mtf_size);
    for (size_t i = num_blocks; i-- > 0;) {
        process_block(i);
    }

    // Forward pass
    mtf_ll_init(&mtf, mtf_size);
    for (size_t i = 0; i < num_blocks; i++) {
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
        min_lambda = min(min_lambda, block_info[i].adjusted_lambda_weights);
        max_lambda = max(max_lambda, block_info[i].adjusted_lambda_weights);
    }

    // Create ASCII representation
    for (int y = 0; y < blocks_height; y++) {
        for (int x = 0; x < blocks_width; x++) {
            int block_index = y * blocks_width + x;
            float normalized_lambda = (block_info[block_index].adjusted_lambda_weights - min_lambda) / (max_lambda - min_lambda);
            
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

static void dual_mtf_pass(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, block_size_descriptor* bsd, uint8_t* all_original_decoded, block_info_t* block_info, float* all_gradients) {
    uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;

    MTF_LL mtf_weights;
    MTF_LL mtf_endpoints;

    // Initialize weight bits table
    uint8_t* weight_bits = (uint8_t*)malloc(NUM_BLOCK_MODES);
    for (size_t i = 0; i < NUM_BLOCK_MODES; ++i) {
        uint8_t block[16];
        block[0] = (i & 255);
        block[1] = ((i >> 8) & 255);
        weight_bits[i] = get_weight_bits(&block[0], block_width, block_height, block_depth);
    }

    // Helper function to process a single block
    auto process_block = [&](size_t block_index) {
        uint8_t *current_block = data + block_index * block_size;
        int128_t current_bits = *((int128_t*)current_block);
        int128_t best_match = current_bits;

        
        int mode = ((uint16_t*)current_block)[0] & 0x7ff;
        int WEIGHT_BITS = weight_bits[mode];
        if (WEIGHT_BITS == 0) {
            return; // Constant color block, skip
        }

        int128_t one = create_from_int(1);
        int128_t weights_mask = shift_left(subtract(shift_left(one, WEIGHT_BITS), one), 128 - WEIGHT_BITS);
        int128_t endpoints_mask = bitwise_not(weights_mask);

        uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

        // Decode the original block to compute initial MSE
        astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
        float original_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            original_mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
        } else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
        }

        float adjusted_lambda_weights = block_info[block_index].adjusted_lambda_weights;
        float adjusted_lambda_endpoints = block_info[block_index].adjusted_lambda_endpoints;
        float best_rd_cost = original_mse + adjusted_lambda_weights * calculate_bit_cost(mtf_ll_peek_position(&mtf_weights, current_bits, weights_mask), current_bits, &mtf_weights, weights_mask) +
                                            adjusted_lambda_endpoints * calculate_bit_cost(mtf_ll_peek_position(&mtf_endpoints, current_bits, endpoints_mask), current_bits, &mtf_endpoints, endpoints_mask);

        struct Candidate {
            int128_t bits;
            float rd_cost;
            float bit_cost;
        };
        Candidate best_weights[16];
        Candidate best_endpoints[16];
        int weights_count = 0;
        int endpoints_count = 0;
        const int best_candidates_count = 16;

        // Find best endpoint candidates
        for (int k = 0; k < mtf_endpoints.size; k++) {
            int128_t candidate_endpoints = mtf_endpoints.list[k];
            int endpoints_mode = ((uint16_t*)&candidate_endpoints)[0] & 0x7ff;
            int endpoints_weight_bits = weight_bits[endpoints_mode];
            //if (endpoints_weight_bits < WEIGHT_BITS) {
                //continue;
            //}
            int128_t weights_mask = shift_left(subtract(shift_left(one, endpoints_weight_bits), one), 128 - endpoints_weight_bits);
            int128_t endpoints_mask = bitwise_not(weights_mask);

            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            int128_t* temp_bits = (int128_t*)temp_block;
            *temp_bits = bitwise_or(
                bitwise_and(current_bits, weights_mask), // TODO: really we should use optimal weights here.... but how?
                bitwise_and(candidate_endpoints, endpoints_mask)
            );

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
                mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
            } else {
                mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
            }

            float bit_cost = adjusted_lambda_endpoints * calculate_bit_cost(k, candidate_endpoints, &mtf_endpoints, endpoints_mask);
            float rd_cost = mse + bit_cost;

            // Insert into best_endpoints if it's one of the best candidates
            if (endpoints_count < best_candidates_count) {
                best_endpoints[endpoints_count++] = {candidate_endpoints, rd_cost};
            } else if (rd_cost < best_endpoints[best_candidates_count - 1].rd_cost) {
                // Find the position to insert
                int insert_pos = best_candidates_count - 1;
                while (insert_pos > 0 && rd_cost < best_endpoints[insert_pos - 1].rd_cost) {
                    best_endpoints[insert_pos] = best_endpoints[insert_pos - 1];
                    insert_pos--;
                }
                best_endpoints[insert_pos] = {candidate_endpoints, rd_cost, bit_cost};
            }
        }

        // Best of best endpoint candidate
        int128_t best_endpoints_bits = best_endpoints[0].bits;
        int best_endpoints_mode = ((uint16_t*)&best_endpoints_bits)[0] & 0x7ff;
        int best_endpoints_weight_bits = weight_bits[best_endpoints_mode];
        int128_t best_endpoints_weight_mask = shift_left(subtract(shift_left(one, best_endpoints_weight_bits), one), 128 - best_endpoints_weight_bits);
        int128_t best_endpoints_endpoint_mask = bitwise_not(best_endpoints_weight_mask);

        // Find best weight candidates
        for (int k = 0; k < mtf_weights.size; k++) {
            int128_t candidate_weights = mtf_weights.list[k];
            int weights_mode = ((uint16_t*)&candidate_weights)[0] & 0x7ff;
            int weights_weight_bits = weight_bits[weights_mode];
            if (weights_weight_bits < best_endpoints_weight_bits) {
                continue;
            }

            uint8_t temp_block[16];
            memcpy(temp_block, current_block, 16);
            int128_t* temp_bits = (int128_t*)temp_block;
            *temp_bits = bitwise_or(
                bitwise_and(candidate_weights, best_endpoints_weight_mask),
                bitwise_and(best_endpoints_bits, best_endpoints_endpoint_mask)
            );

            astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

            float mse;
            if (block_type == ASTCENC_TYPE_U8) {
                mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
            } else {
                mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
            }

            float bit_cost = adjusted_lambda_weights * calculate_bit_cost(k, candidate_weights, &mtf_weights, weights_mask);
            float rd_cost = mse + bit_cost;

            // Insert into best_weights if it's one of the best candidates
            if (weights_count < best_candidates_count) {
                best_weights[weights_count++] = {candidate_weights, rd_cost};
            } else if (rd_cost < best_weights[best_candidates_count - 1].rd_cost) {
                // Find the position to insert
                int insert_pos = best_candidates_count - 1;
                while (insert_pos > 0 && rd_cost < best_weights[insert_pos - 1].rd_cost) {
                    best_weights[insert_pos] = best_weights[insert_pos - 1];
                    insert_pos--;
                }
                best_weights[insert_pos] = {candidate_weights, rd_cost, bit_cost};
            }
        }

        // Search through combinations of best candidates
        for (int i = 0; i < weights_count; i++) {

            for (int j = 0; j < endpoints_count; j++) {
                int128_t candidate_endpoints = best_endpoints[j].bits;
                int endpoints_mode = ((uint16_t*)&candidate_endpoints)[0] & 0x7ff;
                int endpoints_weight_bits = weight_bits[endpoints_mode];

                int128_t weights_mask = shift_left(subtract(shift_left(one, endpoints_weight_bits), one), 128 - endpoints_weight_bits);
                int128_t endpoints_mask = bitwise_not(weights_mask);

                uint8_t temp_block[16];
                memcpy(temp_block, current_block, 16);
                int128_t* temp_bits = (int128_t*)temp_block;
                *temp_bits = bitwise_or(
                    bitwise_and(best_weights[i].bits, weights_mask),
                    bitwise_and(best_endpoints[j].bits, endpoints_mask)
                );

                astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

                float mse;
                if (block_type == ASTCENC_TYPE_U8) {
                    mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
                } else {
                    mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
                }

                float rd_cost = mse + best_weights[i].bit_cost + best_endpoints[j].bit_cost;

                if (rd_cost < best_rd_cost) {
                    best_match = *temp_bits;
                    best_rd_cost = rd_cost;
                }

                // now do the same thing for weights
                int128_t candidate_weights = best_weights[i].bits;
                int weights_mode = ((uint16_t*)&candidate_weights)[0] & 0x7ff;
                int weights_weight_bits = weight_bits[weights_mode];

                weights_mask = shift_left(subtract(shift_left(one, weights_weight_bits), one), 128 - weights_weight_bits);
                endpoints_mask = bitwise_not(weights_mask);

                memcpy(temp_block, current_block, 16);
                *temp_bits = bitwise_or(
                    bitwise_and(best_weights[i].bits, weights_mask),
                    bitwise_and(best_endpoints[j].bits, endpoints_mask)
                );

                astc_decompress_block(*bsd, temp_block, modified_decoded, block_width, block_height, block_depth, block_type);

                if (block_type == ASTCENC_TYPE_U8) {
                    mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4);
                } else {
                    mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4);
                }

                rd_cost = mse + best_weights[i].bit_cost + best_endpoints[j].bit_cost;
                if (rd_cost < best_rd_cost) {
                    best_match = *temp_bits;
                    best_rd_cost = rd_cost;
                }
            }
        }

        if (!is_equal(best_match, current_bits)) {
            *((int128_t*)current_block) = best_match;
        }
		histo_update(&mtf_weights.histogram, best_match, bitwise_not(create_from_int(0)));
		histo_update(&mtf_endpoints.histogram, best_match, bitwise_not(create_from_int(0)));
        mtf_ll_encode(&mtf_weights, best_match, weights_mask);
        mtf_ll_encode(&mtf_endpoints, best_match, endpoints_mask);
    };

    int mtf_size = MAX_MTF_SIZE;

    // Morton order pass
    #if 0
    mtf_ll_init(&mtf_weights, mtf_size);
    mtf_ll_init(&mtf_endpoints, mtf_size);
    if(block_depth > 1) {
        for (int z = 0; z < blocks_z; z++) {
            for (int y = 0; y < blocks_y; y++) {
                for (int x = 0; x < blocks_x; x++) {
                    uint32_t morton = xyz_to_morton(x, y, z);
                    size_t block_index = morton;
                    if (block_index < num_blocks) {
                        process_block(block_index);
                    }
                }
            }
        }
    } else {
        for (int y = 0; y < blocks_y; y++) {
            for (int x = 0; x < blocks_x; x++) {
                uint32_t morton = xy_to_morton(x, y);
                size_t block_index = morton;
                if (block_index < num_blocks) {
                    process_block(block_index);
                }
            }
        }
    }
    #endif

    // Backward pass
    mtf_ll_init(&mtf_weights, mtf_size);
    mtf_ll_init(&mtf_endpoints, mtf_size);
    for (size_t i = num_blocks; i-- > 0;) {
        process_block(i);
    }

    // Forward pass
    mtf_ll_init(&mtf_weights, mtf_size);
    mtf_ll_init(&mtf_endpoints, mtf_size);
    for (size_t i = 0; i < num_blocks; i++) {
        process_block(i);
    }

    // Clean up
    free(modified_decoded);
    free(weight_bits);
}

void optimize_for_lz(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda) {
    if (lambda <= 0.0f) {
        lambda = 10.0f;
    }

    // Map lambda from [10, 40] to [0.5, 1.5]
    float lambda_10 = 0.5f;
    float lambda_40 = 1.5f;
    lambda = lambda_10 + (lambda - 10.0f) * (lambda_40 - lambda_10) / (40.0f - 10.0f);

    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 1.0f, *bsd);

    // Calculate gradient magnitudes and adjusted lambdas for all blocks
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    size_t decoded_block_size = block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
    uint8_t *all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);
    float *all_gradients = (float*)malloc(num_blocks * block_width * block_height * block_depth * sizeof(float));

    // Preserve original blocks
    uint8_t *original_blocks = (uint8_t*)malloc(data_len);
    memcpy(original_blocks, data, data_len);

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
        float adjusted_lambda_weights = lambda * gradient_factor;
        float adjusted_lambda_endpoints = lambda * powf(gradient_factor, 0.5f); // Less aggressive adjustment for endpoints

        block_info[i].adjusted_lambda_weights = adjusted_lambda_weights;
        block_info[i].adjusted_lambda_endpoints = adjusted_lambda_endpoints;
    }

    //print_adjusted_lambda_ascii(block_info, 768/block_width, 512/block_height);


    //mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, create_from_int(0), bsd, all_original_decoded, block_info, all_gradients);
    
    // Process and recalculate each block

    //mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, create_from_int(1), bsd, all_original_decoded, block_info, all_gradients);

    //mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, create_from_int(0), bsd, all_original_decoded, block_info, all_gradients);
    //mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, create_from_int(1), bsd, all_original_decoded, block_info, all_gradients);

    dual_mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_info, all_gradients);
    dual_mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, block_info, all_gradients);

#if 0
    // Allocate temporary memory for decompressed data
    uint8_t* temp_decompressed = (uint8_t*)malloc(6 * 6 * 6 * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

    // Process and recalculate each block
    for (size_t i = 0; i < num_blocks; i++) {
        uint8_t* block_ptr = data + i * 16;  
        const uint8_t* original_block_ptr = original_blocks + i * 16;
        const uint8_t* original_decoded = all_original_decoded + i * decoded_block_size;

        // Use stack allocation for the temporary block
        uint8_t temp_block[16];

        // Copy the original block to temp memory
        memcpy(temp_block, block_ptr, 16);

        // Decode the original block
        astc_decompress_block(*bsd, temp_block, temp_decompressed, block_width, block_height, block_depth, block_type);

        // Calculate original MSE
        float original_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            original_mse = ERROR_FN(original_decoded, temp_decompressed, block_width * block_height * block_depth * 4);
        }
        else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)temp_decompressed, block_width * block_height * block_depth * 4);
        }

        // Process and recalculate the block
        process_and_recalculate_astc_block(*bsd, temp_block, block_width, block_height, block_depth, block_type, original_decoded, original_block_ptr);

        // Decode the modified block
        astc_decompress_block(*bsd, temp_block, temp_decompressed, block_width, block_height, block_depth, block_type);

        // Calculate new MSE
        float new_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            new_mse = ERROR_FN(original_decoded, temp_decompressed, block_width * block_height * block_depth * 4);
        }
        else {
            new_mse = ERROR_FN((float*)original_decoded, (float*)temp_decompressed, block_width * block_height * block_depth * 4);
        }

        // Only accept the changes if the new MSE is better (lower) than the original
        if (new_mse < original_mse) {
            // Copy the improved block back to the main data
            memcpy(block_ptr, temp_block, 16);
        }
    }

    // Free temporary memory
    free(temp_decompressed);
#endif

    // Clean up
    free(bsd);
    free(all_original_decoded);
    free(all_gradients);
    free(original_blocks);
}