#include <thread>
#include <vector>
#include <atomic>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h> // For SSE intrinsics
#ifdef __ARM_NEON
#include <arm_neon.h> // For NEON intrinsics
#endif
#include "astcenc.h"
#include "astcenccli_internal.h"
#include "astcenc_internal_entry.h"
#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

#if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
    typedef __m128i int128_t;

    // Helper functions for MSVC
    static __forceinline uint8_t get_byte(const int128_t& value, int index) {
        return ((uint8_t*)&value)[index];
    }

    static __forceinline uint64_t get_uint64(const int128_t& value, int index) {
        return ((uint64_t*)&value)[index];
    }

    static __forceinline int128_t shift_left(const int128_t value, int shift) {
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

    static __forceinline int128_t bitwise_and(const int128_t& a, const int128_t& b) {
        return _mm_and_si128(a, b);
    }

    static __forceinline int128_t bitwise_or(const int128_t& a, const int128_t& b) {
        return _mm_or_si128(a, b);
    }

    static __forceinline bool is_equal(const int128_t& a, const int128_t& b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
    }

    static __forceinline int128_t create_from_int(long long value) {
        return _mm_set_epi64x(0, value);
    }

    static __forceinline int128_t subtract(const int128_t& a, const int128_t& b) {
        __m128i borrow = _mm_setzero_si128();
        __m128i result = _mm_sub_epi64(a, b);
        borrow = _mm_srli_epi64(_mm_cmpgt_epi64(b, a), 63);
        __m128i high_result = _mm_sub_epi64(_mm_srli_si128(a, 8), _mm_srli_si128(b, 8));
        high_result = _mm_sub_epi64(high_result, borrow);
        return _mm_or_si128(result, _mm_slli_si128(high_result, 8));
    }

    static __forceinline int128_t bitwise_not(const int128_t& a) {
        return _mm_xor_si128(a, _mm_set1_epi32(-1));
    }

    static __forceinline char *to_string(const int128_t& value) {
        static char buffer[257] = { 0 };
        sprintf(buffer, "%016llx%016llx", get_uint64(value, 1), get_uint64(value, 0));
        return buffer;
    }

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
    typedef __int128 int128_t;

    // Helper functions for GCC
    static __forceinline uint8_t get_byte(const int128_t& value, int index) {
        return (value >> (index * 8)) & 0xFF;
    }

    static __forceinline int128_t shift_left(const int128_t& value, int shift) {
        return value << shift;
    }

    static __forceinline int128_t bitwise_and(const int128_t& a, const int128_t& b) {
        return a & b;
    }

    static __forceinline int128_t bitwise_or(const int128_t& a, const int128_t& b) {
        return a | b;
    }

    static __forceinline bool is_equal(const int128_t& a, const int128_t& b) {
        return a == b;
    }

    static __forceinline int128_t create_from_int(long long value) {
        return (int128_t)value;
    }

    static __forceinline int128_t subtract(const int128_t& a, const int128_t& b) {
        return a - b;
    }

    static __forceinline int128_t bitwise_not(const int128_t& a) {
        return ~a;
    }

#else
    #error "No 128-bit integer type available for this platform"
#endif

//#define MAX_MTF_SIZE (256+64+16+1)
#define MAX_MTF_SIZE (1024+256+64+16+1)
#define CACHE_SIZE (4096)  // Should be a power of 2 for efficient modulo operation

static const float EDGE_WEIGHT = 2.0f;
static const float CORNER_WEIGHT = 1.0f;
static const float SIGMOID_CENTER = 0.1f;
static const float SIGMOID_STEEPNESS = 20.0f;

#define ERROR_FN calculate_mse_weighted

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
    uint8_t encoded[16];
    uint8_t decoded[6 * 6 * 6 * 4 * 4];  // Max size for both U8 and float types
    bool valid;
} CachedBlock;

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

static uint32_t hash_128(int128_t value) {
    uint32_t hash = 0;
    for (int i = 0; i < 4; i++) {
        hash ^= ((uint32_t*)&value)[i];
    }
    return hash;
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
    float cost = 1.0f;
    int count = 0;

    for (int i = 0; i < 16; i++) {
        if (get_byte(mask, i)) {
            int c = h->h[get_byte(value, i)] + 1;
            tlb += 1;
            cost *= tlb / c;
            count++;
        }
    }

    return count > 0 ? log2_fast(cost) : 0.0f;
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

template<typename T1, typename T2>
static inline float calculate_mse_weighted(const T1* img1, const T2* img2, int total, const float* weights) {
    float sum = 0.0f;
    static const float channel_weights[4] = {0.299f, 0.587f, 0.114f, 1.0f};  // R, G, B, A weights
    for (int i = 0; i < total; i++) {
        float diff = (float)img1[i] - (float)img2[i];
        sum += diff * diff * weights[i/4] * channel_weights[i%4];
    }
    return sum / total;
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

static void dual_mtf_pass(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, block_size_descriptor* bsd, uint8_t* all_original_decoded, float* all_gradients) {
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    const int max_threads = 8;  // Maximum number of threads
    const int blocks_per_thread = (int)((num_blocks + max_threads - 1) / max_threads);
    const int num_threads = min(max_threads, (int)((num_blocks + blocks_per_thread - 1) / blocks_per_thread));

    std::vector<std::thread> threads;

    auto thread_function = [&](size_t start_block, size_t end_block) {
        uint8_t *modified_decoded = (uint8_t*)malloc(6 * 6 * 6 * 4 * 4);
        CachedBlock* block_cache = (CachedBlock*)calloc(CACHE_SIZE, sizeof(CachedBlock));
        MTF_LL mtf_weights;
        MTF_LL mtf_endpoints;

        // Initialize weight bits table
        uint8_t* weight_bits = (uint8_t*)malloc(2048);
        for (size_t i = 0; i < 2048; ++i) {
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
                histo_update(&mtf_weights.histogram, best_match, bitwise_not(create_from_int(0)));
                histo_update(&mtf_endpoints.histogram, best_match, bitwise_not(create_from_int(0)));
                mtf_ll_encode(&mtf_weights, best_match, create_from_int(0));
                mtf_ll_encode(&mtf_endpoints, best_match, bitwise_not(create_from_int(0)));
                return; // Constant color block, skip
            }

            int128_t one = create_from_int(1);
            int128_t weights_mask = shift_left(subtract(shift_left(one, WEIGHT_BITS), one), 128 - WEIGHT_BITS);
            int128_t endpoints_mask = bitwise_not(weights_mask);

            uint8_t* original_decoded = all_original_decoded + block_index * (block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4));

            // Function to get or compute MSE for a candidate
            auto get_or_compute_mse = [&](const int128_t& candidate_bits) -> float {
                uint32_t hash = hash_128(candidate_bits) & (CACHE_SIZE - 1);  // Modulo CACHE_SIZE

                // Check if the candidate is in the cache
                if (block_cache[hash].valid && is_equal(*((int128_t*)block_cache[hash].encoded), candidate_bits)) {
                    // Compute MSE using cached decoded data
                    if (block_type == ASTCENC_TYPE_U8) {
                        return ERROR_FN(original_decoded, block_cache[hash].decoded, block_width*block_height*block_depth*4, all_gradients);
                    } else {
                        return ERROR_FN((float*)original_decoded, (float*)block_cache[hash].decoded, block_width*block_height*block_depth*4, all_gradients);
                    }
                }

                // If not in cache, compute and cache the result
                uint8_t temp_block[16];
                *((int128_t*)temp_block) = candidate_bits;

                // Decode and compute MSE
                astc_decompress_block(*bsd, temp_block, block_cache[hash].decoded, block_width, block_height, block_depth, block_type);
                memcpy(block_cache[hash].encoded, temp_block, 16);
                block_cache[hash].valid = true;

                if (block_type == ASTCENC_TYPE_U8) {
                    return ERROR_FN(original_decoded, block_cache[hash].decoded, block_width*block_height*block_depth*4, all_gradients);
                } else {
                    return ERROR_FN((float*)original_decoded, (float*)block_cache[hash].decoded, block_width*block_height*block_depth*4, all_gradients);
                }
            };

            // Decode the original block to compute initial MSE
            astc_decompress_block(*bsd, current_block, modified_decoded, block_width, block_height, block_depth, block_type);
            float original_mse;
            if (block_type == ASTCENC_TYPE_U8) {
                original_mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4, all_gradients);
            } else {
                original_mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4, all_gradients);
            }

            float adjusted_lambda_weights = lambda;
            float adjusted_lambda_endpoints = lambda;
            float best_rd_cost = original_mse + adjusted_lambda_weights * calculate_bit_cost(mtf_ll_peek_position(&mtf_weights, current_bits, weights_mask), current_bits, &mtf_weights, weights_mask) +
                                                adjusted_lambda_endpoints * calculate_bit_cost(mtf_ll_peek_position(&mtf_endpoints, current_bits, endpoints_mask), current_bits, &mtf_endpoints, endpoints_mask);

            struct Candidate {
                int128_t bits;
                float rd_cost;
                int mtf_position;
            };
            const int best_candidates_count = 8;
            Candidate best_weights[8];
            Candidate best_endpoints[8];
            int weights_count = 0;
            int endpoints_count = 0;

            // Find best endpoint candidates
            for (int k = 0; k < mtf_endpoints.size; k++) {
                int128_t candidate_endpoints = mtf_endpoints.list[k];
                //int endpoints_mode = ((uint16_t*)&candidate_endpoints)[0] & 0x7ff;
                //int endpoints_weight_bits = weight_bits[endpoints_mode];
                //if (endpoints_weight_bits < WEIGHT_BITS) {
                    //continue;
                //}
                
                float mse = get_or_compute_mse(candidate_endpoints);

                float bit_cost = adjusted_lambda_endpoints * calculate_bit_cost(k, candidate_endpoints, &mtf_endpoints, endpoints_mask);
                float rd_cost = mse + bit_cost;

                // Insert into best_endpoints if it's one of the best candidates
                if (endpoints_count < best_candidates_count || rd_cost < best_endpoints[best_candidates_count - 1].rd_cost) {
                    int insert_pos = endpoints_count < best_candidates_count ? endpoints_count : best_candidates_count - 1;
                    
                    // Find the position to insert
                    while (insert_pos > 0 && rd_cost < best_endpoints[insert_pos - 1].rd_cost) {
                        best_endpoints[insert_pos] = best_endpoints[insert_pos - 1];
                        insert_pos--;
                    }
                    
                    best_endpoints[insert_pos] = {candidate_endpoints, rd_cost, k};
                    
                    if (endpoints_count < best_candidates_count) {
                        endpoints_count++;
                    }
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
                    mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                } else {
                    mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                }

                float bit_cost = adjusted_lambda_weights * calculate_bit_cost(k, candidate_weights, &mtf_weights, weights_mask);
                float rd_cost = mse + bit_cost;

                // Insert into best_weights if it's one of the best candidates
                if (weights_count < best_candidates_count || rd_cost < best_weights[best_candidates_count - 1].rd_cost) {
                    int insert_pos = weights_count < best_candidates_count ? weights_count : best_candidates_count - 1;
                    
                    // Find the position to insert
                    while (insert_pos > 0 && rd_cost < best_weights[insert_pos - 1].rd_cost) {
                        best_weights[insert_pos] = best_weights[insert_pos - 1];
                        insert_pos--;
                    }
                    
                    best_weights[insert_pos] = {candidate_weights, rd_cost, k};
                    
                    if (weights_count < best_candidates_count) {
                        weights_count++;
                    }
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
                        mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                    } else {
                        mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                    }

                    float rd_cost = mse + adjusted_lambda_weights * calculate_bit_cost(best_weights[i].mtf_position, *temp_bits, &mtf_weights, weights_mask) +
                                        adjusted_lambda_endpoints * calculate_bit_cost(best_endpoints[j].mtf_position, *temp_bits, &mtf_endpoints, endpoints_mask);

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
                        mse = ERROR_FN(original_decoded, modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                    } else {
                        mse = ERROR_FN((float*)original_decoded, (float*)modified_decoded, block_width*block_height*block_depth*4, all_gradients);
                    }

                    rd_cost = mse + adjusted_lambda_weights * calculate_bit_cost(best_weights[i].mtf_position, *temp_bits, &mtf_weights, weights_mask) +
                                    adjusted_lambda_endpoints * calculate_bit_cost(best_endpoints[j].mtf_position, *temp_bits, &mtf_endpoints, endpoints_mask);

                    if (rd_cost < best_rd_cost) {
                        best_match = *temp_bits;
                        best_rd_cost = rd_cost;
                    }
                }
            }

            if (!is_equal(best_match, current_bits)) {
                *((int128_t*)current_block) = best_match;
            }

            // Recalculate masks for the best match
            int best_mode = ((uint16_t*)&best_match)[0] & 0x7ff;
            int best_weight_bits = weight_bits[best_mode];
            int128_t best_weights_mask = shift_left(subtract(shift_left(one, best_weight_bits), one), 128 - best_weight_bits);
            int128_t best_endpoints_mask = bitwise_not(best_weights_mask);

            histo_update(&mtf_weights.histogram, best_match, bitwise_not(create_from_int(0)));
            histo_update(&mtf_endpoints.histogram, best_match, bitwise_not(create_from_int(0)));
            mtf_ll_encode(&mtf_weights, best_match, best_weights_mask); 
            mtf_ll_encode(&mtf_endpoints, best_match, best_endpoints_mask);
        };

        int mtf_size = MAX_MTF_SIZE;

        // Backward pass
        mtf_ll_init(&mtf_weights, mtf_size);
        mtf_ll_init(&mtf_endpoints, mtf_size);
        for (size_t i = end_block; i-- > start_block;) {
            process_block(i);
        }

        // Forward pass
        mtf_ll_init(&mtf_weights, mtf_size);
        mtf_ll_init(&mtf_endpoints, mtf_size);
        for (size_t i = start_block; i < end_block; i++) {
            process_block(i);
        }

        // Clean up
        free(modified_decoded);
        free(weight_bits);
    };

    // Start threads
    for (int i = 0; i < num_threads; ++i) {
        size_t start_block = i * blocks_per_thread;
        size_t end_block = min((i + 1) * blocks_per_thread, (int)num_blocks);
        threads.emplace_back(thread_function, start_block, end_block);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}

void jeff_pass(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda, block_size_descriptor* bsd, uint8_t* all_original_decoded, float *all_gradients) {
    //lambda /= 5.f;
    //lambda *= 8.f;
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    const int max_palette = 512; 
    int num_palette = 0;
    const int pixels_per_block = block_width * block_height * block_depth * 4;

    // Allocate memory for clusters
    typedef struct {
        int* block_indexes;
        int num_blocks;
        int capacity;
        float block_average_pixels[6*6*6*4];
        float last_used;
    } cluster_t;

    cluster_t* palette = (cluster_t*)malloc(max_palette * sizeof(cluster_t));
    for (int i = 0; i < max_palette; i++) {
        palette[i].block_indexes = (int*)malloc(16 * sizeof(int));  // Initial capacity
        palette[i].num_blocks = 0;
        palette[i].capacity = 16;
        palette[i].last_used = 0;
        memset(palette[i].block_average_pixels, 0, sizeof(palette[i].block_average_pixels));
    }

    // Prepare compression settings
    astcenc_config config;
    astcenc_context* context;
    astcenc_error status = astcenc_config_init(ASTCENC_PRF_LDR, block_width, block_height, block_depth, ASTCENC_PRE_THOROUGH, 0, &config);
    if (status != ASTCENC_SUCCESS) {
        // Handle error
        return;
    }
    status = astcenc_context_alloc(&config, 1, &context);
    if (status != ASTCENC_SUCCESS) {
        // Handle error
        return;
    }

	compression_working_buffers *temp_buffers = (compression_working_buffers*)malloc(sizeof(compression_working_buffers));

    float current_time = 0;

    // For each block
    for (int i = 0; i < num_blocks; ++i) {
        float best_rd_cost = INFINITY;
        int best_palette_index = -1;

        const uint8_t* original_decoded = all_original_decoded + i * pixels_per_block * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);

        // Find the best palette entry for this block to add to.
        for (int j = 0; j < num_palette; ++j) {
            cluster_t* cluster = &palette[j];

            // Speculatively add this block to the palette entry
            float new_average[6*6*6*4];
            for (int k = 0; k < pixels_per_block; k++) {
                new_average[k] = (cluster->block_average_pixels[k] * cluster->num_blocks + original_decoded[k]) / (cluster->num_blocks + 1);
            }

            // Compute MSE for all blocks in this cluster
            float total_mse = 0;
            for (int k = 0; k < cluster->num_blocks; k++) {
                int block_index = cluster->block_indexes[k];
                uint8_t* original_block = all_original_decoded + block_index * pixels_per_block * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
                
                if (block_type == ASTCENC_TYPE_U8) {
                    total_mse += ERROR_FN(original_block, new_average, pixels_per_block, all_gradients);
                } else {
                    total_mse += ERROR_FN((float*)original_block, (float*)new_average, pixels_per_block, all_gradients);
                }
            }

            // Add MSE for the current block
            if (block_type == ASTCENC_TYPE_U8) {
                total_mse += ERROR_FN(original_decoded, new_average, pixels_per_block, all_gradients);
            } else {
                total_mse += ERROR_FN((float*)original_decoded, (float*)new_average, pixels_per_block, all_gradients);
            }

            // Calculate average MSE
            float avg_mse = total_mse / (cluster->num_blocks + 1);

            // Compute RD cost
            float rd_cost = avg_mse + lambda * log2f(j + 1.f);  // Simple model for bit cost

            // Update best if this is better
            if (rd_cost < best_rd_cost) {
                best_rd_cost = rd_cost;
                best_palette_index = j;
            }
        }

        // Consider adding a new palette entry
        if (num_palette < max_palette) {
            float new_entry_cost = lambda * log2f(num_palette + 2.f);  // Cost of adding a new entry
            if (new_entry_cost < best_rd_cost) {
                best_rd_cost = new_entry_cost;
                best_palette_index = num_palette;
                num_palette++;
            }
        } else {
            // Find the least recently used palette entry
            int lru_index = 0;
            float oldest_time = palette[0].last_used;
            for (int j = 1; j < max_palette; j++) {
                if (palette[j].last_used < oldest_time) {
                    oldest_time = palette[j].last_used;
                    lru_index = j;
                }
            }

            // Consider evicting the LRU entry
            float eviction_cost = lambda * log2f(max_palette + 1.f);  // Cost of evicting and adding a new entry
            if (eviction_cost < best_rd_cost) {
                best_rd_cost = eviction_cost;
                best_palette_index = lru_index;

                // Clear the evicted palette entry
                palette[lru_index].num_blocks = 0;
                memset(palette[lru_index].block_average_pixels, 0, sizeof(palette[lru_index].block_average_pixels));
            }
        }
        
        // Add block to the best palette entry
        cluster_t* best_cluster = &palette[best_palette_index];
        if (best_cluster->num_blocks >= best_cluster->capacity) {
            best_cluster->capacity *= 2;
            best_cluster->block_indexes = (int*)realloc(best_cluster->block_indexes, best_cluster->capacity * sizeof(int));
        }
        best_cluster->block_indexes[best_cluster->num_blocks++] = i;
        best_cluster->last_used = current_time;

        // Update the average for the best palette entry
        for (int k = 0; k < pixels_per_block; k++) {
            best_cluster->block_average_pixels[k] = (best_cluster->block_average_pixels[k] * (best_cluster->num_blocks - 1) + 
                                                    (block_type == ASTCENC_TYPE_U8 ? original_decoded[k] : ((float*)original_decoded)[k])) / 
                                                    best_cluster->num_blocks;
        }

        // Compress the average block once
        image_block blk = { 0 };
        blk.texel_count = static_cast<uint8_t>(block_width * block_height * block_depth);
        blk.grayscale = false;
        blk.channel_weight = vfloat4(1.0f);

        // compute data min/max/mean
        float min = 65535.0f;
        float max = 0.0f;
        float mean = 0.0f;
        for (int i = 0; i < pixels_per_block; i++) {
            float v = best_cluster->block_average_pixels[i] / 255.f * 65535.f;
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            mean += v;
        }
        mean /= pixels_per_block;

        blk.data_min = vfloat4(min);
        blk.data_max = vfloat4(max);
        blk.data_mean = vfloat4(mean);

        if (block_type == ASTCENC_TYPE_U8) {
            for (int i = 0; i < pixels_per_block; i+=4) {
                blk.data_r[i/4] = static_cast<float>(best_cluster->block_average_pixels[i    ]) / 255.f * 65535.f;
				blk.data_g[i/4] = static_cast<float>(best_cluster->block_average_pixels[i + 1]) / 255.f * 65535.f;
				blk.data_b[i/4] = static_cast<float>(best_cluster->block_average_pixels[i + 2]) / 255.f * 65535.f;
				blk.data_a[i/4] = static_cast<float>(best_cluster->block_average_pixels[i + 3]) / 255.f * 65535.f;
            }
        } else {
            for (int i = 0; i < pixels_per_block; i+=4) {
                blk.data_r[i/4] = best_cluster->block_average_pixels[i    ];
                blk.data_g[i/4] = best_cluster->block_average_pixels[i + 1];
                blk.data_b[i/4] = best_cluster->block_average_pixels[i + 2];
                blk.data_a[i/4] = best_cluster->block_average_pixels[i + 3];
            }
        }
        
        blk.origin_texel = vfloat4(blk.data_r[0], blk.data_g[0], blk.data_b[0], blk.data_a[0]);

		uint8_t compressed_block[16];
        compress_block(context->context, blk, compressed_block, *temp_buffers);

        // Update every block in the block index of the current palette entry to the new compressed values.
        for (int k = 0; k < best_cluster->num_blocks; k++) {
            uint8_t* block = data + best_cluster->block_indexes[k] * block_size;
            memcpy(block, compressed_block, 16);
        }

        current_time += 1.0f;
    }

    // Clean up
    for (int i = 0; i < num_palette; i++) {
        free(palette[i].block_indexes);
    }
    free(palette);
    free(temp_buffers);
}

void reconstruct_image(uint8_t* all_original_decoded, int width, int height, int depth, int block_width, int block_height, int block_depth, int block_type, uint8_t* output_image) {
    int blocks_x = (width + block_width - 1) / block_width;
    int blocks_y = (height + block_height - 1) / block_height;
    int blocks_z = (depth + block_depth - 1) / block_depth;
    int channels = 4;
    int pixel_size = (block_type == ASTCENC_TYPE_U8) ? 1 : 4;

    for (int z = 0; z < blocks_z; z++) {
        for (int y = 0; y < blocks_y; y++) {
            for (int x = 0; x < blocks_x; x++) {
                int block_index = (z * blocks_y * blocks_x) + (y * blocks_x) + x;
                uint8_t* block_data = all_original_decoded + block_index * block_width * block_height * block_depth * channels * pixel_size;

                for (int bz = 0; bz < block_depth; bz++) {
                    for (int by = 0; by < block_height; by++) {
                        for (int bx = 0; bx < block_width; bx++) {
                            int image_x = x * block_width + bx;
                            int image_y = y * block_height + by;
                            int image_z = z * block_depth + bz;

                            if (image_x < width && image_y < height && image_z < depth) {
                                int image_index = (image_z * height * width + image_y * width + image_x) * channels * pixel_size;
                                int block_pixel_index = (bz * block_height * block_width + by * block_width + bx) * channels * pixel_size;

                                for (int c = 0; c < channels * pixel_size; c++) {
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
static void generate_gaussian_kernel(float sigma, float* kernel, int* kernel_radius) {
    *kernel_radius = (int)ceil(3.0f * sigma);
    if (*kernel_radius > MAX_KERNEL_SIZE / 2) {
        *kernel_radius = MAX_KERNEL_SIZE / 2;
    }
    
    float sum = 0.0f;
    for (int x = -(*kernel_radius); x <= *kernel_radius; x++) {
        float value = expf(-(x*x) / (2.0f * sigma * sigma));
        kernel[x + *kernel_radius] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (int i = 0; i < 2 * (*kernel_radius) + 1; i++) {
        kernel[i] /= sum;
    }
}

// Apply 1D convolution for 3D images
template <typename T>
static void apply_1d_convolution_3d(const T* input, T* output, int width, int height, int depth, int channels,
                             const float* kernel, int kernel_radius, int direction) {
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum[4] = {0};
                for (int k = -kernel_radius; k <= kernel_radius; k++) {
                    int sx = direction == 0 ? x + k : x;
                    int sy = direction == 1 ? y + k : y;
                    int sz = direction == 2 ? z + k : z;
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height && sz >= 0 && sz < depth) {
                        const T* pixel = input + (sz * height * width + sy * width + sx) * channels;
                        float kvalue = kernel[k + kernel_radius];
                        for (int c = 0; c < channels; c++) {
                            sum[c] += pixel[c] * kvalue;
                        }
                    }
                }
                T* out_pixel = output + (z * height * width + y * width + x) * channels;
                for (int c = 0; c < channels; c++) {
                    if(std::is_same_v<T, uint8_t>) {
                        out_pixel[c] = (uint8_t)(sum[c] + 0.5f);
                    } else {
                        out_pixel[c] = (T)sum[c];
                    }
                }
            }
        }
    }
}

// Separable Gaussian blur for 3D images
template <typename T>
void gaussian_blur_3d(const T* input, T* output, int width, int height, int depth, int channels, float sigma) {
    float kernel[MAX_KERNEL_SIZE];
    int kernel_radius;
    generate_gaussian_kernel(sigma, kernel, &kernel_radius);
    
    T* temp1 = (T*)malloc(width * height * depth * channels * sizeof(T));
    T* temp2 = (T*)malloc(width * height * depth * channels * sizeof(T));
    
    // X direction pass
    apply_1d_convolution_3d(input, temp1, width, height, depth, channels, kernel, kernel_radius, 0);
    
    // Y direction pass
    apply_1d_convolution_3d(temp1, temp2, width, height, depth, channels, kernel, kernel_radius, 1);
    
    // Z direction pass (only if depth > 1)
    if (depth > 1) {
        apply_1d_convolution_3d(temp2, output, width, height, depth, channels, kernel, kernel_radius, 2);
    } else {
        memcpy(output, temp2, width * height * depth * channels * sizeof(T));
    }
    
    free(temp1);
    free(temp2);
}

void high_pass_filter_squared_blurred(const uint8_t* input, float* output, int width, int height, int depth, int channels, float sigma_highpass, float sigma_blur) {
    size_t pixel_count = width * height * depth;
    size_t image_size = pixel_count * channels;
    uint8_t* blurred = (uint8_t*)malloc(image_size);
    float* squared_diff = (float*)malloc(pixel_count * sizeof(float));
    
    // Apply initial Gaussian blur for high-pass filter
    gaussian_blur_3d(input, blurred, width, height, depth, channels, sigma_highpass);
    
    // Calculate squared differences (combined across channels)
    for (size_t i = 0; i < pixel_count; i++) {
        float diff_sum = 0;
        for (int c = 0; c < channels; c++) {
            float diff = (float)input[i * channels + c] - (float)blurred[i * channels + c];
            diff_sum += diff * diff;
        }
        squared_diff[i] = diff_sum;
    }
    
    // Apply second Gaussian blur to the squared differences
    gaussian_blur_3d(squared_diff, output, width, height, depth, 1, sigma_blur);
    
    // Map x |-> C1/(C2 + sqrt(x))
    float C1 = 256.0f*4;
    float C2 = 1.0f;
    float activity_scalar = 0.75f;
    for (size_t i = 0; i < pixel_count; i++) {
        output[i] = C1 / (C2 + activity_scalar * sqrtf(output[i]));
        output[i] = max(output[i], 1.0f);
    }
    
    free(blurred);
    free(squared_diff);
}

void optimize_for_lz(uint8_t* data, size_t data_len, int blocks_x, int blocks_y, int blocks_z, int block_width, int block_height, int block_depth, int block_type, float lambda) {
    if (lambda <= 0.0f) {
        lambda = 10.0f;
    }

    // Map lambda from [10, 40] to ...
    float lambda_10 = 0.025f;
    float lambda_40 = 0.175f;
    lambda = lambda_10 + (lambda - 10.0f) * (lambda_40 - lambda_10) / (40.0f - 10.0f);

    // Initialize block_size_descriptor once
    block_size_descriptor* bsd = (block_size_descriptor*)malloc(sizeof(*bsd));
    init_block_size_descriptor(block_width, block_height, block_depth, false, 4, 43, *bsd);

    // Calculate gradient magnitudes and adjusted lambdas for all blocks
    const int block_size = 16;
    size_t num_blocks = data_len / block_size;
    size_t decoded_block_size = block_width * block_height * block_depth * 4 * (block_type == ASTCENC_TYPE_U8 ? 1 : 4);
    uint8_t *all_original_decoded = (uint8_t*)malloc(num_blocks * decoded_block_size);

    // Preserve original blocks
    uint8_t *original_blocks = (uint8_t*)malloc(data_len);
    memcpy(original_blocks, data, data_len);

    for (size_t i = 0; i < num_blocks; i++) {
        uint8_t *original_block = data + i * block_size;
        uint8_t *decoded_block = all_original_decoded + i * decoded_block_size;
        astc_decompress_block(*bsd, original_block, decoded_block, block_width, block_height, block_depth, block_type);
    }

    // Calculate the full image dimensions
    int width = blocks_x * block_width;
    int height = blocks_y * block_height;
    int depth = blocks_z * block_depth;

    // Allocate memory for the reconstructed image
    size_t image_size = width * height * depth * 4;
    uint8_t* reconstructed_image = (uint8_t*)malloc(image_size);
    float* high_pass_image = (float*)malloc(width * height * depth * sizeof(float)); // Single channel

    // Reconstruct the image from all_original_decoded
    reconstruct_image(all_original_decoded, width, height, depth, block_width, block_height, block_depth, block_type, reconstructed_image);

    // Apply high-pass filter with squared differences and additional blur
    high_pass_filter_squared_blurred(reconstructed_image, high_pass_image, width, height, depth, 4, 2.2f, 1.25f);

    //jeff_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded);

    dual_mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, high_pass_image);
    dual_mtf_pass(data, data_len, blocks_x, blocks_y, blocks_z, block_width, block_height, block_depth, block_type, lambda, bsd, all_original_decoded, high_pass_image);

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
            original_mse = ERROR_FN(original_decoded, temp_decompressed, block_width * block_height * block_depth * 4, high_pass_image);
        }
        else {
            original_mse = ERROR_FN((float*)original_decoded, (float*)temp_decompressed, block_width * block_height * block_depth * 4, high_pass_image);
        }

        // Process and recalculate the block
        process_and_recalculate_astc_block(*bsd, temp_block, block_width, block_height, block_depth, block_type, original_decoded, original_block_ptr);

        // Decode the modified block
        astc_decompress_block(*bsd, temp_block, temp_decompressed, block_width, block_height, block_depth, block_type);

        // Calculate new MSE
        float new_mse;
        if (block_type == ASTCENC_TYPE_U8) {
            new_mse = ERROR_FN(original_decoded, temp_decompressed, block_width * block_height * block_depth * 4, high_pass_image);
        }
        else {
            new_mse = ERROR_FN((float*)original_decoded, (float*)temp_decompressed, block_width * block_height * block_depth * 4, high_pass_image);
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
    free(original_blocks);
    free(reconstructed_image);
    free(high_pass_image);
}