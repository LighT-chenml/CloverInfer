#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <string.h>

#include "common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif
#ifndef KVSLOT_SLIM_QK_CACHE_HEADS
#define KVSLOT_SLIM_QK_CACHE_HEADS 12
#endif
#ifndef KVSLOT_SLIM_QK_CACHE_HEAD_DIM
#define KVSLOT_SLIM_QK_CACHE_HEAD_DIM 64
#endif

__host kvslot_slot_args_t slot_args;
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
__host kvslot_qk_dpu_args_t qk_args;
#endif
__host kvslot_runtime_slot_args_t runtime_slot_args;
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
__host kvslot_runtime_slot_args_t grouped_runtime_slot_args[KVSLOT_MAX_GROUP_SEGMENTS];
#endif
__host kvslot_qk_slot_args_t qk_slot_args;
__host uint32_t qk_slot_head_indices[KVSLOT_MAX_HEADS];
__host uint32_t qk_slot_rowmax_bits[KVSLOT_MAX_HEADS];
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
__host uint32_t grouped_segment_lengths[KVSLOT_MAX_GROUP_SEGMENTS];
__host uint32_t grouped_segment_count;
#endif
__host kvslot_meta_t kvslot_meta;
__host uint32_t kvslot_kernel_command;

#define KVSLOT_UNSET_U32 0xffffffffu

__mram_noinit int32_t k_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t v_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t qk_query[KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
__mram_noinit int32_t qk_keys[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int64_t qk_scores[KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t qk_slot_scores_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t av_weights_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
#endif
__mram_noinit uint32_t av_context_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];

BARRIER_INIT(kvslot_barrier, NR_TASKLETS);

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static int64_t partial_sums[NR_TASKLETS];
#endif
static uint32_t qk_slot_score_local[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
static float qk_slot_query_row[KVSLOT_MAX_HEAD_DIM];
#ifdef KVSLOT_SLIM_QK_SLOT_ONLY
static float qk_slot_query_rows[KVSLOT_SLIM_QK_CACHE_HEADS * KVSLOT_SLIM_QK_CACHE_HEAD_DIM];
#endif
#if defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK) && !defined(KVSLOT_SLIM_QK_SLOT_ONLY)
static int16_t qk_slot_query_i16[KVSLOT_MAX_HEAD_DIM];
static float qk_slot_query_i16_scale;
#endif
#if defined(KVSLOT_SLIM_QK_SLOT_ONLY) && (defined(KVSLOT_EXPERIMENTAL_INT16_KV) || defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK))
static int16_t qk_slot_query_i16_rows[KVSLOT_SLIM_QK_CACHE_HEADS * KVSLOT_SLIM_QK_CACHE_HEAD_DIM];
static float qk_slot_query_i16_scales[KVSLOT_SLIM_QK_CACHE_HEADS];
#endif
static float qk_slot_row_sums[KVSLOT_MAX_HEADS];
static uint32_t qk_phase_profile_enabled;

static float u32_bits_to_float(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } value = {.u = bits};
    return value.f;
}

static uint32_t float_to_u32_bits(float value)
{
    union {
        uint32_t u;
        float f;
    } bits = {.f = value};
    return bits.u;
}

static float fast_exp_approx(float x)
{
    const float ln2 = 0.69314718056f;
    const float inv_ln2 = 1.44269504089f;
    float scaled_kf;
    int k;
    float r;
    float r2;
    float r3;
    float r4;
    float r5;
    float r6;
    float poly;
    union {
        uint32_t u;
        float f;
    } scale;

    if (x <= -80.0f) {
        return 0.0f;
    }
    if (x >= 0.0f) {
        return 1.0f;
    }

    scaled_kf = x * inv_ln2;
    k = (int)(scaled_kf + (scaled_kf >= 0.0f ? 0.5f : -0.5f));
    r = x - ((float)k * ln2);
    r2 = r * r;
    r3 = r2 * r;
    r4 = r2 * r2;
    r5 = r4 * r;
    r6 = r3 * r3;
    poly = 1.0f
        + r
        + (0.5f * r2)
        + ((1.0f / 6.0f) * r3)
        + ((1.0f / 24.0f) * r4)
        + ((1.0f / 120.0f) * r5)
        + ((1.0f / 720.0f) * r6);

    if (k < -126) {
        return 0.0f;
    }
    scale.u = (uint32_t)(k + 127) << 23;
    return poly * scale.f;
}

static float fp16_bits_to_float(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits >> 15) & 0x1u;
    uint32_t exponent = ((uint32_t)bits >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)bits & 0x3ffu;
    uint32_t out_bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            out_bits = sign << 31;
        } else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent -= 1u;
            }
            mantissa &= 0x3ffu;
            out_bits = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        out_bits = (sign << 31) | 0x7f800000u | (mantissa << 13);
    } else {
        out_bits = (sign << 31) | ((exponent + (127u - 15u)) << 23) | (mantissa << 13);
    }
    return u32_bits_to_float(out_bits);
}

static float bf16_bits_to_float(uint16_t bits)
{
    return u32_bits_to_float(((uint32_t)bits) << 16);
}

static uint32_t int8_cache_pair_base(uint32_t word_offset, uint32_t logical_idx)
{
    uint32_t word_idx = word_offset + (logical_idx / 4u);
    return word_idx & ~1u;
}

static int8_t unpack_int8_value(uint64_t packed64, uint32_t word_offset, uint32_t logical_idx)
{
    uint32_t word_idx = word_offset + (logical_idx / 4u);
    uint32_t pair_base = word_idx & ~1u;
    uint32_t byte_in_pair = ((word_idx - pair_base) * 4u) + (logical_idx & 3u);
    uint8_t raw = (uint8_t)((packed64 >> (byte_in_pair * 8u)) & 0xffu);
    return (int8_t)raw;
}

static float unpack_int8_scaled_value(uint64_t packed64, uint32_t word_offset, uint32_t logical_idx, float scale)
{
    return ((float)unpack_int8_value(packed64, word_offset, logical_idx)) * scale;
}

#ifdef KVSLOT_CONTEXT_INT8_I16_WEIGHTS
static int32_t quantize_context_weight_i16(float weight)
{
    int32_t quantized;

    if (weight <= 0.0f) {
        return 0;
    }
    if (weight >= 1.0f) {
        return 32767;
    }
    quantized = (int32_t)((weight * 32767.0f) + 0.5f);
    if (quantized < 0) {
        return 0;
    }
    if (quantized > 32767) {
        return 32767;
    }
    return quantized;
}
#endif

static float read_int8_v_value(uint32_t word_offset, uint32_t logical_idx, float scale)
{
    uint64_t packed64 = 0;
    uint32_t pair_base = int8_cache_pair_base(word_offset, logical_idx);
    mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
    return unpack_int8_scaled_value(packed64, word_offset, logical_idx, scale);
}

static float read_int8_k_value(uint32_t word_offset, uint32_t logical_idx, float scale)
{
    uint64_t packed64 = 0;
    uint32_t pair_base = int8_cache_pair_base(word_offset, logical_idx);
    mram_read(&k_cache[pair_base], &packed64, sizeof(packed64));
    return unpack_int8_scaled_value(packed64, word_offset, logical_idx, scale);
}

static uint16_t read_k_packed_u16_value(uint32_t elem_offset, uint32_t logical_idx)
{
    uint64_t packed64 = 0;
    uint32_t packed = 0;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t pair_base = word_idx & ~1u;
    mram_read(&k_cache[pair_base], &packed64, sizeof(packed64));
    packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
    return (logical_idx & 1u) == 0 ? (uint16_t)(packed & 0xffffu) : (uint16_t)(packed >> 16);
}

static uint16_t read_v_packed_u16_value(uint32_t elem_offset, uint32_t logical_idx)
{
    uint64_t packed64 = 0;
    uint32_t packed = 0;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t pair_base = word_idx & ~1u;
    mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
    packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
    return (logical_idx & 1u) == 0 ? (uint16_t)(packed & 0xffffu) : (uint16_t)(packed >> 16);
}

static float u16_cache_bits_to_float(uint16_t bits, uint32_t dtype_code)
{
    if (dtype_code == KVSLOT_DTYPE_BF16) {
        return bf16_bits_to_float(bits);
    }
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
    if (dtype_code == KVSLOT_DTYPE_INT16) {
        return ((float)(int16_t)bits);
    }
#endif
    return fp16_bits_to_float(bits);
}

#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
static float read_int16_v_value(uint32_t word_offset, uint32_t logical_idx, float scale)
{
    return ((float)(int16_t)read_v_packed_u16_value(word_offset, logical_idx)) * scale;
}

static float read_int16_k_value(uint32_t word_offset, uint32_t logical_idx, float scale)
{
    return ((float)(int16_t)read_k_packed_u16_value(word_offset, logical_idx)) * scale;
}
#endif

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static float read_av_weight(uint32_t logical_idx)
{
    uint64_t packed = 0;
    uint32_t pair_base = logical_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&av_weights_bits[pair_base], &packed, sizeof(packed));
    bits = (logical_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}
#endif

static void write_av_context_pair(uint32_t pair_idx, uint32_t low_bits, uint32_t high_bits)
{
    uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
    mram_write(&packed, &av_context_bits[pair_idx * 2u], sizeof(packed));
}

static void write_av_context_quad(
    uint32_t out_idx0,
    uint32_t bits0,
    uint32_t bits1,
    uint32_t bits2,
    uint32_t bits3)
{
    __dma_aligned uint64_t packed[2];
    packed[0] = ((uint64_t)bits1 << 32) | (uint64_t)bits0;
    packed[1] = ((uint64_t)bits3 << 32) | (uint64_t)bits2;
    mram_write(packed, &av_context_bits[out_idx0], sizeof(packed));
}

#ifdef KVSLOT_CONTEXT_BULK_ROW
static int run_context_fp16_bulk_row(uint32_t num_heads, uint32_t window, uint32_t group_heads, uint32_t head_dim)
{
    uint32_t tasklet_id = me();
    uint32_t v_dtype_code = runtime_slot_args.v_dtype_code == KVSLOT_UNSET_U32 ? runtime_slot_args.dtype_code : runtime_slot_args.v_dtype_code;
    float value_scale = 1.0f;
    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32
        ? runtime_slot_args.elem_offset
        : runtime_slot_args.v_elem_offset;

    if ((v_dtype_code != KVSLOT_DTYPE_FP16
            && v_dtype_code != KVSLOT_DTYPE_BF16
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
            && v_dtype_code != KVSLOT_DTYPE_INT16
#endif
        )
        || num_heads == 0
        || window == 0
        || head_dim == 0
        || head_dim > KVSLOT_SLIM_QK_CACHE_HEAD_DIM
        || (head_dim & 3u) != 0
        || (value_elem_offset & 1u) != 0) {
        return 0;
    }
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
    if (v_dtype_code == KVSLOT_DTYPE_INT16) {
        value_scale = runtime_slot_args.v_scale;
    }
#endif

    for (uint32_t head_idx = tasklet_id; head_idx < num_heads; head_idx += NR_TASKLETS) {
        uint32_t kv_head_idx = qk_slot_head_indices[head_idx];
        float acc[KVSLOT_SLIM_QK_CACHE_HEAD_DIM];
        if (kv_head_idx >= group_heads) {
            continue;
        }
#pragma clang loop unroll(disable)
        for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            acc[dim_idx] = 0.0f;
        }

        for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
            __dma_aligned uint32_t packed_words[KVSLOT_SLIM_QK_CACHE_HEAD_DIM / 2u];
            uint32_t value_word_idx = value_elem_offset
                + ((((token_idx * group_heads) + kv_head_idx) * head_dim) / 2u);
            uint32_t packed_word_count = head_dim / 2u;
            float weight = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx * window + token_idx]);
            mram_read(&v_cache[value_word_idx], packed_words, packed_word_count * sizeof(packed_words[0]));
#pragma clang loop unroll(disable)
            for (uint32_t packed_idx = 0; packed_idx < packed_word_count; ++packed_idx) {
                uint32_t packed = packed_words[packed_idx];
                uint32_t dim_idx = packed_idx * 2u;
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
                if (v_dtype_code == KVSLOT_DTYPE_INT16) {
                    acc[dim_idx] += weight * (float)(int16_t)(packed & 0xffffu);
                    acc[dim_idx + 1u] += weight * (float)(int16_t)(packed >> 16);
                } else
#endif
                {
                    acc[dim_idx] += weight * u16_cache_bits_to_float((uint16_t)(packed & 0xffffu), v_dtype_code);
                    acc[dim_idx + 1u] += weight * u16_cache_bits_to_float((uint16_t)(packed >> 16), v_dtype_code);
                }
            }
        }

        for (uint32_t dim_idx = 0; dim_idx < head_dim; dim_idx += 4u) {
            uint32_t out_idx0 = head_idx * head_dim + dim_idx;
            write_av_context_quad(
                out_idx0,
                float_to_u32_bits(acc[dim_idx] * value_scale),
                float_to_u32_bits(acc[dim_idx + 1u] * value_scale),
                float_to_u32_bits(acc[dim_idx + 2u] * value_scale),
                float_to_u32_bits(acc[dim_idx + 3u] * value_scale));
        }
    }
    return 1;
}
#endif

#ifdef KVSLOT_CONTEXT_TILE16
static int run_context_tile16(uint32_t num_heads, uint32_t window, uint32_t group_heads, uint32_t head_dim)
{
    uint32_t tasklet_id = me();
    uint32_t v_dtype_code = runtime_slot_args.v_dtype_code == KVSLOT_UNSET_U32 ? runtime_slot_args.dtype_code : runtime_slot_args.v_dtype_code;
    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32
        ? runtime_slot_args.elem_offset
        : runtime_slot_args.v_elem_offset;

    if (v_dtype_code != KVSLOT_DTYPE_FP16
        || num_heads == 0
        || window == 0
        || group_heads == 0
        || head_dim == 0
        || (head_dim % 16u) != 0
        || (value_elem_offset & 1u) != 0) {
        return 0;
    }

    uint32_t tiles_per_head = head_dim / 16u;
    uint32_t total_tiles = num_heads * tiles_per_head;
    for (uint32_t tile_idx = tasklet_id; tile_idx < total_tiles; tile_idx += NR_TASKLETS) {
        uint32_t head_idx = tile_idx / tiles_per_head;
        uint32_t dim_idx0 = (tile_idx - (head_idx * tiles_per_head)) * 16u;
        uint32_t kv_head_idx = qk_slot_head_indices[head_idx];
        uint32_t score_idx = head_idx * window;
        float acc[16];

        if (kv_head_idx >= group_heads) {
            continue;
        }

        for (uint32_t acc_idx = 0; acc_idx < 16u; ++acc_idx) {
            acc[acc_idx] = 0.0f;
        }

        uint32_t value_word_stride = (group_heads * head_dim) / 2u;
        uint32_t value_word_idx = value_elem_offset + (((kv_head_idx * head_dim) + dim_idx0) / 2u);
        for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
            __dma_aligned uint32_t packed_words[8];
            float weight = u32_bits_to_float(qk_slot_score_local[score_idx + token_idx]);

            mram_read(&v_cache[value_word_idx], packed_words, sizeof(packed_words));
            value_word_idx += value_word_stride;
#pragma clang loop unroll(disable)
            for (uint32_t packed_idx = 0; packed_idx < 8u; ++packed_idx) {
                uint32_t packed = packed_words[packed_idx];
                uint32_t acc_idx = packed_idx * 2u;
                acc[acc_idx] += weight * fp16_bits_to_float((uint16_t)(packed & 0xffffu));
                acc[acc_idx + 1u] += weight * fp16_bits_to_float((uint16_t)(packed >> 16));
            }
        }

        uint32_t out_idx0 = head_idx * head_dim + dim_idx0;
        write_av_context_quad(
            out_idx0,
            float_to_u32_bits(acc[0]),
            float_to_u32_bits(acc[1]),
            float_to_u32_bits(acc[2]),
            float_to_u32_bits(acc[3]));
        write_av_context_quad(
            out_idx0 + 4u,
            float_to_u32_bits(acc[4]),
            float_to_u32_bits(acc[5]),
            float_to_u32_bits(acc[6]),
            float_to_u32_bits(acc[7]));
        write_av_context_quad(
            out_idx0 + 8u,
            float_to_u32_bits(acc[8]),
            float_to_u32_bits(acc[9]),
            float_to_u32_bits(acc[10]),
            float_to_u32_bits(acc[11]));
        write_av_context_quad(
            out_idx0 + 12u,
            float_to_u32_bits(acc[12]),
            float_to_u32_bits(acc[13]),
            float_to_u32_bits(acc[14]),
            float_to_u32_bits(acc[15]));
    }
    return 1;
}
#endif

static float read_v_value(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx)
{
    uint32_t elem_offset = slot->v_elem_offset;
    uint32_t dtype_code = slot->v_dtype_code;
    if (elem_offset == KVSLOT_UNSET_U32) {
        elem_offset = slot->elem_offset;
    }
    if (dtype_code == KVSLOT_UNSET_U32) {
        dtype_code = slot->dtype_code;
    }
    if (dtype_code == KVSLOT_DTYPE_INT8) {
        return read_int8_v_value(elem_offset, logical_idx, slot->v_scale);
    }
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
    if (dtype_code == KVSLOT_DTYPE_INT16) {
        return read_int16_v_value(elem_offset, logical_idx, slot->v_scale);
    }
#endif
    if (dtype_code == KVSLOT_DTYPE_FP16 || dtype_code == KVSLOT_DTYPE_BF16) {
        return u16_cache_bits_to_float(read_v_packed_u16_value(elem_offset, logical_idx), dtype_code);
    }

    uint64_t packed = 0;
    uint32_t word_idx = elem_offset + logical_idx;
    uint32_t pair_base = word_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&v_cache[pair_base], &packed, sizeof(packed));
    bits = (word_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}

static void read_fp16_v_pair(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx, float *value0, float *value1)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t pair_base = word_idx & ~1u;
    uint32_t packed;

    mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
    packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
    if ((logical_idx & 1u) == 0) {
        *value0 = fp16_bits_to_float((uint16_t)(packed & 0xffffu));
        *value1 = fp16_bits_to_float((uint16_t)(packed >> 16));
        return;
    }

    *value0 = fp16_bits_to_float((uint16_t)(packed >> 16));
    if (((word_idx + 1u) & ~1u) == pair_base) {
        uint32_t next_packed = ((word_idx + 1u) & 1u) == 0
            ? (uint32_t)(packed64 & 0xffffffffu)
            : (uint32_t)(packed64 >> 32);
        *value1 = fp16_bits_to_float((uint16_t)(next_packed & 0xffffu));
    } else {
        uint64_t next_packed64 = 0;
        mram_read(&v_cache[pair_base + 2u], &next_packed64, sizeof(next_packed64));
        *value1 = fp16_bits_to_float((uint16_t)(next_packed64 & 0xffffu));
    }
}

static void read_bf16_v_pair(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx, float *value0, float *value1)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t pair_base = word_idx & ~1u;
    uint32_t packed;

    mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
    packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
    if ((logical_idx & 1u) == 0) {
        *value0 = bf16_bits_to_float((uint16_t)(packed & 0xffffu));
        *value1 = bf16_bits_to_float((uint16_t)(packed >> 16));
        return;
    }

    *value0 = bf16_bits_to_float((uint16_t)(packed >> 16));
    if (((word_idx + 1u) & ~1u) == pair_base) {
        uint32_t next_packed = ((word_idx + 1u) & 1u) == 0
            ? (uint32_t)(packed64 & 0xffffffffu)
            : (uint32_t)(packed64 >> 32);
        *value1 = bf16_bits_to_float((uint16_t)(next_packed & 0xffffu));
    } else {
        uint64_t next_packed64 = 0;
        mram_read(&v_cache[pair_base + 2u], &next_packed64, sizeof(next_packed64));
        *value1 = bf16_bits_to_float((uint16_t)(next_packed64 & 0xffffu));
    }
}

#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
static void read_int16_v_pair(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx, float *value0, float *value1)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t pair_base = word_idx & ~1u;
    uint32_t packed;
    float scale = slot->v_scale;

    mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
    packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
    if ((logical_idx & 1u) == 0) {
        *value0 = ((float)(int16_t)(packed & 0xffffu)) * scale;
        *value1 = ((float)(int16_t)(packed >> 16)) * scale;
        return;
    }

    *value0 = ((float)(int16_t)(packed >> 16)) * scale;
    if (((word_idx + 1u) & ~1u) == pair_base) {
        uint32_t next_packed = ((word_idx + 1u) & 1u) == 0
            ? (uint32_t)(packed64 & 0xffffffffu)
            : (uint32_t)(packed64 >> 32);
        *value1 = ((float)(int16_t)(next_packed & 0xffffu)) * scale;
    } else {
        uint64_t next_packed64 = 0;
        mram_read(&v_cache[pair_base + 2u], &next_packed64, sizeof(next_packed64));
        *value1 = ((float)(int16_t)(next_packed64 & 0xffffu)) * scale;
    }
}
#endif

static void read_fp16_v_quad_aligned(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t logical_idx,
    float *value0,
    float *value1,
    float *value2,
    float *value3)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t low_word;
    uint32_t high_word;

    mram_read(&v_cache[word_idx], &packed64, sizeof(packed64));
    low_word = (uint32_t)(packed64 & 0xffffffffu);
    high_word = (uint32_t)(packed64 >> 32);
    *value0 = fp16_bits_to_float((uint16_t)(low_word & 0xffffu));
    *value1 = fp16_bits_to_float((uint16_t)(low_word >> 16));
    *value2 = fp16_bits_to_float((uint16_t)(high_word & 0xffffu));
    *value3 = fp16_bits_to_float((uint16_t)(high_word >> 16));
}

static void read_bf16_v_quad_aligned(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t logical_idx,
    float *value0,
    float *value1,
    float *value2,
    float *value3)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t low_word;
    uint32_t high_word;

    mram_read(&v_cache[word_idx], &packed64, sizeof(packed64));
    low_word = (uint32_t)(packed64 & 0xffffffffu);
    high_word = (uint32_t)(packed64 >> 32);
    *value0 = bf16_bits_to_float((uint16_t)(low_word & 0xffffu));
    *value1 = bf16_bits_to_float((uint16_t)(low_word >> 16));
    *value2 = bf16_bits_to_float((uint16_t)(high_word & 0xffffu));
    *value3 = bf16_bits_to_float((uint16_t)(high_word >> 16));
}

#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
static void read_int16_v_quad_raw_aligned(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t logical_idx,
    int16_t *value0,
    int16_t *value1,
    int16_t *value2,
    int16_t *value3)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t low_word;
    uint32_t high_word;

    mram_read(&v_cache[word_idx], &packed64, sizeof(packed64));
    low_word = (uint32_t)(packed64 & 0xffffffffu);
    high_word = (uint32_t)(packed64 >> 32);
    *value0 = (int16_t)(low_word & 0xffffu);
    *value1 = (int16_t)(low_word >> 16);
    *value2 = (int16_t)(high_word & 0xffffu);
    *value3 = (int16_t)(high_word >> 16);
}

static void read_int16_v_quad_aligned(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t logical_idx,
    float *value0,
    float *value1,
    float *value2,
    float *value3)
{
    uint64_t packed64 = 0;
    uint32_t elem_offset = slot->v_elem_offset == KVSLOT_UNSET_U32 ? slot->elem_offset : slot->v_elem_offset;
    uint32_t word_idx = elem_offset + (logical_idx / 2u);
    uint32_t low_word;
    uint32_t high_word;
    float scale = slot->v_scale;

    mram_read(&v_cache[word_idx], &packed64, sizeof(packed64));
    low_word = (uint32_t)(packed64 & 0xffffffffu);
    high_word = (uint32_t)(packed64 >> 32);
    *value0 = ((float)(int16_t)(low_word & 0xffffu)) * scale;
    *value1 = ((float)(int16_t)(low_word >> 16)) * scale;
    *value2 = ((float)(int16_t)(high_word & 0xffffu)) * scale;
    *value3 = ((float)(int16_t)(high_word >> 16)) * scale;
}
#endif

static float read_k_value(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx)
{
#ifndef KVSLOT_EXPERIMENTAL_INT8_I16_QK
    if (slot->dtype_code == KVSLOT_DTYPE_INT8) {
        return read_int8_k_value(slot->elem_offset, logical_idx, slot->k_scale);
    }
#endif
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
    if (slot->dtype_code == KVSLOT_DTYPE_INT16) {
        return read_int16_k_value(slot->elem_offset, logical_idx, slot->k_scale);
    }
#endif
    if (slot->dtype_code == KVSLOT_DTYPE_FP16 || slot->dtype_code == KVSLOT_DTYPE_BF16) {
        return u16_cache_bits_to_float(read_k_packed_u16_value(slot->elem_offset, logical_idx), slot->dtype_code);
    }

    uint64_t packed = 0;
    uint32_t word_idx = slot->elem_offset + logical_idx;
    uint32_t pair_base = word_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&k_cache[pair_base], &packed, sizeof(packed));
    bits = (word_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}

#ifdef KVSLOT_EXPERIMENTAL_INT8_I16_QK
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static float quantize_query_row_i16(uint32_t head_dim)
{
    float max_abs = 0.0f;
    float scale;

    for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        float value = qk_slot_query_row[dim_idx];
        float abs_value = value < 0.0f ? -value : value;
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    if (max_abs <= 0.0f) {
        for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            qk_slot_query_i16[dim_idx] = 0;
        }
        return 1.0f;
    }

    scale = max_abs / 32767.0f;
    for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        float scaled = qk_slot_query_row[dim_idx] / scale;
        int32_t rounded = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (rounded > 32767) {
            rounded = 32767;
        } else if (rounded < -32767) {
            rounded = -32767;
        }
        qk_slot_query_i16[dim_idx] = (int16_t)rounded;
    }
    return scale;
}
#endif

static float dot_query_i16_row_with_int8_k_row(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t key_row_base,
    uint32_t head_dim,
    const int16_t *query_row,
    float query_scale)
{
    int64_t acc = 0;
    uint32_t dim_idx = 0;

    if (((key_row_base & 3u) == 0) && (((slot->elem_offset + (key_row_base / 4u)) & 1u) == 0)) {
        uint32_t word_idx = slot->elem_offset + (key_row_base / 4u);
        for (; dim_idx + 8u <= head_dim; dim_idx += 8u, word_idx += 2u) {
            uint64_t packed64 = 0;
            mram_read(&k_cache[word_idx], &packed64, sizeof(packed64));
#pragma clang loop unroll(disable)
            for (uint32_t byte_idx = 0; byte_idx < 8u; ++byte_idx) {
                acc += (int32_t)query_row[dim_idx + byte_idx]
                    * (int32_t)(int8_t)((packed64 >> (byte_idx * 8u)) & 0xffu);
            }
        }
    }

    for (; dim_idx < head_dim; ++dim_idx) {
        uint32_t logical_idx = key_row_base + dim_idx;
        uint32_t pair_base = int8_cache_pair_base(slot->elem_offset, logical_idx);
        uint64_t packed64 = 0;
        mram_read(&k_cache[pair_base], &packed64, sizeof(packed64));
        acc += (int32_t)query_row[dim_idx] * (int32_t)unpack_int8_value(packed64, slot->elem_offset, logical_idx);
    }
    return ((float)acc) * query_scale * slot->k_scale;
}

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static float dot_query_i16_with_int8_k_row(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t key_row_base,
    uint32_t head_dim)
{
    return dot_query_i16_row_with_int8_k_row(
        slot,
        key_row_base,
        head_dim,
        qk_slot_query_i16,
        qk_slot_query_i16_scale);
}
#endif
#endif

static float dot_query_with_k_row(const kvslot_runtime_slot_args_t *slot, uint32_t key_row_base, uint32_t head_dim)
{
    float local_sum = 0.0f;
    uint32_t dim_idx = 0;

#if defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK) && !defined(KVSLOT_SLIM_QK_SLOT_ONLY)
    if (slot->dtype_code == KVSLOT_DTYPE_INT8) {
        return dot_query_i16_with_int8_k_row(slot, key_row_base, head_dim);
    } else
#endif
    if (slot->dtype_code == KVSLOT_DTYPE_FP16 || slot->dtype_code == KVSLOT_DTYPE_BF16) {
        uint32_t word_idx = slot->elem_offset + (key_row_base / 2u);
        if ((key_row_base & 1u) == 0 && (word_idx & 1u) == 0) {
            __dma_aligned uint32_t packed_words[4];
            for (; dim_idx + 8u <= head_dim; dim_idx += 8u, word_idx += 4u) {
                mram_read(&k_cache[word_idx], packed_words, sizeof(packed_words));
#pragma clang loop unroll(disable)
                for (uint32_t packed_idx = 0; packed_idx < 4u; ++packed_idx) {
                    uint32_t packed = packed_words[packed_idx];
                    uint32_t packed_dim_idx = dim_idx + (packed_idx * 2u);
                    local_sum += qk_slot_query_row[packed_dim_idx]
                        * u16_cache_bits_to_float((uint16_t)(packed & 0xffffu), slot->dtype_code);
                    local_sum += qk_slot_query_row[packed_dim_idx + 1u]
                        * u16_cache_bits_to_float((uint16_t)(packed >> 16), slot->dtype_code);
                }
            }
            for (; dim_idx + 4u <= head_dim; dim_idx += 4u, word_idx += 2u) {
                uint64_t packed64 = 0;
                uint32_t low_word;
                uint32_t high_word;
                mram_read(&k_cache[word_idx], &packed64, sizeof(packed64));
                low_word = (uint32_t)(packed64 & 0xffffffffu);
                high_word = (uint32_t)(packed64 >> 32);
                local_sum += qk_slot_query_row[dim_idx]
                    * u16_cache_bits_to_float((uint16_t)(low_word & 0xffffu), slot->dtype_code);
                local_sum += qk_slot_query_row[dim_idx + 1u]
                    * u16_cache_bits_to_float((uint16_t)(low_word >> 16), slot->dtype_code);
                local_sum += qk_slot_query_row[dim_idx + 2u]
                    * u16_cache_bits_to_float((uint16_t)(high_word & 0xffffu), slot->dtype_code);
                local_sum += qk_slot_query_row[dim_idx + 3u]
                    * u16_cache_bits_to_float((uint16_t)(high_word >> 16), slot->dtype_code);
            }
        }
    } else if (slot->dtype_code == KVSLOT_DTYPE_FP32) {
        uint32_t word_idx = slot->elem_offset + key_row_base;
        if ((word_idx & 1u) == 0) {
            for (; dim_idx + 2u <= head_dim; dim_idx += 2u, word_idx += 2u) {
                uint64_t packed64 = 0;
                mram_read(&k_cache[word_idx], &packed64, sizeof(packed64));
                local_sum += qk_slot_query_row[dim_idx] * u32_bits_to_float((uint32_t)(packed64 & 0xffffffffu));
                local_sum += qk_slot_query_row[dim_idx + 1u] * u32_bits_to_float((uint32_t)(packed64 >> 32));
            }
        }
    }

    for (; dim_idx < head_dim; ++dim_idx) {
        local_sum += qk_slot_query_row[dim_idx] * read_k_value(slot, key_row_base + dim_idx);
    }
    return local_sum;
}

#ifdef KVSLOT_SLIM_QK_SLOT_ONLY
static float dot_query_u16_with_query_row(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t key_row_base,
    uint32_t head_dim,
    const float *query_row)
{
    float local_sum = 0.0f;
    uint32_t dim_idx = 0;
    uint32_t word_idx = slot->elem_offset + (key_row_base / 2u);

#ifdef KVSLOT_BULK_FP16_DOT
    if ((key_row_base & 1u) == 0
        && (word_idx & 1u) == 0
        && (head_dim & 3u) == 0
        && head_dim <= KVSLOT_SLIM_QK_CACHE_HEAD_DIM) {
        __dma_aligned uint32_t packed_words[KVSLOT_SLIM_QK_CACHE_HEAD_DIM / 2u];
        uint32_t packed_word_count = (head_dim + 1u) / 2u;
        mram_read(&k_cache[word_idx], packed_words, packed_word_count * sizeof(packed_words[0]));
        for (; dim_idx + 8u <= head_dim; dim_idx += 8u) {
            uint32_t packed0 = packed_words[(dim_idx / 2u)];
            uint32_t packed1 = packed_words[(dim_idx / 2u) + 1u];
            uint32_t packed2 = packed_words[(dim_idx / 2u) + 2u];
            uint32_t packed3 = packed_words[(dim_idx / 2u) + 3u];
            local_sum += query_row[dim_idx] * u16_cache_bits_to_float((uint16_t)(packed0 & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 1u] * u16_cache_bits_to_float((uint16_t)(packed0 >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 2u] * u16_cache_bits_to_float((uint16_t)(packed1 & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 3u] * u16_cache_bits_to_float((uint16_t)(packed1 >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 4u] * u16_cache_bits_to_float((uint16_t)(packed2 & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 5u] * u16_cache_bits_to_float((uint16_t)(packed2 >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 6u] * u16_cache_bits_to_float((uint16_t)(packed3 & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 7u] * u16_cache_bits_to_float((uint16_t)(packed3 >> 16), slot->dtype_code);
        }
        for (; dim_idx < head_dim; ++dim_idx) {
            uint32_t packed = packed_words[dim_idx / 2u];
            uint16_t fp16_bits = (dim_idx & 1u) == 0
                ? (uint16_t)(packed & 0xffffu)
                : (uint16_t)(packed >> 16);
            local_sum += query_row[dim_idx] * u16_cache_bits_to_float(fp16_bits, slot->dtype_code);
        }
        return local_sum;
    }
#endif

    if ((key_row_base & 1u) == 0 && (word_idx & 1u) == 0) {
        __dma_aligned uint32_t packed_words[4];
        for (; dim_idx + 8u <= head_dim; dim_idx += 8u, word_idx += 4u) {
            mram_read(&k_cache[word_idx], packed_words, sizeof(packed_words));
            local_sum += query_row[dim_idx] * u16_cache_bits_to_float((uint16_t)(packed_words[0] & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 1u] * u16_cache_bits_to_float((uint16_t)(packed_words[0] >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 2u] * u16_cache_bits_to_float((uint16_t)(packed_words[1] & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 3u] * u16_cache_bits_to_float((uint16_t)(packed_words[1] >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 4u] * u16_cache_bits_to_float((uint16_t)(packed_words[2] & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 5u] * u16_cache_bits_to_float((uint16_t)(packed_words[2] >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 6u] * u16_cache_bits_to_float((uint16_t)(packed_words[3] & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 7u] * u16_cache_bits_to_float((uint16_t)(packed_words[3] >> 16), slot->dtype_code);
        }
        for (; dim_idx + 4u <= head_dim; dim_idx += 4u, word_idx += 2u) {
            uint64_t packed64 = 0;
            uint32_t low_word;
            uint32_t high_word;
            mram_read(&k_cache[word_idx], &packed64, sizeof(packed64));
            low_word = (uint32_t)(packed64 & 0xffffffffu);
            high_word = (uint32_t)(packed64 >> 32);
            local_sum += query_row[dim_idx] * u16_cache_bits_to_float((uint16_t)(low_word & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 1u] * u16_cache_bits_to_float((uint16_t)(low_word >> 16), slot->dtype_code);
            local_sum += query_row[dim_idx + 2u] * u16_cache_bits_to_float((uint16_t)(high_word & 0xffffu), slot->dtype_code);
            local_sum += query_row[dim_idx + 3u] * u16_cache_bits_to_float((uint16_t)(high_word >> 16), slot->dtype_code);
        }
    }

    for (; dim_idx < head_dim; ++dim_idx) {
        local_sum += query_row[dim_idx] * read_k_value(slot, key_row_base + dim_idx);
    }
    return local_sum;
}

#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
static float dot_query_int16_with_query_row(
    const kvslot_runtime_slot_args_t *slot,
    uint32_t key_row_base,
    uint32_t head_dim,
    const int16_t *query_row,
    float query_scale)
{
    int64_t acc = 0;
    uint32_t dim_idx = 0;
    uint32_t word_idx = slot->elem_offset + (key_row_base / 2u);

#ifdef KVSLOT_BULK_FP16_DOT
    if ((key_row_base & 1u) == 0
        && (word_idx & 1u) == 0
        && (head_dim & 3u) == 0
        && head_dim <= KVSLOT_SLIM_QK_CACHE_HEAD_DIM) {
        __dma_aligned uint32_t packed_words[KVSLOT_SLIM_QK_CACHE_HEAD_DIM / 2u];
        uint32_t packed_word_count = (head_dim + 1u) / 2u;
        mram_read(&k_cache[word_idx], packed_words, packed_word_count * sizeof(packed_words[0]));
        for (; dim_idx + 8u <= head_dim; dim_idx += 8u) {
            uint32_t packed0 = packed_words[(dim_idx / 2u)];
            uint32_t packed1 = packed_words[(dim_idx / 2u) + 1u];
            uint32_t packed2 = packed_words[(dim_idx / 2u) + 2u];
            uint32_t packed3 = packed_words[(dim_idx / 2u) + 3u];
            acc += (int32_t)query_row[dim_idx] * (int32_t)(int16_t)(packed0 & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 1u] * (int32_t)(int16_t)(packed0 >> 16);
            acc += (int32_t)query_row[dim_idx + 2u] * (int32_t)(int16_t)(packed1 & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 3u] * (int32_t)(int16_t)(packed1 >> 16);
            acc += (int32_t)query_row[dim_idx + 4u] * (int32_t)(int16_t)(packed2 & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 5u] * (int32_t)(int16_t)(packed2 >> 16);
            acc += (int32_t)query_row[dim_idx + 6u] * (int32_t)(int16_t)(packed3 & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 7u] * (int32_t)(int16_t)(packed3 >> 16);
        }
        for (; dim_idx < head_dim; ++dim_idx) {
            uint32_t packed = packed_words[dim_idx / 2u];
            int16_t k_value = (dim_idx & 1u) == 0 ? (int16_t)(packed & 0xffffu) : (int16_t)(packed >> 16);
            acc += (int32_t)query_row[dim_idx] * (int32_t)k_value;
        }
        return ((float)acc) * query_scale * slot->k_scale;
    }
#endif

    if ((key_row_base & 1u) == 0 && (word_idx & 1u) == 0) {
        __dma_aligned uint32_t packed_words[4];
        for (; dim_idx + 8u <= head_dim; dim_idx += 8u, word_idx += 4u) {
            mram_read(&k_cache[word_idx], packed_words, sizeof(packed_words));
            acc += (int32_t)query_row[dim_idx] * (int32_t)(int16_t)(packed_words[0] & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 1u] * (int32_t)(int16_t)(packed_words[0] >> 16);
            acc += (int32_t)query_row[dim_idx + 2u] * (int32_t)(int16_t)(packed_words[1] & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 3u] * (int32_t)(int16_t)(packed_words[1] >> 16);
            acc += (int32_t)query_row[dim_idx + 4u] * (int32_t)(int16_t)(packed_words[2] & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 5u] * (int32_t)(int16_t)(packed_words[2] >> 16);
            acc += (int32_t)query_row[dim_idx + 6u] * (int32_t)(int16_t)(packed_words[3] & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 7u] * (int32_t)(int16_t)(packed_words[3] >> 16);
        }
        for (; dim_idx + 4u <= head_dim; dim_idx += 4u, word_idx += 2u) {
            uint64_t packed64 = 0;
            uint32_t low_word;
            uint32_t high_word;
            mram_read(&k_cache[word_idx], &packed64, sizeof(packed64));
            low_word = (uint32_t)(packed64 & 0xffffffffu);
            high_word = (uint32_t)(packed64 >> 32);
            acc += (int32_t)query_row[dim_idx] * (int32_t)(int16_t)(low_word & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 1u] * (int32_t)(int16_t)(low_word >> 16);
            acc += (int32_t)query_row[dim_idx + 2u] * (int32_t)(int16_t)(high_word & 0xffffu);
            acc += (int32_t)query_row[dim_idx + 3u] * (int32_t)(int16_t)(high_word >> 16);
        }
    }

    for (; dim_idx < head_dim; ++dim_idx) {
        acc += (int32_t)query_row[dim_idx]
            * (int32_t)(int16_t)read_k_packed_u16_value(slot->elem_offset, key_row_base + dim_idx);
    }
    return ((float)acc) * query_scale * slot->k_scale;
}
#endif

static int run_qk_slot_slim_all_head_dot(
    uint32_t tasklet_id,
    uint32_t seq_len,
    uint32_t group_heads,
    uint32_t slot_head_dim,
    uint32_t num_heads,
    uint32_t window,
    uint32_t head_dim,
    uint32_t mode)
{
    uint32_t query_pairs_per_head;
    uint32_t total_query_pairs;
    uint32_t total_scores;

    if (runtime_slot_args.dtype_code != KVSLOT_DTYPE_FP16
        && runtime_slot_args.dtype_code != KVSLOT_DTYPE_BF16
#ifdef KVSLOT_EXPERIMENTAL_INT8_I16_QK
        && runtime_slot_args.dtype_code != KVSLOT_DTYPE_INT8
#endif
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
        && runtime_slot_args.dtype_code != KVSLOT_DTYPE_INT16
#endif
    ) {
        return 0;
    }
    if (mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED
        && mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED) {
        return 0;
    }
    if (num_heads == 0 || window == 0 || head_dim == 0 || (head_dim & 1u) != 0) {
        return 0;
    }
    if (num_heads > KVSLOT_MAX_HEADS || head_dim > KVSLOT_MAX_HEAD_DIM || window > KVSLOT_MAX_CAPACITY) {
        return 0;
    }
    if (num_heads > KVSLOT_SLIM_QK_CACHE_HEADS || head_dim > KVSLOT_SLIM_QK_CACHE_HEAD_DIM) {
        return 0;
    }

    query_pairs_per_head = (head_dim + 1u) / 2u;
    total_query_pairs = num_heads * query_pairs_per_head;
    for (uint32_t pair_idx = tasklet_id; pair_idx < total_query_pairs; pair_idx += NR_TASKLETS) {
        uint32_t head_row = pair_idx / query_pairs_per_head;
        uint32_t pair_in_head = pair_idx - (head_row * query_pairs_per_head);
        uint32_t dim_idx0 = pair_in_head * 2u;
        uint32_t dim_idx1 = dim_idx0 + 1u;
        uint32_t query_row_base = head_row * head_dim;
        uint32_t cache_row_base = head_row * KVSLOT_SLIM_QK_CACHE_HEAD_DIM;
        uint64_t packed_q = 0;
        mram_read(&qk_query[query_row_base + dim_idx0], &packed_q, sizeof(packed_q));
        qk_slot_query_rows[cache_row_base + dim_idx0] = u32_bits_to_float((uint32_t)(packed_q & 0xffffffffu));
        if (dim_idx1 < head_dim) {
            qk_slot_query_rows[cache_row_base + dim_idx1] = u32_bits_to_float((uint32_t)(packed_q >> 32));
        }
    }
    barrier_wait(&kvslot_barrier);

#if defined(KVSLOT_EXPERIMENTAL_INT16_KV) || defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK)
    if (0
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
        || runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT16
#endif
#ifdef KVSLOT_EXPERIMENTAL_INT8_I16_QK
        || runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT8
#endif
    ) {
        for (uint32_t head_row = tasklet_id; head_row < num_heads; head_row += NR_TASKLETS) {
            uint32_t cache_row_base = head_row * KVSLOT_SLIM_QK_CACHE_HEAD_DIM;
            float max_abs = 0.0f;
            float scale;
            for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                float value = qk_slot_query_rows[cache_row_base + dim_idx];
                float abs_value = value < 0.0f ? -value : value;
                if (abs_value > max_abs) {
                    max_abs = abs_value;
                }
            }
            if (max_abs <= 0.0f) {
                scale = 1.0f;
                for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    qk_slot_query_i16_rows[cache_row_base + dim_idx] = 0;
                }
            } else {
                scale = max_abs / 32767.0f;
                for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    float scaled = qk_slot_query_rows[cache_row_base + dim_idx] / scale;
                    int32_t rounded = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                    if (rounded > 32767) {
                        rounded = 32767;
                    } else if (rounded < -32767) {
                        rounded = -32767;
                    }
                    qk_slot_query_i16_rows[cache_row_base + dim_idx] = (int16_t)rounded;
                }
            }
            qk_slot_query_i16_scales[head_row] = scale;
        }
        barrier_wait(&kvslot_barrier);
    }
#endif

    total_scores = num_heads * window;
    for (uint32_t score_idx = tasklet_id; score_idx < total_scores; score_idx += NR_TASKLETS) {
        uint32_t head_row = score_idx / window;
        uint32_t token_offset = score_idx - (head_row * window);
        uint32_t local_head_idx = qk_slot_head_indices[head_row];
        float local_sum = 0.0f;
        if (local_head_idx < group_heads) {
            uint32_t token_idx = seq_len - window + token_offset;
            uint32_t key_row_base = (((token_idx * group_heads) + local_head_idx) * slot_head_dim);
            uint32_t query_cache_base = head_row * KVSLOT_SLIM_QK_CACHE_HEAD_DIM;
#ifdef KVSLOT_EXPERIMENTAL_INT8_I16_QK
            if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT8) {
                local_sum = dot_query_i16_row_with_int8_k_row(
                    &runtime_slot_args,
                    key_row_base,
                    head_dim,
                    &qk_slot_query_i16_rows[query_cache_base],
                    qk_slot_query_i16_scales[head_row]);
            } else
#endif
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
            if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT16) {
                local_sum = dot_query_int16_with_query_row(
                    &runtime_slot_args,
                    key_row_base,
                    head_dim,
                    &qk_slot_query_i16_rows[query_cache_base],
                    qk_slot_query_i16_scales[head_row]);
            } else {
#endif
                local_sum = dot_query_u16_with_query_row(
                    &runtime_slot_args,
                    key_row_base,
                    head_dim,
                    &qk_slot_query_rows[query_cache_base]);
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
            }
#endif
        }
        qk_slot_score_local[(size_t)head_row * window + token_offset] = float_to_u32_bits(local_sum);
    }
    barrier_wait(&kvslot_barrier);
    return 1;
}
#endif

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static void run_qk_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t head_dim = qk_args.head_dim;
    uint32_t num_keys = qk_args.num_keys;
    uint32_t key_stride = qk_args.key_stride;

    if (head_dim > KVSLOT_MAX_HEAD_DIM) {
        head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (num_keys > KVSLOT_MAX_CAPACITY) {
        num_keys = KVSLOT_MAX_CAPACITY;
    }
    head_dim = (head_dim / 2) * 2;
    key_stride = (key_stride / 2) * 2;

    for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx) {
        int64_t local_sum = 0;
        for (uint32_t pair_idx = tasklet_id; pair_idx < head_dim / 2; pair_idx += NR_TASKLETS) {
            uint32_t elem_idx = pair_idx * 2;
            uint64_t packed_q;
            uint64_t packed_k;
            mram_read(&qk_query[elem_idx], &packed_q, sizeof(packed_q));
            mram_read(&qk_keys[key_idx * key_stride + elem_idx], &packed_k, sizeof(packed_k));

            int32_t q0 = (int32_t)(packed_q & 0xffffffffu);
            int32_t q1 = (int32_t)(packed_q >> 32);
            int32_t k0 = (int32_t)(packed_k & 0xffffffffu);
            int32_t k1 = (int32_t)(packed_k >> 32);

            local_sum += (int64_t)q0 * (int64_t)k0;
            local_sum += (int64_t)q1 * (int64_t)k1;
        }
        partial_sums[tasklet_id] = local_sum;
        barrier_wait(&kvslot_barrier);

        if (tasklet_id == 0) {
            int64_t total = 0;
            for (uint32_t idx = 0; idx < NR_TASKLETS; ++idx) {
                total += partial_sums[idx];
            }
            qk_scores[key_idx] = total;
        }
        barrier_wait(&kvslot_barrier);
    }
}
#endif

static void run_context_from_local_softmax(uint32_t num_heads, uint32_t window, uint32_t group_heads, uint32_t head_dim)
{
    uint32_t tasklet_id = me();
    uint32_t v_dtype_code = runtime_slot_args.v_dtype_code == KVSLOT_UNSET_U32 ? runtime_slot_args.dtype_code : runtime_slot_args.v_dtype_code;
    uint32_t total_outputs = num_heads * head_dim;
    uint32_t total_pairs = (total_outputs + 1u) / 2u;
    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32
        ? runtime_slot_args.elem_offset
        : runtime_slot_args.v_elem_offset;

#ifdef KVSLOT_CONTEXT_BULK_ROW
    if (run_context_fp16_bulk_row(num_heads, window, group_heads, head_dim)) {
        return;
    }
#endif

#ifdef KVSLOT_CONTEXT_TILE16
    if (run_context_tile16(num_heads, window, group_heads, head_dim)) {
        return;
    }
#endif

#ifdef KVSLOT_CONTEXT_OCTETV
    if (v_dtype_code == KVSLOT_DTYPE_FP16
        && (head_dim % 8u) == 0
        && (total_outputs % 8u) == 0
        && (value_elem_offset & 1u) == 0) {
        uint32_t total_octets = total_outputs / 8u;
        uint32_t value_word_stride = (group_heads * head_dim) / 2u;
        for (uint32_t octet_idx = tasklet_id; octet_idx < total_octets; octet_idx += NR_TASKLETS) {
            uint32_t out_idx0 = octet_idx * 8u;
            uint32_t head_idx = out_idx0 / head_dim;
            uint32_t dim_idx0 = out_idx0 % head_dim;
            uint32_t kv_head_idx = qk_slot_head_indices[head_idx];
            uint32_t score_idx = head_idx * window;
            uint32_t value_word_idx;
            float acc[8];

            if (kv_head_idx >= group_heads) {
                continue;
            }
            for (uint32_t acc_idx = 0; acc_idx < 8u; ++acc_idx) {
                acc[acc_idx] = 0.0f;
            }
            value_word_idx = value_elem_offset + (((kv_head_idx * head_dim) + dim_idx0) / 2u);
            for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                __dma_aligned uint32_t packed_words[4];
                float weight = u32_bits_to_float(qk_slot_score_local[score_idx + token_idx]);
                mram_read(&v_cache[value_word_idx], packed_words, sizeof(packed_words));
                value_word_idx += value_word_stride;
                for (uint32_t packed_idx = 0; packed_idx < 4u; ++packed_idx) {
                    uint32_t packed = packed_words[packed_idx];
                    uint32_t acc_idx = packed_idx * 2u;
                    acc[acc_idx] += weight * fp16_bits_to_float((uint16_t)(packed & 0xffffu));
                    acc[acc_idx + 1u] += weight * fp16_bits_to_float((uint16_t)(packed >> 16));
                }
            }
            write_av_context_quad(
                out_idx0,
                float_to_u32_bits(acc[0]),
                float_to_u32_bits(acc[1]),
                float_to_u32_bits(acc[2]),
                float_to_u32_bits(acc[3]));
            write_av_context_quad(
                out_idx0 + 4u,
                float_to_u32_bits(acc[4]),
                float_to_u32_bits(acc[5]),
                float_to_u32_bits(acc[6]),
                float_to_u32_bits(acc[7]));
        }
        return;
    }
#endif

#ifndef KVSLOT_CONTEXT_OCTETV
    if ((v_dtype_code == KVSLOT_DTYPE_FP16
            || v_dtype_code == KVSLOT_DTYPE_BF16
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
            || v_dtype_code == KVSLOT_DTYPE_INT16
#endif
        )
        && (head_dim % 4u) == 0
        && (total_outputs % 4u) == 0
        && (value_elem_offset & 1u) == 0) {
        uint32_t total_quads = total_outputs / 4u;
        uint32_t value_word_stride = (group_heads * head_dim) / 2u;
        for (uint32_t quad_idx = tasklet_id; quad_idx < total_quads; quad_idx += NR_TASKLETS) {
            uint32_t out_idx0 = quad_idx * 4u;
            uint32_t head_idx = out_idx0 / head_dim;
            uint32_t dim_idx0 = out_idx0 % head_dim;
            uint32_t kv_head_idx = qk_slot_head_indices[head_idx];
            uint32_t score_idx = head_idx * window;
            uint32_t value_word_idx;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            if (kv_head_idx >= group_heads) {
                continue;
            }
            value_word_idx = value_elem_offset + (((kv_head_idx * head_dim) + dim_idx0) / 2u);
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
            if (v_dtype_code == KVSLOT_DTYPE_INT16) {
                float value_scale = runtime_slot_args.v_scale;
                for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                    uint64_t packed64 = 0;
                    uint32_t low_word;
                    uint32_t high_word;
                    float weight = u32_bits_to_float(qk_slot_score_local[score_idx + token_idx]);
                    mram_read(&v_cache[value_word_idx], &packed64, sizeof(packed64));
                    value_word_idx += value_word_stride;
                    low_word = (uint32_t)(packed64 & 0xffffffffu);
                    high_word = (uint32_t)(packed64 >> 32);
                    acc0 += weight * (float)(int16_t)(low_word & 0xffffu);
                    acc1 += weight * (float)(int16_t)(low_word >> 16);
                    acc2 += weight * (float)(int16_t)(high_word & 0xffffu);
                    acc3 += weight * (float)(int16_t)(high_word >> 16);
                }
                acc0 *= value_scale;
                acc1 *= value_scale;
                acc2 *= value_scale;
                acc3 *= value_scale;
            } else
#endif
            if (v_dtype_code == KVSLOT_DTYPE_BF16) {
                for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                    uint64_t packed64 = 0;
                    uint32_t low_word;
                    uint32_t high_word;
                    float weight = u32_bits_to_float(qk_slot_score_local[score_idx + token_idx]);
                    float value0;
                    float value1;
                    float value2;
                    float value3;
                    mram_read(&v_cache[value_word_idx], &packed64, sizeof(packed64));
                    value_word_idx += value_word_stride;
                    low_word = (uint32_t)(packed64 & 0xffffffffu);
                    high_word = (uint32_t)(packed64 >> 32);
                    value0 = bf16_bits_to_float((uint16_t)(low_word & 0xffffu));
                    value1 = bf16_bits_to_float((uint16_t)(low_word >> 16));
                    value2 = bf16_bits_to_float((uint16_t)(high_word & 0xffffu));
                    value3 = bf16_bits_to_float((uint16_t)(high_word >> 16));
                    acc0 += weight * value0;
                    acc1 += weight * value1;
                    acc2 += weight * value2;
                    acc3 += weight * value3;
                }
            } else {
                for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                    uint64_t packed64 = 0;
                    uint32_t low_word;
                    uint32_t high_word;
                    float weight = u32_bits_to_float(qk_slot_score_local[score_idx + token_idx]);
                    float value0;
                    float value1;
                    float value2;
                    float value3;
                    mram_read(&v_cache[value_word_idx], &packed64, sizeof(packed64));
                    value_word_idx += value_word_stride;
                    low_word = (uint32_t)(packed64 & 0xffffffffu);
                    high_word = (uint32_t)(packed64 >> 32);
                    value0 = fp16_bits_to_float((uint16_t)(low_word & 0xffffu));
                    value1 = fp16_bits_to_float((uint16_t)(low_word >> 16));
                    value2 = fp16_bits_to_float((uint16_t)(high_word & 0xffffu));
                    value3 = fp16_bits_to_float((uint16_t)(high_word >> 16));
                    acc0 += weight * value0;
                    acc1 += weight * value1;
                    acc2 += weight * value2;
                    acc3 += weight * value3;
                }
            }
            write_av_context_quad(
                out_idx0,
                float_to_u32_bits(acc0),
                float_to_u32_bits(acc1),
                float_to_u32_bits(acc2),
                float_to_u32_bits(acc3));
        }
        return;
    }
#endif

    for (uint32_t pair_idx = tasklet_id; pair_idx < total_pairs; pair_idx += NR_TASKLETS) {
        uint32_t out_idx0 = pair_idx * 2u;
        uint32_t out_idx1 = out_idx0 + 1u;
        uint32_t head_idx0 = out_idx0 / head_dim;
        uint32_t dim_idx0 = out_idx0 % head_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        uint32_t head_idx1 = 0;
        uint32_t dim_idx1 = 0;
        int has_second = out_idx1 < total_outputs;
        int same_head_pair = 0;
        uint32_t kv_head_idx0 = 0;
        uint32_t kv_head_idx1 = 0;

        if (has_second) {
            head_idx1 = out_idx1 / head_dim;
            dim_idx1 = out_idx1 % head_dim;
            same_head_pair = head_idx1 == head_idx0;
        }
        kv_head_idx0 = qk_slot_head_indices[head_idx0];
        if (kv_head_idx0 >= group_heads) {
            continue;
        }
        if (has_second) {
            kv_head_idx1 = qk_slot_head_indices[head_idx1];
            if (kv_head_idx1 >= group_heads) {
                has_second = 0;
                same_head_pair = 0;
            }
        }

        if (same_head_pair) {
            if (v_dtype_code == KVSLOT_DTYPE_INT8) {
                uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32
                    ? runtime_slot_args.elem_offset
                    : runtime_slot_args.v_elem_offset;
#ifdef KVSLOT_CONTEXT_INT8_I16_WEIGHTS
                int64_t acc0_i64 = 0;
                int64_t acc1_i64 = 0;
                for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                    uint32_t value_idx0 = ((token_idx * group_heads) + kv_head_idx0) * head_dim + dim_idx0;
                    int32_t weight0_i32 = quantize_context_weight_i16(
                        u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]));
                    uint32_t pair_base0 = int8_cache_pair_base(value_elem_offset, value_idx0);
                    uint32_t pair_base1 = int8_cache_pair_base(value_elem_offset, value_idx0 + 1u);
                    uint64_t packed_v0 = 0;
                    uint64_t packed_v1 = 0;
                    mram_read(&v_cache[pair_base0], &packed_v0, sizeof(packed_v0));
                    if (pair_base1 == pair_base0) {
                        packed_v1 = packed_v0;
                    } else {
                        mram_read(&v_cache[pair_base1], &packed_v1, sizeof(packed_v1));
                    }
                    acc0_i64 += (int64_t)weight0_i32 * (int64_t)unpack_int8_value(packed_v0, value_elem_offset, value_idx0);
                    acc1_i64 += (int64_t)weight0_i32 * (int64_t)unpack_int8_value(packed_v1, value_elem_offset, value_idx0 + 1u);
                }
                {
                    float output_scale = runtime_slot_args.v_scale * (1.0f / 32767.0f);
                    acc0 = ((float)acc0_i64) * output_scale;
                    acc1 = ((float)acc1_i64) * output_scale;
                }
#else
                for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                    uint32_t value_idx0 = ((token_idx * group_heads) + kv_head_idx0) * head_dim + dim_idx0;
                    float weight0 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]);
                    uint32_t pair_base0 = int8_cache_pair_base(value_elem_offset, value_idx0);
                    uint32_t pair_base1 = int8_cache_pair_base(value_elem_offset, value_idx0 + 1u);
                    uint64_t packed_v0 = 0;
                    uint64_t packed_v1 = 0;
                    mram_read(&v_cache[pair_base0], &packed_v0, sizeof(packed_v0));
                    if (pair_base1 == pair_base0) {
                        packed_v1 = packed_v0;
                    } else {
                        mram_read(&v_cache[pair_base1], &packed_v1, sizeof(packed_v1));
                    }
                    acc0 += weight0 * (float)unpack_int8_value(packed_v0, value_elem_offset, value_idx0);
                    acc1 += weight0 * (float)unpack_int8_value(packed_v1, value_elem_offset, value_idx0 + 1u);
                }
                acc0 *= runtime_slot_args.v_scale;
                acc1 *= runtime_slot_args.v_scale;
#endif
            } else
            for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                uint32_t value_idx0 = ((token_idx * group_heads) + kv_head_idx0) * head_dim + dim_idx0;
                float weight0 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]);
                if (v_dtype_code == KVSLOT_DTYPE_FP16
                    || v_dtype_code == KVSLOT_DTYPE_BF16
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
                    || v_dtype_code == KVSLOT_DTYPE_INT16
#endif
                ) {
                    float value0;
                    float value1;
                    if (v_dtype_code == KVSLOT_DTYPE_BF16) {
                        read_bf16_v_pair(&runtime_slot_args, value_idx0, &value0, &value1);
#ifdef KVSLOT_EXPERIMENTAL_INT16_KV
                    } else if (v_dtype_code == KVSLOT_DTYPE_INT16) {
                        read_int16_v_pair(&runtime_slot_args, value_idx0, &value0, &value1);
#endif
                    } else {
                        read_fp16_v_pair(&runtime_slot_args, value_idx0, &value0, &value1);
                    }
                    acc0 += weight0 * value0;
                    acc1 += weight0 * value1;
                } else if (v_dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                    uint64_t packed_v = 0;
                    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32 ? runtime_slot_args.elem_offset : runtime_slot_args.v_elem_offset;
                    uint32_t word_idx0 = value_elem_offset + value_idx0;
                    mram_read(&v_cache[word_idx0], &packed_v, sizeof(packed_v));
                    acc0 += weight0 * u32_bits_to_float((uint32_t)(packed_v & 0xffffffffu));
                    acc1 += weight0 * u32_bits_to_float((uint32_t)(packed_v >> 32));
                } else {
                    float value0 = read_v_value(&runtime_slot_args, value_idx0);
                    float value1 = read_v_value(&runtime_slot_args, value_idx0 + 1u);
                    acc0 += weight0 * value0;
                    acc1 += weight0 * value1;
                }
            }
        } else {
            for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                uint32_t value_idx0 = ((token_idx * group_heads) + kv_head_idx0) * head_dim + dim_idx0;
                float weight0 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]);
                float value0 = read_v_value(&runtime_slot_args, value_idx0);
                acc0 += weight0 * value0;
                if (has_second) {
                    uint32_t value_idx1 = ((token_idx * group_heads) + kv_head_idx1) * head_dim + dim_idx1;
                    float weight1 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx1 * window + token_idx]);
                    float value1 = read_v_value(&runtime_slot_args, value_idx1);
                    acc1 += weight1 * value1;
                }
            }
        }
        write_av_context_pair(pair_idx, float_to_u32_bits(acc0), float_to_u32_bits(acc1));
    }
}

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static void run_av_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t seq_len = runtime_slot_args.seq_len;
    uint32_t group_heads = runtime_slot_args.group_heads;
    uint32_t head_dim = runtime_slot_args.head_dim;
    uint32_t v_dtype_code = runtime_slot_args.v_dtype_code == KVSLOT_UNSET_U32 ? runtime_slot_args.dtype_code : runtime_slot_args.v_dtype_code;
    uint32_t total_outputs = group_heads * head_dim;

    if (seq_len > KVSLOT_MAX_CAPACITY) {
        seq_len = KVSLOT_MAX_CAPACITY;
    }
    if (group_heads > KVSLOT_MAX_HEADS) {
        group_heads = KVSLOT_MAX_HEADS;
    }
    if (head_dim > KVSLOT_MAX_HEAD_DIM) {
        head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    total_outputs = group_heads * head_dim;

    uint32_t total_pairs = (total_outputs + 1u) / 2u;

    for (uint32_t pair_idx = tasklet_id; pair_idx < total_pairs; pair_idx += NR_TASKLETS) {
        uint32_t out_idx0 = pair_idx * 2u;
        uint32_t out_idx1 = out_idx0 + 1u;
        uint32_t head_idx0 = out_idx0 / head_dim;
        uint32_t dim_idx0 = out_idx0 % head_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        uint32_t head_idx1 = 0;
        uint32_t dim_idx1 = 0;
        int has_second = out_idx1 < total_outputs;
        int same_head_pair = 0;

        if (has_second) {
            head_idx1 = out_idx1 / head_dim;
            dim_idx1 = out_idx1 % head_dim;
            same_head_pair = head_idx1 == head_idx0;
        }
        if (same_head_pair) {
            uint32_t token_idx = 0;
            uint32_t weight_row_base = head_idx0 * seq_len;
            if (v_dtype_code == KVSLOT_DTYPE_INT8) {
                for (; token_idx + 8u <= seq_len; token_idx += 8u) {
                    uint32_t logical_weight_start = weight_row_base + token_idx;
                    uint32_t weight_pair_base = logical_weight_start & ~1u;
                    uint32_t packed_weight_pairs = (logical_weight_start & 1u) == 0 ? 4u : 5u;
                    __dma_aligned uint64_t packed_weight_tile[5];
                    mram_read(&av_weights_bits[weight_pair_base], packed_weight_tile, packed_weight_pairs * sizeof(uint64_t));
                    for (uint32_t tile_offset = 0; tile_offset < 8u; ++tile_offset) {
                        uint32_t value_idx0 = (((token_idx + tile_offset) * group_heads) + head_idx0) * head_dim + dim_idx0;
                        uint32_t logical_weight_idx = logical_weight_start + tile_offset;
                        uint32_t packed_rel_idx = logical_weight_idx - weight_pair_base;
                        uint64_t packed_weight = packed_weight_tile[packed_rel_idx / 2u];
                        float weight0 = u32_bits_to_float(
                            (packed_rel_idx & 1u) == 0 ? (uint32_t)(packed_weight & 0xffffffffu)
                                                       : (uint32_t)(packed_weight >> 32));
                        uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32 ? runtime_slot_args.elem_offset : runtime_slot_args.v_elem_offset;
                        uint32_t pair_base0 = int8_cache_pair_base(value_elem_offset, value_idx0);
                        uint32_t pair_base1 = int8_cache_pair_base(value_elem_offset, value_idx0 + 1u);
                        uint64_t packed_v0 = 0;
                        uint64_t packed_v1 = 0;
                        mram_read(&v_cache[pair_base0], &packed_v0, sizeof(packed_v0));
                        if (pair_base1 == pair_base0) {
                            packed_v1 = packed_v0;
                        } else {
                            mram_read(&v_cache[pair_base1], &packed_v1, sizeof(packed_v1));
                        }
                        acc0 += weight0 * unpack_int8_scaled_value(packed_v0, value_elem_offset, value_idx0, runtime_slot_args.v_scale);
                        acc1 += weight0 * unpack_int8_scaled_value(packed_v1, value_elem_offset, value_idx0 + 1u, runtime_slot_args.v_scale);
                    }
                }
            } else
            for (; token_idx + 8u <= seq_len; token_idx += 8u) {
                uint32_t logical_weight_start = weight_row_base + token_idx;
                uint32_t weight_pair_base = logical_weight_start & ~1u;
                uint32_t packed_weight_pairs = (logical_weight_start & 1u) == 0 ? 4u : 5u;
                __dma_aligned uint64_t packed_weight_tile[5];
                mram_read(&av_weights_bits[weight_pair_base], packed_weight_tile, packed_weight_pairs * sizeof(uint64_t));
                for (uint32_t tile_offset = 0; tile_offset < 8u; ++tile_offset) {
                    uint32_t value_idx0 = (((token_idx + tile_offset) * group_heads) + head_idx0) * head_dim + dim_idx0;
                    uint32_t logical_weight_idx = logical_weight_start + tile_offset;
                    uint32_t packed_rel_idx = logical_weight_idx - weight_pair_base;
                    uint64_t packed_weight = packed_weight_tile[packed_rel_idx / 2u];
                    float weight0 = u32_bits_to_float(
                        (packed_rel_idx & 1u) == 0 ? (uint32_t)(packed_weight & 0xffffffffu)
                                                   : (uint32_t)(packed_weight >> 32));
                    if (v_dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                        uint64_t packed_v = 0;
                        uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32 ? runtime_slot_args.elem_offset : runtime_slot_args.v_elem_offset;
                        uint32_t word_idx0 = value_elem_offset + value_idx0;
                        mram_read(&v_cache[word_idx0], &packed_v, sizeof(packed_v));
                        acc0 += weight0 * u32_bits_to_float((uint32_t)(packed_v & 0xffffffffu));
                        acc1 += weight0 * u32_bits_to_float((uint32_t)(packed_v >> 32));
                    } else {
                        float value0 = read_v_value(&runtime_slot_args, value_idx0);
                        float value1 = read_v_value(&runtime_slot_args, value_idx0 + 1u);
                        acc0 += weight0 * value0;
                        acc1 += weight0 * value1;
                    }
                }
            }
            for (; token_idx < seq_len; ++token_idx) {
                uint32_t weight_idx0 = head_idx0 * seq_len + token_idx;
                uint32_t value_idx0 = ((token_idx * group_heads) + head_idx0) * head_dim + dim_idx0;
                float weight0 = read_av_weight(weight_idx0);
                if (v_dtype_code == KVSLOT_DTYPE_INT8) {
                    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32 ? runtime_slot_args.elem_offset : runtime_slot_args.v_elem_offset;
                    uint32_t pair_base0 = int8_cache_pair_base(value_elem_offset, value_idx0);
                    uint32_t pair_base1 = int8_cache_pair_base(value_elem_offset, value_idx0 + 1u);
                    uint64_t packed_v0 = 0;
                    uint64_t packed_v1 = 0;
                    mram_read(&v_cache[pair_base0], &packed_v0, sizeof(packed_v0));
                    if (pair_base1 == pair_base0) {
                        packed_v1 = packed_v0;
                    } else {
                        mram_read(&v_cache[pair_base1], &packed_v1, sizeof(packed_v1));
                    }
                    acc0 += weight0 * unpack_int8_scaled_value(packed_v0, value_elem_offset, value_idx0, runtime_slot_args.v_scale);
                    acc1 += weight0 * unpack_int8_scaled_value(packed_v1, value_elem_offset, value_idx0 + 1u, runtime_slot_args.v_scale);
                } else if (v_dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                    uint64_t packed_v = 0;
                    uint32_t value_elem_offset = runtime_slot_args.v_elem_offset == KVSLOT_UNSET_U32 ? runtime_slot_args.elem_offset : runtime_slot_args.v_elem_offset;
                    uint32_t word_idx0 = value_elem_offset + value_idx0;
                    mram_read(&v_cache[word_idx0], &packed_v, sizeof(packed_v));
                    acc0 += weight0 * u32_bits_to_float((uint32_t)(packed_v & 0xffffffffu));
                    acc1 += weight0 * u32_bits_to_float((uint32_t)(packed_v >> 32));
                } else {
                    float value0 = read_v_value(&runtime_slot_args, value_idx0);
                    float value1 = read_v_value(&runtime_slot_args, value_idx0 + 1u);
                    acc0 += weight0 * value0;
                    acc1 += weight0 * value1;
                }
            }
        } else {
            for (uint32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
                uint32_t weight_idx0 = head_idx0 * seq_len + token_idx;
                uint32_t value_idx0 = ((token_idx * group_heads) + head_idx0) * head_dim + dim_idx0;
                float weight0 = read_av_weight(weight_idx0);
                float value0 = read_v_value(&runtime_slot_args, value_idx0);
                acc0 += weight0 * value0;
                if (has_second) {
                    uint32_t weight_idx1 = head_idx1 * seq_len + token_idx;
                    uint32_t value_idx1 = ((token_idx * group_heads) + head_idx1) * head_dim + dim_idx1;
                    float weight1 = read_av_weight(weight_idx1);
                    float value1 = read_v_value(&runtime_slot_args, value_idx1);
                    acc1 += weight1 * value1;
                }
            }
        }
        write_av_context_pair(pair_idx, float_to_u32_bits(acc0), float_to_u32_bits(acc1));
    }
}

static void run_grouped_av_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t group_heads = runtime_slot_args.group_heads;
    uint32_t head_dim = runtime_slot_args.head_dim;
    uint32_t total_outputs = group_heads * head_dim;
    uint32_t total_pairs;
    uint32_t segment_count = grouped_segment_count;

    if (group_heads > KVSLOT_MAX_HEADS) {
        group_heads = KVSLOT_MAX_HEADS;
    }
    if (head_dim > KVSLOT_MAX_HEAD_DIM) {
        head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
        segment_count = KVSLOT_MAX_GROUP_SEGMENTS;
    }
    total_outputs = group_heads * head_dim;
    total_pairs = (total_outputs + 1u) / 2u;

    for (uint32_t pair_idx = tasklet_id; pair_idx < total_pairs; pair_idx += NR_TASKLETS) {
        uint32_t out_idx0 = pair_idx * 2u;
        uint32_t out_idx1 = out_idx0 + 1u;
        uint32_t head_idx0 = out_idx0 / head_dim;
        uint32_t dim_idx0 = out_idx0 % head_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        uint32_t head_idx1 = 0;
        uint32_t dim_idx1 = 0;
        int has_second = out_idx1 < total_outputs;
        int same_head_pair = 0;
        uint32_t weight_row_offset = 0;

        if (has_second) {
            head_idx1 = out_idx1 / head_dim;
            dim_idx1 = out_idx1 % head_dim;
            same_head_pair = head_idx1 == head_idx0;
        }

        for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
            kvslot_runtime_slot_args_t *segment_slot = &grouped_runtime_slot_args[seg_idx];
            uint32_t segment_v_dtype_code = segment_slot->v_dtype_code == KVSLOT_UNSET_U32 ? segment_slot->dtype_code : segment_slot->v_dtype_code;
            uint32_t seq_len = grouped_segment_lengths[seg_idx];
            uint32_t slot_group_heads = segment_slot->group_heads;
            if (seq_len == 0) {
                continue;
            }
            if (slot_group_heads > group_heads) {
                slot_group_heads = group_heads;
            }
            if (same_head_pair) {
                if (segment_v_dtype_code == KVSLOT_DTYPE_INT8) {
                    uint32_t value_elem_offset = segment_slot->v_elem_offset == KVSLOT_UNSET_U32
                        ? segment_slot->elem_offset
                        : segment_slot->v_elem_offset;
                    float segment_acc0 = 0.0f;
                    float segment_acc1 = 0.0f;
                    for (uint32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
                        uint32_t weight_idx0 = head_idx0 * runtime_slot_args.seq_len + weight_row_offset + token_idx;
                        uint32_t value_idx0 = ((token_idx * slot_group_heads) + head_idx0) * segment_slot->head_dim + dim_idx0;
                        float weight0 = read_av_weight(weight_idx0);
                        uint32_t pair_base0 = int8_cache_pair_base(value_elem_offset, value_idx0);
                        uint32_t pair_base1 = int8_cache_pair_base(value_elem_offset, value_idx0 + 1u);
                        uint64_t packed_v0 = 0;
                        uint64_t packed_v1 = 0;
                        mram_read(&v_cache[pair_base0], &packed_v0, sizeof(packed_v0));
                        if (pair_base1 == pair_base0) {
                            packed_v1 = packed_v0;
                        } else {
                            mram_read(&v_cache[pair_base1], &packed_v1, sizeof(packed_v1));
                        }
                        segment_acc0 += weight0 * (float)unpack_int8_value(packed_v0, value_elem_offset, value_idx0);
                        segment_acc1 += weight0 * (float)unpack_int8_value(packed_v1, value_elem_offset, value_idx0 + 1u);
                    }
                    acc0 += segment_acc0 * segment_slot->v_scale;
                    acc1 += segment_acc1 * segment_slot->v_scale;
                    weight_row_offset += seq_len;
                    continue;
                }
                for (uint32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
                    uint32_t weight_idx0 = head_idx0 * runtime_slot_args.seq_len + weight_row_offset + token_idx;
                    uint32_t value_idx0 = ((token_idx * slot_group_heads) + head_idx0) * segment_slot->head_dim + dim_idx0;
                    float weight0 = read_av_weight(weight_idx0);
                    if (segment_v_dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                        uint64_t packed_v = 0;
                        uint32_t value_elem_offset = segment_slot->v_elem_offset == KVSLOT_UNSET_U32 ? segment_slot->elem_offset : segment_slot->v_elem_offset;
                        uint32_t word_idx0 = value_elem_offset + value_idx0;
                        mram_read(&v_cache[word_idx0], &packed_v, sizeof(packed_v));
                        acc0 += weight0 * u32_bits_to_float((uint32_t)(packed_v & 0xffffffffu));
                        acc1 += weight0 * u32_bits_to_float((uint32_t)(packed_v >> 32));
                    } else {
                        float value0 = read_v_value(segment_slot, value_idx0);
                        float value1 = read_v_value(segment_slot, value_idx0 + 1u);
                        acc0 += weight0 * value0;
                        acc1 += weight0 * value1;
                    }
                }
            } else {
                for (uint32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
                    uint32_t weight_idx0 = head_idx0 * runtime_slot_args.seq_len + weight_row_offset + token_idx;
                    uint32_t value_idx0 = ((token_idx * slot_group_heads) + head_idx0) * segment_slot->head_dim + dim_idx0;
                    float weight0 = read_av_weight(weight_idx0);
                    float value0 = read_v_value(segment_slot, value_idx0);
                    acc0 += weight0 * value0;
                    if (has_second) {
                        uint32_t weight_idx1 = head_idx1 * runtime_slot_args.seq_len + weight_row_offset + token_idx;
                        uint32_t value_idx1 = ((token_idx * slot_group_heads) + head_idx1) * segment_slot->head_dim + dim_idx1;
                        float weight1 = read_av_weight(weight_idx1);
                        float value1 = read_v_value(segment_slot, value_idx1);
                        acc1 += weight1 * value1;
                    }
                }
            }
            weight_row_offset += seq_len;
        }
        write_av_context_pair(pair_idx, float_to_u32_bits(acc0), float_to_u32_bits(acc1));
    }
}
#endif

static void run_qk_slot_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t seq_len = runtime_slot_args.seq_len;
    uint32_t group_heads = runtime_slot_args.group_heads;
    uint32_t slot_head_dim = runtime_slot_args.head_dim;
    uint32_t num_heads = qk_slot_args.num_heads;
    uint32_t window = qk_slot_args.window;
    uint32_t head_dim = qk_slot_args.head_dim;
    uint32_t mode = qk_slot_args.mode;
    float score_scale = qk_slot_args.score_scale;
    uint64_t phase_start = 0;

    if (seq_len > KVSLOT_MAX_CAPACITY) {
        seq_len = KVSLOT_MAX_CAPACITY;
    }
    if (group_heads > KVSLOT_MAX_HEADS) {
        group_heads = KVSLOT_MAX_HEADS;
    }
    if (slot_head_dim > KVSLOT_MAX_HEAD_DIM) {
        slot_head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (head_dim > slot_head_dim) {
        head_dim = slot_head_dim;
    }
    if (head_dim > KVSLOT_MAX_HEAD_DIM) {
        head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (window > seq_len) {
        window = seq_len;
    }
    if (num_heads > KVSLOT_MAX_HEADS) {
        num_heads = KVSLOT_MAX_HEADS;
    }
    if (mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED
        && mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED
        && mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED) {
        mode = KVSLOT_QK_SLOT_MODE_RAW_SCORES;
    }
    uint32_t score_stride = (window + 1u) & ~1u;

    if (tasklet_id == 0 && qk_phase_profile_enabled) {
        phase_start = perfcounter_get();
    }
#ifdef KVSLOT_SLIM_QK_SLOT_ONLY
    if (!run_qk_slot_slim_all_head_dot(
            tasklet_id,
            seq_len,
            group_heads,
            slot_head_dim,
            num_heads,
            window,
            head_dim,
            mode)) {
#endif
    for (uint32_t head_row = 0; head_row < num_heads; ++head_row) {
        uint32_t local_head_idx = qk_slot_head_indices[head_row];
        if (local_head_idx >= group_heads) {
            continue;
        }
        {
            uint32_t total_query_pairs = (head_dim + 1u) / 2u;
            uint32_t query_row_base = head_row * head_dim;
            for (uint32_t pair_idx = tasklet_id; pair_idx < total_query_pairs; pair_idx += NR_TASKLETS) {
                uint32_t dim_idx0 = pair_idx * 2u;
                uint32_t dim_idx1 = dim_idx0 + 1u;
                uint64_t packed_q = 0;
                mram_read(&qk_query[query_row_base + dim_idx0], &packed_q, sizeof(packed_q));
                qk_slot_query_row[dim_idx0] = u32_bits_to_float((uint32_t)(packed_q & 0xffffffffu));
                if (dim_idx1 < head_dim) {
                    qk_slot_query_row[dim_idx1] = u32_bits_to_float((uint32_t)(packed_q >> 32));
                }
            }
        }
        barrier_wait(&kvslot_barrier);
#if defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK) && !defined(KVSLOT_SLIM_QK_SLOT_ONLY)
        if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT8) {
            if (tasklet_id == 0) {
                qk_slot_query_i16_scale = quantize_query_row_i16(head_dim);
            }
            barrier_wait(&kvslot_barrier);
        }
#endif
        for (uint32_t token_offset = tasklet_id; token_offset < window; token_offset += NR_TASKLETS) {
            uint32_t token_idx = seq_len - window + token_offset;
            uint32_t key_row_base = (((token_idx * group_heads) + local_head_idx) * slot_head_dim);
            float local_sum = dot_query_with_k_row(&runtime_slot_args, key_row_base, head_dim);
            qk_slot_score_local[(size_t)head_row * window + token_offset] = float_to_u32_bits(local_sum);
        }
        barrier_wait(&kvslot_barrier);
    }
#ifdef KVSLOT_SLIM_QK_SLOT_ONLY
    }
#endif
    if (tasklet_id == 0 && qk_phase_profile_enabled) {
        uint64_t phase_end = perfcounter_get();
        kvslot_meta.qk_dot_cycles += phase_end - phase_start;
        phase_start = phase_end;
    }

    for (uint32_t head_row = tasklet_id; head_row < num_heads; head_row += NR_TASKLETS) {
        float row_max = 0.0f;
        float scaled_row_max = 0.0f;
        float row_sum = 0.0f;
        uint32_t total_pairs = (window + 1u) / 2u;
        if (window > 0) {
            row_max = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window]);
            for (uint32_t pos = 1; pos < window; ++pos) {
                float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]);
                if (value > row_max) {
                    row_max = value;
                }
            }
        }
        qk_slot_rowmax_bits[head_row] = float_to_u32_bits(row_max);
        qk_slot_row_sums[head_row] = 0.0f;
        if ((mode == KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED
                || mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED
                || mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED)
            && window > 0) {
            scaled_row_max = row_max * score_scale;
            row_sum = 0.0f;
            for (uint32_t pos = 0; pos < window; ++pos) {
                float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]);
                float exp_value = fast_exp_approx((value * score_scale) - scaled_row_max);
                qk_slot_score_local[(size_t)head_row * window + pos] = float_to_u32_bits(exp_value);
                row_sum += exp_value;
            }
            qk_slot_row_sums[head_row] = row_sum;
            if (row_sum > 0.0f && mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED) {
                for (uint32_t pos = 0; pos < window; ++pos) {
                    float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]) / row_sum;
                    qk_slot_score_local[(size_t)head_row * window + pos] = float_to_u32_bits(value);
                }
            }
        }
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
        if (mode == KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
            for (uint32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
                uint32_t out_idx0 = pair_idx * 2u;
                uint32_t out_idx1 = out_idx0 + 1u;
                uint32_t low_bits = qk_slot_score_local[(size_t)head_row * window + out_idx0];
                uint32_t high_bits = out_idx1 < window ? qk_slot_score_local[(size_t)head_row * window + out_idx1] : 0u;
                uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
                mram_write(&packed, &qk_slot_scores_bits[(size_t)head_row * score_stride + out_idx0], sizeof(packed));
            }
        }
#endif
    }
    barrier_wait(&kvslot_barrier);
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
    if (mode == KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED && window > 0) {
        uint32_t total_weights = num_heads * window;
        uint32_t total_pairs = (total_weights + 1u) / 2u;
        for (uint32_t pair_idx = tasklet_id; pair_idx < total_pairs; pair_idx += NR_TASKLETS) {
            uint32_t idx0 = pair_idx * 2u;
            uint32_t idx1 = idx0 + 1u;
            uint32_t low_bits = qk_slot_score_local[idx0];
            uint32_t high_bits = idx1 < total_weights ? qk_slot_score_local[idx1] : 0u;
            uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
            mram_write(&packed, &av_weights_bits[idx0], sizeof(packed));
        }
    }
#endif
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0 && qk_phase_profile_enabled) {
        uint64_t phase_end = perfcounter_get();
        kvslot_meta.qk_softmax_cycles += phase_end - phase_start;
        phase_start = phase_end;
    }
    if (mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED
        || mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED) {
        run_context_from_local_softmax(num_heads, window, group_heads, head_dim);
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0 && qk_phase_profile_enabled) {
        uint64_t phase_end = perfcounter_get();
        kvslot_meta.qk_context_cycles += phase_end - phase_start;
        phase_start = phase_end;
    }
    if (tasklet_id == 0) {
        for (uint32_t head_row = num_heads; head_row < KVSLOT_MAX_HEADS; ++head_row) {
            qk_slot_rowmax_bits[head_row] = 0u;
            qk_slot_head_indices[head_row] = 0u;
            qk_slot_row_sums[head_row] = 0.0f;
        }
        for (uint32_t head_row = 0; head_row < num_heads; ++head_row) {
            qk_slot_head_indices[head_row] = float_to_u32_bits(qk_slot_row_sums[head_row]);
        }
    }
    barrier_wait(&kvslot_barrier);
}

#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
static void run_grouped_qk_slot_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t group_heads = runtime_slot_args.group_heads;
    uint32_t slot_head_dim = runtime_slot_args.head_dim;
    uint32_t num_heads = qk_slot_args.num_heads;
    uint32_t window = runtime_slot_args.seq_len;
    uint32_t head_dim = qk_slot_args.head_dim;
    uint32_t segment_count = grouped_segment_count;

    if (group_heads > KVSLOT_MAX_HEADS) {
        group_heads = KVSLOT_MAX_HEADS;
    }
    if (slot_head_dim > KVSLOT_MAX_HEAD_DIM) {
        slot_head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (head_dim > slot_head_dim) {
        head_dim = slot_head_dim;
    }
    if (head_dim > KVSLOT_MAX_HEAD_DIM) {
        head_dim = KVSLOT_MAX_HEAD_DIM;
    }
    if (num_heads > KVSLOT_MAX_HEADS) {
        num_heads = KVSLOT_MAX_HEADS;
    }
    if (segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
        segment_count = KVSLOT_MAX_GROUP_SEGMENTS;
    }
    if (window > KVSLOT_MAX_CAPACITY) {
        window = KVSLOT_MAX_CAPACITY;
    }

    for (uint32_t head_row = 0; head_row < num_heads; ++head_row) {
        uint32_t local_head_idx = qk_slot_head_indices[head_row];
        uint32_t total_query_pairs;
        uint32_t query_row_base;
        uint32_t output_offset = 0;
        if (local_head_idx >= group_heads) {
            continue;
        }
        total_query_pairs = (head_dim + 1u) / 2u;
        query_row_base = head_row * head_dim;
        for (uint32_t pair_idx = tasklet_id; pair_idx < total_query_pairs; pair_idx += NR_TASKLETS) {
            uint32_t dim_idx0 = pair_idx * 2u;
            uint32_t dim_idx1 = dim_idx0 + 1u;
            uint64_t packed_q = 0;
            mram_read(&qk_query[query_row_base + dim_idx0], &packed_q, sizeof(packed_q));
            qk_slot_query_row[dim_idx0] = u32_bits_to_float((uint32_t)(packed_q & 0xffffffffu));
            if (dim_idx1 < head_dim) {
                qk_slot_query_row[dim_idx1] = u32_bits_to_float((uint32_t)(packed_q >> 32));
            }
        }
        barrier_wait(&kvslot_barrier);
#if defined(KVSLOT_EXPERIMENTAL_INT8_I16_QK) && !defined(KVSLOT_SLIM_QK_SLOT_ONLY)
        if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_INT8) {
            if (tasklet_id == 0) {
                qk_slot_query_i16_scale = quantize_query_row_i16(head_dim);
            }
            barrier_wait(&kvslot_barrier);
        }
#endif
        for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
            kvslot_runtime_slot_args_t *segment_slot = &grouped_runtime_slot_args[seg_idx];
            uint32_t seg_window = grouped_segment_lengths[seg_idx];
            uint32_t seg_group_heads = segment_slot->group_heads;
            uint32_t seg_head_dim = segment_slot->head_dim;
            if (seg_window == 0) {
                continue;
            }
            if (seg_group_heads > group_heads) {
                seg_group_heads = group_heads;
            }
            if (local_head_idx >= seg_group_heads) {
                output_offset += seg_window;
                continue;
            }
            if (seg_head_dim > slot_head_dim) {
                seg_head_dim = slot_head_dim;
            }
            for (uint32_t token_offset = tasklet_id; token_offset < seg_window; token_offset += NR_TASKLETS) {
                uint32_t key_row_base = ((token_offset * seg_group_heads) + local_head_idx) * seg_head_dim;
                float local_sum = dot_query_with_k_row(segment_slot, key_row_base, head_dim);
                qk_slot_score_local[(size_t)head_row * window + output_offset + token_offset] = float_to_u32_bits(local_sum);
            }
            barrier_wait(&kvslot_barrier);
            output_offset += seg_window;
        }
    }

    for (uint32_t head_row = tasklet_id; head_row < num_heads; head_row += NR_TASKLETS) {
        float row_max = 0.0f;
        uint32_t total_pairs = (window + 1u) / 2u;
        if (window > 0) {
            row_max = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window]);
            for (uint32_t pos = 1; pos < window; ++pos) {
                float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]);
                if (value > row_max) {
                    row_max = value;
                }
            }
        }
        qk_slot_rowmax_bits[head_row] = float_to_u32_bits(row_max);
        for (uint32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
            uint32_t out_idx0 = pair_idx * 2u;
            uint32_t out_idx1 = out_idx0 + 1u;
            uint32_t low_bits = qk_slot_score_local[(size_t)head_row * window + out_idx0];
            uint32_t high_bits = out_idx1 < window ? qk_slot_score_local[(size_t)head_row * window + out_idx1] : 0u;
            uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
            mram_write(&packed, &qk_slot_scores_bits[(size_t)head_row * ((window + 1u) & ~1u) + out_idx0], sizeof(packed));
        }
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0) {
        for (uint32_t head_row = num_heads; head_row < KVSLOT_MAX_HEADS; ++head_row) {
            qk_slot_rowmax_bits[head_row] = 0u;
            qk_slot_head_indices[head_row] = 0u;
            qk_slot_row_sums[head_row] = 0.0f;
        }
    }
    barrier_wait(&kvslot_barrier);
}
#endif

int main(void)
{
    uint32_t tasklet_id = me();
    uint32_t kernel_command = kvslot_kernel_command & KVSLOT_KERNEL_COMMAND_MASK;
    if (tasklet_id == 0) {
        mem_reset();
        memset(&kvslot_meta, 0, sizeof(kvslot_meta));
        qk_phase_profile_enabled = (kvslot_kernel_command & KVSLOT_KERNEL_PROFILE_FLAG) != 0;
        if (qk_phase_profile_enabled) {
            perfcounter_config(COUNT_CYCLES, true);
        }
    }
    barrier_wait(&kvslot_barrier);
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
    if (kernel_command == KVSLOT_KERNEL_QK) {
        run_qk_kernel();
    } else if (kernel_command == KVSLOT_KERNEL_AV) {
        run_av_kernel();
    } else if (kernel_command == (KVSLOT_KERNEL_AV + 100u)) {
        run_grouped_av_kernel();
    } else if (kernel_command == KVSLOT_KERNEL_QK_SLOT) {
        run_qk_slot_kernel();
    } else if (kernel_command == (KVSLOT_KERNEL_QK_SLOT + 100u)) {
        run_grouped_qk_slot_kernel();
    }
#else
    if (kernel_command == KVSLOT_KERNEL_QK_SLOT) {
        run_qk_slot_kernel();
    }
#endif
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0 && qk_phase_profile_enabled) {
        kvslot_meta.cycles = perfcounter_get();
        kvslot_meta.qk_other_cycles = kvslot_meta.cycles;
        if (kvslot_meta.qk_other_cycles >= kvslot_meta.qk_dot_cycles) {
            kvslot_meta.qk_other_cycles -= kvslot_meta.qk_dot_cycles;
        } else {
            kvslot_meta.qk_other_cycles = 0;
        }
        if (kvslot_meta.qk_other_cycles >= kvslot_meta.qk_softmax_cycles) {
            kvslot_meta.qk_other_cycles -= kvslot_meta.qk_softmax_cycles;
        } else {
            kvslot_meta.qk_other_cycles = 0;
        }
        if (kvslot_meta.qk_other_cycles >= kvslot_meta.qk_context_cycles) {
            kvslot_meta.qk_other_cycles -= kvslot_meta.qk_context_cycles;
        } else {
            kvslot_meta.qk_other_cycles = 0;
        }
    }
    return 0;
}
