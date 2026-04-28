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

__host kvslot_slot_args_t slot_args;
__host kvslot_qk_dpu_args_t qk_args;
__host kvslot_runtime_slot_args_t runtime_slot_args;
__host kvslot_qk_slot_args_t qk_slot_args;
__host uint32_t qk_slot_head_indices[KVSLOT_MAX_HEADS];
__host uint32_t qk_slot_rowmax_bits[KVSLOT_MAX_HEADS];
__host kvslot_meta_t kvslot_meta;
__host uint32_t kvslot_kernel_command;

__mram_noinit int32_t k_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t v_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t qk_query[KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t qk_keys[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int64_t qk_scores[KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t qk_slot_scores_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t av_weights_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t av_context_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];

BARRIER_INIT(kvslot_barrier, NR_TASKLETS);

static int64_t partial_sums[NR_TASKLETS];
static float partial_float_sums[NR_TASKLETS];
static uint32_t qk_slot_score_local[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
static float qk_slot_query_row[KVSLOT_MAX_HEAD_DIM];
static float qk_slot_row_sums[KVSLOT_MAX_HEADS];

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

static float read_av_weight(uint32_t logical_idx)
{
    uint64_t packed = 0;
    uint32_t pair_base = logical_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&av_weights_bits[pair_base], &packed, sizeof(packed));
    bits = (logical_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}

static void write_av_context_pair(uint32_t pair_idx, uint32_t low_bits, uint32_t high_bits)
{
    uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
    mram_write(&packed, &av_context_bits[pair_idx * 2u], sizeof(packed));
}

static float read_v_value(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx)
{
    if (slot->dtype_code == KVSLOT_DTYPE_FP16) {
        uint64_t packed64 = 0;
        uint32_t packed = 0;
        uint32_t word_idx = slot->elem_offset + (logical_idx / 2u);
        uint32_t pair_base = word_idx & ~1u;
        uint16_t fp16_bits;
        mram_read(&v_cache[pair_base], &packed64, sizeof(packed64));
        packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
        fp16_bits = (logical_idx & 1u) == 0 ? (uint16_t)(packed & 0xffffu) : (uint16_t)(packed >> 16);
        return fp16_bits_to_float(fp16_bits);
    }

    uint64_t packed = 0;
    uint32_t word_idx = slot->elem_offset + logical_idx;
    uint32_t pair_base = word_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&v_cache[pair_base], &packed, sizeof(packed));
    bits = (word_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}

static float read_k_value(const kvslot_runtime_slot_args_t *slot, uint32_t logical_idx)
{
    if (slot->dtype_code == KVSLOT_DTYPE_FP16) {
        uint64_t packed64 = 0;
        uint32_t packed = 0;
        uint32_t word_idx = slot->elem_offset + (logical_idx / 2u);
        uint32_t pair_base = word_idx & ~1u;
        uint16_t fp16_bits;
        mram_read(&k_cache[pair_base], &packed64, sizeof(packed64));
        packed = (word_idx & 1u) == 0 ? (uint32_t)(packed64 & 0xffffffffu) : (uint32_t)(packed64 >> 32);
        fp16_bits = (logical_idx & 1u) == 0 ? (uint16_t)(packed & 0xffffu) : (uint16_t)(packed >> 16);
        return fp16_bits_to_float(fp16_bits);
    }

    uint64_t packed = 0;
    uint32_t word_idx = slot->elem_offset + logical_idx;
    uint32_t pair_base = word_idx & ~1u;
    uint32_t bits = 0;
    mram_read(&k_cache[pair_base], &packed, sizeof(packed));
    bits = (word_idx & 1u) == 0 ? (uint32_t)(packed & 0xffffffffu) : (uint32_t)(packed >> 32);
    return u32_bits_to_float(bits);
}

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

static void run_context_from_local_softmax(uint32_t num_heads, uint32_t window, uint32_t group_heads, uint32_t head_dim)
{
    uint32_t tasklet_id = me();
    uint32_t total_outputs = num_heads * head_dim;
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
            for (uint32_t token_idx = 0; token_idx < window; ++token_idx) {
                uint32_t value_idx0 = ((token_idx * group_heads) + head_idx0) * head_dim + dim_idx0;
                float weight0 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]);
                if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                    uint64_t packed_v = 0;
                    uint32_t word_idx0 = runtime_slot_args.elem_offset + value_idx0;
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
                uint32_t value_idx0 = ((token_idx * group_heads) + head_idx0) * head_dim + dim_idx0;
                float weight0 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx0 * window + token_idx]);
                float value0 = read_v_value(&runtime_slot_args, value_idx0);
                acc0 += weight0 * value0;
                if (has_second) {
                    uint32_t value_idx1 = ((token_idx * group_heads) + head_idx1) * head_dim + dim_idx1;
                    float weight1 = u32_bits_to_float(qk_slot_score_local[(size_t)head_idx1 * window + token_idx]);
                    float value1 = read_v_value(&runtime_slot_args, value_idx1);
                    acc1 += weight1 * value1;
                }
            }
        }
        write_av_context_pair(pair_idx, float_to_u32_bits(acc0), float_to_u32_bits(acc1));
    }
}

static void run_av_kernel(void)
{
    uint32_t tasklet_id = me();
    uint32_t seq_len = runtime_slot_args.seq_len;
    uint32_t group_heads = runtime_slot_args.group_heads;
    uint32_t head_dim = runtime_slot_args.head_dim;
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
                    if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                        uint64_t packed_v = 0;
                        uint32_t word_idx0 = runtime_slot_args.elem_offset + value_idx0;
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
                if (runtime_slot_args.dtype_code == KVSLOT_DTYPE_FP32 && (value_idx0 % 2u) == 0) {
                    uint64_t packed_v = 0;
                    uint32_t word_idx0 = runtime_slot_args.elem_offset + value_idx0;
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
    if (num_heads > group_heads) {
        num_heads = group_heads;
    }
    if (num_heads > KVSLOT_MAX_HEADS) {
        num_heads = KVSLOT_MAX_HEADS;
    }
    if (mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED && mode != KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED) {
        mode = KVSLOT_QK_SLOT_MODE_RAW_SCORES;
    }
    uint32_t score_stride = (window + 1u) & ~1u;

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
        for (uint32_t token_offset = tasklet_id; token_offset < window; token_offset += NR_TASKLETS) {
            uint32_t token_idx = seq_len - window + token_offset;
            uint32_t key_row_base = (((token_idx * group_heads) + local_head_idx) * slot_head_dim);
            float local_sum = 0.0f;
            for (uint32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                float key_value = read_k_value(&runtime_slot_args, key_row_base + dim_idx);
                local_sum += qk_slot_query_row[dim_idx] * key_value;
            }
            qk_slot_score_local[(size_t)head_row * window + token_offset] = float_to_u32_bits(local_sum);
        }
        barrier_wait(&kvslot_barrier);
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
        if ((mode == KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED || mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED) && window > 0) {
            scaled_row_max = row_max * score_scale;
            row_sum = 0.0f;
            for (uint32_t pos = 0; pos < window; ++pos) {
                float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]);
                float exp_value = fast_exp_approx((value * score_scale) - scaled_row_max);
                qk_slot_score_local[(size_t)head_row * window + pos] = float_to_u32_bits(exp_value);
                row_sum += exp_value;
            }
            qk_slot_row_sums[head_row] = row_sum;
            if (row_sum > 0.0f) {
                for (uint32_t pos = 0; pos < window; ++pos) {
                    float value = u32_bits_to_float(qk_slot_score_local[(size_t)head_row * window + pos]) / row_sum;
                    qk_slot_score_local[(size_t)head_row * window + pos] = float_to_u32_bits(value);
                }
            }
        }
        if (mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
            for (uint32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
                uint32_t out_idx0 = pair_idx * 2u;
                uint32_t out_idx1 = out_idx0 + 1u;
                uint32_t low_bits = qk_slot_score_local[(size_t)head_row * window + out_idx0];
                uint32_t high_bits = out_idx1 < window ? qk_slot_score_local[(size_t)head_row * window + out_idx1] : 0u;
                uint64_t packed = ((uint64_t)high_bits << 32) | (uint64_t)low_bits;
                mram_write(&packed, &qk_slot_scores_bits[(size_t)head_row * score_stride + out_idx0], sizeof(packed));
            }
        }
    }
    barrier_wait(&kvslot_barrier);
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
    barrier_wait(&kvslot_barrier);
    if (mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED) {
        run_context_from_local_softmax(num_heads, window, group_heads, head_dim);
    }
    barrier_wait(&kvslot_barrier);
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

int main(void)
{
    uint32_t tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
        kvslot_meta.cycles = 0;
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&kvslot_barrier);
    if (kvslot_kernel_command == KVSLOT_KERNEL_QK) {
        run_qk_kernel();
    } else if (kvslot_kernel_command == KVSLOT_KERNEL_AV) {
        run_av_kernel();
    } else if (kvslot_kernel_command == KVSLOT_KERNEL_QK_SLOT) {
        run_qk_slot_kernel();
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0) {
        kvslot_meta.cycles = perfcounter_get();
    }
    return 0;
}
