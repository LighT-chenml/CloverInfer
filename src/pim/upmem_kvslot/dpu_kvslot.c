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
__host kvslot_meta_t kvslot_meta;
__host uint32_t kvslot_kernel_command;

__mram_noinit int32_t k_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t v_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t qk_query[KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t qk_keys[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int64_t qk_scores[KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t av_weights_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_CAPACITY];
__mram_noinit uint32_t av_context_bits[KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];

BARRIER_INIT(kvslot_barrier, NR_TASKLETS);

static int64_t partial_sums[NR_TASKLETS];

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

        if (has_second) {
            head_idx1 = out_idx1 / head_dim;
            dim_idx1 = out_idx1 % head_dim;
        }
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
        write_av_context_pair(pair_idx, float_to_u32_bits(acc0), float_to_u32_bits(acc1));
    }
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
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0) {
        kvslot_meta.cycles = perfcounter_get();
    }
    return 0;
}
