#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

#include "common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

__host qk_args_t qk_args;
__host qk_meta_t qk_meta;

__mram_noinit int32_t query[MAX_HEAD_DIM];
__mram_noinit int32_t keys[MAX_KEYS_PER_DPU * MAX_HEAD_DIM];
__mram_noinit int64_t scores[MAX_KEYS_PER_DPU];

BARRIER_INIT(qk_barrier, NR_TASKLETS);

static int64_t partial_sums[NR_TASKLETS];

int main(void)
{
    uint32_t tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
        qk_meta.cycles = 0;
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&qk_barrier);

    uint32_t head_dim = qk_args.head_dim;
    uint32_t num_keys = qk_args.num_keys;
    uint32_t key_stride = qk_args.key_stride;

    if (head_dim > MAX_HEAD_DIM) {
        head_dim = MAX_HEAD_DIM;
    }
    if (num_keys > MAX_KEYS_PER_DPU) {
        num_keys = MAX_KEYS_PER_DPU;
    }
    head_dim = (head_dim / 2) * 2;
    key_stride = (key_stride / 2) * 2;

    for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx) {
        int64_t local_sum = 0;
        for (uint32_t pair_idx = tasklet_id; pair_idx < head_dim / 2; pair_idx += NR_TASKLETS) {
            uint32_t elem_idx = pair_idx * 2;
            uint64_t packed_q;
            uint64_t packed_k;
            mram_read(&query[elem_idx], &packed_q, sizeof(packed_q));
            mram_read(&keys[key_idx * key_stride + elem_idx], &packed_k, sizeof(packed_k));

            int32_t q0 = (int32_t)(packed_q & 0xffffffffu);
            int32_t q1 = (int32_t)(packed_q >> 32);
            int32_t k0 = (int32_t)(packed_k & 0xffffffffu);
            int32_t k1 = (int32_t)(packed_k >> 32);

            local_sum += (int64_t)q0 * (int64_t)k0;
            local_sum += (int64_t)q1 * (int64_t)k1;
        }
        partial_sums[tasklet_id] = local_sum;
        barrier_wait(&qk_barrier);

        if (tasklet_id == 0) {
            int64_t total = 0;
            for (uint32_t idx = 0; idx < NR_TASKLETS; ++idx) {
                total += partial_sums[idx];
            }
            scores[key_idx] = total;
        }
        barrier_wait(&qk_barrier);
    }

    if (tasklet_id == 0) {
        qk_meta.cycles = perfcounter_get();
    }
    return 0;
}
