#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram_unaligned.h>
#include <perfcounter.h>
#include <stdint.h>

#include "common.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

__host dot_args_t dot_args;
__host dot_result_t dot_result;

__mram_noinit int32_t input_a[MAX_ELEMS_PER_DPU];
__mram_noinit int32_t input_b[MAX_ELEMS_PER_DPU];

BARRIER_INIT(dot_barrier, NR_TASKLETS);

static int64_t partial_sums[NR_TASKLETS];

int main(void)
{
    uint32_t tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
        dot_result.dot = 0;
        dot_result.cycles = 0;
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&dot_barrier);

    uint32_t length = dot_args.length;
    if (length > MAX_ELEMS_PER_DPU) {
        length = MAX_ELEMS_PER_DPU;
    }

    int64_t local_sum = 0;
    for (uint32_t idx = tasklet_id; idx < length; idx += NR_TASKLETS) {
        int32_t a;
        int32_t b;
        mram_read_unaligned(&input_a[idx], &a, sizeof(a));
        mram_read_unaligned(&input_b[idx], &b, sizeof(b));
        local_sum += (int64_t)a * (int64_t)b;
    }
    partial_sums[tasklet_id] = local_sum;
    barrier_wait(&dot_barrier);

    if (tasklet_id == 0) {
        int64_t total = 0;
        for (uint32_t idx = 0; idx < NR_TASKLETS; ++idx) {
            total += partial_sums[idx];
        }
        dot_result.dot = total;
        dot_result.cycles = perfcounter_get();
    }
    return 0;
}
