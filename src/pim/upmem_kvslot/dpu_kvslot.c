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
__host kvslot_meta_t kvslot_meta;

__mram_noinit int32_t k_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];
__mram_noinit int32_t v_cache[KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM];

BARRIER_INIT(kvslot_barrier, NR_TASKLETS);

int main(void)
{
    uint32_t tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
        kvslot_meta.cycles = 0;
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0) {
        uint32_t capacity = slot_args.capacity;
        uint32_t group_heads = slot_args.group_heads;
        uint32_t head_dim = slot_args.head_dim;
        if (capacity > KVSLOT_MAX_CAPACITY) {
            capacity = KVSLOT_MAX_CAPACITY;
        }
        if (group_heads > KVSLOT_MAX_HEADS) {
            group_heads = KVSLOT_MAX_HEADS;
        }
        if (head_dim > KVSLOT_MAX_HEAD_DIM) {
            head_dim = KVSLOT_MAX_HEAD_DIM;
        }
        if (capacity > 0 && group_heads > 0 && head_dim > 0) {
            uint64_t probe = 0;
            mram_read(&k_cache[0], &probe, sizeof(probe));
            mram_write(&probe, &k_cache[0], sizeof(probe));
            mram_read(&v_cache[0], &probe, sizeof(probe));
            mram_write(&probe, &v_cache[0], sizeof(probe));
        }
    }
    barrier_wait(&kvslot_barrier);
    if (tasklet_id == 0) {
        kvslot_meta.cycles = perfcounter_get();
    }
    return 0;
}
