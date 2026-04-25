#ifndef CLOVER_UPMEM_KVSLOT_COMMON_H
#define CLOVER_UPMEM_KVSLOT_COMMON_H

#include <stdint.h>

#define KVSLOT_MAGIC 0x4B56534CU
#define KVSLOT_MAX_HEADS 32
#define KVSLOT_MAX_HEAD_DIM 128
#define KVSLOT_MAX_CAPACITY 256
#define KVSLOT_MAX_SLOTS_PER_DPU 64

#define KVSLOT_CMD_ALLOCATE 1U
#define KVSLOT_CMD_APPEND 2U
#define KVSLOT_CMD_READBACK 3U
#define KVSLOT_CMD_FREE 4U
#define KVSLOT_CMD_GET_STATS 5U
#define KVSLOT_CMD_QK_BATCH 6U

typedef struct {
    uint32_t magic;
    uint32_t command;
    uint32_t slot_id;
    uint32_t reserved;
} kvslot_io_header_t;

typedef struct {
    uint32_t capacity;
    uint32_t seq_len;
    uint32_t group_heads;
    uint32_t head_dim;
    uint32_t dtype_code;
} kvslot_slot_args_t;

typedef struct {
    uint32_t head_dim;
    uint32_t num_keys;
    uint32_t num_queries;
    uint32_t reserved;
} kvslot_qk_args_t;

#define KVSLOT_DTYPE_FP32 0U
#define KVSLOT_DTYPE_FP16 1U

typedef struct {
    uint64_t cycles;
} kvslot_meta_t;

typedef struct {
    uint32_t next_free_elem;
    uint32_t free_range_count;
    uint32_t free_elems_total;
    uint32_t largest_free_range;
    uint32_t live_slot_count;
    uint32_t live_elems_total;
} kvslot_allocator_stats_t;

#endif
