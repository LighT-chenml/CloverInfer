#ifndef CLOVER_UPMEM_KVSLOT_COMMON_H
#define CLOVER_UPMEM_KVSLOT_COMMON_H

#include <stdint.h>

#define KVSLOT_MAGIC 0x4B56534CU
#define KVSLOT_MAX_HEADS 32
#define KVSLOT_MAX_HEAD_DIM 128
#define KVSLOT_MAX_CAPACITY 256
#define KVSLOT_MAX_SLOTS_PER_DPU 32

#define KVSLOT_CMD_ALLOCATE 1U
#define KVSLOT_CMD_APPEND 2U
#define KVSLOT_CMD_READBACK 3U

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
} kvslot_slot_args_t;

typedef struct {
    uint64_t cycles;
} kvslot_meta_t;

#endif
