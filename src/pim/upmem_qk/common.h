#ifndef CLOVER_UPMEM_QK_COMMON_H
#define CLOVER_UPMEM_QK_COMMON_H

#include <stdint.h>

#define MAX_HEAD_DIM 128
#define MAX_KEYS_PER_DPU 128
#define QK_IO_MAGIC 0x514B494FU

typedef struct {
    uint32_t head_dim;
    uint32_t num_keys;
    uint32_t key_stride;
    uint32_t reserved;
} qk_args_t;

typedef struct {
    uint64_t cycles;
} qk_meta_t;

typedef struct {
    uint32_t magic;
    uint32_t head_dim;
    uint32_t num_keys;
    uint32_t reserved;
} qk_io_header_t;

#endif
