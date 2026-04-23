#ifndef CLOVER_UPMEM_DOT_COMMON_H
#define CLOVER_UPMEM_DOT_COMMON_H

#include <stdint.h>

#define MAX_ELEMS_PER_DPU 256

typedef struct {
    uint32_t length;
    uint32_t padding;
} dot_args_t;

typedef struct {
    int64_t dot;
    uint64_t cycles;
} dot_result_t;

#endif
