#ifndef CLOVER_UPMEM_ALLOCATOR_COMMON_H
#define CLOVER_UPMEM_ALLOCATOR_COMMON_H

#include <stdint.h>

#define CLOVER_ALLOCATOR_ALIGNMENT_BYTES 1024u
#define CLOVER_ALLOCATOR_MAX_ALLOCS 1024u
#define CLOVER_ALLOCATOR_REMAP_STRIDE 3u

typedef struct {
    uint32_t ptr_bytes;
    uint32_t size_bytes;
    uint32_t in_use;
} clover_alloc_record_t;

#endif
