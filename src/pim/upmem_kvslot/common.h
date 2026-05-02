#ifndef CLOVER_UPMEM_KVSLOT_COMMON_H
#define CLOVER_UPMEM_KVSLOT_COMMON_H

#include <stdint.h>

#define KVSLOT_MAGIC 0x4B56534CU
#define KVSLOT_MAX_HEADS 32
#define KVSLOT_MAX_HEAD_DIM 128
#define KVSLOT_MAX_CAPACITY 256
#define KVSLOT_MAX_SLOTS_PER_DPU 256

#define KVSLOT_CMD_ALLOCATE 1U
#define KVSLOT_CMD_APPEND 2U
#define KVSLOT_CMD_READBACK 3U
#define KVSLOT_CMD_FREE 4U
#define KVSLOT_CMD_GET_STATS 5U
#define KVSLOT_CMD_QK_BATCH 6U
#define KVSLOT_CMD_AV 7U
#define KVSLOT_CMD_AV_BATCH 8U
#define KVSLOT_CMD_QK_SLOT_BATCH 9U
#define KVSLOT_CMD_SOFTMAX_AV_BATCH 10U
#define KVSLOT_CMD_QK_SOFTMAX_AV_BATCH 11U
#define KVSLOT_CMD_QK_SOFTMAX_AV_PARTIAL_BATCH 12U
#define KVSLOT_CMD_GET_PROFILE 13U
#define KVSLOT_CMD_GET_TOPOLOGY 14U

#define KVSLOT_KERNEL_NONE 0U
#define KVSLOT_KERNEL_QK 1U
#define KVSLOT_KERNEL_AV 2U
#define KVSLOT_KERNEL_QK_SLOT 3U

#define KVSLOT_QK_SLOT_MODE_RAW_SCORES 0U
#define KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED 1U
#define KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED 2U

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

typedef struct {
    uint32_t head_dim;
    uint32_t num_keys;
    uint32_t key_stride;
    uint32_t reserved;
} kvslot_qk_dpu_args_t;

typedef struct {
    uint32_t seq_len;
    uint32_t group_heads;
    uint32_t head_dim;
    uint32_t dtype_code;
    uint32_t elem_offset;
    uint32_t reserved[3];
} kvslot_runtime_slot_args_t;

typedef struct {
    uint32_t num_slots;
    uint32_t reserved[3];
} kvslot_av_batch_args_t;

typedef struct {
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    uint32_t mode;
    float score_scale;
} kvslot_qk_slot_args_t;

typedef struct {
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    uint32_t reserved;
} kvslot_qk_slot_batch_item_args_t;

typedef struct {
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    float score_scale;
} kvslot_qk_softmax_av_batch_item_args_t;

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

typedef struct {
    uint64_t qk_rounds_total;
    uint64_t qk_batched_rounds;
    uint64_t qk_fallback_rounds;
    uint64_t qk_round_items_total;
    uint64_t qk_batched_items_total;
    uint64_t qk_active_ranks_total;
    uint64_t qk_max_round_size;
    uint64_t qk_max_active_ranks;
    uint64_t av_rounds_total;
    uint64_t av_batched_rounds;
    uint64_t av_fallback_rounds;
    uint64_t av_round_items_total;
    uint64_t av_batched_items_total;
    uint64_t av_active_ranks_total;
    uint64_t av_max_round_size;
    uint64_t av_max_active_ranks;
    uint64_t qk_batched_round_total_ns;
    uint64_t qk_batched_xfer_to_ns;
    uint64_t qk_batched_launch_ns;
    uint64_t qk_batched_xfer_from_ns;
    uint64_t qk_fallback_round_total_ns;
    uint64_t qk_fallback_launch_ns;
    uint64_t qk_fallback_sync_ns;
    uint64_t qk_fallback_xfer_from_ns;
    uint64_t av_batched_round_total_ns;
    uint64_t av_batched_xfer_to_ns;
    uint64_t av_batched_launch_ns;
    uint64_t av_batched_xfer_from_ns;
    uint64_t av_fallback_round_total_ns;
    uint64_t av_fallback_launch_ns;
    uint64_t av_fallback_sync_ns;
    uint64_t av_fallback_xfer_from_ns;
} kvslot_profile_stats_t;

typedef struct {
    uint32_t nr_dpus;
    uint32_t nr_ranks;
    uint32_t reserved[2];
} kvslot_topology_header_t;

typedef struct {
    uint32_t logical_dpu_id;
    uint32_t rank_index;
    uint32_t rank_id;
    uint32_t reserved;
} kvslot_topology_item_t;

#endif
