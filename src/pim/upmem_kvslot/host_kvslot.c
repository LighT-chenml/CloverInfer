#include <dpu.h>
#include <dpu_management.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./build/dpu_kvslot"
#endif

#define KVSLOT_QK_MAX_ACTIVE_DPUS 16U

typedef struct {
    uint32_t capacity;
    uint32_t seq_len;
    uint32_t group_heads;
    uint32_t head_dim;
    uint32_t dtype_code;
    uint32_t elem_offset;
    uint32_t elem_count;
} host_slot_t;

typedef struct {
    uint32_t start_elem;
    uint32_t elem_count;
} free_range_t;

typedef struct {
    struct dpu_set_t dpu_set;
    uint32_t nr_dpus;
    uint32_t nr_ranks;
    struct dpu_rank_t **ranks;
    struct dpu_set_t *physical_dpus;
    uint32_t *physical_dpu_rank_indices;
    host_slot_t *slots;
    uint32_t *next_free_elem;
    free_range_t *free_ranges;
    uint32_t *num_free_ranges;
} kvslot_runner_t;

typedef struct {
    struct dpu_set_t target_dpu;
    host_slot_t *slot;
    kvslot_slot_args_t out;
    kvslot_runtime_slot_args_t runtime_args;
    uint32_t slot_id;
    uint32_t physical_dpu_id;
    size_t weight_bytes;
    size_t context_bytes;
    size_t padded_weight_bytes;
    size_t padded_context_bytes;
    float *weights;
    float *context;
    int weights_resident_on_dpu;
    int ready;
} av_item_t;

typedef struct {
    struct dpu_set_t target_dpu;
    host_slot_t *slot;
    kvslot_runtime_slot_args_t runtime_args;
    kvslot_qk_slot_args_t slot_args;
    uint32_t slot_id;
    uint32_t physical_dpu_id;
    uint32_t num_heads;
    uint32_t window;
    uint32_t score_stride;
    uint32_t head_dim;
    uint32_t *local_head_indices;
    float *queries;
    uint32_t *raw_scores;
    uint32_t *raw_row_max_bits;
    int ready;
} qk_slot_item_t;

static int launch_qk_slot_item_async(const qk_slot_item_t *item);
static int finish_qk_slot_item(qk_slot_item_t *item);
static int can_use_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static int execute_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static qk_slot_item_t *find_qk_round_item_for_dpu(
    kvslot_runner_t *runner,
    qk_slot_item_t **items_by_dpu,
    struct dpu_set_t dpu);
static int can_use_batched_av_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static int execute_batched_av_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static av_item_t *find_av_round_item_for_dpu(
    kvslot_runner_t *runner,
    av_item_t **items_by_dpu,
    struct dpu_set_t dpu);
static int prepare_av_item_header(kvslot_runner_t *runner, uint32_t slot_id, av_item_t *item);
static int read_av_item_weights(av_item_t *item);
static int softmax_av_item_scores_inplace(av_item_t *item);
static int prepare_qk_slot_item_header(
    kvslot_runner_t *runner,
    uint32_t slot_id,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    qk_slot_item_t *item);
static int read_qk_slot_item_payload(FILE *file, qk_slot_item_t *item);
static void cleanup_qk_slot_item(qk_slot_item_t *item);
static int fetch_qk_slot_row_maxes(qk_slot_item_t *item);

static size_t slot_table_index(uint32_t physical_dpu_id, uint32_t local_slot_id)
{
    return (size_t)physical_dpu_id * KVSLOT_MAX_SLOTS_PER_DPU + local_slot_id;
}

static uint32_t kvslot_pool_capacity_elems(void)
{
    return KVSLOT_MAX_CAPACITY * KVSLOT_MAX_HEADS * KVSLOT_MAX_HEAD_DIM;
}

static uint32_t kvslot_max_free_ranges_per_dpu(void)
{
    return KVSLOT_MAX_SLOTS_PER_DPU + 1;
}

static float u32_bits_to_float(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } value = {.u = bits};
    return value.f;
}

static free_range_t *runner_free_ranges_for_dpu(kvslot_runner_t *runner, uint32_t physical_dpu_id)
{
    return &runner->free_ranges[(size_t)physical_dpu_id * kvslot_max_free_ranges_per_dpu()];
}

static int read_exact(FILE *file, void *dst, size_t bytes)
{
    return fread(dst, 1, bytes, file) == bytes ? 0 : 1;
}

static int write_exact(FILE *file, const void *src, size_t bytes)
{
    return fwrite(src, 1, bytes, file) == bytes ? 0 : 1;
}

static int flush_exact(FILE *file)
{
    return fflush(file) == 0 ? 0 : 1;
}

static void free_slot(host_slot_t *slot)
{
    if (slot == NULL) {
        return;
    }
    memset(slot, 0, sizeof(*slot));
}

static int ensure_slot(host_slot_t *slot, uint32_t capacity, uint32_t group_heads, uint32_t head_dim)
{
    if (capacity == 0 || capacity > KVSLOT_MAX_CAPACITY || group_heads == 0 || group_heads > KVSLOT_MAX_HEADS || head_dim == 0 || head_dim > KVSLOT_MAX_HEAD_DIM) {
        fprintf(stderr, "Invalid slot shape capacity=%u heads=%u head_dim=%u\n", capacity, group_heads, head_dim);
        return 1;
    }

    free_slot(slot);
    slot->capacity = capacity;
    slot->seq_len = 0;
    slot->group_heads = group_heads;
    slot->head_dim = head_dim;
    slot->dtype_code = KVSLOT_DTYPE_FP32;
    return 0;
}

static size_t kvslot_dtype_elem_size(uint32_t dtype_code)
{
    return dtype_code == KVSLOT_DTYPE_FP16 ? sizeof(uint16_t) : sizeof(int32_t);
}

static uint32_t kvslot_packed_elem_count(uint32_t logical_elems, uint32_t dtype_code)
{
    if (dtype_code == KVSLOT_DTYPE_FP16) {
        return (logical_elems + 1U) / 2U;
    }
    return logical_elems;
}

static void remove_free_range(free_range_t *ranges, uint32_t *count, uint32_t idx)
{
    if (ranges == NULL || count == NULL || idx >= *count) {
        return;
    }
    for (uint32_t pos = idx + 1; pos < *count; ++pos) {
        ranges[pos - 1] = ranges[pos];
    }
    *count -= 1;
}

static int insert_free_range_sorted(
    kvslot_runner_t *runner,
    uint32_t physical_dpu_id,
    uint32_t start_elem,
    uint32_t elem_count)
{
    free_range_t *ranges = runner_free_ranges_for_dpu(runner, physical_dpu_id);
    uint32_t *count = &runner->num_free_ranges[physical_dpu_id];
    uint32_t limit = kvslot_max_free_ranges_per_dpu();
    uint32_t pos = 0;
    if (elem_count == 0) {
        return 0;
    }
    if (*count >= limit) {
        fprintf(stderr, "Too many free ranges on DPU %u\n", physical_dpu_id);
        return 1;
    }
    while (pos < *count && ranges[pos].start_elem < start_elem) {
        pos += 1;
    }
    for (uint32_t move = *count; move > pos; --move) {
        ranges[move] = ranges[move - 1];
    }
    ranges[pos].start_elem = start_elem;
    ranges[pos].elem_count = elem_count;
    *count += 1;
    return 0;
}

static void coalesce_free_ranges(kvslot_runner_t *runner, uint32_t physical_dpu_id)
{
    free_range_t *ranges = runner_free_ranges_for_dpu(runner, physical_dpu_id);
    uint32_t *count = &runner->num_free_ranges[physical_dpu_id];
    uint32_t idx = 0;
    while (idx + 1 < *count) {
        uint32_t end_elem = ranges[idx].start_elem + ranges[idx].elem_count;
        if (end_elem >= ranges[idx + 1].start_elem) {
            uint32_t next_end = ranges[idx + 1].start_elem + ranges[idx + 1].elem_count;
            if (next_end > end_elem) {
                ranges[idx].elem_count = next_end - ranges[idx].start_elem;
            }
            remove_free_range(ranges, count, idx + 1);
            continue;
        }
        idx += 1;
    }
}

static void reclaim_tail_free_range(kvslot_runner_t *runner, uint32_t physical_dpu_id)
{
    free_range_t *ranges = runner_free_ranges_for_dpu(runner, physical_dpu_id);
    uint32_t *count = &runner->num_free_ranges[physical_dpu_id];
    while (*count > 0) {
        uint32_t last_idx = *count - 1;
        uint32_t range_end = ranges[last_idx].start_elem + ranges[last_idx].elem_count;
        if (range_end != runner->next_free_elem[physical_dpu_id]) {
            break;
        }
        runner->next_free_elem[physical_dpu_id] = ranges[last_idx].start_elem;
        *count -= 1;
    }
}

static int reserve_elem_range(
    kvslot_runner_t *runner,
    uint32_t physical_dpu_id,
    uint32_t elem_count,
    uint32_t *elem_offset_out)
{
    free_range_t *ranges = runner_free_ranges_for_dpu(runner, physical_dpu_id);
    uint32_t *count = &runner->num_free_ranges[physical_dpu_id];
    if (elem_offset_out == NULL) {
        return 1;
    }
    for (uint32_t idx = 0; idx < *count; ++idx) {
        if (ranges[idx].elem_count < elem_count) {
            continue;
        }
        *elem_offset_out = ranges[idx].start_elem;
        ranges[idx].start_elem += elem_count;
        ranges[idx].elem_count -= elem_count;
        if (ranges[idx].elem_count == 0) {
            remove_free_range(ranges, count, idx);
        }
        return 0;
    }
    if (runner->next_free_elem[physical_dpu_id] + elem_count > kvslot_pool_capacity_elems()) {
        return 1;
    }
    *elem_offset_out = runner->next_free_elem[physical_dpu_id];
    runner->next_free_elem[physical_dpu_id] += elem_count;
    return 0;
}

static int release_elem_range(
    kvslot_runner_t *runner,
    uint32_t physical_dpu_id,
    uint32_t start_elem,
    uint32_t elem_count)
{
    if (elem_count == 0) {
        return 0;
    }
    if (insert_free_range_sorted(runner, physical_dpu_id, start_elem, elem_count) != 0) {
        return 1;
    }
    coalesce_free_ranges(runner, physical_dpu_id);
    reclaim_tail_free_range(runner, physical_dpu_id);
    return 0;
}

static kvslot_allocator_stats_t collect_allocator_stats(kvslot_runner_t *runner, uint32_t physical_dpu_id)
{
    kvslot_allocator_stats_t stats = {0};
    free_range_t *ranges = runner_free_ranges_for_dpu(runner, physical_dpu_id);
    stats.next_free_elem = runner->next_free_elem[physical_dpu_id];
    stats.free_range_count = runner->num_free_ranges[physical_dpu_id];
    for (uint32_t idx = 0; idx < stats.free_range_count; ++idx) {
        stats.free_elems_total += ranges[idx].elem_count;
        if (ranges[idx].elem_count > stats.largest_free_range) {
            stats.largest_free_range = ranges[idx].elem_count;
        }
    }
    for (uint32_t local_slot_id = 0; local_slot_id < KVSLOT_MAX_SLOTS_PER_DPU; ++local_slot_id) {
        host_slot_t *slot = &runner->slots[slot_table_index(physical_dpu_id, local_slot_id)];
        if (slot->capacity == 0) {
            continue;
        }
        stats.live_slot_count += 1;
        stats.live_elems_total += slot->elem_count;
    }
    return stats;
}

static int runner_get_dpu_and_slot(
    kvslot_runner_t *runner,
    uint32_t slot_id,
    struct dpu_set_t *target_out,
    host_slot_t **slot_out)
{
    uint32_t physical_dpu_id;
    uint32_t local_slot_id;
    if (runner == NULL || target_out == NULL || slot_out == NULL || runner->nr_dpus == 0) {
        return 1;
    }
    physical_dpu_id = slot_id % runner->nr_dpus;
    local_slot_id = slot_id / runner->nr_dpus;
    if (local_slot_id >= KVSLOT_MAX_SLOTS_PER_DPU) {
        return 1;
    }
    if (runner->physical_dpus == NULL) {
        return 1;
    }
    *target_out = runner->physical_dpus[physical_dpu_id];
    *slot_out = &runner->slots[slot_table_index(physical_dpu_id, local_slot_id)];
    return 0;
}

static int runner_init(kvslot_runner_t *runner, uint32_t requested_dpus)
{
    if (runner == NULL) {
        return 1;
    }
    runner->nr_dpus = 0;
    runner->nr_ranks = 0;
    runner->ranks = NULL;
    runner->physical_dpus = NULL;
    runner->physical_dpu_rank_indices = NULL;
    runner->slots = NULL;
    runner->next_free_elem = NULL;
    runner->free_ranges = NULL;
    runner->num_free_ranges = NULL;
    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &runner->dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(runner->dpu_set, &runner->nr_dpus));
    DPU_ASSERT(dpu_get_nr_ranks(runner->dpu_set, &runner->nr_ranks));
    DPU_ASSERT(dpu_load(runner->dpu_set, DPU_BINARY, NULL));
    if (runner->nr_dpus != requested_dpus) {
        fprintf(stderr, "Requested %u DPUs, allocated %u DPUs\n", requested_dpus, runner->nr_dpus);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    runner->ranks = calloc(runner->nr_ranks, sizeof(*runner->ranks));
    runner->physical_dpus = calloc(runner->nr_dpus, sizeof(*runner->physical_dpus));
    runner->physical_dpu_rank_indices = calloc(runner->nr_dpus, sizeof(*runner->physical_dpu_rank_indices));
    if (runner->ranks == NULL || runner->physical_dpus == NULL || runner->physical_dpu_rank_indices == NULL) {
        fprintf(stderr, "Failed to allocate DPU topology metadata\n");
        free(runner->physical_dpu_rank_indices);
        runner->physical_dpu_rank_indices = NULL;
        free(runner->physical_dpus);
        runner->physical_dpus = NULL;
        free(runner->ranks);
        runner->ranks = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    {
        struct dpu_set_t rank_set;
        uint32_t each_rank = 0;
        DPU_RANK_FOREACH(runner->dpu_set, rank_set, each_rank)
        {
            runner->ranks[each_rank] = dpu_rank_from_set(rank_set);
        }
    }
    {
        struct dpu_set_t dpu;
        uint32_t each_dpu = 0;
        DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
        {
            struct dpu_t *dpu_ptr = dpu_from_set(dpu);
            struct dpu_rank_t *rank_ptr = dpu_get_rank(dpu_ptr);
            uint32_t rank_idx = 0;
            runner->physical_dpus[each_dpu] = dpu;
            for (; rank_idx < runner->nr_ranks; ++rank_idx) {
                if (runner->ranks[rank_idx] == rank_ptr) {
                    break;
                }
            }
            if (rank_idx == runner->nr_ranks) {
                fprintf(stderr, "Failed to resolve rank for DPU %u\n", each_dpu);
                free(runner->physical_dpu_rank_indices);
                runner->physical_dpu_rank_indices = NULL;
                free(runner->physical_dpus);
                runner->physical_dpus = NULL;
                free(runner->ranks);
                runner->ranks = NULL;
                dpu_free(runner->dpu_set);
                runner->nr_dpus = 0;
                runner->nr_ranks = 0;
                return 1;
            }
            runner->physical_dpu_rank_indices[each_dpu] = rank_idx;
        }
    }
    runner->slots = calloc((size_t)runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU, sizeof(host_slot_t));
    if (runner->slots == NULL) {
        fprintf(stderr, "Failed to allocate slot table\n");
        free(runner->physical_dpu_rank_indices);
        runner->physical_dpu_rank_indices = NULL;
        free(runner->physical_dpus);
        runner->physical_dpus = NULL;
        free(runner->ranks);
        runner->ranks = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    runner->next_free_elem = calloc(runner->nr_dpus, sizeof(uint32_t));
    if (runner->next_free_elem == NULL) {
        fprintf(stderr, "Failed to allocate next_free_elem table\n");
        free(runner->slots);
        runner->slots = NULL;
        free(runner->physical_dpu_rank_indices);
        runner->physical_dpu_rank_indices = NULL;
        free(runner->physical_dpus);
        runner->physical_dpus = NULL;
        free(runner->ranks);
        runner->ranks = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    runner->free_ranges = calloc((size_t)runner->nr_dpus * kvslot_max_free_ranges_per_dpu(), sizeof(free_range_t));
    if (runner->free_ranges == NULL) {
        fprintf(stderr, "Failed to allocate free_ranges table\n");
        free(runner->next_free_elem);
        runner->next_free_elem = NULL;
        free(runner->slots);
        runner->slots = NULL;
        free(runner->physical_dpu_rank_indices);
        runner->physical_dpu_rank_indices = NULL;
        free(runner->physical_dpus);
        runner->physical_dpus = NULL;
        free(runner->ranks);
        runner->ranks = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    runner->num_free_ranges = calloc(runner->nr_dpus, sizeof(uint32_t));
    if (runner->num_free_ranges == NULL) {
        fprintf(stderr, "Failed to allocate num_free_ranges table\n");
        free(runner->free_ranges);
        runner->free_ranges = NULL;
        free(runner->next_free_elem);
        runner->next_free_elem = NULL;
        free(runner->slots);
        runner->slots = NULL;
        free(runner->physical_dpu_rank_indices);
        runner->physical_dpu_rank_indices = NULL;
        free(runner->physical_dpus);
        runner->physical_dpus = NULL;
        free(runner->ranks);
        runner->ranks = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    return 0;
}

static void runner_destroy(kvslot_runner_t *runner)
{
    if (runner == NULL) {
        return;
    }
    if (runner->slots != NULL) {
        for (uint32_t idx = 0; idx < runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU; ++idx) {
            free_slot(&runner->slots[idx]);
        }
        free(runner->slots);
        runner->slots = NULL;
    }
    free(runner->next_free_elem);
    runner->next_free_elem = NULL;
    free(runner->free_ranges);
    runner->free_ranges = NULL;
    free(runner->num_free_ranges);
    runner->num_free_ranges = NULL;
    free(runner->physical_dpu_rank_indices);
    runner->physical_dpu_rank_indices = NULL;
    free(runner->physical_dpus);
    runner->physical_dpus = NULL;
    free(runner->ranks);
    runner->ranks = NULL;
    if (runner->nr_dpus > 0) {
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
    }
    runner->nr_ranks = 0;
}

static int handle_allocate(kvslot_runner_t *runner, uint32_t slot_id)
{
    struct dpu_set_t target_dpu;
    host_slot_t *slot = NULL;
    uint32_t physical_dpu_id = 0;
    kvslot_slot_args_t args;
    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read allocate args\n");
        return 1;
    }
    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for allocate\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    physical_dpu_id = slot_id % runner->nr_dpus;
    if (ensure_slot(slot, args.capacity, args.group_heads, args.head_dim) != 0) {
        return 1;
    }

    size_t logical_elems = (size_t)args.seq_len * args.group_heads * args.head_dim;
    size_t elem_size = kvslot_dtype_elem_size(args.dtype_code);
    size_t bytes = logical_elems * elem_size;
    uint32_t slot_total_logical_elems = args.capacity * args.group_heads * args.head_dim;
    uint32_t slot_total_elems = kvslot_packed_elem_count(slot_total_logical_elems, args.dtype_code);
    if (args.seq_len > args.capacity) {
        fprintf(stderr, "Initial seq_len exceeds capacity\n");
        return 1;
    }
    void *k_data = NULL;
    void *v_data = NULL;
    if (logical_elems > 0) {
        k_data = calloc(logical_elems, elem_size);
        v_data = calloc(logical_elems, elem_size);
        if (k_data == NULL || v_data == NULL) {
            fprintf(stderr, "Failed to allocate allocate payload buffers\n");
            free(k_data);
            free(v_data);
            return 1;
        }
        if (read_exact(stdin, k_data, bytes) != 0
            || read_exact(stdin, v_data, bytes) != 0) {
            fprintf(stderr, "Failed to read allocate payload\n");
            free(k_data);
            free(v_data);
            return 1;
        }
    }
    slot->dtype_code = args.dtype_code;
    DPU_ASSERT(dpu_broadcast_to(target_dpu, "slot_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    if (reserve_elem_range(runner, physical_dpu_id, slot_total_elems, &slot->elem_offset) != 0) {
        fprintf(stderr, "DPU %u out of reusable kvslot capacity\n", physical_dpu_id);
        free_slot(slot);
        free(k_data);
        free(v_data);
        return 1;
    }
    slot->elem_count = slot_total_elems;
    if (bytes > 0) {
        size_t byte_offset = (size_t)slot->elem_offset * sizeof(int32_t);
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "k_cache", byte_offset, bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "v_cache", byte_offset, bytes, DPU_XFER_DEFAULT));
    }
    slot->seq_len = args.seq_len;
    free(k_data);
    free(v_data);

    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
        .dtype_code = slot->dtype_code,
    };
    if (write_exact(stdout, &out, sizeof(out)) != 0 || flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to write allocate response\n");
        return 1;
    }
    return 0;
}

static int handle_append(kvslot_runner_t *runner, uint32_t slot_id)
{
    struct dpu_set_t target_dpu;
    host_slot_t *slot = NULL;
    kvslot_slot_args_t args;
    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read append args\n");
        return 1;
    }
    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for append\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    if (slot->capacity == 0) {
        fprintf(stderr, "Append on uninitialized slot %u\n", slot_id);
        return 1;
    }
    if (args.seq_len != 1 || args.group_heads != slot->group_heads || args.head_dim != slot->head_dim || args.dtype_code != slot->dtype_code) {
        fprintf(stderr, "Append args mismatch for slot %u\n", slot_id);
        return 1;
    }
    if (slot->seq_len + 1 > slot->capacity) {
        fprintf(stderr, "Slot %u capacity exceeded\n", slot_id);
        return 1;
    }

    size_t token_elems = (size_t)slot->group_heads * slot->head_dim;
    size_t token_elem_size = kvslot_dtype_elem_size(slot->dtype_code);
    size_t token_bytes = token_elems * token_elem_size;
    void *k_token = calloc(token_elems, token_elem_size);
    void *v_token = calloc(token_elems, token_elem_size);
    if (k_token == NULL || v_token == NULL) {
        fprintf(stderr, "Failed to allocate append buffers\n");
        free(k_token);
        free(v_token);
        return 1;
    }
    if (read_exact(stdin, k_token, token_bytes) != 0
        || read_exact(stdin, v_token, token_bytes) != 0) {
        fprintf(stderr, "Failed to read append payload\n");
        free(k_token);
        free(v_token);
        return 1;
    }
    size_t byte_offset = ((size_t)slot->elem_offset * sizeof(int32_t)) + ((size_t)slot->seq_len * token_bytes);
    DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_token));
    DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "k_cache", byte_offset, token_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_token));
    DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "v_cache", byte_offset, token_bytes, DPU_XFER_DEFAULT));
    slot->seq_len += 1;
    free(k_token);
    free(v_token);

    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
        .dtype_code = slot->dtype_code,
    };
    if (write_exact(stdout, &out, sizeof(out)) != 0 || flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to write append response\n");
        return 1;
    }
    return 0;
}

static int handle_readback(kvslot_runner_t *runner, uint32_t slot_id)
{
    struct dpu_set_t target_dpu;
    host_slot_t *slot = NULL;
    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for readback\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    if (slot->capacity == 0) {
        fprintf(stderr, "Readback on uninitialized slot %u\n", slot_id);
        return 1;
    }
    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
        .dtype_code = slot->dtype_code,
    };
    size_t elems = (size_t)slot->seq_len * slot->group_heads * slot->head_dim;
    size_t elem_size = kvslot_dtype_elem_size(slot->dtype_code);
    size_t bytes = elems * elem_size;
    void *k_data = NULL;
    void *v_data = NULL;
    if (elems > 0) {
        k_data = calloc(elems, elem_size);
        v_data = calloc(elems, elem_size);
        if (k_data == NULL || v_data == NULL) {
            fprintf(stderr, "Failed to allocate readback buffers\n");
            free(k_data);
            free(v_data);
            return 1;
        }
        size_t byte_offset = (size_t)slot->elem_offset * sizeof(int32_t);
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "k_cache", byte_offset, bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "v_cache", byte_offset, bytes, DPU_XFER_DEFAULT));
    }
    if (write_exact(stdout, &out, sizeof(out)) != 0
        || (elems > 0 && write_exact(stdout, k_data, bytes) != 0)
        || (elems > 0 && write_exact(stdout, v_data, bytes) != 0)
        || flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to write readback response\n");
        free(k_data);
        free(v_data);
        return 1;
    }
    free(k_data);
    free(v_data);
    return 0;
}

static int handle_free(kvslot_runner_t *runner, uint32_t slot_id)
{
    struct dpu_set_t target_dpu;
    host_slot_t *slot = NULL;
    kvslot_slot_args_t zero_args = {0};
    uint32_t physical_dpu_id = 0;
    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for free\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    physical_dpu_id = slot_id % runner->nr_dpus;
    if (release_elem_range(runner, physical_dpu_id, slot->elem_offset, slot->elem_count) != 0) {
        fprintf(stderr, "Failed to release DPU %u range for slot %u\n", physical_dpu_id, slot_id);
        return 1;
    }
    free_slot(slot);
    DPU_ASSERT(dpu_broadcast_to(target_dpu, "slot_args", 0, &zero_args, sizeof(zero_args), DPU_XFER_DEFAULT));
    return write_exact(stdout, &zero_args, sizeof(zero_args)) == 0 && flush_exact(stdout) == 0 ? 0 : 1;
}

static int handle_get_stats(kvslot_runner_t *runner)
{
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        kvslot_allocator_stats_t stats = collect_allocator_stats(runner, physical_dpu_id);
        if (write_exact(stdout, &stats, sizeof(stats)) != 0) {
            fprintf(stderr, "Failed to write allocator stats for DPU %u\n", physical_dpu_id);
            return 1;
        }
    }
    return flush_exact(stdout);
}

static int handle_qk_batch(kvslot_runner_t *runner)
{
    kvslot_qk_args_t args;
    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read qk batch args\n");
        return 1;
    }
    if (args.head_dim == 0 || args.head_dim > KVSLOT_MAX_HEAD_DIM || (args.head_dim % 2) != 0) {
        fprintf(stderr, "Invalid qk batch head_dim=%u\n", args.head_dim);
        return 1;
    }
    if (args.num_keys == 0 || args.num_keys > KVSLOT_MAX_CAPACITY) {
        fprintf(stderr, "Invalid qk batch num_keys=%u\n", args.num_keys);
        return 1;
    }
    if (args.num_queries == 0 || args.num_queries > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid qk batch num_queries=%u\n", args.num_queries);
        return 1;
    }

    size_t query_elems = (size_t)args.num_queries * args.head_dim;
    size_t key_elems = (size_t)args.num_queries * args.num_keys * args.head_dim;
    size_t score_elems = (size_t)args.num_queries * args.num_keys;
    int32_t *queries_in = calloc(query_elems, sizeof(*queries_in));
    int32_t *keys_in = calloc(key_elems, sizeof(*keys_in));
    int64_t *scores_out = calloc(score_elems, sizeof(*scores_out));
    if (queries_in == NULL || keys_in == NULL || scores_out == NULL) {
        fprintf(stderr, "Failed to allocate qk batch buffers\n");
        free(queries_in);
        free(keys_in);
        free(scores_out);
        return 1;
    }

    if (read_exact(stdin, queries_in, query_elems * sizeof(*queries_in)) != 0
        || read_exact(stdin, keys_in, key_elems * sizeof(*keys_in)) != 0) {
        fprintf(stderr, "Failed to read qk batch payload\n");
        free(queries_in);
        free(keys_in);
        free(scores_out);
        return 1;
    }

    uint32_t active_dpus = runner->nr_dpus;
    if (active_dpus > args.num_keys) {
        active_dpus = args.num_keys;
    }
    if (active_dpus > KVSLOT_QK_MAX_ACTIVE_DPUS) {
        active_dpus = KVSLOT_QK_MAX_ACTIVE_DPUS;
    }
    uint32_t keys_per_dpu = (args.num_keys + active_dpus - 1) / active_dpus;
    if (keys_per_dpu > KVSLOT_MAX_CAPACITY) {
        fprintf(stderr, "Too many keys per DPU after partitioning: %u\n", keys_per_dpu);
        free(queries_in);
        free(keys_in);
        free(scores_out);
        return 1;
    }

    int32_t *queries_partitioned = calloc((size_t)active_dpus * args.head_dim, sizeof(*queries_partitioned));
    int32_t *keys_partitioned = calloc((size_t)active_dpus * keys_per_dpu * args.head_dim, sizeof(*keys_partitioned));
    int64_t *scores_partitioned = calloc((size_t)active_dpus * keys_per_dpu, sizeof(*scores_partitioned));
    kvslot_meta_t *metas = calloc(active_dpus, sizeof(*metas));
    if (queries_partitioned == NULL || keys_partitioned == NULL || scores_partitioned == NULL || metas == NULL) {
        fprintf(stderr, "Failed to allocate qk DPU-partitioned buffers\n");
        free(queries_in);
        free(keys_in);
        free(scores_out);
        free(queries_partitioned);
        free(keys_partitioned);
        free(scores_partitioned);
        free(metas);
        return 1;
    }

    int rc = 0;
    struct dpu_set_t active_dpu_sets[KVSLOT_QK_MAX_ACTIVE_DPUS];
    memset(active_dpu_sets, 0, sizeof(active_dpu_sets));
    {
        struct dpu_set_t dpu;
        uint32_t each_dpu;
        uint32_t selected = 0;
        DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
        {
            if (selected >= active_dpus) {
                break;
            }
            active_dpu_sets[selected++] = dpu;
        }
        if (selected != active_dpus) {
            fprintf(stderr, "Failed to collect enough active DPUs for qk batch\n");
            free(queries_in);
            free(keys_in);
            free(scores_out);
            free(queries_partitioned);
            free(keys_partitioned);
            free(scores_partitioned);
            free(metas);
            return 1;
        }
    }

    for (uint32_t query_idx = 0; query_idx < args.num_queries && rc == 0; ++query_idx) {
        memset(queries_partitioned, 0, (size_t)active_dpus * args.head_dim * sizeof(*queries_partitioned));
        memset(keys_partitioned, 0, (size_t)active_dpus * keys_per_dpu * args.head_dim * sizeof(*keys_partitioned));
        memset(scores_partitioned, 0, (size_t)active_dpus * keys_per_dpu * sizeof(*scores_partitioned));
        memset(metas, 0, (size_t)active_dpus * sizeof(*metas));

        const int32_t *query = &queries_in[(size_t)query_idx * args.head_dim];
        const int32_t *query_keys = &keys_in[(size_t)query_idx * args.num_keys * args.head_dim];

        for (uint32_t dpu_idx = 0; dpu_idx < active_dpus; ++dpu_idx) {
            memcpy(&queries_partitioned[(size_t)dpu_idx * args.head_dim], query, args.head_dim * sizeof(int32_t));
            for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                if (global_key < args.num_keys) {
                    memcpy(
                        &keys_partitioned[((size_t)dpu_idx * keys_per_dpu + local_key) * args.head_dim],
                        &query_keys[(size_t)global_key * args.head_dim],
                        args.head_dim * sizeof(int32_t));
                }
            }
        }

        kvslot_qk_dpu_args_t dpu_args = {
            .head_dim = args.head_dim,
            .num_keys = keys_per_dpu,
            .key_stride = args.head_dim,
            .reserved = 0,
        };
        uint32_t kernel_command = KVSLOT_KERNEL_QK;
        for (uint32_t dpu_idx = 0; dpu_idx < active_dpus; ++dpu_idx) {
            struct dpu_set_t target = active_dpu_sets[dpu_idx];
            DPU_ASSERT(dpu_copy_to(target, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command)));
            DPU_ASSERT(dpu_copy_to(target, "qk_args", 0, &dpu_args, sizeof(dpu_args)));
            DPU_ASSERT(dpu_copy_to(
                target,
                "qk_query",
                0,
                &queries_partitioned[(size_t)dpu_idx * args.head_dim],
                (size_t)args.head_dim * sizeof(int32_t)));
            DPU_ASSERT(dpu_copy_to(
                target,
                "qk_keys",
                0,
                &keys_partitioned[(size_t)dpu_idx * keys_per_dpu * args.head_dim],
                (size_t)keys_per_dpu * args.head_dim * sizeof(int32_t)));
            DPU_ASSERT(dpu_launch(target, DPU_ASYNCHRONOUS));
        }
        for (uint32_t dpu_idx = 0; dpu_idx < active_dpus; ++dpu_idx) {
            struct dpu_set_t target = active_dpu_sets[dpu_idx];
            DPU_ASSERT(dpu_sync(target));
            DPU_ASSERT(dpu_copy_from(
                target,
                "qk_scores",
                0,
                &scores_partitioned[(size_t)dpu_idx * keys_per_dpu],
                (size_t)keys_per_dpu * sizeof(int64_t)));
            DPU_ASSERT(dpu_copy_from(target, "kvslot_meta", 0, &metas[dpu_idx], sizeof(kvslot_meta_t)));
        }

        for (uint32_t dpu_idx = 0; dpu_idx < active_dpus; ++dpu_idx) {
            for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                if (global_key < args.num_keys) {
                    scores_out[(size_t)query_idx * args.num_keys + global_key] =
                        scores_partitioned[(size_t)dpu_idx * keys_per_dpu + local_key];
                }
            }
        }
    }

    if (rc == 0 && (write_exact(stdout, &args, sizeof(args)) != 0
        || write_exact(stdout, scores_out, score_elems * sizeof(*scores_out)) != 0
        || flush_exact(stdout) != 0)) {
        fprintf(stderr, "Failed to write qk batch response\n");
        rc = 1;
    }

    free(queries_in);
    free(keys_in);
    free(scores_out);
    free(queries_partitioned);
    free(keys_partitioned);
    free(scores_partitioned);
    free(metas);
    return rc;
}

static int handle_qk_slot_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    qk_slot_item_t *items = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read qk slot batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid qk slot batch num_slots=%u\n", args.num_slots);
        return 1;
    }

    if (write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write qk slot batch response header\n");
        return 1;
    }

    items = calloc(args.num_slots, sizeof(*items));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (items == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate qk slot batch state\n");
        rc = 1;
        goto cleanup;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        kvslot_qk_slot_batch_item_args_t item_args;
        uint32_t slot_id;
        uint32_t window;
        uint32_t head_dim;
        uint32_t num_heads;
        qk_slot_item_t *item = &items[idx];

        if (read_exact(stdin, &slot_id, sizeof(slot_id)) != 0) {
            fprintf(stderr, "Failed to read qk slot batch slot id %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (read_exact(stdin, &item_args, sizeof(item_args)) != 0) {
            fprintf(stderr, "Failed to read qk slot batch item args %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        num_heads = item_args.num_heads;
        window = item_args.window;
        head_dim = item_args.head_dim;

        {
            kvslot_qk_softmax_av_batch_item_args_t fused_args;
            fused_args.num_heads = num_heads;
            fused_args.window = window;
            fused_args.head_dim = head_dim;
            fused_args.score_scale = 1.0f;
            if (prepare_qk_slot_item_header(runner, slot_id, &fused_args, item) != 0) {
                rc = 1;
                goto cleanup;
            }
            if (read_qk_slot_item_payload(stdin, item) != 0) {
                rc = 1;
                goto cleanup;
            }
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_HEADS];
        uint32_t round_count = 0;

        memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            if (processed[idx]) {
                continue;
            }
            if (used_dpus[items[idx].physical_dpu_id]) {
                continue;
            }
            used_dpus[items[idx].physical_dpu_id] = 1;
            round_indices[round_count++] = idx;
        }
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk slot batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_qk_round(runner, items, round_indices, round_count)) {
            if (execute_batched_qk_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched qk round\n");
                rc = 1;
            }
        } else {
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to launch qk batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to finish qk batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        qk_slot_item_t *item = &items[idx];
        uint32_t item_header[4] = {item->num_heads, item->window, 0, 0};
        if (write_exact(stdout, item_header, sizeof(item_header)) != 0) {
            fprintf(stderr, "Failed to write qk slot batch item header %u\n", idx);
            rc = 1;
            break;
        }
        for (uint32_t head_idx = 0; head_idx < item->num_heads && rc == 0; ++head_idx) {
            for (uint32_t pos = 0; pos < item->window; ++pos) {
                union {
                    uint32_t u;
                    float f;
                } bits = {.u = item->raw_scores[(size_t)head_idx * item->score_stride + pos]};
                if (write_exact(stdout, &bits.f, sizeof(bits.f)) != 0) {
                    fprintf(stderr, "Failed to write qk slot batch score %u:%u:%u\n", idx, head_idx, pos);
                    rc = 1;
                    break;
                }
            }
        }
    }

cleanup:
    if (items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_qk_slot_item(&items[idx]);
        }
    }
    free(items);
    free(processed);
    free(used_dpus);

    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush qk slot batch response\n");
        rc = 1;
    }
    return rc;
}

static int prepare_qk_slot_item_header(
    kvslot_runner_t *runner,
    uint32_t slot_id,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    qk_slot_item_t *item)
{
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;

    if (runner == NULL || item_args == NULL || item == NULL) {
        return 1;
    }
    memset(item, 0, sizeof(*item));
    num_heads = item_args->num_heads;
    window = item_args->window;
    head_dim = item_args->head_dim;

    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for qk slot batch\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &item->target_dpu, &item->slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    if (item->slot->capacity == 0) {
        fprintf(stderr, "QK slot batch on uninitialized slot %u\n", slot_id);
        return 1;
    }
    if (num_heads == 0 || num_heads > item->slot->group_heads || num_heads > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid qk slot batch num_heads=%u for slot %u\n", num_heads, slot_id);
        return 1;
    }
    if (head_dim == 0 || head_dim > item->slot->head_dim || head_dim > KVSLOT_MAX_HEAD_DIM) {
        fprintf(stderr, "Invalid qk slot batch head_dim=%u for slot %u\n", head_dim, slot_id);
        return 1;
    }
    if (window > item->slot->seq_len) {
        window = item->slot->seq_len;
    }

    item->slot_id = slot_id;
    item->physical_dpu_id = slot_id % runner->nr_dpus;
    item->num_heads = num_heads;
    item->window = window;
    item->score_stride = (window + 1u) & ~1u;
    item->head_dim = head_dim;
    item->local_head_indices = calloc(num_heads, sizeof(*item->local_head_indices));
    item->queries = calloc((size_t)num_heads * head_dim, sizeof(*item->queries));
    item->raw_scores = calloc(
        (size_t)(item->score_stride > 0 ? item->score_stride : 1u) * num_heads,
        sizeof(*item->raw_scores)
    );
    item->raw_row_max_bits = calloc(num_heads > 0 ? num_heads : 1u, sizeof(*item->raw_row_max_bits));
    if (item->local_head_indices == NULL || item->queries == NULL || item->raw_scores == NULL || item->raw_row_max_bits == NULL) {
        fprintf(stderr, "Failed to allocate qk slot batch buffers\n");
        return 1;
    }

    item->runtime_args.seq_len = item->slot->seq_len;
    item->runtime_args.group_heads = item->slot->group_heads;
    item->runtime_args.head_dim = item->slot->head_dim;
    item->runtime_args.dtype_code = item->slot->dtype_code;
    item->runtime_args.elem_offset = item->slot->elem_offset;
    item->runtime_args.reserved[0] = 0;
    item->runtime_args.reserved[1] = 0;
    item->runtime_args.reserved[2] = 0;

    item->slot_args.num_heads = num_heads;
    item->slot_args.window = window;
    item->slot_args.head_dim = head_dim;
    item->slot_args.mode = KVSLOT_QK_SLOT_MODE_RAW_SCORES;
    item->slot_args.score_scale = item_args->score_scale;
    item->ready = 1;
    return 0;
}

static int read_qk_slot_item_payload(FILE *file, qk_slot_item_t *item)
{
    if (file == NULL || item == NULL || !item->ready) {
        return 1;
    }
    if (read_exact(file, item->local_head_indices, (size_t)item->num_heads * sizeof(*item->local_head_indices)) != 0
        || read_exact(file, item->queries, (size_t)item->num_heads * item->head_dim * sizeof(*item->queries)) != 0) {
        fprintf(stderr, "Failed to read qk slot batch payload\n");
        return 1;
    }
    for (uint32_t head_idx = 0; head_idx < item->num_heads; ++head_idx) {
        if (item->local_head_indices[head_idx] >= item->slot->group_heads) {
            fprintf(stderr, "Invalid local head idx %u for slot %u at row %u\n", item->local_head_indices[head_idx], item->slot_id, head_idx);
            return 1;
        }
    }
    return 0;
}

static void cleanup_qk_slot_item(qk_slot_item_t *item)
{
    if (item == NULL) {
        return;
    }
    free(item->local_head_indices);
    free(item->queries);
    free(item->raw_scores);
    free(item->raw_row_max_bits);
    memset(item, 0, sizeof(*item));
}

static int launch_qk_slot_item_async(const qk_slot_item_t *item)
{
    uint32_t kernel_command = KVSLOT_KERNEL_QK_SLOT;

    if (item == NULL || !item->ready) {
        return 1;
    }
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "runtime_slot_args", 0, &item->runtime_args, sizeof(item->runtime_args)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "qk_slot_args", 0, &item->slot_args, sizeof(item->slot_args)));
    DPU_ASSERT(dpu_copy_to(
        item->target_dpu,
        "qk_slot_head_indices",
        0,
        item->local_head_indices,
        (size_t)item->num_heads * sizeof(*item->local_head_indices)
    ));
    DPU_ASSERT(dpu_copy_to(
        item->target_dpu,
        "qk_query",
        0,
        item->queries,
        (size_t)item->num_heads * item->head_dim * sizeof(*item->queries)
    ));
    DPU_ASSERT(dpu_launch(item->target_dpu, DPU_ASYNCHRONOUS));
    return 0;
}

static int finish_qk_slot_item(qk_slot_item_t *item)
{
    size_t score_bytes;
    size_t aligned_score_bytes;
    uint32_t *aligned_scores = NULL;

    if (item == NULL || !item->ready) {
        return 1;
    }
    DPU_ASSERT(dpu_sync(item->target_dpu));
    if (item->slot_args.mode == KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
        return 0;
    }
    if (item->window > 0) {
        score_bytes = (size_t)item->num_heads * item->score_stride * sizeof(*item->raw_scores);
        aligned_score_bytes = (score_bytes + 7u) & ~((size_t)7u);
        if (aligned_score_bytes == score_bytes) {
            DPU_ASSERT(dpu_copy_from(
                item->target_dpu,
                "qk_slot_scores_bits",
                0,
                item->raw_scores,
                score_bytes
            ));
            return 0;
        }
        aligned_scores = calloc(1, aligned_score_bytes);
        if (aligned_scores == NULL) {
            fprintf(stderr, "Failed to allocate aligned qk slot score buffer (%zu bytes)\n", aligned_score_bytes);
            return 1;
        }
        DPU_ASSERT(dpu_copy_from(
            item->target_dpu,
            "qk_slot_scores_bits",
            0,
            aligned_scores,
            aligned_score_bytes
        ));
        memcpy(item->raw_scores, aligned_scores, score_bytes);
        free(aligned_scores);
    }
    if (fetch_qk_slot_row_maxes(item) != 0) {
        return 1;
    }
    return 0;
}

static int fetch_qk_slot_row_maxes(qk_slot_item_t *item)
{
    if (item == NULL || !item->ready || item->raw_row_max_bits == NULL) {
        return 1;
    }
    if (item->num_heads == 0) {
        return 0;
    }
    DPU_ASSERT(dpu_copy_from(
        item->target_dpu,
        "qk_slot_rowmax_bits",
        0,
        item->raw_row_max_bits,
        (size_t)item->num_heads * sizeof(*item->raw_row_max_bits)
    ));
    return 0;
}

static int can_use_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    uint32_t mode;
    uint32_t active_rank_count = 0;
    size_t query_bytes;
    size_t score_bytes;
    uint8_t *rank_used = NULL;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count <= 1) {
        return 0;
    }
    if (runner->nr_ranks == 0 || runner->physical_dpu_rank_indices == NULL) {
        return 0;
    }

    num_heads = items[round_indices[0]].num_heads;
    window = items[round_indices[0]].window;
    head_dim = items[round_indices[0]].head_dim;
    mode = items[round_indices[0]].slot_args.mode;
    query_bytes = (size_t)num_heads * head_dim * sizeof(float);
    score_bytes = (size_t)num_heads * items[round_indices[0]].score_stride * sizeof(uint32_t);

    if ((query_bytes % 8u) != 0 || (score_bytes % 8u) != 0) {
        return 0;
    }

    rank_used = calloc(runner->nr_ranks, sizeof(*rank_used));
    if (rank_used == NULL) {
        return 0;
    }
    for (uint32_t pos = 1; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        if (!item->ready || item->num_heads != num_heads || item->window != window || item->head_dim != head_dim) {
            free(rank_used);
            return 0;
        }
        if (item->slot_args.mode != mode) {
            free(rank_used);
            return 0;
        }
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        uint32_t rank_idx;
        if (item->physical_dpu_id >= runner->nr_dpus) {
            free(rank_used);
            return 0;
        }
        rank_idx = runner->physical_dpu_rank_indices[item->physical_dpu_id];
        if (rank_idx >= runner->nr_ranks) {
            free(rank_used);
            return 0;
        }
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    free(rank_used);
    if (active_rank_count > 16u) {
        return 0;
    }
    return 1;
}

static int execute_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    qk_slot_item_t **items_by_dpu = NULL;
    uint32_t *dummy_head_indices = NULL;
    float *dummy_queries = NULL;
    uint32_t *dummy_scores = NULL;
    uint32_t *dummy_row_max_bits = NULL;
    kvslot_runtime_slot_args_t zero_runtime_args;
    kvslot_qk_slot_args_t zero_slot_args;
    uint32_t kernel_command = KVSLOT_KERNEL_QK_SLOT;
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    uint32_t active_rank_count = 0;
    size_t head_index_bytes;
    size_t query_bytes;
    size_t score_bytes;
    uint8_t *rank_used = NULL;
    struct dpu_rank_t **active_ranks = NULL;
    struct dpu_set_t launch_set;
    struct dpu_set_t dpu;

    if (!can_use_batched_qk_round(runner, items, round_indices, round_count)) {
        return 1;
    }

    items_by_dpu = calloc(runner->nr_dpus, sizeof(*items_by_dpu));
    if (items_by_dpu == NULL) {
        fprintf(stderr, "Failed to allocate batched qk round map\n");
        return 1;
    }

    memset(&zero_runtime_args, 0, sizeof(zero_runtime_args));
    memset(&zero_slot_args, 0, sizeof(zero_slot_args));
    num_heads = items[round_indices[0]].num_heads;
    window = items[round_indices[0]].window;
    head_dim = items[round_indices[0]].head_dim;
    head_index_bytes = (size_t)num_heads * sizeof(uint32_t);
    query_bytes = (size_t)num_heads * head_dim * sizeof(float);
    score_bytes = (size_t)num_heads * window * sizeof(uint32_t);
    score_bytes = (size_t)num_heads * items[round_indices[0]].score_stride * sizeof(uint32_t);
    memset(&launch_set, 0, sizeof(launch_set));

    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        if (item->physical_dpu_id >= runner->nr_dpus) {
            fprintf(stderr, "Invalid physical dpu id %u in batched qk round\n", item->physical_dpu_id);
            free(items_by_dpu);
            return 1;
        }
        items_by_dpu[item->physical_dpu_id] = item;
    }
    rank_used = calloc(runner->nr_ranks, sizeof(*rank_used));
    if (rank_used == NULL) {
        fprintf(stderr, "Failed to allocate batched qk rank mask\n");
        free(items_by_dpu);
        return 1;
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t rank_idx = runner->physical_dpu_rank_indices[items[round_indices[pos]].physical_dpu_id];
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    active_ranks = calloc(active_rank_count > 0 ? active_rank_count : 1u, sizeof(*active_ranks));
    if (active_ranks == NULL) {
        fprintf(stderr, "Failed to allocate batched qk active ranks\n");
        free(rank_used);
        free(items_by_dpu);
        return 1;
    }
    if (active_rank_count == runner->nr_ranks) {
        launch_set = runner->dpu_set;
    } else {
        uint32_t out_rank = 0;
        for (uint32_t rank_idx = 0; rank_idx < runner->nr_ranks; ++rank_idx) {
            if (rank_used[rank_idx]) {
                active_ranks[out_rank++] = runner->ranks[rank_idx];
            }
        }
        launch_set.kind = DPU_SET_RANKS;
        launch_set.list.nr_ranks = active_rank_count;
        launch_set.list.ranks = active_ranks;
    }

    if (head_index_bytes > 0) {
        dummy_head_indices = calloc(1, head_index_bytes);
        if (dummy_head_indices == NULL) {
            fprintf(stderr, "Failed to allocate batched qk dummy head indices\n");
            free(active_ranks);
            free(rank_used);
            free(items_by_dpu);
            return 1;
        }
    }
    if (query_bytes > 0) {
        dummy_queries = calloc(1, query_bytes);
        if (dummy_queries == NULL) {
            fprintf(stderr, "Failed to allocate batched qk dummy queries\n");
            free(dummy_head_indices);
            free(active_ranks);
            free(rank_used);
            free(items_by_dpu);
            return 1;
        }
    }
    if (score_bytes > 0) {
        dummy_scores = calloc(1, score_bytes);
        if (dummy_scores == NULL) {
            fprintf(stderr, "Failed to allocate batched qk dummy scores\n");
            free(dummy_queries);
            free(dummy_head_indices);
            free(active_ranks);
            free(rank_used);
            free(items_by_dpu);
            return 1;
        }
    }
    if (num_heads > 0) {
        dummy_row_max_bits = calloc(num_heads, sizeof(*dummy_row_max_bits));
        if (dummy_row_max_bits == NULL) {
            fprintf(stderr, "Failed to allocate batched qk dummy row maxes\n");
            free(dummy_scores);
            free(dummy_queries);
            free(dummy_head_indices);
            free(active_ranks);
            free(rank_used);
            free(items_by_dpu);
            return 1;
        }
    }

    DPU_ASSERT(dpu_broadcast_to(launch_set, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command), DPU_XFER_DEFAULT));

    DPU_FOREACH(launch_set, dpu) {
        qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
        DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)&item->runtime_args : (void *)&zero_runtime_args));
    }
    DPU_ASSERT(dpu_push_xfer(
        launch_set,
        DPU_XFER_TO_DPU,
        "runtime_slot_args",
        0,
        sizeof(zero_runtime_args),
        DPU_XFER_DEFAULT));

    DPU_FOREACH(launch_set, dpu) {
        qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
        DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)&item->slot_args : (void *)&zero_slot_args));
    }
    DPU_ASSERT(dpu_push_xfer(
        launch_set,
        DPU_XFER_TO_DPU,
        "qk_slot_args",
        0,
        sizeof(zero_slot_args),
        DPU_XFER_DEFAULT));

    if (head_index_bytes > 0) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->local_head_indices : (void *)dummy_head_indices));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "qk_slot_head_indices",
            0,
            head_index_bytes,
            DPU_XFER_DEFAULT));
    }

    if (query_bytes > 0) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->queries : (void *)dummy_queries));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "qk_query",
            0,
            query_bytes,
            DPU_XFER_DEFAULT));
    }

    DPU_ASSERT(dpu_launch(launch_set, DPU_SYNCHRONOUS));

    if (score_bytes > 0 && items[round_indices[0]].slot_args.mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->raw_scores : (void *)dummy_scores));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "qk_slot_scores_bits",
            0,
            score_bytes,
            DPU_XFER_DEFAULT));
    }
    if (num_heads > 0 && items[round_indices[0]].slot_args.mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->raw_row_max_bits : (void *)dummy_row_max_bits));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "qk_slot_rowmax_bits",
            0,
            (size_t)num_heads * sizeof(*dummy_row_max_bits),
            DPU_XFER_DEFAULT));
    }

    free(active_ranks);
    free(rank_used);
    free(dummy_row_max_bits);
    free(dummy_scores);
    free(dummy_queries);
    free(dummy_head_indices);
    free(items_by_dpu);
    return 0;
}

static qk_slot_item_t *find_qk_round_item_for_dpu(
    kvslot_runner_t *runner,
    qk_slot_item_t **items_by_dpu,
    struct dpu_set_t dpu)
{
    struct dpu_t *target_ptr;
    if (runner == NULL || items_by_dpu == NULL || runner->physical_dpus == NULL) {
        return NULL;
    }
    target_ptr = dpu_from_set(dpu);
    if (target_ptr == NULL) {
        return NULL;
    }
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        if (items_by_dpu[physical_dpu_id] == NULL) {
            continue;
        }
        if (dpu_from_set(runner->physical_dpus[physical_dpu_id]) == target_ptr) {
            return items_by_dpu[physical_dpu_id];
        }
    }
    return NULL;
}

static void cleanup_av_item(av_item_t *item)
{
    if (item == NULL) {
        return;
    }
    free(item->weights);
    free(item->context);
    memset(item, 0, sizeof(*item));
}

static int prepare_av_item_header(kvslot_runner_t *runner, uint32_t slot_id, av_item_t *item)
{
    host_slot_t *slot = NULL;

    if (runner == NULL || item == NULL) {
        return 1;
    }
    memset(item, 0, sizeof(*item));

    if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
        fprintf(stderr, "Invalid slot id %u for av\n", slot_id);
        return 1;
    }
    if (runner_get_dpu_and_slot(runner, slot_id, &item->target_dpu, &slot) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    if (slot->capacity == 0) {
        fprintf(stderr, "AV on uninitialized slot %u\n", slot_id);
        return 1;
    }

    item->slot_id = slot_id;
    item->slot = slot;
    item->physical_dpu_id = slot_id % runner->nr_dpus;
    item->weight_bytes = (size_t)slot->seq_len * slot->group_heads * sizeof(float);
    item->context_bytes = (size_t)slot->group_heads * slot->head_dim * sizeof(float);
    item->padded_weight_bytes = ((item->weight_bytes + 7u) / 8u) * 8u;
    item->padded_context_bytes = ((item->context_bytes + 7u) / 8u) * 8u;

    if (item->weight_bytes > 0) {
        item->weights = calloc(1, item->padded_weight_bytes);
        if (item->weights == NULL) {
            fprintf(stderr, "Failed to allocate av weights buffer\n");
            return 1;
        }
    }

    if (item->context_bytes > 0) {
        item->context = calloc(1, item->padded_context_bytes);
        if (item->context == NULL) {
            fprintf(stderr, "Failed to allocate av context buffer\n");
            return 1;
        }
    }

    item->runtime_args.seq_len = slot->seq_len;
    item->runtime_args.group_heads = slot->group_heads;
    item->runtime_args.head_dim = slot->head_dim;
    item->runtime_args.dtype_code = slot->dtype_code;
    item->runtime_args.elem_offset = slot->elem_offset;
    item->runtime_args.reserved[0] = 0;
    item->runtime_args.reserved[1] = 0;
    item->runtime_args.reserved[2] = 0;

    item->out.capacity = slot->capacity;
    item->out.seq_len = slot->seq_len;
    item->out.group_heads = slot->group_heads;
    item->out.head_dim = slot->head_dim;
    item->out.dtype_code = slot->dtype_code;
    item->ready = 1;
    return 0;
}

static int read_av_item_weights(av_item_t *item)
{
    if (item == NULL || !item->ready) {
        return 1;
    }
    if (item->weights_resident_on_dpu) {
        return 0;
    }
    if (item->weight_bytes == 0) {
        return 0;
    }
    if (read_exact(stdin, item->weights, item->weight_bytes) != 0) {
        fprintf(stderr, "Failed to read av weights payload\n");
        return 1;
    }
    return 0;
}

static int softmax_av_item_scores_inplace(av_item_t *item)
{
    uint32_t group_heads;
    uint32_t seq_len;

    if (item == NULL || !item->ready) {
        return 1;
    }
    group_heads = item->slot->group_heads;
    seq_len = item->slot->seq_len;
    for (uint32_t head = 0; head < group_heads; ++head) {
        float row_max;
        float row_sum = 0.0f;
        size_t base = (size_t)head * seq_len;
        if (seq_len == 0) {
            continue;
        }
        row_max = item->weights[base];
        for (uint32_t pos = 1; pos < seq_len; ++pos) {
            float value = item->weights[base + pos];
            if (value > row_max) {
                row_max = value;
            }
        }
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            float exp_value = expf(item->weights[base + pos] - row_max);
            item->weights[base + pos] = exp_value;
            row_sum += exp_value;
        }
        if (row_sum <= 0.0f) {
            fprintf(stderr, "Invalid softmax row sum for slot %u head %u\n", item->slot_id, head);
            return 1;
        }
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            item->weights[base + pos] /= row_sum;
        }
    }
    return 0;
}

static int softmax_av_item_from_qk_scores(
    const qk_slot_item_t *qk_item,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    av_item_t *av_item)
{
    (void)item_args;
    if (qk_item == NULL || item_args == NULL || av_item == NULL || !qk_item->ready || !av_item->ready) {
        return 1;
    }
    if (av_item->slot != qk_item->slot) {
        fprintf(stderr, "Mismatched slot between qk and av items for slot %u\n", qk_item->slot_id);
        return 1;
    }
    if (av_item->slot->group_heads != qk_item->num_heads) {
        fprintf(stderr, "QK/AV head count mismatch for slot %u: qk=%u av=%u\n", qk_item->slot_id, qk_item->num_heads, av_item->slot->group_heads);
        return 1;
    }
    if (av_item->slot->seq_len != qk_item->window) {
        fprintf(stderr, "QK/AV window mismatch for slot %u: qk=%u av=%u\n", qk_item->slot_id, qk_item->window, av_item->slot->seq_len);
        return 1;
    }
    if (qk_item->slot_args.mode != KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
        fprintf(stderr, "QK-softmax-av item %u did not request normalized DPU weights\n", qk_item->slot_id);
        return 1;
    }
    av_item->weights_resident_on_dpu = 1;
    return 0;
}

static int prepare_av_item(kvslot_runner_t *runner, uint32_t slot_id, av_item_t *item)
{
    if (prepare_av_item_header(runner, slot_id, item) != 0) {
        return 1;
    }
    if (read_av_item_weights(item) != 0) {
        return 1;
    }
    return 0;
}

static int launch_av_item_async(const av_item_t *item)
{
    uint32_t kernel_command = KVSLOT_KERNEL_AV;

    if (item == NULL || !item->ready) {
        return 1;
    }
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "runtime_slot_args", 0, &item->runtime_args, sizeof(item->runtime_args)));
    if (item->weight_bytes > 0 && !item->weights_resident_on_dpu) {
        DPU_ASSERT(dpu_copy_to(item->target_dpu, "av_weights_bits", 0, item->weights, item->padded_weight_bytes));
    }
    DPU_ASSERT(dpu_launch(item->target_dpu, DPU_ASYNCHRONOUS));
    return 0;
}

static int finish_av_item(av_item_t *item)
{
    if (item == NULL || !item->ready) {
        return 1;
    }
    DPU_ASSERT(dpu_sync(item->target_dpu));
    if (item->context_bytes > 0) {
        DPU_ASSERT(dpu_copy_from(item->target_dpu, "av_context_bits", 0, item->context, item->padded_context_bytes));
    }
    return 0;
}

static int can_use_batched_av_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    size_t padded_weight_bytes;
    size_t padded_context_bytes;
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count <= 1) {
        return 0;
    }
    if (runner->nr_ranks == 0 || runner->physical_dpu_rank_indices == NULL) {
        return 0;
    }

    padded_weight_bytes = items[round_indices[0]].padded_weight_bytes;
    padded_context_bytes = items[round_indices[0]].padded_context_bytes;
    for (uint32_t pos = 1; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        if (!item->ready || item->padded_weight_bytes != padded_weight_bytes || item->padded_context_bytes != padded_context_bytes) {
            return 0;
        }
        if (item->weights_resident_on_dpu != items[round_indices[0]].weights_resident_on_dpu) {
            return 0;
        }
    }

    rank_used = calloc(runner->nr_ranks, sizeof(*rank_used));
    if (rank_used == NULL) {
        return 0;
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        uint32_t rank_idx;
        if (item->physical_dpu_id >= runner->nr_dpus) {
            free(rank_used);
            return 0;
        }
        rank_idx = runner->physical_dpu_rank_indices[item->physical_dpu_id];
        if (rank_idx >= runner->nr_ranks) {
            free(rank_used);
            return 0;
        }
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    free(rank_used);
    if (active_rank_count > KVSLOT_QK_MAX_ACTIVE_DPUS) {
        return 0;
    }
    return 1;
}

static int execute_batched_av_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    av_item_t **items_by_dpu = NULL;
    float *dummy_weights = NULL;
    float *dummy_context = NULL;
    kvslot_runtime_slot_args_t zero_runtime_args;
    size_t padded_weight_bytes;
    size_t padded_context_bytes;
    uint32_t kernel_command = KVSLOT_KERNEL_AV;
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;
    struct dpu_rank_t **active_ranks = NULL;
    struct dpu_set_t launch_set;
    struct dpu_set_t dpu;

    if (!can_use_batched_av_round(runner, items, round_indices, round_count)) {
        return 1;
    }

    items_by_dpu = calloc(runner->nr_dpus, sizeof(*items_by_dpu));
    if (items_by_dpu == NULL) {
        fprintf(stderr, "Failed to allocate batched av round map\n");
        return 1;
    }
    memset(&zero_runtime_args, 0, sizeof(zero_runtime_args));
    padded_weight_bytes = items[round_indices[0]].padded_weight_bytes;
    padded_context_bytes = items[round_indices[0]].padded_context_bytes;
    memset(&launch_set, 0, sizeof(launch_set));

    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        if (item->physical_dpu_id >= runner->nr_dpus) {
            fprintf(stderr, "Invalid physical dpu id %u in batched av round\n", item->physical_dpu_id);
            free(items_by_dpu);
            return 1;
        }
        items_by_dpu[item->physical_dpu_id] = item;
    }

    rank_used = calloc(runner->nr_ranks, sizeof(*rank_used));
    if (rank_used == NULL) {
        fprintf(stderr, "Failed to allocate batched av rank mask\n");
        free(items_by_dpu);
        return 1;
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t rank_idx = runner->physical_dpu_rank_indices[items[round_indices[pos]].physical_dpu_id];
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    active_ranks = calloc(active_rank_count > 0 ? active_rank_count : 1u, sizeof(*active_ranks));
    if (active_ranks == NULL) {
        fprintf(stderr, "Failed to allocate batched av active ranks\n");
        free(rank_used);
        free(items_by_dpu);
        return 1;
    }
    if (active_rank_count == runner->nr_ranks) {
        launch_set = runner->dpu_set;
    } else {
        uint32_t out_rank = 0;
        for (uint32_t rank_idx = 0; rank_idx < runner->nr_ranks; ++rank_idx) {
            if (rank_used[rank_idx]) {
                active_ranks[out_rank++] = runner->ranks[rank_idx];
            }
        }
        launch_set.kind = DPU_SET_RANKS;
        launch_set.list.nr_ranks = active_rank_count;
        launch_set.list.ranks = active_ranks;
    }

    if (padded_weight_bytes > 0 && !items[round_indices[0]].weights_resident_on_dpu) {
        dummy_weights = calloc(1, padded_weight_bytes);
        if (dummy_weights == NULL) {
            fprintf(stderr, "Failed to allocate batched av dummy weights\n");
            free(active_ranks);
            free(rank_used);
            free(items_by_dpu);
            return 1;
        }
    }
    if (padded_context_bytes > 0) {
        dummy_context = calloc(1, padded_context_bytes);
        if (dummy_context == NULL) {
            fprintf(stderr, "Failed to allocate batched av dummy context\n");
            free(active_ranks);
            free(rank_used);
            free(dummy_weights);
            free(items_by_dpu);
            return 1;
        }
    }

    DPU_ASSERT(dpu_broadcast_to(launch_set, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command), DPU_XFER_DEFAULT));

    DPU_FOREACH(launch_set, dpu) {
        av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
        DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)&item->runtime_args : (void *)&zero_runtime_args));
    }
    DPU_ASSERT(dpu_push_xfer(
        launch_set,
        DPU_XFER_TO_DPU,
        "runtime_slot_args",
        0,
        sizeof(zero_runtime_args),
        DPU_XFER_DEFAULT));

    if (padded_weight_bytes > 0 && !items[round_indices[0]].weights_resident_on_dpu) {
        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->weights : (void *)dummy_weights));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "av_weights_bits",
            0,
            padded_weight_bytes,
            DPU_XFER_DEFAULT));
    }

    DPU_ASSERT(dpu_launch(launch_set, DPU_SYNCHRONOUS));

    if (padded_context_bytes > 0) {
        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->context : (void *)dummy_context));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "av_context_bits",
            0,
            padded_context_bytes,
            DPU_XFER_DEFAULT));
    }

    free(active_ranks);
    free(rank_used);
    free(dummy_context);
    free(dummy_weights);
    free(items_by_dpu);
    return 0;
}

static av_item_t *find_av_round_item_for_dpu(
    kvslot_runner_t *runner,
    av_item_t **items_by_dpu,
    struct dpu_set_t dpu)
{
    struct dpu_t *target_ptr;
    if (runner == NULL || items_by_dpu == NULL || runner->physical_dpus == NULL) {
        return NULL;
    }
    target_ptr = dpu_from_set(dpu);
    if (target_ptr == NULL) {
        return NULL;
    }
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        if (items_by_dpu[physical_dpu_id] == NULL) {
            continue;
        }
        if (dpu_from_set(runner->physical_dpus[physical_dpu_id]) == target_ptr) {
            return items_by_dpu[physical_dpu_id];
        }
    }
    return NULL;
}

static int write_av_item_response(const av_item_t *item)
{
    if (item == NULL || !item->ready) {
        return 1;
    }
    if (write_exact(stdout, &item->out, sizeof(item->out)) != 0) {
        fprintf(stderr, "Failed to write av response header\n");
        return 1;
    }
    if (item->context_bytes > 0 && write_exact(stdout, item->context, item->context_bytes) != 0) {
        fprintf(stderr, "Failed to write av response payload\n");
        return 1;
    }
    return 0;
}

static int handle_av(kvslot_runner_t *runner, uint32_t slot_id)
{
    av_item_t item;
    int rc = 0;

    if (prepare_av_item(runner, slot_id, &item) != 0) {
        cleanup_av_item(&item);
        return 1;
    }
    if (launch_av_item_async(&item) != 0 || finish_av_item(&item) != 0 || write_av_item_response(&item) != 0 || flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to execute av for slot %u\n", slot_id);
        rc = 1;
    }
    cleanup_av_item(&item);
    return rc;
}

static int handle_av_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    av_item_t *items = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read av batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid av batch num_slots=%u\n", args.num_slots);
        return 1;
    }

    items = calloc(args.num_slots, sizeof(*items));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (items == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate av batch state\n");
        rc = 1;
        goto cleanup;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        kvslot_io_header_t header;
        if (read_exact(stdin, &header, sizeof(header)) != 0) {
            fprintf(stderr, "Failed to read av batch item header %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (header.magic != KVSLOT_MAGIC || header.command != KVSLOT_CMD_AV) {
            fprintf(stderr, "Invalid av batch item header %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (prepare_av_item(runner, header.slot_id, &items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare av batch item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_HEADS];
        uint32_t round_count = 0;

        memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            if (processed[idx]) {
                continue;
            }
            if (used_dpus[items[idx].physical_dpu_id]) {
                continue;
            }
            used_dpus[items[idx].physical_dpu_id] = 1;
            round_indices[round_count++] = idx;
        }
        if (round_count == 0) {
            fprintf(stderr, "Failed to build av batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_av_round(runner, items, round_indices, round_count)) {
            if (execute_batched_av_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched av round\n");
                rc = 1;
            }
        } else {
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to launch av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to finish av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (rc == 0 && write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write av batch response header\n");
        rc = 1;
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (write_av_item_response(&items[idx]) != 0) {
            fprintf(stderr, "Failed to write av batch item %u\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush av batch response\n");
        rc = 1;
    }

cleanup:
    if (items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_av_item(&items[idx]);
        }
    }
    free(items);
    free(processed);
    free(used_dpus);
    return rc;
}

static int handle_softmax_av_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    av_item_t *items = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read softmax av batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid softmax av batch num_slots=%u\n", args.num_slots);
        return 1;
    }

    items = calloc(args.num_slots, sizeof(*items));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (items == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate softmax av batch state\n");
        rc = 1;
        goto cleanup;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        kvslot_io_header_t header;
        if (read_exact(stdin, &header, sizeof(header)) != 0) {
            fprintf(stderr, "Failed to read softmax av batch item header %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (header.magic != KVSLOT_MAGIC || header.command != KVSLOT_CMD_SOFTMAX_AV_BATCH) {
            fprintf(stderr, "Invalid softmax av batch item header %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (prepare_av_item_header(runner, header.slot_id, &items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare softmax av batch item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (read_av_item_weights(&items[idx]) != 0) {
            fprintf(stderr, "Failed to read softmax av batch scores %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (softmax_av_item_scores_inplace(&items[idx]) != 0) {
            fprintf(stderr, "Failed to softmax softmax av batch item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_HEADS];
        uint32_t round_count = 0;

        memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            if (processed[idx]) {
                continue;
            }
            if (used_dpus[items[idx].physical_dpu_id]) {
                continue;
            }
            used_dpus[items[idx].physical_dpu_id] = 1;
            round_indices[round_count++] = idx;
        }
        if (round_count == 0) {
            fprintf(stderr, "Failed to build softmax av batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_av_round(runner, items, round_indices, round_count)) {
            if (execute_batched_av_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched softmax av round\n");
                rc = 1;
            }
        } else {
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to launch softmax av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to finish softmax av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (rc == 0 && write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write softmax av batch response header\n");
        rc = 1;
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (write_av_item_response(&items[idx]) != 0) {
            fprintf(stderr, "Failed to write softmax av batch item %u\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush softmax av batch response\n");
        rc = 1;
    }

cleanup:
    if (items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_av_item(&items[idx]);
        }
    }
    free(items);
    free(processed);
    free(used_dpus);
    return rc;
}

static int handle_qk_softmax_av_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    qk_slot_item_t *qk_items = NULL;
    av_item_t *av_items = NULL;
    kvslot_qk_softmax_av_batch_item_args_t *item_args = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read qk-softmax-av batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid qk-softmax-av batch num_slots=%u\n", args.num_slots);
        return 1;
    }

    qk_items = calloc(args.num_slots, sizeof(*qk_items));
    av_items = calloc(args.num_slots, sizeof(*av_items));
    item_args = calloc(args.num_slots, sizeof(*item_args));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (qk_items == NULL || av_items == NULL || item_args == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate qk-softmax-av batch state\n");
        rc = 1;
        goto cleanup;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        kvslot_qk_softmax_av_batch_item_args_t current_args;
        uint32_t slot_id;
        if (read_exact(stdin, &slot_id, sizeof(slot_id)) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av slot id %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (read_exact(stdin, &current_args, sizeof(current_args)) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av item args %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        item_args[idx] = current_args;
        if (prepare_qk_slot_item_header(runner, slot_id, &current_args, &qk_items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare qk-softmax-av qk item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        qk_items[idx].slot_args.mode = KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED;
        if (read_qk_slot_item_payload(stdin, &qk_items[idx]) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av payload %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (prepare_av_item_header(runner, slot_id, &av_items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare qk-softmax-av av item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (av_items[idx].slot->seq_len != qk_items[idx].window || av_items[idx].slot->group_heads != qk_items[idx].num_heads) {
            fprintf(stderr, "QK-softmax-av slot shape mismatch at item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_HEADS];
        uint32_t round_count = 0;

        memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            if (processed[idx]) {
                continue;
            }
            if (used_dpus[qk_items[idx].physical_dpu_id]) {
                continue;
            }
            used_dpus[qk_items[idx].physical_dpu_id] = 1;
            round_indices[round_count++] = idx;
        }
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk-softmax-av qk round\n");
            rc = 1;
            break;
        }
        if (can_use_batched_qk_round(runner, qk_items, round_indices, round_count)) {
            if (execute_batched_qk_round(runner, qk_items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched qk-softmax-av qk round\n");
                rc = 1;
            }
        } else {
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&qk_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to launch qk-softmax-av qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&qk_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to finish qk-softmax-av qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (softmax_av_item_from_qk_scores(&qk_items[idx], &item_args[idx], &av_items[idx]) != 0) {
            fprintf(stderr, "Failed to fuse qk-softmax-av item %u\n", idx);
            rc = 1;
            break;
        }
    }

    processed_count = 0;
    if (processed != NULL) {
        memset(processed, 0, args.num_slots * sizeof(*processed));
    }
    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_HEADS];
        uint32_t round_count = 0;

        memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            if (processed[idx]) {
                continue;
            }
            if (used_dpus[av_items[idx].physical_dpu_id]) {
                continue;
            }
            used_dpus[av_items[idx].physical_dpu_id] = 1;
            round_indices[round_count++] = idx;
        }
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk-softmax-av av round\n");
            rc = 1;
            break;
        }
        if (can_use_batched_av_round(runner, av_items, round_indices, round_count)) {
            if (execute_batched_av_round(runner, av_items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched qk-softmax-av av round\n");
                rc = 1;
            }
        } else {
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&av_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to launch qk-softmax-av av item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&av_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to finish qk-softmax-av av item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (rc == 0 && write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write qk-softmax-av batch response header\n");
        rc = 1;
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (write_av_item_response(&av_items[idx]) != 0) {
            fprintf(stderr, "Failed to write qk-softmax-av batch item %u\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush qk-softmax-av batch response\n");
        rc = 1;
    }

cleanup:
    if (qk_items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_qk_slot_item(&qk_items[idx]);
        }
    }
    if (av_items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_av_item(&av_items[idx]);
        }
    }
    free(qk_items);
    free(av_items);
    free(item_args);
    free(processed);
    free(used_dpus);
    return rc;
}

static int run_stdio_mode(uint32_t requested_dpus)
{
    kvslot_runner_t runner;
    int rc = runner_init(&runner, requested_dpus);
    if (rc != 0) {
        return rc;
    }

    for (;;) {
        kvslot_io_header_t header;
        if (read_exact(stdin, &header, sizeof(header)) != 0) {
            if (feof(stdin)) {
                rc = 0;
            } else {
                fprintf(stderr, "Failed to read kvslot header\n");
                rc = 1;
            }
            break;
        }
        if (header.magic != KVSLOT_MAGIC) {
            fprintf(stderr, "Invalid kvslot magic\n");
            rc = 1;
            break;
        }
        if (header.command == KVSLOT_CMD_ALLOCATE) {
            rc = handle_allocate(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_APPEND) {
            rc = handle_append(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_READBACK) {
            rc = handle_readback(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_FREE) {
            rc = handle_free(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_GET_STATS) {
            rc = handle_get_stats(&runner);
        } else if (header.command == KVSLOT_CMD_QK_BATCH) {
            rc = handle_qk_batch(&runner);
        } else if (header.command == KVSLOT_CMD_AV) {
            rc = handle_av(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_AV_BATCH) {
            rc = handle_av_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SLOT_BATCH) {
            rc = handle_qk_slot_batch(&runner);
        } else if (header.command == KVSLOT_CMD_SOFTMAX_AV_BATCH) {
            rc = handle_softmax_av_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SOFTMAX_AV_BATCH) {
            rc = handle_qk_softmax_av_batch(&runner);
        } else {
            fprintf(stderr, "Unknown kvslot command %u\n", header.command);
            rc = 1;
        }
        if (rc != 0) {
            break;
        }
    }

    runner_destroy(&runner);
    return rc;
}

int main(int argc, char **argv)
{
    uint32_t requested_dpus = 1;
    if (argc >= 2 && strcmp(argv[1], "--stdio") == 0) {
        if (argc >= 4 && strcmp(argv[2], "--num-dpus") == 0) {
            requested_dpus = (uint32_t)strtoul(argv[3], NULL, 10);
        }
        return run_stdio_mode(requested_dpus);
    }
    fprintf(stderr, "Usage: %s --stdio [--num-dpus N]\n", argv[0]);
    return 1;
}
