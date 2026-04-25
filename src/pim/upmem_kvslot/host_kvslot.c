#include <dpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./build/dpu_kvslot"
#endif

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
    host_slot_t *slots;
    uint32_t *next_free_elem;
    free_range_t *free_ranges;
    uint32_t *num_free_ranges;
} kvslot_runner_t;

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
    struct dpu_set_t dpu;
    uint32_t each_dpu;
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
    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        if (each_dpu == physical_dpu_id) {
            *target_out = dpu;
            *slot_out = &runner->slots[slot_table_index(physical_dpu_id, local_slot_id)];
            return 0;
        }
    }
    return 1;
}

static int runner_init(kvslot_runner_t *runner, uint32_t requested_dpus)
{
    if (runner == NULL) {
        return 1;
    }
    runner->nr_dpus = 0;
    runner->slots = NULL;
    runner->next_free_elem = NULL;
    runner->free_ranges = NULL;
    runner->num_free_ranges = NULL;
    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &runner->dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(runner->dpu_set, &runner->nr_dpus));
    DPU_ASSERT(dpu_load(runner->dpu_set, DPU_BINARY, NULL));
    if (runner->nr_dpus != requested_dpus) {
        fprintf(stderr, "Requested %u DPUs, allocated %u DPUs\n", requested_dpus, runner->nr_dpus);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        return 1;
    }
    runner->slots = calloc((size_t)runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU, sizeof(host_slot_t));
    if (runner->slots == NULL) {
        fprintf(stderr, "Failed to allocate slot table\n");
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        return 1;
    }
    runner->next_free_elem = calloc(runner->nr_dpus, sizeof(uint32_t));
    if (runner->next_free_elem == NULL) {
        fprintf(stderr, "Failed to allocate next_free_elem table\n");
        free(runner->slots);
        runner->slots = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        return 1;
    }
    runner->free_ranges = calloc((size_t)runner->nr_dpus * kvslot_max_free_ranges_per_dpu(), sizeof(free_range_t));
    if (runner->free_ranges == NULL) {
        fprintf(stderr, "Failed to allocate free_ranges table\n");
        free(runner->next_free_elem);
        runner->next_free_elem = NULL;
        free(runner->slots);
        runner->slots = NULL;
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
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
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
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
    if (runner->nr_dpus > 0) {
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
    }
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
    (void)runner;
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
    int32_t *queries = calloc(query_elems, sizeof(*queries));
    int32_t *keys = calloc(key_elems, sizeof(*keys));
    int64_t *scores = calloc(score_elems, sizeof(*scores));
    if (queries == NULL || keys == NULL || scores == NULL) {
        fprintf(stderr, "Failed to allocate qk batch buffers\n");
        free(queries);
        free(keys);
        free(scores);
        return 1;
    }

    if (read_exact(stdin, queries, query_elems * sizeof(*queries)) != 0
        || read_exact(stdin, keys, key_elems * sizeof(*keys)) != 0) {
        fprintf(stderr, "Failed to read qk batch payload\n");
        free(queries);
        free(keys);
        free(scores);
        return 1;
    }

    for (uint32_t query_idx = 0; query_idx < args.num_queries; ++query_idx) {
        const int32_t *query = &queries[(size_t)query_idx * args.head_dim];
        const int32_t *query_keys = &keys[(size_t)query_idx * args.num_keys * args.head_dim];
        for (uint32_t key_idx = 0; key_idx < args.num_keys; ++key_idx) {
            int64_t dot = 0;
            const int32_t *key = &query_keys[(size_t)key_idx * args.head_dim];
            for (uint32_t dim = 0; dim < args.head_dim; ++dim) {
                dot += (int64_t)query[dim] * (int64_t)key[dim];
            }
            scores[(size_t)query_idx * args.num_keys + key_idx] = dot;
        }
    }

    if (write_exact(stdout, &args, sizeof(args)) != 0
        || write_exact(stdout, scores, score_elems * sizeof(*scores)) != 0
        || flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to write qk batch response\n");
        free(queries);
        free(keys);
        free(scores);
        return 1;
    }

    free(queries);
    free(keys);
    free(scores);
    return 0;
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
