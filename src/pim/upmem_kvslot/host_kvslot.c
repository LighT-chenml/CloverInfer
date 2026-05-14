#include <dpu.h>
#include <dpu_management.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./build/dpu_kvslot"
#endif

#ifndef KVSLOT_QK_MAX_ACTIVE_DPUS
#define KVSLOT_QK_MAX_ACTIVE_DPUS 16U
#endif

#ifndef KVSLOT_MAX_BATCH_ITEMS
#define KVSLOT_MAX_BATCH_ITEMS 32U
#endif

typedef struct {
    uint32_t capacity;
    uint32_t seq_len;
    uint32_t group_heads;
    uint32_t head_dim;
    uint32_t dtype_code;
    uint32_t v_dtype_code;
    float k_scale;
    float v_scale;
    uint32_t elem_offset;
    uint32_t elem_count;
} host_slot_t;

typedef struct {
    uint32_t start_elem;
    uint32_t elem_count;
} free_range_t;

typedef struct av_item av_item_t;
typedef struct qk_slot_item qk_slot_item_t;

typedef struct {
    qk_slot_item_t **qk_items_by_dpu;
    size_t qk_items_by_dpu_bytes;
    av_item_t **av_items_by_dpu;
    size_t av_items_by_dpu_bytes;
    uint8_t *rank_used;
    size_t rank_used_bytes;
    struct dpu_rank_t **active_ranks;
    size_t active_ranks_bytes;
    void **xfer_buffers;
    size_t xfer_buffers_bytes;
    uint32_t *dummy_head_indices;
    size_t dummy_head_indices_bytes;
    float *dummy_queries;
    size_t dummy_queries_bytes;
    uint32_t *dummy_scores;
    size_t dummy_scores_bytes;
    uint32_t *dummy_row_max_bits;
    size_t dummy_row_max_bits_bytes;
    uint32_t *dummy_row_sum_bits;
    size_t dummy_row_sum_bits_bytes;
    float *dummy_weights;
    size_t dummy_weights_bytes;
    float *dummy_context;
    size_t dummy_context_bytes;
    kvslot_runtime_slot_args_t *dummy_segment_runtime_args;
    size_t dummy_segment_runtime_args_bytes;
    uint32_t *dummy_segment_lengths;
    size_t dummy_segment_lengths_bytes;
} kvslot_round_scratch_t;

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
    kvslot_profile_stats_t profile;
    kvslot_round_scratch_t scratch;
} kvslot_runner_t;

struct av_item {
    struct dpu_set_t target_dpu;
    host_slot_t *slot;
    host_slot_t *segment_slots[KVSLOT_MAX_GROUP_SEGMENTS];
    kvslot_slot_args_t out;
    kvslot_runtime_slot_args_t runtime_args;
    kvslot_runtime_slot_args_t segment_runtime_args[KVSLOT_MAX_GROUP_SEGMENTS];
    uint32_t slot_id;
    uint32_t physical_dpu_id;
    uint32_t segment_slot_ids[KVSLOT_MAX_GROUP_SEGMENTS];
    uint32_t segment_lengths[KVSLOT_MAX_GROUP_SEGMENTS];
    uint32_t segment_count;
    size_t weight_bytes;
    size_t context_bytes;
    size_t padded_weight_bytes;
    size_t padded_context_bytes;
    uint32_t output_heads;
    float *weights;
    float *context;
    int weights_resident_on_dpu;
    int context_from_qk_kernel;
    int context_prefetched;
    int ready;
};

struct qk_slot_item {
    struct dpu_set_t target_dpu;
    host_slot_t *slot;
    host_slot_t *segment_slots[KVSLOT_MAX_GROUP_SEGMENTS];
    kvslot_runtime_slot_args_t runtime_args;
    kvslot_runtime_slot_args_t segment_runtime_args[KVSLOT_MAX_GROUP_SEGMENTS];
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
    uint32_t *raw_row_sum_bits;
    uint32_t segment_slot_ids[KVSLOT_MAX_GROUP_SEGMENTS];
    uint32_t segment_lengths[KVSLOT_MAX_GROUP_SEGMENTS];
    uint32_t segment_count;
    int ready;
};

static int launch_qk_slot_item_async(const qk_slot_item_t *item, kvslot_profile_stats_t *profile);
static int finish_qk_slot_item(qk_slot_item_t *item, kvslot_profile_stats_t *profile);
static int can_use_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static uint32_t batched_qk_round_max_window(
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static int execute_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    av_item_t *context_items);
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
static int fetch_batched_context_fused_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static av_item_t *find_av_round_item_for_dpu(
    kvslot_runner_t *runner,
    av_item_t **items_by_dpu,
    struct dpu_set_t dpu);
static int prepare_av_item_header(kvslot_runner_t *runner, uint32_t slot_id, av_item_t *item);
static int prepare_grouped_av_item_header(
    kvslot_runner_t *runner,
    const uint32_t *slot_ids,
    const uint32_t *segment_lengths,
    uint32_t segment_count,
    av_item_t *item);
static int read_av_item_weights(av_item_t *item);
static int softmax_av_item_scores_inplace(av_item_t *item);
static int prepare_qk_slot_item_header(
    kvslot_runner_t *runner,
    uint32_t slot_id,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    qk_slot_item_t *item);
static int prepare_grouped_qk_slot_item_header(
    kvslot_runner_t *runner,
    const uint32_t *slot_ids,
    const uint32_t *segment_lengths,
    uint32_t segment_count,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    qk_slot_item_t *item);
static int read_qk_slot_item_payload(FILE *file, qk_slot_item_t *item);
static void cleanup_qk_slot_item(qk_slot_item_t *item);
static int fetch_qk_slot_row_maxes(qk_slot_item_t *item);
static int fetch_qk_slot_row_sums(qk_slot_item_t *item);
static uint32_t count_round_active_ranks_qk(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static uint32_t count_round_active_ranks_av(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count);
static void record_qk_round_profile(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    int used_batched);
static void record_av_round_profile(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    int used_batched);
static void record_qk_round_timing(
    kvslot_runner_t *runner,
    int used_batched,
    uint64_t round_total_ns,
    uint64_t xfer_to_ns,
    uint64_t launch_ns,
    uint64_t xfer_from_ns,
    uint64_t sync_ns);
static void record_av_round_timing(
    kvslot_runner_t *runner,
    int used_batched,
    uint64_t round_total_ns,
    uint64_t xfer_to_ns,
    uint64_t launch_ns,
    uint64_t xfer_from_ns,
    uint64_t sync_ns);

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

static int ensure_scratch_buffer(void **buffer, size_t *capacity_bytes, size_t required_bytes)
{
    void *new_buffer;

    if (required_bytes == 0) {
        return 0;
    }
    if (buffer == NULL || capacity_bytes == NULL) {
        return 1;
    }
    if (*capacity_bytes >= required_bytes && *buffer != NULL) {
        return 0;
    }
    new_buffer = realloc(*buffer, required_bytes);
    if (new_buffer == NULL) {
        return 1;
    }
    if (required_bytes > *capacity_bytes) {
        memset((uint8_t *)new_buffer + *capacity_bytes, 0, required_bytes - *capacity_bytes);
    }
    *buffer = new_buffer;
    *capacity_bytes = required_bytes;
    return 0;
}

static int ensure_round_scratch_qk_items(kvslot_runner_t *runner)
{
    size_t bytes;
    if (runner == NULL || runner->nr_dpus == 0) {
        return 1;
    }
    bytes = (size_t)runner->nr_dpus * sizeof(*runner->scratch.qk_items_by_dpu);
    if (ensure_scratch_buffer((void **)&runner->scratch.qk_items_by_dpu, &runner->scratch.qk_items_by_dpu_bytes, bytes)
        != 0) {
        return 1;
    }
    memset(runner->scratch.qk_items_by_dpu, 0, bytes);
    return 0;
}

static int ensure_round_scratch_av_items(kvslot_runner_t *runner)
{
    size_t bytes;
    if (runner == NULL || runner->nr_dpus == 0) {
        return 1;
    }
    bytes = (size_t)runner->nr_dpus * sizeof(*runner->scratch.av_items_by_dpu);
    if (ensure_scratch_buffer((void **)&runner->scratch.av_items_by_dpu, &runner->scratch.av_items_by_dpu_bytes, bytes)
        != 0) {
        return 1;
    }
    memset(runner->scratch.av_items_by_dpu, 0, bytes);
    return 0;
}

static int ensure_round_scratch_rank_used(kvslot_runner_t *runner)
{
    size_t bytes;
    if (runner == NULL || runner->nr_ranks == 0) {
        return 1;
    }
    bytes = (size_t)runner->nr_ranks * sizeof(*runner->scratch.rank_used);
    if (ensure_scratch_buffer((void **)&runner->scratch.rank_used, &runner->scratch.rank_used_bytes, bytes) != 0) {
        return 1;
    }
    memset(runner->scratch.rank_used, 0, bytes);
    return 0;
}

static int ensure_round_scratch_active_ranks(kvslot_runner_t *runner, uint32_t active_rank_count)
{
    size_t rank_count;
    size_t bytes;

    if (runner == NULL) {
        return 1;
    }
    rank_count = active_rank_count > 0 ? active_rank_count : 1u;
    bytes = rank_count * sizeof(*runner->scratch.active_ranks);
    if (ensure_scratch_buffer((void **)&runner->scratch.active_ranks, &runner->scratch.active_ranks_bytes, bytes) != 0) {
        return 1;
    }
    memset(runner->scratch.active_ranks, 0, bytes);
    return 0;
}

static int ensure_round_scratch_xfer_buffers(kvslot_runner_t *runner)
{
    size_t bytes;
    if (runner == NULL || runner->nr_dpus == 0) {
        return 1;
    }
    bytes = (size_t)runner->nr_dpus * sizeof(*runner->scratch.xfer_buffers);
    if (ensure_scratch_buffer((void **)&runner->scratch.xfer_buffers, &runner->scratch.xfer_buffers_bytes, bytes) != 0) {
        return 1;
    }
    memset(runner->scratch.xfer_buffers, 0, bytes);
    return 0;
}

static void free_round_scratch_xfer_payloads(kvslot_runner_t *runner)
{
    if (runner == NULL || runner->scratch.xfer_buffers == NULL) {
        return;
    }
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        free(runner->scratch.xfer_buffers[physical_dpu_id]);
        runner->scratch.xfer_buffers[physical_dpu_id] = NULL;
    }
}

static int ensure_round_scratch_dummy_head_indices(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer(
            (void **)&runner->scratch.dummy_head_indices,
            &runner->scratch.dummy_head_indices_bytes,
            bytes)
        != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_head_indices, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_queries(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer((void **)&runner->scratch.dummy_queries, &runner->scratch.dummy_queries_bytes, bytes)
        != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_queries, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_scores(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer((void **)&runner->scratch.dummy_scores, &runner->scratch.dummy_scores_bytes, bytes)
        != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_scores, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_rows(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer(
            (void **)&runner->scratch.dummy_row_max_bits,
            &runner->scratch.dummy_row_max_bits_bytes,
            bytes)
            != 0
        || ensure_scratch_buffer(
               (void **)&runner->scratch.dummy_row_sum_bits,
               &runner->scratch.dummy_row_sum_bits_bytes,
               bytes)
               != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_row_max_bits, 0, bytes);
        memset(runner->scratch.dummy_row_sum_bits, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_weights(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer((void **)&runner->scratch.dummy_weights, &runner->scratch.dummy_weights_bytes, bytes)
        != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_weights, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_context(kvslot_runner_t *runner, size_t bytes)
{
    if (runner == NULL) {
        return 1;
    }
    if (ensure_scratch_buffer((void **)&runner->scratch.dummy_context, &runner->scratch.dummy_context_bytes, bytes)
        != 0) {
        return 1;
    }
    if (bytes > 0) {
        memset(runner->scratch.dummy_context, 0, bytes);
    }
    return 0;
}

static int ensure_round_scratch_dummy_segments(kvslot_runner_t *runner, uint32_t segment_count)
{
    size_t runtime_bytes;
    size_t length_bytes;

    if (runner == NULL) {
        return 1;
    }
    runtime_bytes = (size_t)segment_count * sizeof(*runner->scratch.dummy_segment_runtime_args);
    length_bytes = (size_t)segment_count * sizeof(*runner->scratch.dummy_segment_lengths);
    if (ensure_scratch_buffer(
            (void **)&runner->scratch.dummy_segment_runtime_args,
            &runner->scratch.dummy_segment_runtime_args_bytes,
            runtime_bytes)
            != 0
        || ensure_scratch_buffer(
               (void **)&runner->scratch.dummy_segment_lengths,
               &runner->scratch.dummy_segment_lengths_bytes,
               length_bytes)
               != 0) {
        return 1;
    }
    if (runtime_bytes > 0) {
        memset(runner->scratch.dummy_segment_runtime_args, 0, runtime_bytes);
    }
    if (length_bytes > 0) {
        memset(runner->scratch.dummy_segment_lengths, 0, length_bytes);
    }
    return 0;
}

static void free_round_scratch(kvslot_round_scratch_t *scratch)
{
    if (scratch == NULL) {
        return;
    }
    free(scratch->qk_items_by_dpu);
    free(scratch->av_items_by_dpu);
    free(scratch->rank_used);
    free(scratch->active_ranks);
    free(scratch->xfer_buffers);
    free(scratch->dummy_head_indices);
    free(scratch->dummy_queries);
    free(scratch->dummy_scores);
    free(scratch->dummy_row_max_bits);
    free(scratch->dummy_row_sum_bits);
    free(scratch->dummy_weights);
    free(scratch->dummy_context);
    free(scratch->dummy_segment_runtime_args);
    free(scratch->dummy_segment_lengths);
    memset(scratch, 0, sizeof(*scratch));
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

static uint64_t monotonic_time_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t elapsed_ns_since(uint64_t start_ns)
{
    uint64_t end_ns;
    if (start_ns == 0) {
        return 0;
    }
    end_ns = monotonic_time_ns();
    if (end_ns < start_ns) {
        return 0;
    }
    return end_ns - start_ns;
}

static uint32_t env_u32_or_default(const char *name, uint32_t default_value)
{
    const char *value;
    char *end = NULL;
    unsigned long parsed;

    value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }
    parsed = strtoul(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    if (parsed > UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)parsed;
}

static uint32_t qk_max_round_items_limit(void)
{
    uint32_t limit = env_u32_or_default("CLOVER_KVSLOT_QK_MAX_ROUND_ITEMS", KVSLOT_MAX_BATCH_ITEMS);
    if (limit == 0 || limit > KVSLOT_MAX_BATCH_ITEMS) {
        limit = KVSLOT_MAX_BATCH_ITEMS;
    }
    return limit;
}

static uint32_t qk_max_active_ranks_limit(void)
{
    uint32_t limit = env_u32_or_default("CLOVER_KVSLOT_QK_MAX_ACTIVE_RANKS", 16u);
    if (limit == 0 || limit > 16u) {
        limit = 16u;
    }
    return limit;
}

static int context_fused_experiment_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_CONTEXT_FUSED");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int shape_rounds_experiment_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_SHAPE_ROUNDS");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int rank_spread_alloc_experiment_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_RANK_SPREAD_ALLOC");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int rank_spread_multi_rank_batch_experiment_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_ALLOW_RANK_SPREAD_MULTI_RANK_BATCH");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int rank_local_rounds_experiment_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_RANK_LOCAL_ROUNDS");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int dpu_phase_profile_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_DPU_PHASE_PROFILE");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return 1;
}

static int init_runner_topology_storage(kvslot_runner_t *runner)
{
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
        return 1;
    }
    return 0;
}

static void destroy_runner_topology_storage(kvslot_runner_t *runner)
{
    free(runner->physical_dpu_rank_indices);
    runner->physical_dpu_rank_indices = NULL;
    free(runner->physical_dpus);
    runner->physical_dpus = NULL;
    free(runner->ranks);
    runner->ranks = NULL;
}

static int collect_runner_ranks(kvslot_runner_t *runner)
{
    struct dpu_set_t rank_set;
    uint32_t each_rank = 0;
    DPU_RANK_FOREACH(runner->dpu_set, rank_set, each_rank)
    {
        if (each_rank >= runner->nr_ranks) {
            fprintf(stderr, "Collected too many ranks while initializing topology\n");
            return 1;
        }
        runner->ranks[each_rank] = dpu_rank_from_set(rank_set);
    }
    if (each_rank != runner->nr_ranks) {
        fprintf(stderr, "Failed to collect all allocated ranks: expected=%u actual=%u\n", runner->nr_ranks, each_rank);
        return 1;
    }
    return 0;
}

static int collect_runner_physical_dpus_default(kvslot_runner_t *runner)
{
    struct dpu_set_t dpu;
    uint32_t each_dpu = 0;
    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        struct dpu_t *dpu_ptr = dpu_from_set(dpu);
        struct dpu_rank_t *rank_ptr = dpu_get_rank(dpu_ptr);
        uint32_t rank_idx = 0;
        if (each_dpu >= runner->nr_dpus) {
            fprintf(stderr, "Collected too many DPUs while initializing topology\n");
            return 1;
        }
        runner->physical_dpus[each_dpu] = dpu;
        for (; rank_idx < runner->nr_ranks; ++rank_idx) {
            if (runner->ranks[rank_idx] == rank_ptr) {
                break;
            }
        }
        if (rank_idx == runner->nr_ranks) {
            fprintf(stderr, "Failed to resolve rank for DPU %u\n", each_dpu);
            return 1;
        }
        runner->physical_dpu_rank_indices[each_dpu] = rank_idx;
    }
    if (each_dpu != runner->nr_dpus) {
        fprintf(stderr, "Collected unexpected DPU count: expected=%u actual=%u\n", runner->nr_dpus, each_dpu);
        return 1;
    }
    return 0;
}

static int collect_runner_physical_dpus_rank_spread(kvslot_runner_t *runner)
{
    uint32_t logical_dpu = 0;
    uint32_t rank_pass = 0;

    while (logical_dpu < runner->nr_dpus) {
        uint32_t added_this_pass = 0;
        struct dpu_set_t rank_set;
        uint32_t rank_idx = 0;
        DPU_RANK_FOREACH(runner->dpu_set, rank_set, rank_idx)
        {
            struct dpu_set_t dpu;
            uint32_t dpu_in_rank = 0;
            int selected = 0;
            DPU_FOREACH(rank_set, dpu, dpu_in_rank)
            {
                if (dpu_in_rank != rank_pass) {
                    continue;
                }
                runner->physical_dpus[logical_dpu] = dpu;
                runner->physical_dpu_rank_indices[logical_dpu] = rank_idx;
                logical_dpu += 1;
                added_this_pass += 1;
                selected = 1;
                break;
            }
            if (!selected) {
                continue;
            }
            if (logical_dpu >= runner->nr_dpus) {
                break;
            }
        }
        if (added_this_pass == 0) {
            fprintf(stderr, "Failed to spread %u logical DPUs across allocated ranks\n", runner->nr_dpus);
            return 1;
        }
        rank_pass += 1;
    }
    return 0;
}

static int qk_items_round_compatible(const qk_slot_item_t *seed, const qk_slot_item_t *item)
{
    if (seed == NULL || item == NULL || !seed->ready || !item->ready) {
        return 0;
    }
    /*
     * Batched QK rounds already size score transfers to the max window in the
     * round and then copy each item's compact score slice back out afterwards.
     * Keeping the window equality requirement here prevents concurrent decode
     * requests with slightly different context lengths from sharing a round at
     * all, which is especially harmful for Qwen continuous batching.
     */
    if (item->num_heads != seed->num_heads || item->head_dim != seed->head_dim) {
        return 0;
    }
    if (item->slot_args.mode != seed->slot_args.mode) {
        return 0;
    }
    return 1;
}

static int av_items_round_compatible(const av_item_t *seed, const av_item_t *item)
{
    if (seed == NULL || item == NULL || !seed->ready || !item->ready) {
        return 0;
    }
    if (item->runtime_args.group_heads != seed->runtime_args.group_heads
        || item->runtime_args.head_dim != seed->runtime_args.head_dim
        || item->runtime_args.dtype_code != seed->runtime_args.dtype_code
        || item->runtime_args.v_dtype_code != seed->runtime_args.v_dtype_code) {
        return 0;
    }
    if (item->padded_context_bytes != seed->padded_context_bytes) {
        return 0;
    }
    if (item->weights_resident_on_dpu != seed->weights_resident_on_dpu) {
        return 0;
    }
    if (item->context_from_qk_kernel != seed->context_from_qk_kernel) {
        return 0;
    }
    if ((item->segment_count > 1) != (seed->segment_count > 1)) {
        return 0;
    }
    return 1;
}

static int shape_rounds_default_enabled(void)
{
    const char *value = getenv("CLOVER_KVSLOT_SHAPE_ROUNDS");
    if (value == NULL || value[0] == '\0') {
        return 1;
    }
    return strcmp(value, "0") != 0;
}

static int same_rank_qk_item(kvslot_runner_t *runner, const qk_slot_item_t *lhs, const qk_slot_item_t *rhs)
{
    if (runner == NULL || lhs == NULL || rhs == NULL || runner->physical_dpu_rank_indices == NULL || runner->nr_ranks == 0) {
        return 0;
    }
    if (lhs->physical_dpu_id >= runner->nr_dpus || rhs->physical_dpu_id >= runner->nr_dpus) {
        return 0;
    }
    return runner->physical_dpu_rank_indices[lhs->physical_dpu_id]
        == runner->physical_dpu_rank_indices[rhs->physical_dpu_id];
}

static int same_rank_av_item(kvslot_runner_t *runner, const av_item_t *lhs, const av_item_t *rhs)
{
    if (runner == NULL || lhs == NULL || rhs == NULL || runner->physical_dpu_rank_indices == NULL || runner->nr_ranks == 0) {
        return 0;
    }
    if (lhs->physical_dpu_id >= runner->nr_dpus || rhs->physical_dpu_id >= runner->nr_dpus) {
        return 0;
    }
    return runner->physical_dpu_rank_indices[lhs->physical_dpu_id]
        == runner->physical_dpu_rank_indices[rhs->physical_dpu_id];
}

static int qk_item_can_join_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    uint32_t seed_idx,
    uint32_t item_idx,
    const uint8_t *processed,
    const uint8_t *used_dpus,
    uint8_t *rank_used,
    uint32_t active_rank_count,
    uint32_t max_active_ranks,
    int shape_rounds_enabled,
    int same_rank_only,
    uint32_t *out_rank_idx,
    int *out_adds_rank)
{
    qk_slot_item_t *seed;
    qk_slot_item_t *item;
    uint32_t rank_idx = UINT32_MAX;
    int adds_rank = 0;

    if (runner == NULL || items == NULL || processed == NULL || used_dpus == NULL) {
        return 0;
    }
    if (seed_idx == item_idx || processed[item_idx]) {
        return 0;
    }
    seed = &items[seed_idx];
    item = &items[item_idx];
    if (item->physical_dpu_id >= runner->nr_dpus || used_dpus[item->physical_dpu_id]) {
        return 0;
    }
    if (same_rank_only && !same_rank_qk_item(runner, seed, item)) {
        return 0;
    }
    if (shape_rounds_enabled && !qk_items_round_compatible(seed, item)) {
        return 0;
    }
    if (rank_used != NULL) {
        if (item->physical_dpu_id >= runner->nr_dpus) {
            return 0;
        }
        rank_idx = runner->physical_dpu_rank_indices[item->physical_dpu_id];
        adds_rank = rank_idx < runner->nr_ranks && !rank_used[rank_idx];
        if (adds_rank && active_rank_count >= max_active_ranks) {
            return 0;
        }
    }
    if (out_rank_idx != NULL) {
        *out_rank_idx = rank_idx;
    }
    if (out_adds_rank != NULL) {
        *out_adds_rank = adds_rank;
    }
    return 1;
}

static uint32_t estimate_qk_round_size_for_seed(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    uint32_t num_items,
    const uint8_t *processed,
    uint32_t seed_idx,
    uint32_t max_round_items,
    uint32_t max_active_ranks,
    uint8_t *used_dpus,
    uint8_t *rank_used,
    int shape_rounds_enabled,
    int rank_local_rounds_enabled)
{
    uint32_t round_count = 1;
    uint32_t active_rank_count = 0;

    if (runner == NULL || items == NULL || processed == NULL || used_dpus == NULL
        || seed_idx >= num_items || processed[seed_idx]
        || items[seed_idx].physical_dpu_id >= runner->nr_dpus) {
        return 0;
    }
    memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
    if (rank_used != NULL) {
        memset(rank_used, 0, runner->nr_ranks * sizeof(*rank_used));
    }
    used_dpus[items[seed_idx].physical_dpu_id] = 1;
    if (rank_used != NULL) {
        uint32_t seed_rank = runner->physical_dpu_rank_indices[items[seed_idx].physical_dpu_id];
        if (seed_rank < runner->nr_ranks) {
            rank_used[seed_rank] = 1;
            active_rank_count = 1;
        }
    }

    for (uint32_t pass = 0; pass < 2; ++pass) {
        int same_rank_only = pass == 0;
        if (pass == 1 && rank_local_rounds_enabled) {
            break;
        }
        for (uint32_t idx = 0; idx < num_items; ++idx) {
            uint32_t rank_idx = UINT32_MAX;
            int adds_rank = 0;
            if (round_count >= max_round_items) {
                return round_count;
            }
            if (!qk_item_can_join_round(
                    runner,
                    items,
                    seed_idx,
                    idx,
                    processed,
                    used_dpus,
                    rank_used,
                    active_rank_count,
                    max_active_ranks,
                    shape_rounds_enabled,
                    same_rank_only,
                    &rank_idx,
                    &adds_rank)) {
                continue;
            }
            used_dpus[items[idx].physical_dpu_id] = 1;
            if (adds_rank) {
                rank_used[rank_idx] = 1;
                active_rank_count += 1;
            }
            round_count += 1;
        }
    }
    return round_count;
}

static uint32_t select_qk_round_seed(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    uint32_t num_items,
    const uint8_t *processed,
    uint32_t max_round_items,
    uint32_t max_active_ranks,
    uint8_t *used_dpus,
    uint8_t *rank_used,
    int shape_rounds_enabled,
    int rank_local_rounds_enabled)
{
    uint32_t best_idx = UINT32_MAX;
    uint32_t best_score = 0;

    for (uint32_t idx = 0; idx < num_items; ++idx) {
        uint32_t score;
        if (processed[idx]) {
            continue;
        }
        score = estimate_qk_round_size_for_seed(
            runner,
            items,
            num_items,
            processed,
            idx,
            max_round_items,
            max_active_ranks,
            used_dpus,
            rank_used,
            shape_rounds_enabled,
            rank_local_rounds_enabled);
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
            if (best_score >= max_round_items) {
                break;
            }
        }
    }
    return best_idx;
}

static uint32_t build_qk_launch_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    uint32_t num_items,
    const uint8_t *processed,
    uint8_t *used_dpus,
    uint32_t *round_indices)
{
    uint32_t seed_idx = UINT32_MAX;
    uint32_t round_count = 0;
    uint32_t max_round_items = qk_max_round_items_limit();
    uint32_t max_active_ranks = qk_max_active_ranks_limit();
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;
    int shape_rounds_enabled = shape_rounds_default_enabled();
    int rank_local_rounds_enabled = rank_local_rounds_experiment_enabled();

    if (runner == NULL || items == NULL || processed == NULL || used_dpus == NULL || round_indices == NULL) {
        return 0;
    }

    memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
    if (runner->nr_ranks > 0 && runner->physical_dpu_rank_indices != NULL) {
        if (ensure_round_scratch_rank_used(runner) != 0) {
            return 0;
        }
        rank_used = runner->scratch.rank_used;
    }
    seed_idx = select_qk_round_seed(
        runner,
        items,
        num_items,
        processed,
        max_round_items,
        max_active_ranks,
        used_dpus,
        rank_used,
        shape_rounds_enabled,
        rank_local_rounds_enabled);
    if (seed_idx == UINT32_MAX) {
        return 0;
    }
    memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
    if (rank_used != NULL) {
        memset(rank_used, 0, runner->nr_ranks * sizeof(*rank_used));
    }

    used_dpus[items[seed_idx].physical_dpu_id] = 1;
    if (rank_used != NULL && items[seed_idx].physical_dpu_id < runner->nr_dpus) {
        uint32_t seed_rank = runner->physical_dpu_rank_indices[items[seed_idx].physical_dpu_id];
        if (seed_rank < runner->nr_ranks) {
            rank_used[seed_rank] = 1;
            active_rank_count = 1;
        }
    }
    round_indices[round_count++] = seed_idx;
    for (uint32_t idx = 0; idx < num_items; ++idx) {
        uint32_t rank_idx = UINT32_MAX;
        int adds_rank = 0;
        if (round_count >= max_round_items) {
            break;
        }
        if (!qk_item_can_join_round(
                runner,
                items,
                seed_idx,
                idx,
                processed,
                used_dpus,
                rank_used,
                active_rank_count,
                max_active_ranks,
                shape_rounds_enabled,
                1,
                &rank_idx,
                &adds_rank)) {
            continue;
        }
        used_dpus[items[idx].physical_dpu_id] = 1;
        if (adds_rank) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
        round_indices[round_count++] = idx;
    }
    for (uint32_t idx = 0; idx < num_items; ++idx) {
        uint32_t rank_idx = UINT32_MAX;
        int adds_rank = 0;
        if (round_count >= max_round_items) {
            break;
        }
        if (rank_local_rounds_enabled) {
            break;
        }
        if (!qk_item_can_join_round(
                runner,
                items,
                seed_idx,
                idx,
                processed,
                used_dpus,
                rank_used,
                active_rank_count,
                max_active_ranks,
                shape_rounds_enabled,
                0,
                &rank_idx,
                &adds_rank)) {
            continue;
        }
        used_dpus[items[idx].physical_dpu_id] = 1;
        if (adds_rank) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
        round_indices[round_count++] = idx;
    }
    return round_count;
}

static uint32_t build_av_launch_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    uint32_t num_items,
    const uint8_t *processed,
    uint8_t *used_dpus,
    uint32_t *round_indices)
{
    uint32_t seed_idx = UINT32_MAX;
    uint32_t round_count = 0;
    int shape_rounds_enabled = shape_rounds_default_enabled();
    int rank_local_rounds_enabled = rank_local_rounds_experiment_enabled();

    if (runner == NULL || items == NULL || processed == NULL || used_dpus == NULL || round_indices == NULL) {
        return 0;
    }

    memset(used_dpus, 0, runner->nr_dpus * sizeof(*used_dpus));
    for (uint32_t idx = 0; idx < num_items; ++idx) {
        if (!processed[idx]) {
            seed_idx = idx;
            break;
        }
    }
    if (seed_idx == UINT32_MAX) {
        return 0;
    }

    used_dpus[items[seed_idx].physical_dpu_id] = 1;
    round_indices[round_count++] = seed_idx;
    for (uint32_t idx = 0; idx < num_items; ++idx) {
        if (idx == seed_idx || processed[idx]) {
            continue;
        }
        if (used_dpus[items[idx].physical_dpu_id]) {
            continue;
        }
        if (!same_rank_av_item(runner, &items[seed_idx], &items[idx])) {
            continue;
        }
        if (shape_rounds_enabled && !av_items_round_compatible(&items[seed_idx], &items[idx])) {
            continue;
        }
        used_dpus[items[idx].physical_dpu_id] = 1;
        round_indices[round_count++] = idx;
    }
    for (uint32_t idx = 0; idx < num_items; ++idx) {
        if (rank_local_rounds_enabled) {
            break;
        }
        if (idx == seed_idx || processed[idx]) {
            continue;
        }
        if (used_dpus[items[idx].physical_dpu_id]) {
            continue;
        }
        if (shape_rounds_enabled && !av_items_round_compatible(&items[seed_idx], &items[idx])) {
            continue;
        }
        used_dpus[items[idx].physical_dpu_id] = 1;
        round_indices[round_count++] = idx;
    }
    return round_count;
}

static uint32_t count_round_active_ranks_qk(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    uint8_t *rank_used = NULL;
    uint32_t active_rank_count = 0;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count == 0 || runner->nr_ranks == 0) {
        return 0;
    }
    if (ensure_round_scratch_rank_used(runner) != 0) {
        return 0;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t physical_dpu_id = items[round_indices[pos]].physical_dpu_id;
        uint32_t rank_idx;
        if (physical_dpu_id >= runner->nr_dpus) {
            continue;
        }
        rank_idx = runner->physical_dpu_rank_indices[physical_dpu_id];
        if (rank_idx < runner->nr_ranks && !rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    return active_rank_count;
}

static uint32_t count_qk_round_launch_dpus(kvslot_runner_t *runner, uint32_t active_rank_count)
{
    uint32_t launch_dpus = 0;

    if (runner == NULL || runner->nr_ranks == 0 || active_rank_count == 0) {
        return 0;
    }
    if (active_rank_count >= runner->nr_ranks) {
        return runner->nr_dpus;
    }
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        if (runner->physical_dpu_rank_indices[physical_dpu_id] < runner->nr_ranks
            && runner->scratch.rank_used[runner->physical_dpu_rank_indices[physical_dpu_id]]) {
            launch_dpus += 1;
        }
    }
    return launch_dpus;
}

static uint32_t count_round_active_ranks_av(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    uint8_t *rank_used = NULL;
    uint32_t active_rank_count = 0;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count == 0 || runner->nr_ranks == 0) {
        return 0;
    }
    if (ensure_round_scratch_rank_used(runner) != 0) {
        return 0;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t physical_dpu_id = items[round_indices[pos]].physical_dpu_id;
        uint32_t rank_idx;
        if (physical_dpu_id >= runner->nr_dpus) {
            continue;
        }
        rank_idx = runner->physical_dpu_rank_indices[physical_dpu_id];
        if (rank_idx < runner->nr_ranks && !rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    return active_rank_count;
}

static void record_qk_round_profile(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    int used_batched)
{
    uint32_t active_ranks;
    if (runner == NULL || round_count == 0) {
        return;
    }
    active_ranks = count_round_active_ranks_qk(runner, items, round_indices, round_count);
    runner->profile.qk_rounds_total += 1;
    runner->profile.qk_round_items_total += round_count;
    runner->profile.qk_active_ranks_total += active_ranks;
    if ((uint64_t)round_count > runner->profile.qk_max_round_size) {
        runner->profile.qk_max_round_size = round_count;
    }
    if ((uint64_t)active_ranks > runner->profile.qk_max_active_ranks) {
        runner->profile.qk_max_active_ranks = active_ranks;
    }
    runner->profile.qk_round_window_total += batched_qk_round_max_window(items, round_indices, round_count);
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        uint64_t head_window = (uint64_t)item->num_heads * (uint64_t)item->window;
        runner->profile.qk_round_heads_total += item->num_heads;
        runner->profile.qk_round_head_window_total += head_window;
        if ((uint64_t)item->num_heads > runner->profile.qk_max_heads_per_item) {
            runner->profile.qk_max_heads_per_item = item->num_heads;
        }
        if (head_window > runner->profile.qk_max_head_window_per_item) {
            runner->profile.qk_max_head_window_per_item = head_window;
        }
        runner->profile.qk_round_item_window_total += item->window;
        if ((uint64_t)item->window > runner->profile.qk_max_window) {
            runner->profile.qk_max_window = item->window;
        }
        if (item->segment_count > 1) {
            runner->profile.qk_segmented_items_total += 1;
        }
        if ((uint64_t)item->segment_count > runner->profile.qk_max_segment_count) {
            runner->profile.qk_max_segment_count = item->segment_count;
        }
    }
    if (used_batched) {
        uint32_t launch_dpus = count_qk_round_launch_dpus(runner, active_ranks);
        uint32_t dummy_dpus = launch_dpus > round_count ? launch_dpus - round_count : 0;
        runner->profile.qk_batched_rounds += 1;
        runner->profile.qk_batched_items_total += round_count;
        runner->profile.qk_batched_launch_dpus_total += launch_dpus;
        runner->profile.qk_batched_dummy_dpus_total += dummy_dpus;
        if ((uint64_t)launch_dpus > runner->profile.qk_batched_max_launch_dpus) {
            runner->profile.qk_batched_max_launch_dpus = launch_dpus;
        }
        if ((uint64_t)dummy_dpus > runner->profile.qk_batched_max_dummy_dpus) {
            runner->profile.qk_batched_max_dummy_dpus = dummy_dpus;
        }
    } else {
        runner->profile.qk_fallback_rounds += 1;
    }
}

static void record_av_round_profile(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    int used_batched)
{
    uint32_t active_ranks;
    if (runner == NULL || round_count == 0) {
        return;
    }
    active_ranks = count_round_active_ranks_av(runner, items, round_indices, round_count);
    runner->profile.av_rounds_total += 1;
    runner->profile.av_round_items_total += round_count;
    runner->profile.av_active_ranks_total += active_ranks;
    if ((uint64_t)round_count > runner->profile.av_max_round_size) {
        runner->profile.av_max_round_size = round_count;
    }
    if ((uint64_t)active_ranks > runner->profile.av_max_active_ranks) {
        runner->profile.av_max_active_ranks = active_ranks;
    }
    if (used_batched) {
        runner->profile.av_batched_rounds += 1;
        runner->profile.av_batched_items_total += round_count;
    } else {
        runner->profile.av_fallback_rounds += 1;
    }
}

static void record_qk_round_timing(
    kvslot_runner_t *runner,
    int used_batched,
    uint64_t round_total_ns,
    uint64_t xfer_to_ns,
    uint64_t launch_ns,
    uint64_t xfer_from_ns,
    uint64_t sync_ns)
{
    if (runner == NULL) {
        return;
    }
    if (used_batched) {
        runner->profile.qk_batched_round_total_ns += round_total_ns;
        runner->profile.qk_batched_xfer_to_ns += xfer_to_ns;
        runner->profile.qk_batched_launch_ns += launch_ns;
        runner->profile.qk_batched_xfer_from_ns += xfer_from_ns;
    } else {
        runner->profile.qk_fallback_round_total_ns += round_total_ns;
        runner->profile.qk_fallback_launch_ns += launch_ns;
        runner->profile.qk_fallback_sync_ns += sync_ns;
        runner->profile.qk_fallback_xfer_from_ns += xfer_from_ns;
    }
}

static void record_qk_dpu_meta(kvslot_runner_t *runner, const kvslot_meta_t *meta)
{
    if (runner == NULL || meta == NULL || meta->cycles == 0) {
        return;
    }
    runner->profile.qk_dpu_cycles_total += meta->cycles;
    runner->profile.qk_dpu_dot_cycles_total += meta->qk_dot_cycles;
    runner->profile.qk_dpu_softmax_cycles_total += meta->qk_softmax_cycles;
    runner->profile.qk_dpu_context_cycles_total += meta->qk_context_cycles;
    runner->profile.qk_dpu_other_cycles_total += meta->qk_other_cycles;
    runner->profile.qk_dpu_profiled_dpus += 1;
}

static void record_av_round_timing(
    kvslot_runner_t *runner,
    int used_batched,
    uint64_t round_total_ns,
    uint64_t xfer_to_ns,
    uint64_t launch_ns,
    uint64_t xfer_from_ns,
    uint64_t sync_ns)
{
    if (runner == NULL) {
        return;
    }
    if (used_batched) {
        runner->profile.av_batched_round_total_ns += round_total_ns;
        runner->profile.av_batched_xfer_to_ns += xfer_to_ns;
        runner->profile.av_batched_launch_ns += launch_ns;
        runner->profile.av_batched_xfer_from_ns += xfer_from_ns;
    } else {
        runner->profile.av_fallback_round_total_ns += round_total_ns;
        runner->profile.av_fallback_launch_ns += launch_ns;
        runner->profile.av_fallback_sync_ns += sync_ns;
        runner->profile.av_fallback_xfer_from_ns += xfer_from_ns;
    }
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
    slot->v_dtype_code = KVSLOT_DTYPE_FP32;
    slot->k_scale = 1.0f;
    slot->v_scale = 1.0f;
    return 0;
}

static uint32_t kvslot_align_words_for_mram_xfer(uint32_t words)
{
    return (words + 1U) & ~1U;
}

static size_t kvslot_dtype_elem_size(uint32_t dtype_code)
{
    if (dtype_code == KVSLOT_DTYPE_INT8) {
        return sizeof(int8_t);
    }
    if (dtype_code == KVSLOT_DTYPE_FP16 || dtype_code == KVSLOT_DTYPE_BF16 || dtype_code == KVSLOT_DTYPE_INT16) {
        return sizeof(uint16_t);
    }
    return sizeof(int32_t);
}

static uint32_t kvslot_packed_elem_count(uint32_t logical_elems, uint32_t dtype_code)
{
    uint32_t packed_words;
    if (dtype_code == KVSLOT_DTYPE_INT8) {
        packed_words = (logical_elems + 3U) / 4U;
        return kvslot_align_words_for_mram_xfer(packed_words);
    }
    if (dtype_code == KVSLOT_DTYPE_FP16 || dtype_code == KVSLOT_DTYPE_BF16 || dtype_code == KVSLOT_DTYPE_INT16) {
        packed_words = (logical_elems + 1U) / 2U;
        return kvslot_align_words_for_mram_xfer(packed_words);
    }
    return kvslot_align_words_for_mram_xfer(logical_elems);
}

static uint32_t kvslot_logical_elem_offset_to_packed_words(uint32_t logical_offset, uint32_t dtype_code)
{
    if (dtype_code == KVSLOT_DTYPE_INT8) {
        return logical_offset / 4U;
    }
    if (dtype_code == KVSLOT_DTYPE_FP16 || dtype_code == KVSLOT_DTYPE_BF16 || dtype_code == KVSLOT_DTYPE_INT16) {
        return logical_offset / 2U;
    }
    return logical_offset;
}

static int kvslot_dtype_supported(uint32_t dtype_code)
{
    return dtype_code == KVSLOT_DTYPE_FP32
        || dtype_code == KVSLOT_DTYPE_FP16
        || dtype_code == KVSLOT_DTYPE_INT8
        || dtype_code == KVSLOT_DTYPE_BF16
        || dtype_code == KVSLOT_DTYPE_INT16;
}

static uint32_t kvslot_normalize_v_dtype_code(uint32_t dtype_code, uint32_t v_dtype_code)
{
    if (v_dtype_code == 0u || v_dtype_code == 0xffffffffu) {
        return dtype_code;
    }
    return v_dtype_code;
}

static float kvslot_sanitize_scale(float scale)
{
    if (!isfinite(scale) || scale <= 0.0f) {
        return 1.0f;
    }
    return scale;
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
    int rank_spread_enabled;
    if (runner == NULL) {
        return 1;
    }
    memset(runner, 0, sizeof(*runner));
    runner->nr_dpus = 0;
    runner->nr_ranks = 0;
    runner->ranks = NULL;
    runner->physical_dpus = NULL;
    runner->physical_dpu_rank_indices = NULL;
    runner->slots = NULL;
    runner->next_free_elem = NULL;
    runner->free_ranges = NULL;
    runner->num_free_ranges = NULL;
    rank_spread_enabled = rank_spread_alloc_experiment_enabled();
    if (rank_spread_enabled) {
        /*
         * dpu_alloc_ranks() takes a rank count, not a DPU count.  For the
         * rank-spread experiment we allocate all ranks and then expose exactly
         * requested_dpus logical DPUs by interleaving one physical DPU from
         * each rank at a time in collect_runner_physical_dpus_rank_spread().
         * This intentionally over-reserves the PIM node, but the Ray placement
         * layer already treats attention_pim as an exclusive resource and this
         * keeps 32-DPU experiments from silently collapsing onto one rank.
         */
        DPU_ASSERT(dpu_alloc_ranks(DPU_ALLOCATE_ALL, NULL, &runner->dpu_set));
        runner->nr_dpus = requested_dpus;
    } else {
        DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &runner->dpu_set));
        DPU_ASSERT(dpu_get_nr_dpus(runner->dpu_set, &runner->nr_dpus));
    }
    DPU_ASSERT(dpu_get_nr_ranks(runner->dpu_set, &runner->nr_ranks));
    DPU_ASSERT(dpu_load(runner->dpu_set, DPU_BINARY, NULL));
    if (runner->nr_dpus != requested_dpus) {
        fprintf(stderr, "Requested %u DPUs, allocated %u DPUs\n", requested_dpus, runner->nr_dpus);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    if (init_runner_topology_storage(runner) != 0) {
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    if (collect_runner_ranks(runner) != 0
        || (rank_spread_enabled ? collect_runner_physical_dpus_rank_spread(runner)
                                : collect_runner_physical_dpus_default(runner))
            != 0) {
        destroy_runner_topology_storage(runner);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        runner->nr_ranks = 0;
        return 1;
    }
    runner->slots = calloc((size_t)runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU, sizeof(host_slot_t));
    if (runner->slots == NULL) {
        fprintf(stderr, "Failed to allocate slot table\n");
        destroy_runner_topology_storage(runner);
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
        destroy_runner_topology_storage(runner);
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
        destroy_runner_topology_storage(runner);
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
        destroy_runner_topology_storage(runner);
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
    free_round_scratch_xfer_payloads(runner);
    free_round_scratch(&runner->scratch);
    destroy_runner_topology_storage(runner);
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
    args.v_dtype_code = kvslot_normalize_v_dtype_code(args.dtype_code, args.v_dtype_code);
    if (!kvslot_dtype_supported(args.dtype_code) || !kvslot_dtype_supported(args.v_dtype_code)) {
        fprintf(stderr, "Unsupported kvslot dtype_code=%u v_dtype_code=%u\n", args.dtype_code, args.v_dtype_code);
        return 1;
    }

    size_t logical_elems = (size_t)args.seq_len * args.group_heads * args.head_dim;
    size_t k_elem_size = kvslot_dtype_elem_size(args.dtype_code);
    size_t v_elem_size = kvslot_dtype_elem_size(args.v_dtype_code);
    size_t k_bytes = logical_elems * k_elem_size;
    size_t v_bytes = logical_elems * v_elem_size;
    uint32_t slot_total_logical_elems = args.capacity * args.group_heads * args.head_dim;
    uint32_t k_slot_total_elems = kvslot_packed_elem_count(slot_total_logical_elems, args.dtype_code);
    uint32_t v_slot_total_elems = kvslot_packed_elem_count(slot_total_logical_elems, args.v_dtype_code);
    uint32_t slot_total_elems = k_slot_total_elems > v_slot_total_elems ? k_slot_total_elems : v_slot_total_elems;
    if (args.seq_len > args.capacity) {
        fprintf(stderr, "Initial seq_len exceeds capacity\n");
        return 1;
    }
    void *k_data = NULL;
    void *v_data = NULL;
    if (logical_elems > 0) {
        k_data = calloc(logical_elems, k_elem_size);
        v_data = calloc(logical_elems, v_elem_size);
        if (k_data == NULL || v_data == NULL) {
            fprintf(stderr, "Failed to allocate allocate payload buffers\n");
            free(k_data);
            free(v_data);
            return 1;
        }
        if (read_exact(stdin, k_data, k_bytes) != 0
            || read_exact(stdin, v_data, v_bytes) != 0) {
            fprintf(stderr, "Failed to read allocate payload\n");
            free(k_data);
            free(v_data);
            return 1;
        }
    }
    slot->dtype_code = args.dtype_code;
    slot->v_dtype_code = args.v_dtype_code;
    slot->k_scale = kvslot_sanitize_scale(args.k_scale);
    slot->v_scale = kvslot_sanitize_scale(args.v_scale);
    args.k_scale = slot->k_scale;
    args.v_scale = slot->v_scale;
    DPU_ASSERT(dpu_broadcast_to(target_dpu, "slot_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    if (reserve_elem_range(runner, physical_dpu_id, slot_total_elems, &slot->elem_offset) != 0) {
        fprintf(stderr, "DPU %u out of reusable kvslot capacity\n", physical_dpu_id);
        free_slot(slot);
        free(k_data);
        free(v_data);
        return 1;
    }
    slot->elem_count = slot_total_elems;
    if (logical_elems > 0) {
        size_t byte_offset = (size_t)slot->elem_offset * sizeof(int32_t);
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "k_cache", byte_offset, k_bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "v_cache", byte_offset, v_bytes, DPU_XFER_DEFAULT));
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
        .k_scale = slot->k_scale,
        .v_scale = slot->v_scale,
        .v_dtype_code = slot->v_dtype_code,
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
    args.v_dtype_code = kvslot_normalize_v_dtype_code(args.dtype_code, args.v_dtype_code);
    if (args.seq_len != 1
        || args.group_heads != slot->group_heads
        || args.head_dim != slot->head_dim
        || args.dtype_code != slot->dtype_code
        || args.v_dtype_code != slot->v_dtype_code) {
        fprintf(stderr, "Append args mismatch for slot %u\n", slot_id);
        return 1;
    }
    if (slot->seq_len + 1 > slot->capacity) {
        fprintf(stderr, "Slot %u capacity exceeded\n", slot_id);
        return 1;
    }

    size_t token_elems = (size_t)slot->group_heads * slot->head_dim;
    size_t k_token_elem_size = kvslot_dtype_elem_size(slot->dtype_code);
    size_t v_token_elem_size = kvslot_dtype_elem_size(slot->v_dtype_code);
    size_t k_token_bytes = token_elems * k_token_elem_size;
    size_t v_token_bytes = token_elems * v_token_elem_size;
    void *k_token = calloc(token_elems, k_token_elem_size);
    void *v_token = calloc(token_elems, v_token_elem_size);
    if (k_token == NULL || v_token == NULL) {
        fprintf(stderr, "Failed to allocate append buffers\n");
        free(k_token);
        free(v_token);
        return 1;
    }
    if (read_exact(stdin, k_token, k_token_bytes) != 0
        || read_exact(stdin, v_token, v_token_bytes) != 0) {
        fprintf(stderr, "Failed to read append payload\n");
        free(k_token);
        free(v_token);
        return 1;
    }
    size_t k_byte_offset = ((size_t)slot->elem_offset * sizeof(int32_t)) + ((size_t)slot->seq_len * k_token_bytes);
    size_t v_byte_offset = ((size_t)slot->elem_offset * sizeof(int32_t)) + ((size_t)slot->seq_len * v_token_bytes);
    DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_token));
    DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "k_cache", k_byte_offset, k_token_bytes, DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_token));
    DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "v_cache", v_byte_offset, v_token_bytes, DPU_XFER_DEFAULT));
    slot->seq_len += 1;
    free(k_token);
    free(v_token);

    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
        .dtype_code = slot->dtype_code,
        .k_scale = slot->k_scale,
        .v_scale = slot->v_scale,
        .v_dtype_code = slot->v_dtype_code,
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
        .k_scale = slot->k_scale,
        .v_scale = slot->v_scale,
        .v_dtype_code = slot->v_dtype_code,
    };
    size_t elems = (size_t)slot->seq_len * slot->group_heads * slot->head_dim;
    size_t k_elem_size = kvslot_dtype_elem_size(slot->dtype_code);
    size_t v_elem_size = kvslot_dtype_elem_size(slot->v_dtype_code);
    size_t k_bytes = elems * k_elem_size;
    size_t v_bytes = elems * v_elem_size;
    void *k_data = NULL;
    void *v_data = NULL;
    if (elems > 0) {
        k_data = calloc(elems, k_elem_size);
        v_data = calloc(elems, v_elem_size);
        if (k_data == NULL || v_data == NULL) {
            fprintf(stderr, "Failed to allocate readback buffers\n");
            free(k_data);
            free(v_data);
            return 1;
        }
        size_t byte_offset = (size_t)slot->elem_offset * sizeof(int32_t);
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "k_cache", byte_offset, k_bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "v_cache", byte_offset, v_bytes, DPU_XFER_DEFAULT));
    }
    if (write_exact(stdout, &out, sizeof(out)) != 0
        || (elems > 0 && write_exact(stdout, k_data, k_bytes) != 0)
        || (elems > 0 && write_exact(stdout, v_data, v_bytes) != 0)
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

static int handle_get_profile(kvslot_runner_t *runner)
{
    if (runner == NULL) {
        return 1;
    }
    if (write_exact(stdout, &runner->profile, sizeof(runner->profile)) != 0) {
        fprintf(stderr, "Failed to write kvslot profile stats\n");
        return 1;
    }
    return flush_exact(stdout);
}

static int slim_qk_slot_command_supported(uint32_t command)
{
#ifndef KVSLOT_SLIM_QK_SLOT_ONLY
    (void)command;
    return 1;
#else
    switch (command) {
    case KVSLOT_CMD_ALLOCATE:
    case KVSLOT_CMD_APPEND:
    case KVSLOT_CMD_READBACK:
    case KVSLOT_CMD_FREE:
    case KVSLOT_CMD_GET_STATS:
    case KVSLOT_CMD_GET_PROFILE:
    case KVSLOT_CMD_GET_TOPOLOGY:
    case KVSLOT_CMD_QK_SOFTMAX_AV_PARTIAL_BATCH:
        return 1;
    default:
        return 0;
    }
#endif
}

static int handle_get_topology(kvslot_runner_t *runner)
{
    kvslot_topology_header_t header;
    if (runner == NULL) {
        return 1;
    }
    memset(&header, 0, sizeof(header));
    header.nr_dpus = runner->nr_dpus;
    header.nr_ranks = runner->nr_ranks;
    if (write_exact(stdout, &header, sizeof(header)) != 0) {
        fprintf(stderr, "Failed to write kvslot topology header\n");
        return 1;
    }
    for (uint32_t physical_dpu_id = 0; physical_dpu_id < runner->nr_dpus; ++physical_dpu_id) {
        kvslot_topology_item_t item;
        struct dpu_t *dpu_ptr = dpu_from_set(runner->physical_dpus[physical_dpu_id]);
        struct dpu_rank_t *rank_ptr = dpu_ptr != NULL ? dpu_get_rank(dpu_ptr) : NULL;
        memset(&item, 0, sizeof(item));
        item.logical_dpu_id = physical_dpu_id;
        item.rank_index = physical_dpu_id < runner->nr_dpus ? runner->physical_dpu_rank_indices[physical_dpu_id] : 0;
        item.rank_id = rank_ptr != NULL ? (uint32_t)dpu_get_rank_id(rank_ptr) : 0;
        if (write_exact(stdout, &item, sizeof(item)) != 0) {
            fprintf(stderr, "Failed to write kvslot topology item %u\n", physical_dpu_id);
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
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid qk slot batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
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
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_qk_launch_round(
            runner,
            items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk slot batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_qk_round(runner, items, round_indices, round_count)) {
            record_qk_round_profile(runner, items, round_indices, round_count, 1);
            if (execute_batched_qk_round(runner, items, round_indices, round_count, NULL) != 0) {
                fprintf(stderr, "Failed to execute batched qk round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_qk_round_profile(runner, items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch qk batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish qk batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_qk_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
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

static int handle_qk_slot_grouped_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    qk_slot_item_t *items = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read grouped qk slot batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid grouped qk slot batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
        return 1;
    }

    if (write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write grouped qk slot batch response header\n");
        return 1;
    }

    items = calloc(args.num_slots, sizeof(*items));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (items == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate grouped qk slot batch state\n");
        free(used_dpus);
        free(processed);
        free(items);
        return 1;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        uint32_t segment_count = 0;
        uint32_t slot_ids[KVSLOT_MAX_GROUP_SEGMENTS];
        uint32_t segment_lengths[KVSLOT_MAX_GROUP_SEGMENTS];
        kvslot_qk_slot_batch_item_args_t first_item_args;
        int have_first = 0;
        if (read_exact(stdin, &segment_count, sizeof(segment_count)) != 0) {
            fprintf(stderr, "Failed to read grouped qk segment count %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (segment_count == 0 || segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
            fprintf(stderr, "Invalid grouped qk segment_count=%u at item %u\n", segment_count, idx);
            rc = 1;
            goto cleanup;
        }
        memset(&first_item_args, 0, sizeof(first_item_args));
        for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
            kvslot_io_header_t header;
            kvslot_qk_slot_batch_item_args_t item_args;
            if (read_exact(stdin, &header, sizeof(header)) != 0) {
                fprintf(stderr, "Failed to read grouped qk item header %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            if (header.magic != KVSLOT_MAGIC || header.command != KVSLOT_CMD_QK_SLOT_GROUPED_BATCH) {
                fprintf(stderr, "Invalid grouped qk item header %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            if (read_exact(stdin, &item_args, sizeof(item_args)) != 0) {
                fprintf(stderr, "Failed to read grouped qk item args %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            if (!have_first) {
                first_item_args = item_args;
                have_first = 1;
            } else if (item_args.num_heads != first_item_args.num_heads || item_args.head_dim != first_item_args.head_dim) {
                fprintf(stderr, "Grouped qk item shape mismatch at %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            slot_ids[seg_idx] = header.slot_id;
            segment_lengths[seg_idx] = item_args.window;
        }
        {
            kvslot_qk_softmax_av_batch_item_args_t fused_args;
            fused_args.num_heads = first_item_args.num_heads;
            fused_args.window = 0;
            fused_args.head_dim = first_item_args.head_dim;
            fused_args.score_scale = 1.0f;
            if (prepare_grouped_qk_slot_item_header(runner, slot_ids, segment_lengths, segment_count, &fused_args, &items[idx]) != 0) {
                fprintf(stderr, "Failed to prepare grouped qk item %u\n", idx);
                rc = 1;
                goto cleanup;
            }
            if (read_qk_slot_item_payload(stdin, &items[idx]) != 0) {
                fprintf(stderr, "Failed to read grouped qk payload %u\n", idx);
                rc = 1;
                goto cleanup;
            }
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_qk_launch_round(
            runner,
            items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build grouped qk slot batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_qk_round(runner, items, round_indices, round_count)) {
            record_qk_round_profile(runner, items, round_indices, round_count, 1);
            if (execute_batched_qk_round(runner, items, round_indices, round_count, NULL) != 0) {
                fprintf(stderr, "Failed to execute batched grouped qk round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_qk_round_profile(runner, items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch grouped qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish grouped qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_qk_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
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
            fprintf(stderr, "Failed to write grouped qk item header %u\n", idx);
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
                    fprintf(stderr, "Failed to write grouped qk score %u:%u:%u\n", idx, head_idx, pos);
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
    free(used_dpus);
    free(processed);
    free(items);
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush grouped qk slot batch response\n");
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
    if (num_heads == 0 || num_heads > KVSLOT_MAX_HEADS) {
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
    item->raw_row_sum_bits = calloc(num_heads > 0 ? num_heads : 1u, sizeof(*item->raw_row_sum_bits));
    if (item->local_head_indices == NULL || item->queries == NULL || item->raw_scores == NULL || item->raw_row_max_bits == NULL || item->raw_row_sum_bits == NULL) {
        fprintf(stderr, "Failed to allocate qk slot batch buffers\n");
        return 1;
    }

    item->runtime_args.seq_len = item->slot->seq_len;
    item->runtime_args.group_heads = item->slot->group_heads;
    item->runtime_args.head_dim = item->slot->head_dim;
    item->runtime_args.dtype_code = item->slot->dtype_code;
    item->runtime_args.elem_offset = item->slot->elem_offset;
    item->runtime_args.v_elem_offset = item->slot->elem_offset;
    item->runtime_args.k_scale = item->slot->k_scale;
    item->runtime_args.v_scale = item->slot->v_scale;
    item->runtime_args.v_dtype_code = item->slot->v_dtype_code;

    item->slot_args.num_heads = num_heads;
    item->slot_args.window = window;
    item->slot_args.head_dim = head_dim;
    item->slot_args.mode = KVSLOT_QK_SLOT_MODE_RAW_SCORES;
    item->slot_args.score_scale = item_args->score_scale;
    item->ready = 1;
    return 0;
}

static void restrict_qk_slot_item_to_tail_window(qk_slot_item_t *item)
{
    uint32_t token_offset;

    if (item == NULL || item->slot == NULL || item->window >= item->slot->seq_len) {
        return;
    }

    token_offset = item->slot->seq_len - item->window;
    item->runtime_args.seq_len = item->window;
    item->runtime_args.elem_offset = item->slot->elem_offset
        + kvslot_logical_elem_offset_to_packed_words(
            token_offset * item->slot->group_heads * item->slot->head_dim,
            item->slot->dtype_code);
    item->runtime_args.v_elem_offset = item->slot->elem_offset
        + kvslot_logical_elem_offset_to_packed_words(
            token_offset * item->slot->group_heads * item->slot->head_dim,
            item->slot->v_dtype_code);
}

static void restrict_av_item_to_qk_tail_window(av_item_t *item, const qk_slot_item_t *qk_item)
{
    uint32_t token_offset;

    if (item == NULL || qk_item == NULL || item->slot == NULL || qk_item->window >= item->slot->seq_len) {
        return;
    }

    token_offset = item->slot->seq_len - qk_item->window;
    item->runtime_args.seq_len = qk_item->window;
    item->runtime_args.elem_offset = item->slot->elem_offset
        + kvslot_logical_elem_offset_to_packed_words(
            token_offset * item->slot->group_heads * item->slot->head_dim,
            item->slot->dtype_code);
    item->runtime_args.v_elem_offset = item->slot->elem_offset
        + kvslot_logical_elem_offset_to_packed_words(
            token_offset * item->slot->group_heads * item->slot->head_dim,
            item->slot->v_dtype_code);
    item->out.seq_len = qk_item->window;
}

static int prepare_grouped_qk_slot_item_header(
    kvslot_runner_t *runner,
    const uint32_t *slot_ids,
    const uint32_t *segment_lengths,
    uint32_t segment_count,
    const kvslot_qk_softmax_av_batch_item_args_t *item_args,
    qk_slot_item_t *item)
{
    uint32_t total_window = 0;
    host_slot_t *first_slot = NULL;
    struct dpu_set_t first_target_dpu;
    uint32_t first_physical_dpu_id = 0;
    uint32_t num_heads;
    uint32_t head_dim;

    if (runner == NULL || slot_ids == NULL || segment_lengths == NULL || item_args == NULL || item == NULL) {
        return 1;
    }
    if (segment_count == 0 || segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
        return 1;
    }
    memset(item, 0, sizeof(*item));

    num_heads = item_args->num_heads;
    head_dim = item_args->head_dim;
    for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
        struct dpu_set_t target_dpu;
        host_slot_t *slot = NULL;
        uint32_t slot_id = slot_ids[seg_idx];
        uint32_t physical_dpu_id;
        if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
            fprintf(stderr, "Invalid grouped qk slot id %u\n", slot_id);
            return 1;
        }
        if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0 || slot == NULL || slot->capacity == 0) {
            fprintf(stderr, "Failed to locate grouped qk slot %u\n", slot_id);
            return 1;
        }
        physical_dpu_id = slot_id % runner->nr_dpus;
        if (seg_idx == 0) {
            first_slot = slot;
            first_target_dpu = target_dpu;
            first_physical_dpu_id = physical_dpu_id;
            item->slot = slot;
            item->target_dpu = target_dpu;
            item->slot_id = slot_id;
            item->physical_dpu_id = physical_dpu_id;
        } else {
            if (physical_dpu_id != first_physical_dpu_id) {
                fprintf(stderr, "Grouped qk item spans multiple physical DPUs\n");
                return 1;
            }
            if (slot->group_heads != first_slot->group_heads
                || slot->head_dim != first_slot->head_dim
                || slot->dtype_code != first_slot->dtype_code
                || slot->v_dtype_code != first_slot->v_dtype_code
                || slot->k_scale != first_slot->k_scale
                || slot->v_scale != first_slot->v_scale) {
                fprintf(stderr, "Grouped qk item shape mismatch across segments\n");
                return 1;
            }
            item->target_dpu = first_target_dpu;
        }
        if (segment_lengths[seg_idx] > slot->seq_len) {
            fprintf(stderr, "Grouped qk segment length exceeds slot seq len\n");
            return 1;
        }
        if (segment_lengths[seg_idx] > KVSLOT_MAX_CAPACITY || total_window + segment_lengths[seg_idx] > KVSLOT_MAX_CAPACITY) {
            fprintf(stderr, "Grouped qk total window exceeds DPU capacity %u\n", KVSLOT_MAX_CAPACITY);
            return 1;
        }
        item->segment_slots[seg_idx] = slot;
        item->segment_slot_ids[seg_idx] = slot_id;
        item->segment_lengths[seg_idx] = segment_lengths[seg_idx];
        item->segment_runtime_args[seg_idx].seq_len = segment_lengths[seg_idx];
        item->segment_runtime_args[seg_idx].group_heads = slot->group_heads;
        item->segment_runtime_args[seg_idx].head_dim = slot->head_dim;
        item->segment_runtime_args[seg_idx].dtype_code = slot->dtype_code;
        item->segment_runtime_args[seg_idx].elem_offset = slot->elem_offset
            + kvslot_logical_elem_offset_to_packed_words(
                (slot->seq_len - segment_lengths[seg_idx]) * slot->group_heads * slot->head_dim,
                slot->dtype_code);
        item->segment_runtime_args[seg_idx].v_elem_offset = slot->elem_offset
            + kvslot_logical_elem_offset_to_packed_words(
                (slot->seq_len - segment_lengths[seg_idx]) * slot->group_heads * slot->head_dim,
                slot->v_dtype_code);
        item->segment_runtime_args[seg_idx].k_scale = slot->k_scale;
        item->segment_runtime_args[seg_idx].v_scale = slot->v_scale;
        item->segment_runtime_args[seg_idx].v_dtype_code = slot->v_dtype_code;
        total_window += segment_lengths[seg_idx];
    }

    if (num_heads == 0 || num_heads > KVSLOT_MAX_HEADS) {
        fprintf(stderr, "Invalid grouped qk num_heads=%u for slot %u\n", num_heads, item->slot_id);
        return 1;
    }
    if (head_dim == 0 || head_dim > first_slot->head_dim || head_dim > KVSLOT_MAX_HEAD_DIM) {
        fprintf(stderr, "Invalid grouped qk head_dim=%u for slot %u\n", head_dim, item->slot_id);
        return 1;
    }

    item->num_heads = num_heads;
    item->window = total_window;
    item->score_stride = (total_window + 1u) & ~1u;
    item->head_dim = head_dim;
    item->segment_count = segment_count;
    item->local_head_indices = calloc(num_heads, sizeof(*item->local_head_indices));
    item->queries = calloc((size_t)num_heads * head_dim, sizeof(*item->queries));
    item->raw_scores = calloc(
        (size_t)(item->score_stride > 0 ? item->score_stride : 1u) * num_heads,
        sizeof(*item->raw_scores)
    );
    item->raw_row_max_bits = calloc(num_heads > 0 ? num_heads : 1u, sizeof(*item->raw_row_max_bits));
    item->raw_row_sum_bits = calloc(num_heads > 0 ? num_heads : 1u, sizeof(*item->raw_row_sum_bits));
    if (item->local_head_indices == NULL || item->queries == NULL || item->raw_scores == NULL || item->raw_row_max_bits == NULL || item->raw_row_sum_bits == NULL) {
        fprintf(stderr, "Failed to allocate grouped qk slot buffers\n");
        return 1;
    }

    item->runtime_args.seq_len = total_window;
    item->runtime_args.group_heads = first_slot->group_heads;
    item->runtime_args.head_dim = first_slot->head_dim;
    item->runtime_args.dtype_code = first_slot->dtype_code;
    item->runtime_args.elem_offset = 0;
    item->runtime_args.v_elem_offset = 0;
    item->runtime_args.k_scale = first_slot->k_scale;
    item->runtime_args.v_scale = first_slot->v_scale;
    item->runtime_args.v_dtype_code = first_slot->v_dtype_code;

    item->slot_args.num_heads = num_heads;
    item->slot_args.window = total_window;
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
    free(item->raw_row_sum_bits);
    memset(item, 0, sizeof(*item));
}

static int launch_qk_slot_item_async(const qk_slot_item_t *item, kvslot_profile_stats_t *profile)
{
    uint32_t kernel_command = item != NULL && item->segment_count > 1 ? (KVSLOT_KERNEL_QK_SLOT + 100u) : KVSLOT_KERNEL_QK_SLOT;
    int collect_dpu_phase_profile = dpu_phase_profile_enabled();
    uint64_t start_ns;
    uint64_t launch_start_ns;

    if (item == NULL || !item->ready) {
        return 1;
    }
    if (collect_dpu_phase_profile) {
        kernel_command |= KVSLOT_KERNEL_PROFILE_FLAG;
    }
    start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "runtime_slot_args", 0, &item->runtime_args, sizeof(item->runtime_args)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "qk_slot_args", 0, &item->slot_args, sizeof(item->slot_args)));
    if (item->segment_count > 1) {
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_runtime_slot_args",
            0,
            item->segment_runtime_args,
            (size_t)item->segment_count * sizeof(*item->segment_runtime_args)
        ));
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_segment_lengths",
            0,
            item->segment_lengths,
            (size_t)item->segment_count * sizeof(*item->segment_lengths)
        ));
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_segment_count",
            0,
            &item->segment_count,
            sizeof(item->segment_count)
        ));
    }
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
    if (profile != NULL) {
        profile->qk_fallback_launch_ns += elapsed_ns_since(start_ns);
    }
    launch_start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_launch(item->target_dpu, DPU_ASYNCHRONOUS));
    if (profile != NULL) {
        profile->qk_fallback_launch_ns += elapsed_ns_since(launch_start_ns);
    }
    return 0;
}

static int finish_qk_slot_item(qk_slot_item_t *item, kvslot_profile_stats_t *profile)
{
    size_t score_bytes;
    size_t aligned_score_bytes;
    uint32_t *aligned_scores = NULL;
    uint64_t sync_start_ns;
    uint64_t xfer_start_ns;

    if (item == NULL || !item->ready) {
        return 1;
    }
    sync_start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_sync(item->target_dpu));
    if (profile != NULL) {
        profile->qk_fallback_sync_ns += elapsed_ns_since(sync_start_ns);
    }
    if (profile != NULL && dpu_phase_profile_enabled()) {
        kvslot_meta_t meta;
        memset(&meta, 0, sizeof(meta));
        DPU_ASSERT(dpu_copy_from(item->target_dpu, "kvslot_meta", 0, &meta, sizeof(meta)));
        profile->qk_dpu_cycles_total += meta.cycles;
        profile->qk_dpu_dot_cycles_total += meta.qk_dot_cycles;
        profile->qk_dpu_softmax_cycles_total += meta.qk_softmax_cycles;
        profile->qk_dpu_context_cycles_total += meta.qk_context_cycles;
        profile->qk_dpu_other_cycles_total += meta.qk_other_cycles;
        profile->qk_dpu_profiled_launches += 1;
        profile->qk_dpu_profiled_dpus += 1;
    }
    if (item->slot_args.mode != KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
        return 0;
    }
    if (item->window > 0) {
        score_bytes = (size_t)item->num_heads * item->score_stride * sizeof(*item->raw_scores);
        aligned_score_bytes = (score_bytes + 7u) & ~((size_t)7u);
        if (aligned_score_bytes == score_bytes) {
            xfer_start_ns = monotonic_time_ns();
            DPU_ASSERT(dpu_copy_from(
                item->target_dpu,
                "qk_slot_scores_bits",
                0,
                item->raw_scores,
                score_bytes
            ));
            if (profile != NULL) {
                profile->qk_fallback_xfer_from_ns += elapsed_ns_since(xfer_start_ns);
            }
            return 0;
        }
        aligned_scores = calloc(1, aligned_score_bytes);
        if (aligned_scores == NULL) {
            fprintf(stderr, "Failed to allocate aligned qk slot score buffer (%zu bytes)\n", aligned_score_bytes);
            return 1;
        }
        xfer_start_ns = monotonic_time_ns();
        DPU_ASSERT(dpu_copy_from(
            item->target_dpu,
            "qk_slot_scores_bits",
            0,
            aligned_scores,
            aligned_score_bytes
        ));
        if (profile != NULL) {
            profile->qk_fallback_xfer_from_ns += elapsed_ns_since(xfer_start_ns);
        }
        memcpy(item->raw_scores, aligned_scores, score_bytes);
        free(aligned_scores);
    }
    xfer_start_ns = monotonic_time_ns();
    if (fetch_qk_slot_row_maxes(item) != 0) {
        return 1;
    }
    if (profile != NULL) {
        profile->qk_fallback_xfer_from_ns += elapsed_ns_since(xfer_start_ns);
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

static int fetch_qk_slot_row_sums(qk_slot_item_t *item)
{
    if (item == NULL || !item->ready || item->raw_row_sum_bits == NULL) {
        return 1;
    }
    if (item->num_heads == 0) {
        return 0;
    }
    DPU_ASSERT(dpu_copy_from(
        item->target_dpu,
        "qk_slot_head_indices",
        0,
        item->raw_row_sum_bits,
        (size_t)item->num_heads * sizeof(*item->raw_row_sum_bits)
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
    uint32_t max_window;
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
    max_window = batched_qk_round_max_window(items, round_indices, round_count);
    head_dim = items[round_indices[0]].head_dim;
    mode = items[round_indices[0]].slot_args.mode;
    query_bytes = (size_t)num_heads * head_dim * sizeof(float);
    score_bytes = (size_t)num_heads * (((size_t)max_window + 1u) & ~1u) * sizeof(uint32_t);

    if ((query_bytes % 8u) != 0 || (score_bytes % 8u) != 0) {
        return 0;
    }

    if (ensure_round_scratch_rank_used(runner) != 0) {
        return 0;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 1; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        if (!item->ready || item->num_heads != num_heads || item->head_dim != head_dim) {
            return 0;
        }
        if (item->slot_args.mode != mode) {
            return 0;
        }
        if ((item->segment_count > 1) != (items[round_indices[0]].segment_count > 1)) {
            return 0;
        }
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        uint32_t rank_idx;
        if (item->physical_dpu_id >= runner->nr_dpus) {
            return 0;
        }
        rank_idx = runner->physical_dpu_rank_indices[item->physical_dpu_id];
        if (rank_idx >= runner->nr_ranks) {
            return 0;
        }
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    if (rank_spread_alloc_experiment_enabled()
        && active_rank_count > 1
        && !rank_spread_multi_rank_batch_experiment_enabled()) {
        /*
         * Clover rank-spread alloc backs logical DPUs with one selected DPU per
         * physical rank. A batched round over DPU_SET_RANKS would iterate every
         * DPU in each active rank and push dummy payloads to the unused ones,
         * which is much more expensive than launching the selected DPUs
         * individually. Keep the original batched path for single-rank rounds.
         */
        return 0;
    }
    if (active_rank_count > qk_max_active_ranks_limit()) {
        return 0;
    }
    return 1;
}

static uint32_t batched_qk_round_max_window(
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    uint32_t max_window = 0;

    if (items == NULL || round_indices == NULL) {
        return 0;
    }
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        if (item->window > max_window) {
            max_window = item->window;
        }
    }
    return max_window;
}

static int execute_batched_qk_round(
    kvslot_runner_t *runner,
    qk_slot_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count,
    av_item_t *context_items)
{
    qk_slot_item_t **items_by_dpu = NULL;
    uint32_t *dummy_head_indices = NULL;
    float *dummy_queries = NULL;
    uint32_t *dummy_scores = NULL;
    uint32_t *dummy_row_max_bits = NULL;
    uint32_t *dummy_row_sum_bits = NULL;
    kvslot_runtime_slot_args_t *dummy_segment_runtime_args = NULL;
    uint32_t *dummy_segment_lengths = NULL;
    void **score_xfer_buffers = NULL;
    kvslot_runtime_slot_args_t zero_runtime_args;
    kvslot_qk_slot_args_t zero_slot_args;
    uint32_t kernel_command = KVSLOT_KERNEL_QK_SLOT;
    uint32_t num_heads;
    uint32_t max_window;
    uint32_t head_dim;
    uint32_t max_segment_count = 0;
    uint32_t zero_segment_count = 0;
    int grouped_round = 0;
    uint32_t active_rank_count = 0;
    size_t head_index_bytes;
    size_t query_bytes;
    size_t score_bytes;
    size_t context_bytes = 0;
    uint8_t *rank_used = NULL;
    struct dpu_rank_t **active_ranks = NULL;
    struct dpu_set_t launch_set;
    struct dpu_set_t dpu;
    uint64_t round_start_ns;
    uint64_t xfer_to_start_ns;
    uint64_t launch_start_ns;
    uint64_t xfer_from_start_ns;
    uint64_t xfer_to_ns = 0;
    uint64_t launch_ns = 0;
    uint64_t xfer_from_ns = 0;
    int collect_dpu_phase_profile = dpu_phase_profile_enabled();

    if (!can_use_batched_qk_round(runner, items, round_indices, round_count)) {
        return 1;
    }
    round_start_ns = monotonic_time_ns();

    if (ensure_round_scratch_qk_items(runner) != 0) {
        fprintf(stderr, "Failed to allocate batched qk round map\n");
        return 1;
    }
    items_by_dpu = runner->scratch.qk_items_by_dpu;

    memset(&zero_runtime_args, 0, sizeof(zero_runtime_args));
    memset(&zero_slot_args, 0, sizeof(zero_slot_args));
    num_heads = items[round_indices[0]].num_heads;
    max_window = batched_qk_round_max_window(items, round_indices, round_count);
    head_dim = items[round_indices[0]].head_dim;
    head_index_bytes = (size_t)num_heads * sizeof(uint32_t);
    query_bytes = (size_t)num_heads * head_dim * sizeof(float);
    score_bytes = (size_t)num_heads * (((size_t)max_window + 1u) & ~1u) * sizeof(uint32_t);
    grouped_round = items[round_indices[0]].segment_count > 1;
    memset(&launch_set, 0, sizeof(launch_set));
    if (context_items != NULL
        && items[round_indices[0]].slot_args.mode != KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
        context_bytes = context_items[round_indices[0]].padded_context_bytes;
        for (uint32_t pos = 0; pos < round_count; ++pos) {
            av_item_t *context_item = &context_items[round_indices[pos]];
            if (!context_item->ready || !context_item->context_from_qk_kernel
                || context_item->context == NULL
                || context_item->padded_context_bytes != context_bytes) {
                context_bytes = 0;
                break;
            }
        }
    }

    for (uint32_t pos = 0; pos < round_count; ++pos) {
        qk_slot_item_t *item = &items[round_indices[pos]];
        if (item->physical_dpu_id >= runner->nr_dpus) {
            fprintf(stderr, "Invalid physical dpu id %u in batched qk round\n", item->physical_dpu_id);
            return 1;
        }
        if ((item->segment_count > 1) != grouped_round) {
            fprintf(stderr, "Mixed grouped/regular qk items in a batched round\n");
            return 1;
        }
        items_by_dpu[item->physical_dpu_id] = item;
        if (item->segment_count > max_segment_count) {
            max_segment_count = item->segment_count;
        }
    }
    if (ensure_round_scratch_rank_used(runner) != 0) {
        fprintf(stderr, "Failed to allocate batched qk rank mask\n");
        return 1;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t rank_idx = runner->physical_dpu_rank_indices[items[round_indices[pos]].physical_dpu_id];
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    if (ensure_round_scratch_active_ranks(runner, active_rank_count) != 0) {
        fprintf(stderr, "Failed to allocate batched qk active ranks\n");
        return 1;
    }
    active_ranks = runner->scratch.active_ranks;
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
        if (ensure_round_scratch_dummy_head_indices(runner, head_index_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched qk dummy head indices\n");
            return 1;
        }
        dummy_head_indices = runner->scratch.dummy_head_indices;
    }
    if (query_bytes > 0) {
        if (ensure_round_scratch_dummy_queries(runner, query_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched qk dummy queries\n");
            return 1;
        }
        dummy_queries = runner->scratch.dummy_queries;
    }
    if (score_bytes > 0) {
        if (ensure_round_scratch_dummy_scores(runner, score_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched qk dummy scores\n");
            return 1;
        }
        dummy_scores = runner->scratch.dummy_scores;
        if (ensure_round_scratch_xfer_buffers(runner) != 0) {
            fprintf(stderr, "Failed to allocate batched qk padded score buffers\n");
            return 1;
        }
        score_xfer_buffers = runner->scratch.xfer_buffers;
    }
    if (context_bytes > 0) {
        if (ensure_round_scratch_dummy_context(runner, context_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched qk fused context dummy buffer\n");
            return 1;
        }
    }
    if (num_heads > 0) {
        if (ensure_round_scratch_dummy_rows(runner, (size_t)num_heads * sizeof(*dummy_row_max_bits)) != 0) {
            fprintf(stderr, "Failed to allocate batched qk dummy row maxes\n");
            return 1;
        }
        dummy_row_max_bits = runner->scratch.dummy_row_max_bits;
        dummy_row_sum_bits = runner->scratch.dummy_row_sum_bits;
    }
    if (grouped_round && max_segment_count > 0) {
        kernel_command = KVSLOT_KERNEL_QK_SLOT + 100u;
        if (ensure_round_scratch_dummy_segments(runner, max_segment_count) != 0) {
            fprintf(stderr, "Failed to allocate grouped qk dummy metadata\n");
            return 1;
        }
        dummy_segment_runtime_args = runner->scratch.dummy_segment_runtime_args;
        dummy_segment_lengths = runner->scratch.dummy_segment_lengths;
    }
    if (collect_dpu_phase_profile) {
        kernel_command |= KVSLOT_KERNEL_PROFILE_FLAG;
    }

    xfer_to_start_ns = monotonic_time_ns();
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

    if (grouped_round && max_segment_count > 0) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)item->segment_runtime_args : (void *)dummy_segment_runtime_args));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_runtime_slot_args",
            0,
            (size_t)max_segment_count * sizeof(*dummy_segment_runtime_args),
            DPU_XFER_DEFAULT));

        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)item->segment_lengths : (void *)dummy_segment_lengths));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_segment_lengths",
            0,
            (size_t)max_segment_count * sizeof(*dummy_segment_lengths),
            DPU_XFER_DEFAULT));

        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)&item->segment_count : (void *)&zero_segment_count));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_segment_count",
            0,
            sizeof(zero_segment_count),
            DPU_XFER_DEFAULT));
    }

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
    xfer_to_ns += elapsed_ns_since(xfer_to_start_ns);

    launch_start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_launch(launch_set, DPU_SYNCHRONOUS));
    launch_ns += elapsed_ns_since(launch_start_ns);

    if (collect_dpu_phase_profile) {
        kvslot_meta_t meta;
        for (uint32_t pos = 0; pos < round_count; ++pos) {
            qk_slot_item_t *item = &items[round_indices[pos]];
            memset(&meta, 0, sizeof(meta));
            DPU_ASSERT(dpu_copy_from(item->target_dpu, "kvslot_meta", 0, &meta, sizeof(meta)));
            record_qk_dpu_meta(runner, &meta);
        }
        runner->profile.qk_dpu_profiled_launches += 1;
    }

    xfer_from_start_ns = monotonic_time_ns();
    if (score_bytes > 0 && items[round_indices[0]].slot_args.mode == KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            void *score_ptr = (void *)dummy_scores;
            if (item != NULL) {
                if ((size_t)item->num_heads * item->score_stride * sizeof(*item->raw_scores) == score_bytes) {
                    score_ptr = (void *)item->raw_scores;
                } else {
                    void *padded_scores = calloc(1, score_bytes);
                    if (padded_scores == NULL) {
                        fprintf(stderr, "Failed to allocate padded batched qk scores\n");
                        free_round_scratch_xfer_payloads(runner);
                        return 1;
                    }
                    score_xfer_buffers[item->physical_dpu_id] = padded_scores;
                    score_ptr = padded_scores;
                }
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, score_ptr));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "qk_slot_scores_bits",
            0,
            score_bytes,
            DPU_XFER_DEFAULT));
    }
    if (num_heads > 0) {
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
    if (num_heads > 0 && items[round_indices[0]].slot_args.mode != KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(dpu, item != NULL ? (void *)item->raw_row_sum_bits : (void *)dummy_row_sum_bits));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "qk_slot_head_indices",
            0,
            (size_t)num_heads * sizeof(*dummy_row_sum_bits),
            DPU_XFER_DEFAULT));
    }
    if (context_bytes > 0) {
        DPU_FOREACH(launch_set, dpu) {
            qk_slot_item_t *item = find_qk_round_item_for_dpu(runner, items_by_dpu, dpu);
            void *context_ptr = (void *)runner->scratch.dummy_context;
            if (item != NULL) {
                av_item_t *context_item = &context_items[item - items];
                context_ptr = (void *)context_item->context;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, context_ptr));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_FROM_DPU,
            "av_context_bits",
            0,
            context_bytes,
            DPU_XFER_DEFAULT));
        for (uint32_t pos = 0; pos < round_count; ++pos) {
            context_items[round_indices[pos]].context_prefetched = 1;
        }
    }
    xfer_from_ns += elapsed_ns_since(xfer_from_start_ns);

    if (score_bytes > 0 && items[round_indices[0]].slot_args.mode == KVSLOT_QK_SLOT_MODE_RAW_SCORES) {
        for (uint32_t pos = 0; pos < round_count; ++pos) {
            qk_slot_item_t *item = &items[round_indices[pos]];
            void *buffer = score_xfer_buffers != NULL ? score_xfer_buffers[item->physical_dpu_id] : NULL;
            size_t item_score_bytes = (size_t)item->num_heads * item->score_stride * sizeof(*item->raw_scores);
            if (buffer != NULL && item_score_bytes > 0) {
                memcpy(item->raw_scores, buffer, item_score_bytes);
            }
        }
    }
    record_qk_round_timing(
        runner,
        1,
        elapsed_ns_since(round_start_ns),
        xfer_to_ns,
        launch_ns,
        xfer_from_ns,
        0);

    free_round_scratch_xfer_payloads(runner);
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
    item->output_heads = slot->group_heads;

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
    item->runtime_args.v_elem_offset = slot->elem_offset;
    item->runtime_args.k_scale = slot->k_scale;
    item->runtime_args.v_scale = slot->v_scale;
    item->runtime_args.v_dtype_code = slot->v_dtype_code;

    item->out.capacity = slot->capacity;
    item->out.seq_len = slot->seq_len;
    item->out.group_heads = slot->group_heads;
    item->out.head_dim = slot->head_dim;
    item->out.dtype_code = slot->dtype_code;
    item->out.k_scale = slot->k_scale;
    item->out.v_scale = slot->v_scale;
    item->out.v_dtype_code = slot->v_dtype_code;
    item->context_prefetched = 0;
    item->ready = 1;
    return 0;
}

static int prepare_grouped_av_item_header(
    kvslot_runner_t *runner,
    const uint32_t *slot_ids,
    const uint32_t *segment_lengths,
    uint32_t segment_count,
    av_item_t *item)
{
    uint32_t total_seq_len = 0;
    host_slot_t *first_slot = NULL;
    struct dpu_set_t first_target_dpu;
    uint32_t first_physical_dpu_id = 0;

    if (runner == NULL || slot_ids == NULL || segment_lengths == NULL || item == NULL) {
        return 1;
    }
    if (segment_count == 0 || segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
        return 1;
    }
    memset(item, 0, sizeof(*item));

    for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
        struct dpu_set_t target_dpu;
        host_slot_t *slot = NULL;
        uint32_t slot_id = slot_ids[seg_idx];
        uint32_t physical_dpu_id;
        if (slot_id >= runner->nr_dpus * KVSLOT_MAX_SLOTS_PER_DPU) {
            fprintf(stderr, "Invalid grouped av slot id %u\n", slot_id);
            return 1;
        }
        if (runner_get_dpu_and_slot(runner, slot_id, &target_dpu, &slot) != 0 || slot == NULL || slot->capacity == 0) {
            fprintf(stderr, "Failed to locate grouped av slot %u\n", slot_id);
            return 1;
        }
        physical_dpu_id = slot_id % runner->nr_dpus;
        if (seg_idx == 0) {
            first_slot = slot;
            first_target_dpu = target_dpu;
            first_physical_dpu_id = physical_dpu_id;
            item->slot = slot;
            item->target_dpu = target_dpu;
            item->slot_id = slot_id;
            item->physical_dpu_id = physical_dpu_id;
            item->out.capacity = 0;
            item->out.group_heads = slot->group_heads;
            item->output_heads = slot->group_heads;
            item->out.head_dim = slot->head_dim;
            item->out.dtype_code = slot->dtype_code;
            item->out.k_scale = slot->k_scale;
            item->out.v_scale = slot->v_scale;
            item->out.v_dtype_code = slot->v_dtype_code;
        } else {
            if (physical_dpu_id != first_physical_dpu_id) {
                fprintf(stderr, "Grouped av item spans multiple physical DPUs\n");
                return 1;
            }
            if (slot->group_heads != first_slot->group_heads
                || slot->head_dim != first_slot->head_dim
                || slot->dtype_code != first_slot->dtype_code
                || slot->v_dtype_code != first_slot->v_dtype_code
                || slot->k_scale != first_slot->k_scale
                || slot->v_scale != first_slot->v_scale) {
                fprintf(stderr, "Grouped av item shape mismatch across segments\n");
                return 1;
            }
            item->target_dpu = first_target_dpu;
        }
        if (segment_lengths[seg_idx] > slot->seq_len) {
            fprintf(stderr, "Grouped av segment length exceeds slot seq len\n");
            return 1;
        }
        if (segment_lengths[seg_idx] > KVSLOT_MAX_CAPACITY || total_seq_len + segment_lengths[seg_idx] > KVSLOT_MAX_CAPACITY) {
            fprintf(stderr, "Grouped av total seq len exceeds DPU capacity %u\n", KVSLOT_MAX_CAPACITY);
            return 1;
        }
        item->segment_slots[seg_idx] = slot;
        item->segment_slot_ids[seg_idx] = slot_id;
        item->segment_lengths[seg_idx] = segment_lengths[seg_idx];
        item->segment_runtime_args[seg_idx].seq_len = segment_lengths[seg_idx];
        item->segment_runtime_args[seg_idx].group_heads = slot->group_heads;
        item->segment_runtime_args[seg_idx].head_dim = slot->head_dim;
        item->segment_runtime_args[seg_idx].dtype_code = slot->dtype_code;
        item->segment_runtime_args[seg_idx].elem_offset = slot->elem_offset
            + kvslot_logical_elem_offset_to_packed_words(
                (slot->seq_len - segment_lengths[seg_idx]) * slot->group_heads * slot->head_dim,
                slot->dtype_code);
        item->segment_runtime_args[seg_idx].v_elem_offset = slot->elem_offset
            + kvslot_logical_elem_offset_to_packed_words(
                (slot->seq_len - segment_lengths[seg_idx]) * slot->group_heads * slot->head_dim,
                slot->v_dtype_code);
        item->segment_runtime_args[seg_idx].k_scale = slot->k_scale;
        item->segment_runtime_args[seg_idx].v_scale = slot->v_scale;
        item->segment_runtime_args[seg_idx].v_dtype_code = slot->v_dtype_code;
        total_seq_len += segment_lengths[seg_idx];
        item->out.capacity += slot->capacity;
    }

    item->segment_count = segment_count;
    item->runtime_args.seq_len = total_seq_len;
    item->runtime_args.group_heads = first_slot->group_heads;
    item->runtime_args.head_dim = first_slot->head_dim;
    item->runtime_args.dtype_code = first_slot->dtype_code;
    item->runtime_args.elem_offset = 0;
    item->runtime_args.v_elem_offset = 0;
    item->runtime_args.k_scale = first_slot->k_scale;
    item->runtime_args.v_scale = first_slot->v_scale;
    item->runtime_args.v_dtype_code = first_slot->v_dtype_code;
    item->out.seq_len = total_seq_len;
    item->out.k_scale = first_slot->k_scale;
    item->out.v_scale = first_slot->v_scale;
    item->out.v_dtype_code = first_slot->v_dtype_code;
    item->weight_bytes = (size_t)total_seq_len * first_slot->group_heads * sizeof(float);
    item->context_bytes = (size_t)first_slot->group_heads * first_slot->head_dim * sizeof(float);
    item->padded_weight_bytes = ((item->weight_bytes + 7u) / 8u) * 8u;
    item->padded_context_bytes = ((item->context_bytes + 7u) / 8u) * 8u;
    item->output_heads = first_slot->group_heads;
    if (item->weight_bytes > 0) {
        item->weights = calloc(1, item->padded_weight_bytes);
        if (item->weights == NULL) {
            fprintf(stderr, "Failed to allocate grouped av weights buffer\n");
            return 1;
        }
    }
    if (item->context_bytes > 0) {
        item->context = calloc(1, item->padded_context_bytes);
        if (item->context == NULL) {
            fprintf(stderr, "Failed to allocate grouped av context buffer\n");
            return 1;
        }
    }
    item->context_prefetched = 0;
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

static int resize_av_item_context_rows(av_item_t *item, uint32_t output_heads)
{
    size_t context_bytes;
    size_t padded_context_bytes;
    float *context = NULL;

    if (item == NULL || item->slot == NULL || output_heads == 0 || output_heads > KVSLOT_MAX_HEADS) {
        return 1;
    }

    context_bytes = (size_t)output_heads * item->slot->head_dim * sizeof(float);
    padded_context_bytes = ((context_bytes + 7u) / 8u) * 8u;
    if (padded_context_bytes != item->padded_context_bytes) {
        context = calloc(1, padded_context_bytes);
        if (context == NULL) {
            fprintf(stderr, "Failed to allocate resized av context buffer\n");
            return 1;
        }
        free(item->context);
        item->context = context;
    } else if (item->context != NULL && padded_context_bytes > 0) {
        memset(item->context, 0, padded_context_bytes);
    }

    item->output_heads = output_heads;
    item->context_bytes = context_bytes;
    item->padded_context_bytes = padded_context_bytes;
    item->out.group_heads = output_heads;
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
    if (av_item->runtime_args.seq_len != qk_item->window) {
        fprintf(stderr, "QK/AV window mismatch for slot %u: qk=%u av=%u\n", qk_item->slot_id, qk_item->window, av_item->runtime_args.seq_len);
        return 1;
    }
    if (resize_av_item_context_rows(av_item, qk_item->num_heads) != 0) {
        return 1;
    }
    if (qk_item->slot_args.mode == KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED) {
        if (av_item->slot->group_heads != qk_item->num_heads) {
            fprintf(stderr, "QK/AV softmax-normalized head mismatch for slot %u: qk=%u av=%u\n", qk_item->slot_id, qk_item->num_heads, av_item->slot->group_heads);
            return 1;
        }
        av_item->weights_resident_on_dpu = 1;
        return 0;
    }
    if (qk_item->slot_args.mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED
        || qk_item->slot_args.mode == KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED) {
        av_item->context_from_qk_kernel = 1;
        return 0;
    }
    fprintf(stderr, "QK-softmax-av item %u did not request a fused softmax/av mode\n", qk_item->slot_id);
    return 1;
}

static int finish_av_item(av_item_t *item, kvslot_profile_stats_t *profile)
{
    uint64_t sync_start_ns;
    uint64_t xfer_start_ns;

    if (item == NULL || !item->ready) {
        return 1;
    }
    if (item->context_prefetched) {
        return 0;
    }
    if (!item->context_from_qk_kernel) {
        sync_start_ns = monotonic_time_ns();
        DPU_ASSERT(dpu_sync(item->target_dpu));
        if (profile != NULL) {
            profile->av_fallback_sync_ns += elapsed_ns_since(sync_start_ns);
        }
    }
    if (item->context_bytes > 0) {
        xfer_start_ns = monotonic_time_ns();
        DPU_ASSERT(dpu_copy_from(item->target_dpu, "av_context_bits", 0, item->context, item->padded_context_bytes));
        if (profile != NULL) {
            profile->av_fallback_xfer_from_ns += elapsed_ns_since(xfer_start_ns);
        }
    }
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

static int launch_av_item_async(const av_item_t *item, kvslot_profile_stats_t *profile)
{
    uint32_t kernel_command = item != NULL && item->segment_count > 1 ? (KVSLOT_KERNEL_AV + 100u) : KVSLOT_KERNEL_AV;
    uint64_t start_ns;
    uint64_t launch_start_ns;

    if (item == NULL || !item->ready) {
        return 1;
    }
    start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "kvslot_kernel_command", 0, &kernel_command, sizeof(kernel_command)));
    DPU_ASSERT(dpu_copy_to(item->target_dpu, "runtime_slot_args", 0, &item->runtime_args, sizeof(item->runtime_args)));
    if (item->segment_count > 1) {
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_runtime_slot_args",
            0,
            item->segment_runtime_args,
            (size_t)item->segment_count * sizeof(*item->segment_runtime_args)
        ));
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_segment_lengths",
            0,
            item->segment_lengths,
            (size_t)item->segment_count * sizeof(*item->segment_lengths)
        ));
        DPU_ASSERT(dpu_copy_to(
            item->target_dpu,
            "grouped_segment_count",
            0,
            &item->segment_count,
            sizeof(item->segment_count)
        ));
    }
    if (item->weight_bytes > 0 && !item->weights_resident_on_dpu) {
        DPU_ASSERT(dpu_copy_to(item->target_dpu, "av_weights_bits", 0, item->weights, item->padded_weight_bytes));
    }
    if (profile != NULL) {
        profile->av_fallback_launch_ns += elapsed_ns_since(start_ns);
    }
    launch_start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_launch(item->target_dpu, DPU_ASYNCHRONOUS));
    if (profile != NULL) {
        profile->av_fallback_launch_ns += elapsed_ns_since(launch_start_ns);
    }
    return 0;
}

static int can_use_batched_av_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    size_t padded_context_bytes;
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count <= 1) {
        return 0;
    }
    if (runner->nr_ranks == 0 || runner->physical_dpu_rank_indices == NULL) {
        return 0;
    }

    padded_context_bytes = items[round_indices[0]].padded_context_bytes;
    for (uint32_t pos = 1; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        if (!item->ready) {
            return 0;
        }
        if (item->runtime_args.group_heads != items[round_indices[0]].runtime_args.group_heads
            || item->runtime_args.head_dim != items[round_indices[0]].runtime_args.head_dim
            || item->runtime_args.dtype_code != items[round_indices[0]].runtime_args.dtype_code
            || item->runtime_args.v_dtype_code != items[round_indices[0]].runtime_args.v_dtype_code
            || item->padded_context_bytes != padded_context_bytes
            || item->weights_resident_on_dpu != items[round_indices[0]].weights_resident_on_dpu
            || item->context_from_qk_kernel != items[round_indices[0]].context_from_qk_kernel
            || ((item->segment_count > 1) != (items[round_indices[0]].segment_count > 1))) {
            return 0;
        }
    }

    if (ensure_round_scratch_rank_used(runner) != 0) {
        return 0;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        uint32_t rank_idx;
        if (item->physical_dpu_id >= runner->nr_dpus) {
            return 0;
        }
        rank_idx = runner->physical_dpu_rank_indices[item->physical_dpu_id];
        if (rank_idx >= runner->nr_ranks) {
            return 0;
        }
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    if (rank_spread_alloc_experiment_enabled()
        && active_rank_count > 1
        && !rank_spread_multi_rank_batch_experiment_enabled()) {
        return 0;
    }
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
    void **weight_xfer_buffers = NULL;
    float *dummy_weights = NULL;
    float *dummy_context = NULL;
    kvslot_runtime_slot_args_t zero_runtime_args;
    kvslot_runtime_slot_args_t *dummy_segment_runtime_args = NULL;
    uint32_t *dummy_segment_lengths = NULL;
    size_t padded_weight_bytes;
    size_t padded_context_bytes;
    uint32_t max_segment_count = 0;
    uint32_t zero_segment_count = 0;
    uint32_t kernel_command = KVSLOT_KERNEL_AV;
    int grouped_round = 0;
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;
    struct dpu_rank_t **active_ranks = NULL;
    struct dpu_set_t launch_set;
    struct dpu_set_t dpu;
    uint64_t round_start_ns;
    uint64_t xfer_to_start_ns;
    uint64_t launch_start_ns;
    uint64_t xfer_from_start_ns;
    uint64_t xfer_to_ns = 0;
    uint64_t launch_ns = 0;
    uint64_t xfer_from_ns = 0;

    if (!can_use_batched_av_round(runner, items, round_indices, round_count)) {
        return 1;
    }
    round_start_ns = monotonic_time_ns();

    if (ensure_round_scratch_av_items(runner) != 0) {
        fprintf(stderr, "Failed to allocate batched av round map\n");
        return 1;
    }
    items_by_dpu = runner->scratch.av_items_by_dpu;
    memset(&zero_runtime_args, 0, sizeof(zero_runtime_args));
    padded_weight_bytes = 0;
    padded_context_bytes = 0;
    grouped_round = items[round_indices[0]].segment_count > 1;
    memset(&launch_set, 0, sizeof(launch_set));

    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        if (item->physical_dpu_id >= runner->nr_dpus) {
            fprintf(stderr, "Invalid physical dpu id %u in batched av round\n", item->physical_dpu_id);
            return 1;
        }
        if ((item->segment_count > 1) != grouped_round) {
            fprintf(stderr, "Mixed grouped/regular av items in a batched round\n");
            return 1;
        }
        items_by_dpu[item->physical_dpu_id] = item;
        if (item->padded_weight_bytes > padded_weight_bytes) {
            padded_weight_bytes = item->padded_weight_bytes;
        }
        if (item->padded_context_bytes > padded_context_bytes) {
            padded_context_bytes = item->padded_context_bytes;
        }
        if (item->segment_count > max_segment_count) {
            max_segment_count = item->segment_count;
        }
    }

    if (ensure_round_scratch_rank_used(runner) != 0) {
        fprintf(stderr, "Failed to allocate batched av rank mask\n");
        return 1;
    }
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        uint32_t rank_idx = runner->physical_dpu_rank_indices[items[round_indices[pos]].physical_dpu_id];
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }
    if (ensure_round_scratch_active_ranks(runner, active_rank_count) != 0) {
        fprintf(stderr, "Failed to allocate batched av active ranks\n");
        return 1;
    }
    active_ranks = runner->scratch.active_ranks;
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
        if (ensure_round_scratch_dummy_weights(runner, padded_weight_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched av dummy weights\n");
            return 1;
        }
        dummy_weights = runner->scratch.dummy_weights;
        if (ensure_round_scratch_xfer_buffers(runner) != 0) {
            fprintf(stderr, "Failed to allocate batched av weight transfer buffers\n");
            return 1;
        }
        weight_xfer_buffers = runner->scratch.xfer_buffers;
    }
    if (padded_context_bytes > 0) {
        if (ensure_round_scratch_dummy_context(runner, padded_context_bytes) != 0) {
            fprintf(stderr, "Failed to allocate batched av dummy context\n");
            return 1;
        }
        dummy_context = runner->scratch.dummy_context;
    }
    if (grouped_round && max_segment_count > 0) {
        kernel_command = KVSLOT_KERNEL_AV + 100u;
        if (ensure_round_scratch_dummy_segments(runner, max_segment_count) != 0) {
            fprintf(stderr, "Failed to allocate grouped av dummy metadata\n");
            return 1;
        }
        dummy_segment_runtime_args = runner->scratch.dummy_segment_runtime_args;
        dummy_segment_lengths = runner->scratch.dummy_segment_lengths;
    }

    xfer_to_start_ns = monotonic_time_ns();
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

    if (grouped_round && max_segment_count > 0) {
        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)item->segment_runtime_args : (void *)dummy_segment_runtime_args));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_runtime_slot_args",
            0,
            (size_t)max_segment_count * sizeof(*dummy_segment_runtime_args),
            DPU_XFER_DEFAULT));

        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)item->segment_lengths : (void *)dummy_segment_lengths));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_segment_lengths",
            0,
            (size_t)max_segment_count * sizeof(*dummy_segment_lengths),
            DPU_XFER_DEFAULT));

        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            DPU_ASSERT(dpu_prepare_xfer(
                dpu,
                item != NULL ? (void *)&item->segment_count : (void *)&zero_segment_count));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "grouped_segment_count",
            0,
            sizeof(zero_segment_count),
            DPU_XFER_DEFAULT));
    }

    if (padded_weight_bytes > 0 && !items[round_indices[0]].weights_resident_on_dpu) {
        DPU_FOREACH(launch_set, dpu) {
            av_item_t *item = find_av_round_item_for_dpu(runner, items_by_dpu, dpu);
            void *weight_ptr = dummy_weights;
            if (item != NULL) {
                if (item->padded_weight_bytes == padded_weight_bytes) {
                    weight_ptr = (void *)item->weights;
                } else {
                    void *padded_weights = calloc(1, padded_weight_bytes);
                    if (padded_weights == NULL) {
                        fprintf(stderr, "Failed to allocate padded batched av weights\n");
                        free_round_scratch_xfer_payloads(runner);
                        return 1;
                    }
                    memcpy(padded_weights, item->weights, item->padded_weight_bytes);
                    weight_xfer_buffers[item->physical_dpu_id] = padded_weights;
                    weight_ptr = padded_weights;
                }
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, weight_ptr));
        }
        DPU_ASSERT(dpu_push_xfer(
            launch_set,
            DPU_XFER_TO_DPU,
            "av_weights_bits",
            0,
            padded_weight_bytes,
            DPU_XFER_DEFAULT));
    }
    xfer_to_ns += elapsed_ns_since(xfer_to_start_ns);

    launch_start_ns = monotonic_time_ns();
    DPU_ASSERT(dpu_launch(launch_set, DPU_SYNCHRONOUS));
    launch_ns += elapsed_ns_since(launch_start_ns);

    xfer_from_start_ns = monotonic_time_ns();
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
    xfer_from_ns += elapsed_ns_since(xfer_from_start_ns);
    record_av_round_timing(
        runner,
        1,
        elapsed_ns_since(round_start_ns),
        xfer_to_ns,
        launch_ns,
        xfer_from_ns,
        0);

    free_round_scratch_xfer_payloads(runner);
    return 0;
}

static int fetch_batched_context_fused_round(
    kvslot_runner_t *runner,
    av_item_t *items,
    const uint32_t *round_indices,
    uint32_t round_count)
{
    av_item_t **items_by_dpu = NULL;
    float *dummy_context = NULL;
    uint32_t active_rank_count = 0;
    uint8_t *rank_used = NULL;
    struct dpu_rank_t **active_ranks = NULL;
    struct dpu_set_t launch_set;
    struct dpu_set_t dpu;
    size_t padded_context_bytes;
    uint64_t round_start_ns;
    uint64_t xfer_from_start_ns;
    uint64_t xfer_from_ns = 0;

    if (runner == NULL || items == NULL || round_indices == NULL || round_count == 0) {
        return 1;
    }
    round_start_ns = monotonic_time_ns();
    padded_context_bytes = items[round_indices[0]].padded_context_bytes;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        if (!item->ready || !item->context_from_qk_kernel || item->padded_context_bytes != padded_context_bytes) {
            return 1;
        }
    }

    if (ensure_round_scratch_av_items(runner) != 0 || ensure_round_scratch_rank_used(runner) != 0) {
        return 1;
    }
    items_by_dpu = runner->scratch.av_items_by_dpu;
    rank_used = runner->scratch.rank_used;
    for (uint32_t pos = 0; pos < round_count; ++pos) {
        av_item_t *item = &items[round_indices[pos]];
        uint32_t physical_dpu_id = item->physical_dpu_id;
        uint32_t rank_idx;
        if (physical_dpu_id >= runner->nr_dpus) {
            return 1;
        }
        items_by_dpu[physical_dpu_id] = item;
        rank_idx = runner->physical_dpu_rank_indices[physical_dpu_id];
        if (!rank_used[rank_idx]) {
            rank_used[rank_idx] = 1;
            active_rank_count += 1;
        }
    }

    if (ensure_round_scratch_active_ranks(runner, active_rank_count) != 0) {
        return 1;
    }
    active_ranks = runner->scratch.active_ranks;
    memset(&launch_set, 0, sizeof(launch_set));
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

    if (padded_context_bytes > 0) {
        if (ensure_round_scratch_dummy_context(runner, padded_context_bytes) != 0) {
            return 1;
        }
        dummy_context = runner->scratch.dummy_context;
        xfer_from_start_ns = monotonic_time_ns();
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
        xfer_from_ns += elapsed_ns_since(xfer_from_start_ns);
    }

    for (uint32_t pos = 0; pos < round_count; ++pos) {
        items[round_indices[pos]].context_prefetched = 1;
    }

    record_av_round_timing(
        runner,
        1,
        elapsed_ns_since(round_start_ns),
        0,
        0,
        xfer_from_ns,
        0);

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

static int write_av_item_response_with_softmax_stats(
    const av_item_t *item,
    const qk_slot_item_t *qk_item)
{
    if (item == NULL || !item->ready || qk_item == NULL || !qk_item->ready) {
        return 1;
    }
    if (write_av_item_response(item) != 0) {
        return 1;
    }
    if (qk_item->num_heads > 0) {
        if (write_exact(stdout, qk_item->raw_row_max_bits, (size_t)qk_item->num_heads * sizeof(*qk_item->raw_row_max_bits)) != 0) {
            fprintf(stderr, "Failed to write qk-softmax-av partial row max payload\n");
            return 1;
        }
        if (write_exact(stdout, qk_item->raw_row_sum_bits, (size_t)qk_item->num_heads * sizeof(*qk_item->raw_row_sum_bits)) != 0) {
            fprintf(stderr, "Failed to write qk-softmax-av partial row sum payload\n");
            return 1;
        }
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
    if (
        launch_av_item_async(&item, NULL) != 0
        || finish_av_item(&item, NULL) != 0
        || write_av_item_response(&item) != 0
        || flush_exact(stdout) != 0
    ) {
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
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid av batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
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
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_av_launch_round(
            runner,
            items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build av batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_av_round(runner, items, round_indices, round_count)) {
            record_av_round_profile(runner, items, round_indices, round_count, 1);
            if (execute_batched_av_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched av round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_av_round_profile(runner, items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_av_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
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
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid softmax av batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
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
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_av_launch_round(
            runner,
            items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build softmax av batch launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_av_round(runner, items, round_indices, round_count)) {
            record_av_round_profile(runner, items, round_indices, round_count, 1);
            if (execute_batched_av_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched softmax av round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_av_round_profile(runner, items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch softmax av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish softmax av batch item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_av_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
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

static int handle_av_grouped_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    av_item_t *items = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read grouped av batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid grouped av batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
        return 1;
    }

    items = calloc(args.num_slots, sizeof(*items));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (items == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate grouped av batch state\n");
        free(used_dpus);
        free(processed);
        free(items);
        return 1;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        uint32_t segment_count = 0;
        uint32_t slot_ids[KVSLOT_MAX_GROUP_SEGMENTS];
        uint32_t segment_lengths[KVSLOT_MAX_GROUP_SEGMENTS];
        uint32_t total_seq_len = 0;
        if (read_exact(stdin, &segment_count, sizeof(segment_count)) != 0) {
            fprintf(stderr, "Failed to read grouped av segment count %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (segment_count == 0 || segment_count > KVSLOT_MAX_GROUP_SEGMENTS) {
            fprintf(stderr, "Invalid grouped av segment_count=%u at item %u\n", segment_count, idx);
            rc = 1;
            goto cleanup;
        }
        for (uint32_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
            kvslot_io_header_t header;
            if (read_exact(stdin, &header, sizeof(header)) != 0) {
                fprintf(stderr, "Failed to read grouped av item header %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            if (header.magic != KVSLOT_MAGIC || header.command != KVSLOT_CMD_AV_GROUPED_BATCH) {
                fprintf(stderr, "Invalid grouped av item header %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            slot_ids[seg_idx] = header.slot_id;
            if (read_exact(stdin, &segment_lengths[seg_idx], sizeof(segment_lengths[seg_idx])) != 0) {
                fprintf(stderr, "Failed to read grouped av segment len %u/%u\n", idx, seg_idx);
                rc = 1;
                goto cleanup;
            }
            total_seq_len += segment_lengths[seg_idx];
        }
        if (prepare_grouped_av_item_header(runner, slot_ids, segment_lengths, segment_count, &items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare grouped av item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (items[idx].weight_bytes > 0 && read_exact(stdin, items[idx].weights, items[idx].weight_bytes) != 0) {
            fprintf(stderr, "Failed to read grouped av weights %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_av_launch_round(
            runner,
            items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build grouped av launch round\n");
            rc = 1;
            break;
        }

        if (can_use_batched_av_round(runner, items, round_indices, round_count)) {
            record_av_round_profile(runner, items, round_indices, round_count, 1);
            if (execute_batched_av_round(runner, items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to execute batched grouped av round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_av_round_profile(runner, items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_av_item_async(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch grouped av item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_av_item(&items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish grouped av item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_av_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (rc == 0 && write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write grouped av batch response header\n");
        rc = 1;
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (write_av_item_response(&items[idx]) != 0) {
            fprintf(stderr, "Failed to write grouped av batch item %u\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush grouped av batch response\n");
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
    int use_context_fused = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read qk-softmax-av batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid qk-softmax-av batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
        return 1;
    }
    use_context_fused = context_fused_experiment_enabled();

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
        qk_items[idx].slot_args.mode =
            use_context_fused ? KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED : KVSLOT_QK_SLOT_MODE_SOFTMAX_NORMALIZED;
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
        if (av_items[idx].slot->seq_len != qk_items[idx].window) {
            fprintf(stderr, "QK-softmax-av slot shape mismatch at item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_qk_launch_round(
            runner,
            qk_items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk-softmax-av qk round\n");
            rc = 1;
            break;
        }
        if (can_use_batched_qk_round(runner, qk_items, round_indices, round_count)) {
            record_qk_round_profile(runner, qk_items, round_indices, round_count, 1);
            if (execute_batched_qk_round(runner, qk_items, round_indices, round_count, NULL) != 0) {
                fprintf(stderr, "Failed to execute batched qk-softmax-av qk round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_qk_round_profile(runner, qk_items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&qk_items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch qk-softmax-av qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&qk_items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish qk-softmax-av qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_qk_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
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
    if (use_context_fused) {
        processed_count = 0;
        if (processed != NULL) {
            memset(processed, 0, args.num_slots * sizeof(*processed));
        }
        while (processed_count < args.num_slots && rc == 0) {
            uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
            uint32_t round_count = build_av_launch_round(
                runner,
                av_items,
                args.num_slots,
                processed,
                used_dpus,
                round_indices);
            if (round_count == 0) {
                fprintf(stderr, "Failed to build qk-softmax-av partial context fetch round\n");
                rc = 1;
                break;
            }
            record_av_round_profile(runner, av_items, round_indices, round_count, 1);
            if (fetch_batched_context_fused_round(runner, av_items, round_indices, round_count) != 0) {
                fprintf(stderr, "Failed to fetch batched qk-softmax-av partial context round\n");
                rc = 1;
                break;
            }
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (finish_av_item(&av_items[round_indices[pos]], NULL) != 0) {
                    fprintf(stderr, "Failed to finish qk-softmax-av partial context-fused item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                processed[round_indices[pos]] = 1;
                processed_count += 1;
            }
        }
    } else {
        while (processed_count < args.num_slots && rc == 0) {
            uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
            uint32_t round_count = build_av_launch_round(
                runner,
                av_items,
                args.num_slots,
                processed,
                used_dpus,
                round_indices);
            if (round_count == 0) {
                fprintf(stderr, "Failed to build qk-softmax-av av round\n");
                rc = 1;
                break;
            }
            if (can_use_batched_av_round(runner, av_items, round_indices, round_count)) {
                record_av_round_profile(runner, av_items, round_indices, round_count, 1);
                if (execute_batched_av_round(runner, av_items, round_indices, round_count) != 0) {
                    fprintf(stderr, "Failed to execute batched qk-softmax-av av round\n");
                    rc = 1;
                }
            } else {
                uint64_t round_start_ns = monotonic_time_ns();
                record_av_round_profile(runner, av_items, round_indices, round_count, 0);
                for (uint32_t pos = 0; pos < round_count; ++pos) {
                    if (launch_av_item_async(&av_items[round_indices[pos]], &runner->profile) != 0) {
                        fprintf(stderr, "Failed to launch qk-softmax-av av item %u\n", round_indices[pos]);
                        rc = 1;
                        break;
                    }
                }
                for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                    if (finish_av_item(&av_items[round_indices[pos]], &runner->profile) != 0) {
                        fprintf(stderr, "Failed to finish qk-softmax-av av item %u\n", round_indices[pos]);
                        rc = 1;
                        break;
                    }
                }
                record_av_round_timing(
                    runner,
                    0,
                    elapsed_ns_since(round_start_ns),
                    0,
                    0,
                    0,
                    0);
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                processed[round_indices[pos]] = 1;
                processed_count += 1;
            }
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

static int handle_qk_softmax_av_partial_batch(kvslot_runner_t *runner)
{
    kvslot_av_batch_args_t args;
    qk_slot_item_t *qk_items = NULL;
    av_item_t *av_items = NULL;
    kvslot_qk_softmax_av_batch_item_args_t *item_args = NULL;
    uint8_t *processed = NULL;
    uint8_t *used_dpus = NULL;
    uint32_t processed_count = 0;
    int av_contexts_bound = 0;
    int rc = 0;

    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read qk-softmax-av partial batch args\n");
        return 1;
    }
    if (args.num_slots == 0 || args.num_slots > KVSLOT_MAX_BATCH_ITEMS) {
        fprintf(stderr, "Invalid qk-softmax-av partial batch num_slots=%u max=%u\n", args.num_slots, KVSLOT_MAX_BATCH_ITEMS);
        return 1;
    }

    qk_items = calloc(args.num_slots, sizeof(*qk_items));
    av_items = calloc(args.num_slots, sizeof(*av_items));
    item_args = calloc(args.num_slots, sizeof(*item_args));
    processed = calloc(args.num_slots, sizeof(*processed));
    used_dpus = calloc(runner->nr_dpus, sizeof(*used_dpus));
    if (qk_items == NULL || av_items == NULL || item_args == NULL || processed == NULL || used_dpus == NULL) {
        fprintf(stderr, "Failed to allocate qk-softmax-av partial batch state\n");
        rc = 1;
        goto cleanup;
    }

    for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
        kvslot_qk_softmax_av_batch_item_args_t current_args;
        uint32_t slot_id;
        if (read_exact(stdin, &slot_id, sizeof(slot_id)) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av partial slot id %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (read_exact(stdin, &current_args, sizeof(current_args)) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av partial item args %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        item_args[idx] = current_args;
        if (prepare_qk_slot_item_header(runner, slot_id, &current_args, &qk_items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare qk-softmax-av partial qk item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        restrict_qk_slot_item_to_tail_window(&qk_items[idx]);
        qk_items[idx].slot_args.mode = KVSLOT_QK_SLOT_MODE_CONTEXT_FUSED_UNNORMALIZED;
        if (read_qk_slot_item_payload(stdin, &qk_items[idx]) != 0) {
            fprintf(stderr, "Failed to read qk-softmax-av partial payload %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        if (prepare_av_item_header(runner, slot_id, &av_items[idx]) != 0) {
            fprintf(stderr, "Failed to prepare qk-softmax-av partial av item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
        restrict_av_item_to_qk_tail_window(&av_items[idx], &qk_items[idx]);
        if (av_items[idx].runtime_args.seq_len != qk_items[idx].window) {
            fprintf(stderr, "QK-softmax-av partial slot shape mismatch at item %u\n", idx);
            rc = 1;
            goto cleanup;
        }
    }

    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (softmax_av_item_from_qk_scores(&qk_items[idx], &item_args[idx], &av_items[idx]) != 0) {
            fprintf(stderr, "Failed to bind qk-softmax-av partial item %u before qk launch\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0) {
        av_contexts_bound = 1;
    }

    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_qk_launch_round(
            runner,
            qk_items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk-softmax-av partial qk round\n");
            rc = 1;
            break;
        }
        if (can_use_batched_qk_round(runner, qk_items, round_indices, round_count)) {
            record_qk_round_profile(runner, qk_items, round_indices, round_count, 1);
            if (execute_batched_qk_round(runner, qk_items, round_indices, round_count, av_items) != 0) {
                fprintf(stderr, "Failed to execute batched qk-softmax-av partial qk round\n");
                rc = 1;
            }
        } else {
            uint64_t round_start_ns = monotonic_time_ns();
            record_qk_round_profile(runner, qk_items, round_indices, round_count, 0);
            for (uint32_t pos = 0; pos < round_count; ++pos) {
                if (launch_qk_slot_item_async(&qk_items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to launch qk-softmax-av partial qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
                if (finish_qk_slot_item(&qk_items[round_indices[pos]], &runner->profile) != 0) {
                    fprintf(stderr, "Failed to finish qk-softmax-av partial qk item %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
                if (fetch_qk_slot_row_maxes(&qk_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to fetch qk-softmax-av partial row maxes %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
                if (fetch_qk_slot_row_sums(&qk_items[round_indices[pos]]) != 0) {
                    fprintf(stderr, "Failed to fetch qk-softmax-av partial row sums %u\n", round_indices[pos]);
                    rc = 1;
                    break;
                }
            }
            record_qk_round_timing(
                runner,
                0,
                elapsed_ns_since(round_start_ns),
                0,
                0,
                0,
                0);
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (!av_contexts_bound) {
        for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
            if (softmax_av_item_from_qk_scores(&qk_items[idx], &item_args[idx], &av_items[idx]) != 0) {
                fprintf(stderr, "Failed to fuse qk-softmax-av partial item %u\n", idx);
                rc = 1;
                break;
            }
        }
    }

    processed_count = 0;
    if (processed != NULL) {
        memset(processed, 0, args.num_slots * sizeof(*processed));
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (av_items[idx].context_prefetched) {
            processed[idx] = 1;
            processed_count += 1;
        }
    }
    while (processed_count < args.num_slots && rc == 0) {
        uint32_t round_indices[KVSLOT_MAX_BATCH_ITEMS];
        uint32_t round_count = build_av_launch_round(
            runner,
            av_items,
            args.num_slots,
            processed,
            used_dpus,
            round_indices);
        if (round_count == 0) {
            fprintf(stderr, "Failed to build qk-softmax-av partial context fetch round\n");
            rc = 1;
            break;
        }
        record_av_round_profile(runner, av_items, round_indices, round_count, 1);
        if (fetch_batched_context_fused_round(runner, av_items, round_indices, round_count) != 0) {
            fprintf(stderr, "Failed to fetch batched qk-softmax-av partial context round\n");
            rc = 1;
            break;
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            if (finish_av_item(&av_items[round_indices[pos]], NULL) != 0) {
                fprintf(stderr, "Failed to finish qk-softmax-av partial context-fused item %u\n", round_indices[pos]);
                rc = 1;
                break;
            }
        }
        for (uint32_t pos = 0; pos < round_count && rc == 0; ++pos) {
            processed[round_indices[pos]] = 1;
            processed_count += 1;
        }
    }

    if (rc == 0 && write_exact(stdout, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to write qk-softmax-av partial batch response header\n");
        rc = 1;
    }
    for (uint32_t idx = 0; idx < args.num_slots && rc == 0; ++idx) {
        if (write_av_item_response_with_softmax_stats(&av_items[idx], &qk_items[idx]) != 0) {
            fprintf(stderr, "Failed to write qk-softmax-av partial batch item %u\n", idx);
            rc = 1;
            break;
        }
    }
    if (rc == 0 && flush_exact(stdout) != 0) {
        fprintf(stderr, "Failed to flush qk-softmax-av partial batch response\n");
        rc = 1;
    }

cleanup:
    free(used_dpus);
    free(processed);
    free(item_args);
    if (av_items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_av_item(&av_items[idx]);
        }
    }
    if (qk_items != NULL) {
        for (uint32_t idx = 0; idx < args.num_slots; ++idx) {
            cleanup_qk_slot_item(&qk_items[idx]);
        }
    }
    free(av_items);
    free(qk_items);
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
        if (!slim_qk_slot_command_supported(header.command)) {
            fprintf(stderr,
                "kvslot slim QK slot helper does not support command %u; rebuild without "
                "KVSLOT_SLIM_QK_SLOT_ONLY for standalone QK/AV or grouped legacy paths\n",
                header.command);
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
        } else if (header.command == KVSLOT_CMD_GET_PROFILE) {
            rc = handle_get_profile(&runner);
        } else if (header.command == KVSLOT_CMD_GET_TOPOLOGY) {
            rc = handle_get_topology(&runner);
        } else if (header.command == KVSLOT_CMD_QK_BATCH) {
            rc = handle_qk_batch(&runner);
        } else if (header.command == KVSLOT_CMD_AV) {
            rc = handle_av(&runner, header.slot_id);
        } else if (header.command == KVSLOT_CMD_AV_BATCH) {
            rc = handle_av_batch(&runner);
        } else if (header.command == KVSLOT_CMD_AV_GROUPED_BATCH) {
            rc = handle_av_grouped_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SLOT_BATCH) {
            rc = handle_qk_slot_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SLOT_GROUPED_BATCH) {
            rc = handle_qk_slot_grouped_batch(&runner);
        } else if (header.command == KVSLOT_CMD_SOFTMAX_AV_BATCH) {
            rc = handle_softmax_av_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SOFTMAX_AV_BATCH) {
            rc = handle_qk_softmax_av_batch(&runner);
        } else if (header.command == KVSLOT_CMD_QK_SOFTMAX_AV_PARTIAL_BATCH) {
            rc = handle_qk_softmax_av_partial_batch(&runner);
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
