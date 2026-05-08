#ifndef CLOVER_ATTENTION_REDUCER_H
#define CLOVER_ATTENTION_REDUCER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t num_requests;
    uint32_t d_head;
    float *local_max;     /* [num_dpus][num_requests] */
    float *local_sum;     /* [num_dpus][num_requests] */
    float *local_output;  /* [num_dpus][num_requests][d_head] */
} clover_attention_reduce_input_t;

typedef struct {
    uint32_t num_requests;
    uint32_t d_head;
    float *global_output; /* [num_requests][d_head] */
    float *global_max;    /* [num_requests] */
    float *global_sum;    /* [num_requests] */
} clover_attention_reduce_output_t;

typedef enum {
    CLOVER_ATTENTION_REDUCE_SLOT_FREE = 0,
    CLOVER_ATTENTION_REDUCE_SLOT_COPYING = 1,
    CLOVER_ATTENTION_REDUCE_SLOT_REDUCING = 2,
    CLOVER_ATTENTION_REDUCE_SLOT_READY = 3,
    CLOVER_ATTENTION_REDUCE_SLOT_FAILED = 4
} clover_attention_reduce_slot_state_t;

typedef struct {
    uint32_t slot_id;
    uint32_t micro_batch_id;
    uint32_t busy;
    uint32_t state;
    uint32_t num_dpus;
    uint32_t num_requests;
    uint32_t d_head;
    int32_t status_code;
    uint64_t copy_latency_ns;
    uint64_t reduce_latency_ns;
    uint64_t total_latency_ns;
} clover_attention_reduce_slot_t;

typedef struct {
    uint32_t num_slots;
    uint32_t next_submit_slot;
    uint32_t inflight_count;
    uint32_t ready_count;
    uint64_t submissions;
    uint64_t completions;
    uintptr_t impl_handle;
    clover_attention_reduce_slot_t slots[2];
} clover_attention_reduce_engine_t;

typedef int (*clover_attention_reduce_fetch_fn)(
    void *user_ctx,
    uint32_t micro_batch_id,
    uint32_t num_dpus,
    uint32_t num_requests,
    uint32_t d_head,
    float *local_max_out,
    float *local_sum_out,
    float *local_output_out);

typedef struct {
    uint32_t slot_id;
    uint32_t micro_batch_id;
    uint32_t num_dpus;
    uint32_t num_requests;
    uint32_t d_head;
    int32_t status_code;
    uint32_t state;
    const float *global_output; /* valid when status_code == 0 */
    const float *global_max;    /* valid when status_code == 0 */
    const float *global_sum;    /* valid when status_code == 0 */
} clover_attention_reduce_completion_t;

void clover_attention_reduce_engine_init(clover_attention_reduce_engine_t *engine);
void clover_attention_reduce_engine_destroy(clover_attention_reduce_engine_t *engine);
int clover_attention_reduce_group(
    uint32_t num_dpus,
    const clover_attention_reduce_input_t *input,
    clover_attention_reduce_output_t *output);
int clover_attention_reduce_engine_submit(
    clover_attention_reduce_engine_t *engine,
    uint32_t micro_batch_id,
    uint32_t num_dpus,
    uint32_t num_requests,
    uint32_t d_head,
    clover_attention_reduce_fetch_fn fetch_fn,
    void *fetch_user_ctx,
    uint32_t *slot_id_out);
int clover_attention_reduce_engine_try_pop_ready(
    clover_attention_reduce_engine_t *engine,
    clover_attention_reduce_completion_t *completion_out);
int clover_attention_reduce_engine_wait_ready(
    clover_attention_reduce_engine_t *engine,
    uint32_t timeout_ms,
    clover_attention_reduce_completion_t *completion_out);
int clover_attention_reduce_engine_release_slot(
    clover_attention_reduce_engine_t *engine,
    uint32_t slot_id);

/*
 * Performance sketch:
 * - Copy/fetch cost scales with G * num_requests * (d_head + 2) fp32 values.
 * - Reduction cost scales with O(G * num_requests * d_head).
 * - The double-buffered engine hides most host work when one slot's
 *   copy+reduce latency fits under the next micro-batch's DPU compute time.
 * Production UPMEM integration should implement `fetch_fn` with batched
 * `dpu_prepare_xfer` / `dpu_push_xfer` reads from WRAM or MRAM.
 */

#ifdef __cplusplus
}
#endif

#endif
