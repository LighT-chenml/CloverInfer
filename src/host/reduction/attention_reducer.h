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

typedef struct {
    uint32_t slot_id;
    uint32_t micro_batch_id;
    uint32_t busy;
} clover_attention_reduce_slot_t;

typedef struct {
    uint32_t num_slots;
    clover_attention_reduce_slot_t slots[2];
} clover_attention_reduce_engine_t;

void clover_attention_reduce_engine_init(clover_attention_reduce_engine_t *engine);
int clover_attention_reduce_group(
    uint32_t num_dpus,
    const clover_attention_reduce_input_t *input,
    clover_attention_reduce_output_t *output);

#ifdef __cplusplus
}
#endif

#endif
