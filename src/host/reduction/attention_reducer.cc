#include "attention_reducer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

void clover_attention_reduce_engine_init(clover_attention_reduce_engine_t *engine)
{
    if (engine == NULL) {
        return;
    }
    engine->num_slots = 2u;
    for (uint32_t idx = 0; idx < 2u; ++idx) {
        engine->slots[idx].slot_id = idx;
        engine->slots[idx].micro_batch_id = 0u;
        engine->slots[idx].busy = 0u;
    }
}

int clover_attention_reduce_group(
    uint32_t num_dpus,
    const clover_attention_reduce_input_t *input,
    clover_attention_reduce_output_t *output)
{
    if (num_dpus == 0u || input == NULL || output == NULL) {
        return 1;
    }
    if (input->num_requests == 0u || input->d_head == 0u) {
        return 1;
    }
    if (input->local_max == NULL || input->local_sum == NULL || input->local_output == NULL) {
        return 1;
    }
    if (output->global_output == NULL || output->global_max == NULL || output->global_sum == NULL) {
        return 1;
    }

    const uint32_t num_requests = input->num_requests;
    const uint32_t d_head = input->d_head;
    const size_t request_stride = (size_t)d_head;
    const size_t dpu_request_count = (size_t)num_requests;
    const size_t dpu_output_stride = (size_t)num_requests * (size_t)d_head;

    memset(output->global_output, 0, (size_t)num_requests * (size_t)d_head * sizeof(float));
    for (uint32_t req = 0; req < num_requests; ++req) {
        float max_value = -FLT_MAX;
        for (uint32_t dpu = 0; dpu < num_dpus; ++dpu) {
            const float local = input->local_max[(size_t)dpu * dpu_request_count + req];
            if (local > max_value) {
                max_value = local;
            }
        }
        output->global_max[req] = max_value;

        float global_sum = 0.0f;
        for (uint32_t dpu = 0; dpu < num_dpus; ++dpu) {
            const float local_max = input->local_max[(size_t)dpu * dpu_request_count + req];
            const float local_sum = input->local_sum[(size_t)dpu * dpu_request_count + req];
            const float correction = expf(local_max - max_value);
            global_sum += local_sum * correction;
        }
        output->global_sum[req] = global_sum;

        if (global_sum <= 0.0f) {
            continue;
        }
        for (uint32_t dpu = 0; dpu < num_dpus; ++dpu) {
            const float local_max = input->local_max[(size_t)dpu * dpu_request_count + req];
            const float local_sum = input->local_sum[(size_t)dpu * dpu_request_count + req];
            if (local_sum <= 0.0f) {
                continue;
            }
            const float scale = expf(local_max - max_value) / global_sum;
            const size_t base = (size_t)dpu * dpu_output_stride + (size_t)req * request_stride;
            const size_t out_base = (size_t)req * request_stride;
            for (uint32_t dim = 0; dim < d_head; ++dim) {
                output->global_output[out_base + dim] += input->local_output[base + dim] * scale;
            }
        }
    }
    return 0;
}

