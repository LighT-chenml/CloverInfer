#include <dpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./build/dpu_dot"
#endif

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [num_dpus] [length_per_dpu]\n", prog);
}

int main(int argc, char **argv)
{
    uint32_t requested_dpus = 1;
    uint32_t length = 128;

    if (argc > 3) {
        usage(argv[0]);
        return 2;
    }
    if (argc >= 2) {
        requested_dpus = (uint32_t)strtoul(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        length = (uint32_t)strtoul(argv[2], NULL, 10);
    }
    if (requested_dpus == 0 || length == 0 || length > MAX_ELEMS_PER_DPU || (length % 2) != 0) {
        usage(argv[0]);
        fprintf(stderr, "num_dpus must be > 0 and length_per_dpu must be an even value in [2, %u]\n", MAX_ELEMS_PER_DPU);
        return 2;
    }

    struct dpu_set_t dpu_set;
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t nr_dpus = 0;

    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    int32_t *inputs_a = calloc((size_t)nr_dpus * length, sizeof(*inputs_a));
    int32_t *inputs_b = calloc((size_t)nr_dpus * length, sizeof(*inputs_b));
    dot_result_t *results = calloc(nr_dpus, sizeof(*results));
    if (inputs_a == NULL || inputs_b == NULL || results == NULL) {
        fprintf(stderr, "Failed to allocate host buffers\n");
        dpu_free(dpu_set);
        free(inputs_a);
        free(inputs_b);
        free(results);
        return 1;
    }

    int64_t expected_total = 0;
    for (uint32_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
        for (uint32_t idx = 0; idx < length; ++idx) {
            int32_t a = (int32_t)((dpu_idx + 1) * 3 + idx % 17);
            int32_t b = (int32_t)((dpu_idx + 2) * 5 - (idx % 11));
            inputs_a[(size_t)dpu_idx * length + idx] = a;
            inputs_b[(size_t)dpu_idx * length + idx] = b;
            expected_total += (int64_t)a * (int64_t)b;
        }
    }

    dot_args_t args = {.length = length, .padding = 0};
    DPU_ASSERT(dpu_broadcast_to(dpu_set, "dot_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &inputs_a[(size_t)each_dpu * length]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_a", 0, length * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &inputs_b[(size_t)each_dpu * length]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_b", 0, length * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &results[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "dot_result", 0, sizeof(dot_result_t), DPU_XFER_DEFAULT));

    int64_t actual_total = 0;
    uint64_t max_cycles = 0;
    for (uint32_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
        actual_total += results[dpu_idx].dot;
        if (results[dpu_idx].cycles > max_cycles) {
            max_cycles = results[dpu_idx].cycles;
        }
    }

    printf("UPMEM dot smoke test\n");
    printf("binary=%s\n", DPU_BINARY);
    printf("dpus=%u length_per_dpu=%u\n", nr_dpus, length);
    printf("expected=%" PRId64 " actual=%" PRId64 "\n", expected_total, actual_total);
    printf("max_dpu_cycles=%" PRIu64 "\n", max_cycles);

    int ok = expected_total == actual_total;
    printf("status=%s\n", ok ? "PASS" : "FAIL");

    dpu_free(dpu_set);
    free(inputs_a);
    free(inputs_b);
    free(results);
    return ok ? 0 : 1;
}
