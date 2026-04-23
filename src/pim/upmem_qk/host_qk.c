#include <dpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./build/dpu_qk"
#endif

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [num_dpus] [head_dim] [keys_per_dpu]\n", prog);
}

int main(int argc, char **argv)
{
    uint32_t requested_dpus = 2;
    uint32_t head_dim = 64;
    uint32_t keys_per_dpu = 8;

    if (argc > 4) {
        usage(argv[0]);
        return 2;
    }
    if (argc >= 2) {
        requested_dpus = (uint32_t)strtoul(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        head_dim = (uint32_t)strtoul(argv[2], NULL, 10);
    }
    if (argc >= 4) {
        keys_per_dpu = (uint32_t)strtoul(argv[3], NULL, 10);
    }
    if (requested_dpus == 0 || head_dim == 0 || keys_per_dpu == 0 || head_dim > MAX_HEAD_DIM || keys_per_dpu > MAX_KEYS_PER_DPU || (head_dim % 2) != 0) {
        usage(argv[0]);
        fprintf(stderr, "num_dpus > 0, head_dim must be even in [2, %u], keys_per_dpu in [1, %u]\n", MAX_HEAD_DIM, MAX_KEYS_PER_DPU);
        return 2;
    }

    struct dpu_set_t dpu_set;
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t nr_dpus = 0;

    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    size_t query_elems = head_dim;
    size_t key_elems = (size_t)keys_per_dpu * head_dim;
    int32_t *queries = calloc((size_t)nr_dpus * query_elems, sizeof(*queries));
    int32_t *keys = calloc((size_t)nr_dpus * key_elems, sizeof(*keys));
    int64_t *scores = calloc((size_t)nr_dpus * keys_per_dpu, sizeof(*scores));
    qk_meta_t *metas = calloc(nr_dpus, sizeof(*metas));
    int64_t *expected = calloc((size_t)nr_dpus * keys_per_dpu, sizeof(*expected));
    if (!queries || !keys || !scores || !metas || !expected) {
        fprintf(stderr, "Failed to allocate host buffers\n");
        dpu_free(dpu_set);
        free(queries);
        free(keys);
        free(scores);
        free(metas);
        free(expected);
        return 1;
    }

    for (uint32_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
        int32_t *q = &queries[(size_t)dpu_idx * query_elems];
        int32_t *k = &keys[(size_t)dpu_idx * key_elems];
        for (uint32_t dim = 0; dim < head_dim; ++dim) {
            q[dim] = (int32_t)((int)dpu_idx + 3 - (int)(dim % 13));
        }
        for (uint32_t key_idx = 0; key_idx < keys_per_dpu; ++key_idx) {
            int64_t dot = 0;
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                int32_t value = (int32_t)(((int)key_idx + 5) * 2 + (int)(dim % 7) - (int)dpu_idx);
                k[key_idx * head_dim + dim] = value;
                dot += (int64_t)q[dim] * (int64_t)value;
            }
            expected[(size_t)dpu_idx * keys_per_dpu + key_idx] = dot;
        }
    }

    qk_args_t args = {
        .head_dim = head_dim,
        .num_keys = keys_per_dpu,
        .key_stride = head_dim,
        .reserved = 0,
    };
    DPU_ASSERT(dpu_broadcast_to(dpu_set, "qk_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &queries[(size_t)each_dpu * query_elems]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "query", 0, query_elems * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &keys[(size_t)each_dpu * key_elems]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "keys", 0, key_elems * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &scores[(size_t)each_dpu * keys_per_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "scores", 0, keys_per_dpu * sizeof(int64_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &metas[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "qk_meta", 0, sizeof(qk_meta_t), DPU_XFER_DEFAULT));

    uint64_t max_cycles = 0;
    int ok = 1;
    for (uint32_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
        if (metas[dpu_idx].cycles > max_cycles) {
            max_cycles = metas[dpu_idx].cycles;
        }
        for (uint32_t key_idx = 0; key_idx < keys_per_dpu; ++key_idx) {
            size_t off = (size_t)dpu_idx * keys_per_dpu + key_idx;
            if (scores[off] != expected[off]) {
                ok = 0;
                fprintf(stderr, "Mismatch dpu=%u key=%u expected=%" PRId64 " actual=%" PRId64 "\n", dpu_idx, key_idx, expected[off], scores[off]);
            }
        }
    }

    printf("UPMEM qk smoke test\n");
    printf("binary=%s\n", DPU_BINARY);
    printf("dpus=%u head_dim=%u keys_per_dpu=%u\n", nr_dpus, head_dim, keys_per_dpu);
    printf("max_dpu_cycles=%" PRIu64 "\n", max_cycles);
    printf("status=%s\n", ok ? "PASS" : "FAIL");

    dpu_free(dpu_set);
    free(queries);
    free(keys);
    free(scores);
    free(metas);
    free(expected);
    return ok ? 0 : 1;
}
