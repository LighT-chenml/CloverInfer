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
    fprintf(stderr, "       %s --input qk_input.bin --output scores.bin [--num-dpus N]\n", prog);
    fprintf(stderr, "       %s --stdio [--num-dpus N]\n", prog);
}

typedef struct {
    struct dpu_set_t dpu_set;
    uint32_t nr_dpus;
} qk_runner_t;

static int qk_runner_init(qk_runner_t *runner, uint32_t requested_dpus)
{
    if (runner == NULL) {
        return 1;
    }

    runner->nr_dpus = 0;
    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &runner->dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(runner->dpu_set, &runner->nr_dpus));
    DPU_ASSERT(dpu_load(runner->dpu_set, DPU_BINARY, NULL));

    if (runner->nr_dpus != requested_dpus) {
        fprintf(stderr, "Requested %u DPUs, allocated %u DPUs\n", requested_dpus, runner->nr_dpus);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        return 1;
    }
    return 0;
}

static void qk_runner_destroy(qk_runner_t *runner)
{
    if (runner != NULL && runner->nr_dpus > 0) {
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
    }
}

static int qk_runner_run(
    qk_runner_t *runner,
    uint32_t head_dim,
    uint32_t keys_per_dpu,
    const int32_t *all_queries,
    const int32_t *all_keys,
    int64_t *all_scores,
    uint64_t *max_cycles_out)
{
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    if (runner == NULL || runner->nr_dpus == 0) {
        return 1;
    }
    uint32_t nr_dpus = runner->nr_dpus;

    qk_meta_t *metas = calloc(nr_dpus, sizeof(*metas));
    if (metas == NULL) {
        fprintf(stderr, "Failed to allocate metadata buffers\n");
        return 1;
    }

    size_t query_elems = head_dim;
    size_t key_elems = (size_t)keys_per_dpu * head_dim;

    qk_args_t args = {
        .head_dim = head_dim,
        .num_keys = keys_per_dpu,
        .key_stride = head_dim,
        .reserved = 0,
    };
    DPU_ASSERT(dpu_broadcast_to(runner->dpu_set, "qk_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));

    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&all_queries[(size_t)each_dpu * query_elems]));
    }
    DPU_ASSERT(dpu_push_xfer(runner->dpu_set, DPU_XFER_TO_DPU, "query", 0, query_elems * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)&all_keys[(size_t)each_dpu * key_elems]));
    }
    DPU_ASSERT(dpu_push_xfer(runner->dpu_set, DPU_XFER_TO_DPU, "keys", 0, key_elems * sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(runner->dpu_set, DPU_SYNCHRONOUS));

    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &all_scores[(size_t)each_dpu * keys_per_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(runner->dpu_set, DPU_XFER_FROM_DPU, "scores", 0, keys_per_dpu * sizeof(int64_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &metas[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(runner->dpu_set, DPU_XFER_FROM_DPU, "qk_meta", 0, sizeof(qk_meta_t), DPU_XFER_DEFAULT));

    uint64_t max_cycles = 0;
    for (uint32_t dpu_idx = 0; dpu_idx < nr_dpus; ++dpu_idx) {
        if (metas[dpu_idx].cycles > max_cycles) {
            max_cycles = metas[dpu_idx].cycles;
        }
    }
    if (max_cycles_out != NULL) {
        *max_cycles_out = max_cycles;
    }

    free(metas);
    return 0;
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

static int run_stdio_mode(uint32_t requested_dpus)
{
    qk_runner_t runner;
    int rc = qk_runner_init(&runner, requested_dpus);
    if (rc != 0) {
        return rc;
    }

    for (;;) {
        qk_io_header_t header;
        if (read_exact(stdin, &header, sizeof(header)) != 0) {
            if (feof(stdin)) {
                rc = 0;
            } else {
                fprintf(stderr, "Failed to read stdio header\n");
                rc = 1;
            }
            break;
        }

        if (header.magic != QK_IO_MAGIC) {
            fprintf(stderr, "Invalid stdio magic\n");
            rc = 1;
            break;
        }
        if (header.head_dim == 0 || header.num_keys == 0 || header.head_dim > MAX_HEAD_DIM || header.num_keys > MAX_KEYS_PER_DPU || (header.head_dim % 2) != 0) {
            fprintf(stderr, "Invalid stdio header parameters\n");
            rc = 1;
            break;
        }

        uint32_t head_dim = header.head_dim;
        uint32_t total_keys = header.num_keys;
        uint32_t num_queries = header.reserved == 0 ? 1 : header.reserved;
        uint32_t num_dpus = runner.nr_dpus;
        if (num_dpus > total_keys) {
            num_dpus = total_keys;
        }
        uint32_t keys_per_dpu = (total_keys + num_dpus - 1) / num_dpus;
        if (keys_per_dpu > MAX_KEYS_PER_DPU) {
            fprintf(stderr, "Too many keys per DPU after partitioning: %u\n", keys_per_dpu);
            rc = 1;
            break;
        }

        int32_t *queries_in = calloc((size_t)num_queries * head_dim, sizeof(*queries_in));
        int32_t *keys_in = calloc((size_t)num_queries * total_keys * head_dim, sizeof(*keys_in));
        int32_t *queries = calloc((size_t)num_dpus * head_dim, sizeof(*queries));
        int32_t *keys_partitioned = calloc((size_t)num_dpus * keys_per_dpu * head_dim, sizeof(*keys_partitioned));
        int64_t *scores_partitioned = calloc((size_t)num_dpus * keys_per_dpu, sizeof(*scores_partitioned));
        int64_t *scores_out = calloc((size_t)num_queries * total_keys, sizeof(*scores_out));
        if (!queries_in || !keys_in || !queries || !keys_partitioned || !scores_partitioned || !scores_out) {
            fprintf(stderr, "Failed to allocate stdio buffers\n");
            free(queries_in);
            free(keys_in);
            free(queries);
            free(keys_partitioned);
            free(scores_partitioned);
            free(scores_out);
            rc = 1;
            break;
        }

        if (read_exact(stdin, queries_in, (size_t)num_queries * head_dim * sizeof(int32_t)) != 0
            || read_exact(stdin, keys_in, (size_t)num_queries * total_keys * head_dim * sizeof(int32_t)) != 0) {
            fprintf(stderr, "Failed to read stdio payload\n");
            free(queries_in);
            free(keys_in);
            free(queries);
            free(keys_partitioned);
            free(scores_partitioned);
            free(scores_out);
            rc = 1;
            break;
        }

        for (uint32_t query_idx = 0; query_idx < num_queries && rc == 0; ++query_idx) {
            memset(queries, 0, (size_t)num_dpus * head_dim * sizeof(*queries));
            memset(keys_partitioned, 0, (size_t)num_dpus * keys_per_dpu * head_dim * sizeof(*keys_partitioned));
            memset(scores_partitioned, 0, (size_t)num_dpus * keys_per_dpu * sizeof(*scores_partitioned));

            const int32_t *query = &queries_in[(size_t)query_idx * head_dim];
            const int32_t *query_keys = &keys_in[(size_t)query_idx * total_keys * head_dim];

            for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
                memcpy(&queries[(size_t)dpu_idx * head_dim], query, head_dim * sizeof(int32_t));
                for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                    uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                    if (global_key < total_keys) {
                        memcpy(
                            &keys_partitioned[((size_t)dpu_idx * keys_per_dpu + local_key) * head_dim],
                            &query_keys[(size_t)global_key * head_dim],
                            head_dim * sizeof(int32_t));
                    }
                }
            }

            uint64_t query_cycles = 0;
            rc = qk_runner_run(&runner, head_dim, keys_per_dpu, queries, keys_partitioned, scores_partitioned, &query_cycles);
            if (rc == 0) {
                for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
                    for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                        uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                        if (global_key < total_keys) {
                            scores_out[(size_t)query_idx * total_keys + global_key] =
                                scores_partitioned[(size_t)dpu_idx * keys_per_dpu + local_key];
                        }
                    }
                }
            }
        }

        if (rc == 0) {
            qk_io_header_t out_header = {
                .magic = QK_IO_MAGIC,
                .head_dim = head_dim,
                .num_keys = total_keys,
                .reserved = num_queries,
            };
            if (write_exact(stdout, &out_header, sizeof(out_header)) != 0
                || write_exact(stdout, scores_out, (size_t)num_queries * total_keys * sizeof(int64_t)) != 0
                || flush_exact(stdout) != 0) {
                fprintf(stderr, "Failed to write stdio output\n");
                rc = 1;
            }
        }

        free(queries_in);
        free(keys_in);
        free(queries);
        free(keys_partitioned);
        free(scores_partitioned);
        free(scores_out);

        if (rc != 0) {
            break;
        }
    }

    qk_runner_destroy(&runner);
    return rc;
}

static int run_file_mode(const char *input_path, const char *output_path, uint32_t requested_dpus)
{
    FILE *input = fopen(input_path, "rb");
    if (input == NULL) {
        perror("fopen input");
        return 1;
    }

    qk_io_header_t header;
    if (read_exact(input, &header, sizeof(header)) != 0) {
        fprintf(stderr, "Failed to read qk input header\n");
        fclose(input);
        return 1;
    }
    if (header.magic != QK_IO_MAGIC || header.head_dim == 0 || header.num_keys == 0 || header.head_dim > MAX_HEAD_DIM || header.num_keys > MAX_KEYS_PER_DPU || (header.head_dim % 2) != 0) {
        fprintf(stderr, "Invalid qk input header\n");
        fclose(input);
        return 1;
    }

    uint32_t head_dim = header.head_dim;
    uint32_t total_keys = header.num_keys;
    uint32_t num_queries = header.reserved == 0 ? 1 : header.reserved;
    uint32_t num_dpus = requested_dpus;
    if (num_dpus == 0) {
        num_dpus = 1;
    }
    if (num_dpus > total_keys) {
        num_dpus = total_keys;
    }
    uint32_t keys_per_dpu = (total_keys + num_dpus - 1) / num_dpus;
    if (keys_per_dpu > MAX_KEYS_PER_DPU) {
        fprintf(stderr, "Too many keys per DPU after partitioning: %u\n", keys_per_dpu);
        fclose(input);
        return 1;
    }

    int32_t *queries_in = calloc((size_t)num_queries * head_dim, sizeof(*queries_in));
    int32_t *keys_in = calloc((size_t)num_queries * total_keys * head_dim, sizeof(*keys_in));
    int32_t *queries = calloc((size_t)num_dpus * head_dim, sizeof(*queries));
    int32_t *keys_partitioned = calloc((size_t)num_dpus * keys_per_dpu * head_dim, sizeof(*keys_partitioned));
    int64_t *scores_partitioned = calloc((size_t)num_dpus * keys_per_dpu, sizeof(*scores_partitioned));
    int64_t *scores_out = calloc((size_t)num_queries * total_keys, sizeof(*scores_out));
    if (!queries_in || !keys_in || !queries || !keys_partitioned || !scores_partitioned || !scores_out) {
        fprintf(stderr, "Failed to allocate file-mode buffers\n");
        fclose(input);
        free(queries_in);
        free(keys_in);
        free(queries);
        free(keys_partitioned);
        free(scores_partitioned);
        free(scores_out);
        return 1;
    }

    if (read_exact(input, queries_in, (size_t)num_queries * head_dim * sizeof(int32_t)) != 0
        || read_exact(input, keys_in, (size_t)num_queries * total_keys * head_dim * sizeof(int32_t)) != 0) {
        fprintf(stderr, "Failed to read qk input payload\n");
        fclose(input);
        free(queries_in);
        free(keys_in);
        free(queries);
        free(keys_partitioned);
        free(scores_partitioned);
        free(scores_out);
        return 1;
    }
    fclose(input);

    qk_runner_t runner;
    int rc = qk_runner_init(&runner, num_dpus);
    if (rc != 0) {
        free(queries_in);
        free(keys_in);
        free(queries);
        free(keys_partitioned);
        free(scores_partitioned);
        free(scores_out);
        return rc;
    }

    uint64_t max_cycles = 0;
    for (uint32_t query_idx = 0; query_idx < num_queries && rc == 0; ++query_idx) {
        memset(queries, 0, (size_t)num_dpus * head_dim * sizeof(*queries));
        memset(keys_partitioned, 0, (size_t)num_dpus * keys_per_dpu * head_dim * sizeof(*keys_partitioned));
        memset(scores_partitioned, 0, (size_t)num_dpus * keys_per_dpu * sizeof(*scores_partitioned));

        const int32_t *query = &queries_in[(size_t)query_idx * head_dim];
        const int32_t *query_keys = &keys_in[(size_t)query_idx * total_keys * head_dim];

        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            memcpy(&queries[(size_t)dpu_idx * head_dim], query, head_dim * sizeof(int32_t));
            for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                if (global_key < total_keys) {
                    memcpy(
                        &keys_partitioned[((size_t)dpu_idx * keys_per_dpu + local_key) * head_dim],
                        &query_keys[(size_t)global_key * head_dim],
                        head_dim * sizeof(int32_t));
                }
            }
        }

        uint64_t query_cycles = 0;
        rc = qk_runner_run(&runner, head_dim, keys_per_dpu, queries, keys_partitioned, scores_partitioned, &query_cycles);
        if (query_cycles > max_cycles) {
            max_cycles = query_cycles;
        }
        if (rc == 0) {
            for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
                for (uint32_t local_key = 0; local_key < keys_per_dpu; ++local_key) {
                    uint32_t global_key = dpu_idx * keys_per_dpu + local_key;
                    if (global_key < total_keys) {
                        scores_out[(size_t)query_idx * total_keys + global_key] =
                            scores_partitioned[(size_t)dpu_idx * keys_per_dpu + local_key];
                    }
                }
            }
        }
    }
    qk_runner_destroy(&runner);

    FILE *output = fopen(output_path, "wb");
    if (output == NULL) {
        perror("fopen output");
        rc = 1;
    } else {
        qk_io_header_t out_header = {
            .magic = QK_IO_MAGIC,
            .head_dim = head_dim,
            .num_keys = total_keys,
            .reserved = num_queries,
        };
        if (write_exact(output, &out_header, sizeof(out_header)) != 0
            || write_exact(output, scores_out, (size_t)num_queries * total_keys * sizeof(int64_t)) != 0) {
            fprintf(stderr, "Failed to write qk output\n");
            rc = 1;
        }
        fclose(output);
    }

    free(queries_in);
    free(keys_in);
    free(queries);
    free(keys_partitioned);
    free(scores_partitioned);
    free(scores_out);
    return rc;
}

int main(int argc, char **argv)
{
    uint32_t requested_dpus = 2;
    uint32_t head_dim = 64;
    uint32_t keys_per_dpu = 8;
    const char *input_path = NULL;
    const char *output_path = NULL;
    int stdio_mode = 0;

    for (int idx = 1; idx < argc; ++idx) {
        if (strcmp(argv[idx], "--input") == 0 && idx + 1 < argc) {
            input_path = argv[++idx];
        } else if (strcmp(argv[idx], "--output") == 0 && idx + 1 < argc) {
            output_path = argv[++idx];
        } else if (strcmp(argv[idx], "--num-dpus") == 0 && idx + 1 < argc) {
            requested_dpus = (uint32_t)strtoul(argv[++idx], NULL, 10);
        } else if (strcmp(argv[idx], "--stdio") == 0) {
            stdio_mode = 1;
        }
    }
    if (stdio_mode) {
        return run_stdio_mode(requested_dpus);
    }
    if (input_path != NULL || output_path != NULL) {
        if (input_path == NULL || output_path == NULL) {
            usage(argv[0]);
            return 2;
        }
        return run_file_mode(input_path, output_path, requested_dpus);
    }

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

    size_t query_elems = head_dim;
    size_t key_elems_per_dpu = (size_t)keys_per_dpu * head_dim;
    int32_t *queries = calloc((size_t)requested_dpus * query_elems, sizeof(*queries));
    int32_t *keys = calloc((size_t)requested_dpus * key_elems_per_dpu, sizeof(*keys));
    int64_t *scores = calloc((size_t)requested_dpus * keys_per_dpu, sizeof(*scores));
    int64_t *expected = calloc((size_t)requested_dpus * keys_per_dpu, sizeof(*expected));
    if (!queries || !keys || !scores || !expected) {
        fprintf(stderr, "Failed to allocate host buffers\n");
        free(queries);
        free(keys);
        free(scores);
        free(expected);
        return 1;
    }

    for (uint32_t dpu_idx = 0; dpu_idx < requested_dpus; ++dpu_idx) {
        int32_t *q = &queries[(size_t)dpu_idx * query_elems];
        int32_t *k = &keys[(size_t)dpu_idx * key_elems_per_dpu];
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

    uint64_t max_cycles = 0;
    qk_runner_t runner;
    int rc = qk_runner_init(&runner, requested_dpus);
    if (rc != 0) {
        free(queries);
        free(keys);
        free(scores);
        free(expected);
        return rc;
    }
    rc = qk_runner_run(&runner, head_dim, keys_per_dpu, queries, keys, scores, &max_cycles);
    qk_runner_destroy(&runner);
    if (rc != 0) {
        free(queries);
        free(keys);
        free(scores);
        free(expected);
        return rc;
    }

    int ok = 1;
    for (uint32_t dpu_idx = 0; dpu_idx < requested_dpus; ++dpu_idx) {
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
    printf("dpus=%u head_dim=%u keys_per_dpu=%u\n", requested_dpus, head_dim, keys_per_dpu);
    printf("max_dpu_cycles=%" PRIu64 "\n", max_cycles);
    printf("status=%s\n", ok ? "PASS" : "FAIL");

    free(queries);
    free(keys);
    free(scores);
    free(expected);
    return ok ? 0 : 1;
}
