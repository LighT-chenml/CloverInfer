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
} host_slot_t;

typedef struct {
    struct dpu_set_t dpu_set;
    uint32_t nr_dpus;
    host_slot_t *slots;
} kvslot_runner_t;

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
    return 0;
}

static int runner_get_dpu(kvslot_runner_t *runner, uint32_t slot_id, struct dpu_set_t *target_out)
{
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    if (runner == NULL || target_out == NULL || slot_id >= runner->nr_dpus) {
        return 1;
    }
    DPU_FOREACH(runner->dpu_set, dpu, each_dpu)
    {
        if (each_dpu == slot_id) {
            *target_out = dpu;
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
    DPU_ASSERT(dpu_alloc(requested_dpus, NULL, &runner->dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(runner->dpu_set, &runner->nr_dpus));
    DPU_ASSERT(dpu_load(runner->dpu_set, DPU_BINARY, NULL));
    if (runner->nr_dpus != requested_dpus) {
        fprintf(stderr, "Requested %u DPUs, allocated %u DPUs\n", requested_dpus, runner->nr_dpus);
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
        return 1;
    }
    runner->slots = calloc(runner->nr_dpus, sizeof(host_slot_t));
    if (runner->slots == NULL) {
        fprintf(stderr, "Failed to allocate slot table\n");
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
        for (uint32_t idx = 0; idx < runner->nr_dpus; ++idx) {
            free_slot(&runner->slots[idx]);
        }
        free(runner->slots);
        runner->slots = NULL;
    }
    if (runner->nr_dpus > 0) {
        dpu_free(runner->dpu_set);
        runner->nr_dpus = 0;
    }
}

static int handle_allocate(kvslot_runner_t *runner, uint32_t slot_id)
{
    struct dpu_set_t target_dpu;
    kvslot_slot_args_t args;
    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read allocate args\n");
        return 1;
    }
    if (slot_id >= runner->nr_dpus) {
        fprintf(stderr, "Invalid slot id %u for allocate\n", slot_id);
        return 1;
    }
    if (runner_get_dpu(runner, slot_id, &target_dpu) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    host_slot_t *slot = &runner->slots[slot_id];
    if (ensure_slot(slot, args.capacity, args.group_heads, args.head_dim) != 0) {
        return 1;
    }

    size_t elems = (size_t)args.seq_len * args.group_heads * args.head_dim;
    size_t bytes = elems * sizeof(int32_t);
    if (args.seq_len > args.capacity) {
        fprintf(stderr, "Initial seq_len exceeds capacity\n");
        return 1;
    }
    int32_t *k_data = NULL;
    int32_t *v_data = NULL;
    if (elems > 0) {
        k_data = calloc(elems, sizeof(int32_t));
        v_data = calloc(elems, sizeof(int32_t));
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
    DPU_ASSERT(dpu_broadcast_to(target_dpu, "slot_args", 0, &args, sizeof(args), DPU_XFER_DEFAULT));
    if (bytes > 0) {
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "k_cache", 0, bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_TO_DPU, "v_cache", 0, bytes, DPU_XFER_DEFAULT));
    }
    slot->seq_len = args.seq_len;
    free(k_data);
    free(v_data);

    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
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
    kvslot_slot_args_t args;
    if (read_exact(stdin, &args, sizeof(args)) != 0) {
        fprintf(stderr, "Failed to read append args\n");
        return 1;
    }
    if (slot_id >= runner->nr_dpus) {
        fprintf(stderr, "Invalid slot id %u for append\n", slot_id);
        return 1;
    }
    if (runner_get_dpu(runner, slot_id, &target_dpu) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    host_slot_t *slot = &runner->slots[slot_id];
    if (slot->capacity == 0) {
        fprintf(stderr, "Append on uninitialized slot %u\n", slot_id);
        return 1;
    }
    if (args.seq_len != 1 || args.group_heads != slot->group_heads || args.head_dim != slot->head_dim) {
        fprintf(stderr, "Append args mismatch for slot %u\n", slot_id);
        return 1;
    }
    if (slot->seq_len + 1 > slot->capacity) {
        fprintf(stderr, "Slot %u capacity exceeded\n", slot_id);
        return 1;
    }

    size_t token_elems = (size_t)slot->group_heads * slot->head_dim;
    size_t token_bytes = token_elems * sizeof(int32_t);
    int32_t *k_token = calloc(token_elems, sizeof(int32_t));
    int32_t *v_token = calloc(token_elems, sizeof(int32_t));
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
    size_t byte_offset = (size_t)slot->seq_len * token_bytes;
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
    if (slot_id >= runner->nr_dpus) {
        fprintf(stderr, "Invalid slot id %u for readback\n", slot_id);
        return 1;
    }
    if (runner_get_dpu(runner, slot_id, &target_dpu) != 0) {
        fprintf(stderr, "Failed to locate DPU for slot %u\n", slot_id);
        return 1;
    }
    host_slot_t *slot = &runner->slots[slot_id];
    if (slot->capacity == 0) {
        fprintf(stderr, "Readback on uninitialized slot %u\n", slot_id);
        return 1;
    }
    kvslot_slot_args_t out = {
        .capacity = slot->capacity,
        .seq_len = slot->seq_len,
        .group_heads = slot->group_heads,
        .head_dim = slot->head_dim,
    };
    size_t elems = (size_t)slot->seq_len * slot->group_heads * slot->head_dim;
    size_t bytes = elems * sizeof(int32_t);
    int32_t *k_data = NULL;
    int32_t *v_data = NULL;
    if (elems > 0) {
        k_data = calloc(elems, sizeof(int32_t));
        v_data = calloc(elems, sizeof(int32_t));
        if (k_data == NULL || v_data == NULL) {
            fprintf(stderr, "Failed to allocate readback buffers\n");
            free(k_data);
            free(v_data);
            return 1;
        }
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, k_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "k_cache", 0, bytes, DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_prepare_xfer(target_dpu, v_data));
        DPU_ASSERT(dpu_push_xfer(target_dpu, DPU_XFER_FROM_DPU, "v_cache", 0, bytes, DPU_XFER_DEFAULT));
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
