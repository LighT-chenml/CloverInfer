#include "attention_reducer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace {

using steady_clock_t = std::chrono::steady_clock;

static uint64_t now_ns()
{
    const auto now = steady_clock_t::now().time_since_epoch();
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

struct SlotBuffers {
    std::vector<float> local_max;
    std::vector<float> local_sum;
    std::vector<float> local_output;
    std::vector<float> global_output;
    std::vector<float> global_max;
    std::vector<float> global_sum;

    void ensure_sizes(uint32_t num_dpus, uint32_t num_requests, uint32_t d_head)
    {
        local_max.resize((size_t)num_dpus * (size_t)num_requests);
        local_sum.resize((size_t)num_dpus * (size_t)num_requests);
        local_output.resize((size_t)num_dpus * (size_t)num_requests * (size_t)d_head);
        global_output.resize((size_t)num_requests * (size_t)d_head);
        global_max.resize((size_t)num_requests);
        global_sum.resize((size_t)num_requests);
    }
};

struct SlotRuntime {
    SlotBuffers buffers;
    clover_attention_reduce_fetch_fn fetch_fn = nullptr;
    void *fetch_user_ctx = nullptr;
    std::thread worker;
};

struct EngineRuntime {
    std::mutex mu;
    std::condition_variable cv;
    SlotRuntime slot_runtime[2];
    uint32_t ready_queue[2] = {0u, 0u};
    uint32_t ready_head = 0u;
    uint32_t ready_tail = 0u;
    uint32_t ready_size = 0u;
    bool shutting_down = false;
};

static EngineRuntime *get_runtime(clover_attention_reduce_engine_t *engine)
{
    if (engine == nullptr || engine->impl_handle == 0u) {
        return nullptr;
    }
    return reinterpret_cast<EngineRuntime *>(engine->impl_handle);
}

static int reduce_group_impl(
    uint32_t num_dpus,
    const clover_attention_reduce_input_t *input,
    clover_attention_reduce_output_t *output)
{
    if (num_dpus == 0u || input == nullptr || output == nullptr) {
        return 1;
    }
    if (input->num_requests == 0u || input->d_head == 0u) {
        return 1;
    }
    if (input->local_max == nullptr || input->local_sum == nullptr || input->local_output == nullptr) {
        return 1;
    }
    if (output->global_output == nullptr || output->global_max == nullptr || output->global_sum == nullptr) {
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

static void publish_ready_slot(
    clover_attention_reduce_engine_t *engine,
    EngineRuntime *runtime,
    uint32_t slot_id)
{
    runtime->ready_queue[runtime->ready_tail] = slot_id;
    runtime->ready_tail = (runtime->ready_tail + 1u) % 2u;
    runtime->ready_size += 1u;
    engine->ready_count = runtime->ready_size;
    runtime->cv.notify_all();
}

static void worker_run_slot(
    clover_attention_reduce_engine_t *engine,
    EngineRuntime *runtime,
    uint32_t slot_id)
{
    SlotRuntime *slot_runtime = &runtime->slot_runtime[slot_id];
    clover_attention_reduce_slot_t *slot = &engine->slots[slot_id];
    const uint64_t total_start_ns = now_ns();
    uint64_t copy_start_ns = total_start_ns;
    uint64_t reduce_start_ns = total_start_ns;
    int status = 0;

    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        slot->state = CLOVER_ATTENTION_REDUCE_SLOT_COPYING;
    }

    try {
        if (slot_runtime->fetch_fn == nullptr) {
            status = 2;
        } else {
            slot_runtime->buffers.ensure_sizes(slot->num_dpus, slot->num_requests, slot->d_head);
            copy_start_ns = now_ns();
            status = slot_runtime->fetch_fn(
                slot_runtime->fetch_user_ctx,
                slot->micro_batch_id,
                slot->num_dpus,
                slot->num_requests,
                slot->d_head,
                slot_runtime->buffers.local_max.data(),
                slot_runtime->buffers.local_sum.data(),
                slot_runtime->buffers.local_output.data());
            slot->copy_latency_ns = now_ns() - copy_start_ns;
        }

        if (status == 0) {
            clover_attention_reduce_input_t input;
            clover_attention_reduce_output_t output;
            input.num_requests = slot->num_requests;
            input.d_head = slot->d_head;
            input.local_max = slot_runtime->buffers.local_max.data();
            input.local_sum = slot_runtime->buffers.local_sum.data();
            input.local_output = slot_runtime->buffers.local_output.data();
            output.num_requests = slot->num_requests;
            output.d_head = slot->d_head;
            output.global_output = slot_runtime->buffers.global_output.data();
            output.global_max = slot_runtime->buffers.global_max.data();
            output.global_sum = slot_runtime->buffers.global_sum.data();

            {
                std::lock_guard<std::mutex> lock(runtime->mu);
                slot->state = CLOVER_ATTENTION_REDUCE_SLOT_REDUCING;
            }
            reduce_start_ns = now_ns();
            status = reduce_group_impl(slot->num_dpus, &input, &output);
            slot->reduce_latency_ns = now_ns() - reduce_start_ns;
        } else {
            slot->reduce_latency_ns = 0u;
        }
    } catch (const std::bad_alloc &) {
        status = 3;
        slot->reduce_latency_ns = 0u;
    } catch (...) {
        status = 4;
        slot->reduce_latency_ns = 0u;
    }

    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        slot->status_code = status;
        slot->busy = 1u;
        slot->total_latency_ns = now_ns() - total_start_ns;
        slot->state = (status == 0) ? CLOVER_ATTENTION_REDUCE_SLOT_READY : CLOVER_ATTENTION_REDUCE_SLOT_FAILED;
        publish_ready_slot(engine, runtime, slot_id);
    }
}

static int build_completion(
    clover_attention_reduce_engine_t *engine,
    EngineRuntime *runtime,
    uint32_t slot_id,
    clover_attention_reduce_completion_t *completion_out)
{
    if (completion_out == nullptr || slot_id >= 2u) {
        return 1;
    }
    clover_attention_reduce_slot_t *slot = &engine->slots[slot_id];
    SlotRuntime *slot_runtime = &runtime->slot_runtime[slot_id];
    completion_out->slot_id = slot_id;
    completion_out->micro_batch_id = slot->micro_batch_id;
    completion_out->num_dpus = slot->num_dpus;
    completion_out->num_requests = slot->num_requests;
    completion_out->d_head = slot->d_head;
    completion_out->status_code = slot->status_code;
    completion_out->state = slot->state;
    completion_out->global_output = slot->status_code == 0 ? slot_runtime->buffers.global_output.data() : nullptr;
    completion_out->global_max = slot->status_code == 0 ? slot_runtime->buffers.global_max.data() : nullptr;
    completion_out->global_sum = slot->status_code == 0 ? slot_runtime->buffers.global_sum.data() : nullptr;
    return 0;
}

}  // namespace

void clover_attention_reduce_engine_init(clover_attention_reduce_engine_t *engine)
{
    if (engine == nullptr) {
        return;
    }
    memset(engine, 0, sizeof(*engine));
    engine->num_slots = 2u;
    engine->next_submit_slot = 0u;
    try {
        EngineRuntime *runtime = new EngineRuntime();
        engine->impl_handle = reinterpret_cast<uintptr_t>(runtime);
    } catch (const std::bad_alloc &) {
        engine->impl_handle = 0u;
    }
    for (uint32_t idx = 0; idx < 2u; ++idx) {
        engine->slots[idx].slot_id = idx;
        engine->slots[idx].micro_batch_id = 0u;
        engine->slots[idx].busy = 0u;
        engine->slots[idx].state = CLOVER_ATTENTION_REDUCE_SLOT_FREE;
        engine->slots[idx].num_dpus = 0u;
        engine->slots[idx].num_requests = 0u;
        engine->slots[idx].d_head = 0u;
        engine->slots[idx].status_code = 0;
        engine->slots[idx].copy_latency_ns = 0u;
        engine->slots[idx].reduce_latency_ns = 0u;
        engine->slots[idx].total_latency_ns = 0u;
    }
}

void clover_attention_reduce_engine_destroy(clover_attention_reduce_engine_t *engine)
{
    if (engine == nullptr) {
        return;
    }
    EngineRuntime *runtime = get_runtime(engine);
    if (runtime != nullptr) {
        std::thread workers[2];
        {
            std::lock_guard<std::mutex> lock(runtime->mu);
            runtime->shutting_down = true;
            runtime->cv.notify_all();
            for (uint32_t idx = 0; idx < 2u; ++idx) {
                workers[idx] = std::move(runtime->slot_runtime[idx].worker);
            }
        }
        for (uint32_t idx = 0; idx < 2u; ++idx) {
            if (workers[idx].joinable()) {
                workers[idx].join();
            }
        }
        delete runtime;
    }
    engine->impl_handle = 0u;
    engine->num_slots = 0u;
    engine->next_submit_slot = 0u;
    engine->inflight_count = 0u;
    engine->ready_count = 0u;
}

int clover_attention_reduce_group(
    uint32_t num_dpus,
    const clover_attention_reduce_input_t *input,
    clover_attention_reduce_output_t *output)
{
    return reduce_group_impl(num_dpus, input, output);
}

int clover_attention_reduce_engine_submit(
    clover_attention_reduce_engine_t *engine,
    uint32_t micro_batch_id,
    uint32_t num_dpus,
    uint32_t num_requests,
    uint32_t d_head,
    clover_attention_reduce_fetch_fn fetch_fn,
    void *fetch_user_ctx,
    uint32_t *slot_id_out)
{
    EngineRuntime *runtime = get_runtime(engine);
    if (engine == nullptr || runtime == nullptr || fetch_fn == nullptr) {
        return 1;
    }
    if (num_dpus == 0u || num_requests == 0u || d_head == 0u) {
        return 1;
    }

    uint32_t chosen_slot = 2u;
    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        for (uint32_t offset = 0; offset < 2u; ++offset) {
            const uint32_t slot_id = (engine->next_submit_slot + offset) % 2u;
            if (engine->slots[slot_id].state == CLOVER_ATTENTION_REDUCE_SLOT_FREE) {
                chosen_slot = slot_id;
                break;
            }
        }
        if (chosen_slot >= 2u) {
            return 2;
        }

        SlotRuntime *slot_runtime = &runtime->slot_runtime[chosen_slot];
        clover_attention_reduce_slot_t *slot = &engine->slots[chosen_slot];
        slot_runtime->fetch_fn = fetch_fn;
        slot_runtime->fetch_user_ctx = fetch_user_ctx;
        slot->micro_batch_id = micro_batch_id;
        slot->busy = 1u;
        slot->state = CLOVER_ATTENTION_REDUCE_SLOT_COPYING;
        slot->num_dpus = num_dpus;
        slot->num_requests = num_requests;
        slot->d_head = d_head;
        slot->status_code = 0;
        slot->copy_latency_ns = 0u;
        slot->reduce_latency_ns = 0u;
        slot->total_latency_ns = 0u;
        engine->next_submit_slot = (chosen_slot + 1u) % 2u;
        engine->inflight_count += 1u;
        engine->submissions += 1u;
    }

    try {
        runtime->slot_runtime[chosen_slot].worker = std::thread(worker_run_slot, engine, runtime, chosen_slot);
    } catch (...) {
        std::lock_guard<std::mutex> lock(runtime->mu);
        clover_attention_reduce_slot_t *slot = &engine->slots[chosen_slot];
        slot->busy = 0u;
        slot->state = CLOVER_ATTENTION_REDUCE_SLOT_FREE;
        slot->num_dpus = 0u;
        slot->num_requests = 0u;
        slot->d_head = 0u;
        slot->status_code = 0;
        slot->copy_latency_ns = 0u;
        slot->reduce_latency_ns = 0u;
        slot->total_latency_ns = 0u;
        if (engine->inflight_count > 0u) {
            engine->inflight_count -= 1u;
        }
        runtime->slot_runtime[chosen_slot].fetch_fn = nullptr;
        runtime->slot_runtime[chosen_slot].fetch_user_ctx = nullptr;
        return 3;
    }
    if (slot_id_out != nullptr) {
        *slot_id_out = chosen_slot;
    }
    return 0;
}

int clover_attention_reduce_engine_try_pop_ready(
    clover_attention_reduce_engine_t *engine,
    clover_attention_reduce_completion_t *completion_out)
{
    EngineRuntime *runtime = get_runtime(engine);
    if (engine == nullptr || runtime == nullptr || completion_out == nullptr) {
        return 1;
    }

    std::lock_guard<std::mutex> lock(runtime->mu);
    if (runtime->ready_size == 0u) {
        return 2;
    }
    const uint32_t slot_id = runtime->ready_queue[runtime->ready_head];
    runtime->ready_head = (runtime->ready_head + 1u) % 2u;
    runtime->ready_size -= 1u;
    engine->ready_count = runtime->ready_size;
    engine->completions += 1u;
    return build_completion(engine, runtime, slot_id, completion_out);
}

int clover_attention_reduce_engine_wait_ready(
    clover_attention_reduce_engine_t *engine,
    uint32_t timeout_ms,
    clover_attention_reduce_completion_t *completion_out)
{
    EngineRuntime *runtime = get_runtime(engine);
    if (engine == nullptr || runtime == nullptr || completion_out == nullptr) {
        return 1;
    }

    std::unique_lock<std::mutex> lock(runtime->mu);
    if (runtime->ready_size == 0u) {
        const auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!runtime->cv.wait_for(lock, timeout, [runtime]() { return runtime->ready_size > 0u || runtime->shutting_down; })) {
            return 2;
        }
        if (runtime->ready_size == 0u) {
            return 2;
        }
    }
    const uint32_t slot_id = runtime->ready_queue[runtime->ready_head];
    runtime->ready_head = (runtime->ready_head + 1u) % 2u;
    runtime->ready_size -= 1u;
    engine->ready_count = runtime->ready_size;
    engine->completions += 1u;
    return build_completion(engine, runtime, slot_id, completion_out);
}

int clover_attention_reduce_engine_release_slot(
    clover_attention_reduce_engine_t *engine,
    uint32_t slot_id)
{
    EngineRuntime *runtime = get_runtime(engine);
    if (engine == nullptr || runtime == nullptr || slot_id >= 2u) {
        return 1;
    }

    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        clover_attention_reduce_slot_t *slot = &engine->slots[slot_id];
        if (slot->state != CLOVER_ATTENTION_REDUCE_SLOT_READY && slot->state != CLOVER_ATTENTION_REDUCE_SLOT_FAILED) {
            return 2;
        }
        worker = std::move(runtime->slot_runtime[slot_id].worker);
    }
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        clover_attention_reduce_slot_t *slot = &engine->slots[slot_id];
        slot->busy = 0u;
        slot->state = CLOVER_ATTENTION_REDUCE_SLOT_FREE;
        slot->num_dpus = 0u;
        slot->num_requests = 0u;
        slot->d_head = 0u;
        slot->status_code = 0;
        slot->copy_latency_ns = 0u;
        slot->reduce_latency_ns = 0u;
        slot->total_latency_ns = 0u;
        if (engine->inflight_count > 0u) {
            engine->inflight_count -= 1u;
        }
        runtime->slot_runtime[slot_id].fetch_fn = nullptr;
        runtime->slot_runtime[slot_id].fetch_user_ctx = nullptr;
    }
    return 0;
}
