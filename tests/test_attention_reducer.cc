#include "src/host/reduction/attention_reducer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct FakeFetchCtx {
    uint32_t delay_ms;
    std::vector<float> local_max;
    std::vector<float> local_sum;
    std::vector<float> local_output;
};

static bool approx_eq(float lhs, float rhs, float tol = 1e-5f)
{
    return std::fabs(lhs - rhs) <= tol;
}

static int fake_fetch(
    void *user_ctx,
    uint32_t micro_batch_id,
    uint32_t num_dpus,
    uint32_t num_requests,
    uint32_t d_head,
    float *local_max_out,
    float *local_sum_out,
    float *local_output_out)
{
    (void)micro_batch_id;
    FakeFetchCtx *ctx = reinterpret_cast<FakeFetchCtx *>(user_ctx);
    if (ctx == nullptr) {
        return 7;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(ctx->delay_ms));
    const size_t max_count = (size_t)num_dpus * (size_t)num_requests;
    const size_t output_count = max_count * (size_t)d_head;
    std::memcpy(local_max_out, ctx->local_max.data(), max_count * sizeof(float));
    std::memcpy(local_sum_out, ctx->local_sum.data(), max_count * sizeof(float));
    std::memcpy(local_output_out, ctx->local_output.data(), output_count * sizeof(float));
    return 0;
}

static int test_sync_reduce()
{
    std::vector<float> local_max = {
        2.0f, 1.0f,
        3.0f, 0.5f,
    };
    std::vector<float> local_sum = {
        1.0f, 2.0f,
        4.0f, 3.0f,
    };
    std::vector<float> local_output = {
        10.0f, 20.0f,
        1.0f,  2.0f,
        6.0f,  8.0f,
        3.0f,  6.0f,
    };
    std::vector<float> global_output(4, 0.0f);
    std::vector<float> global_max(2, 0.0f);
    std::vector<float> global_sum(2, 0.0f);

    clover_attention_reduce_input_t input;
    input.num_requests = 2u;
    input.d_head = 2u;
    input.local_max = local_max.data();
    input.local_sum = local_sum.data();
    input.local_output = local_output.data();

    clover_attention_reduce_output_t output;
    output.num_requests = 2u;
    output.d_head = 2u;
    output.global_output = global_output.data();
    output.global_max = global_max.data();
    output.global_sum = global_sum.data();

    if (clover_attention_reduce_group(2u, &input, &output) != 0) {
        std::fprintf(stderr, "sync reduce returned non-zero\n");
        return 1;
    }

    const float req0_sum = 1.0f * std::exp(2.0f - 3.0f) + 4.0f;
    const float req0_scale0 = std::exp(2.0f - 3.0f) / req0_sum;
    const float req0_scale1 = 1.0f / req0_sum;
    const float req1_sum = 2.0f + 3.0f * std::exp(0.5f - 1.0f);
    const float req1_scale0 = 1.0f / req1_sum;
    const float req1_scale1 = std::exp(0.5f - 1.0f) / req1_sum;

    if (!approx_eq(global_max[0], 3.0f) || !approx_eq(global_max[1], 1.0f)) {
        std::fprintf(stderr, "unexpected global_max values\n");
        return 2;
    }
    if (!approx_eq(global_sum[0], req0_sum) || !approx_eq(global_sum[1], req1_sum)) {
        std::fprintf(stderr, "unexpected global_sum values\n");
        return 3;
    }
    if (!approx_eq(global_output[0], 10.0f * req0_scale0 + 6.0f * req0_scale1)) {
        std::fprintf(stderr, "unexpected global_output[0]\n");
        return 4;
    }
    if (!approx_eq(global_output[1], 20.0f * req0_scale0 + 8.0f * req0_scale1)) {
        std::fprintf(stderr, "unexpected global_output[1]\n");
        return 5;
    }
    if (!approx_eq(global_output[2], 1.0f * req1_scale0 + 3.0f * req1_scale1)) {
        std::fprintf(stderr, "unexpected global_output[2]\n");
        return 6;
    }
    if (!approx_eq(global_output[3], 2.0f * req1_scale0 + 6.0f * req1_scale1)) {
        std::fprintf(stderr, "unexpected global_output[3]\n");
        return 7;
    }
    return 0;
}

static int test_async_double_buffer()
{
    FakeFetchCtx ctx0;
    ctx0.delay_ms = 20u;
    ctx0.local_max = {1.0f, 2.0f};
    ctx0.local_sum = {2.0f, 3.0f};
    ctx0.local_output = {4.0f, 6.0f};

    FakeFetchCtx ctx1 = ctx0;
    ctx1.delay_ms = 5u;
    ctx1.local_output = {8.0f, 10.0f};

    clover_attention_reduce_engine_t engine;
    clover_attention_reduce_engine_init(&engine);
    if (engine.impl_handle == 0u) {
        std::fprintf(stderr, "engine init failed\n");
        return 10;
    }

    uint32_t slot0 = 99u;
    uint32_t slot1 = 99u;
    if (clover_attention_reduce_engine_submit(&engine, 100u, 1u, 1u, 2u, fake_fetch, &ctx0, &slot0) != 0) {
        std::fprintf(stderr, "submit slot0 failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 11;
    }
    if (clover_attention_reduce_engine_submit(&engine, 101u, 1u, 1u, 2u, fake_fetch, &ctx1, &slot1) != 0) {
        std::fprintf(stderr, "submit slot1 failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 12;
    }
    if (slot0 == slot1) {
        std::fprintf(stderr, "double buffer reused same slot unexpectedly\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 13;
    }
    if (clover_attention_reduce_engine_submit(&engine, 102u, 1u, 1u, 2u, fake_fetch, &ctx1, nullptr) != 2) {
        std::fprintf(stderr, "expected no-free-slot rejection\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 14;
    }

    clover_attention_reduce_completion_t completion;
    if (clover_attention_reduce_engine_wait_ready(&engine, 200u, &completion) != 0) {
        std::fprintf(stderr, "wait_ready failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 15;
    }
    if (completion.status_code != 0 || completion.global_output == nullptr) {
        std::fprintf(stderr, "completion invalid\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 16;
    }
    if (completion.micro_batch_id != 100u && completion.micro_batch_id != 101u) {
        std::fprintf(stderr, "unexpected micro_batch_id\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 17;
    }
    if (clover_attention_reduce_engine_release_slot(&engine, completion.slot_id) != 0) {
        std::fprintf(stderr, "release after completion failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 18;
    }

    if (clover_attention_reduce_engine_wait_ready(&engine, 200u, &completion) != 0) {
        std::fprintf(stderr, "second wait_ready failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 19;
    }
    if (clover_attention_reduce_engine_release_slot(&engine, completion.slot_id) != 0) {
        std::fprintf(stderr, "second release failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 20;
    }
    if (engine.inflight_count != 0u) {
        std::fprintf(stderr, "inflight_count did not return to zero\n");
        clover_attention_reduce_engine_destroy(&engine);
        return 21;
    }

    clover_attention_reduce_engine_destroy(&engine);
    return 0;
}

}  // namespace

int main()
{
    int rc = test_sync_reduce();
    if (rc != 0) {
        return rc;
    }
    rc = test_async_double_buffer();
    if (rc != 0) {
        return rc;
    }
    std::puts("attention_reducer smoke ok");
    return 0;
}
