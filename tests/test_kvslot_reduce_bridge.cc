#include "src/host/reduction/attention_reducer.h"
#include "src/host/reduction/kvslot_reduce_bridge.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

static bool approx_eq(float lhs, float rhs, float tol = 1e-4f)
{
    return std::fabs(lhs - rhs) <= tol;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s /abs/path/to/host_kvslot [num_dpus]\n", argv[0]);
        return 2;
    }
    const std::string helper_path = argv[1];
    const uint32_t num_dpus = argc >= 3 ? (uint32_t)std::stoul(argv[2]) : 2u;

    CloverKvslotHelperClient client(helper_path, num_dpus);
    std::string error;
    if (!client.start(&error)) {
        std::fprintf(stderr, "failed to start helper: %s\n", error.c_str());
        return 3;
    }

    const uint32_t seq_len = 2u;
    const uint32_t group_heads = 1u;
    const uint32_t head_dim = 2u;
    const uint32_t capacity = 4u;

    const std::vector<float> k0 = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const std::vector<float> v0 = {
        10.0f, 0.0f,
        0.0f, 20.0f,
    };
    const std::vector<float> k1 = {
        1.0f, 1.0f,
        2.0f, 0.0f,
    };
    const std::vector<float> v1 = {
        5.0f, 5.0f,
        8.0f, 1.0f,
    };

    const uint32_t slot0 = 0u;
    const uint32_t slot1 = 1u;
    if (!client.allocate_slot(slot0, capacity, seq_len, group_heads, head_dim, k0.data(), v0.data(), &error)) {
        std::fprintf(stderr, "allocate slot0 failed: %s\n", error.c_str());
        return 4;
    }
    if (!client.allocate_slot(slot1, capacity, seq_len, group_heads, head_dim, k1.data(), v1.data(), &error)) {
        std::fprintf(stderr, "allocate slot1 failed: %s\n", error.c_str());
        client.free_slot(slot0, nullptr);
        return 5;
    }

    CloverKvslotReducerFetchContext fetch_ctx;
    fetch_ctx.client = &client;
    CloverKvslotReducerRequest request;
    request.slot_ids_by_dpu = {slot0, slot1};
    request.window = seq_len;
    const float score_scale = 0.5f;
    request.score_scale = score_scale;
    request.local_head_indices = {0u};
    request.queries = {1.0f, 0.0f};
    fetch_ctx.requests.push_back(request);

    clover_attention_reduce_engine_t engine;
    clover_attention_reduce_engine_init(&engine);
    uint32_t slot_id = 0u;
    if (clover_attention_reduce_engine_submit(
            &engine,
            1u,
            num_dpus,
            1u,
            head_dim,
            clover_attention_reduce_fetch_from_kvslot_partial,
            &fetch_ctx,
            &slot_id) != 0) {
        std::fprintf(stderr, "engine submit failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        client.free_slot(slot1, nullptr);
        client.free_slot(slot0, nullptr);
        return 6;
    }

    clover_attention_reduce_completion_t completion;
    if (clover_attention_reduce_engine_wait_ready(&engine, 5000u, &completion) != 0) {
        std::fprintf(stderr, "engine wait failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        client.free_slot(slot1, nullptr);
        client.free_slot(slot0, nullptr);
        return 7;
    }
    if (completion.status_code != 0 || completion.global_output == nullptr) {
        std::fprintf(stderr, "completion invalid status=%d\n", completion.status_code);
        clover_attention_reduce_engine_destroy(&engine);
        client.free_slot(slot1, nullptr);
        client.free_slot(slot0, nullptr);
        return 8;
    }

    const float score0_0 = 1.0f * score_scale;
    const float score0_1 = 0.0f * score_scale;
    const float row_max0 = std::max(score0_0, score0_1);
    const float row_sum0 = std::exp(score0_0 - row_max0) + std::exp(score0_1 - row_max0);
    const float weight00 = std::exp(score0_0 - row_max0) / row_sum0;
    const float weight01 = std::exp(score0_1 - row_max0) / row_sum0;
    const float partial0_x = weight00 * 10.0f + weight01 * 0.0f;
    const float partial0_y = weight00 * 0.0f + weight01 * 20.0f;

    const float score1_0 = 1.0f * score_scale;
    const float score1_1 = 2.0f * score_scale;
    const float row_max1 = std::max(score1_0, score1_1);
    const float row_sum1 = std::exp(score1_0 - row_max1) + std::exp(score1_1 - row_max1);
    const float weight10 = std::exp(score1_0 - row_max1) / row_sum1;
    const float weight11 = std::exp(score1_1 - row_max1) / row_sum1;
    const float partial1_x = weight10 * 5.0f + weight11 * 8.0f;
    const float partial1_y = weight10 * 5.0f + weight11 * 1.0f;

    const float global_max = std::max(row_max0, row_max1);
    const float scaled_sum0 = row_sum0 * std::exp(row_max0 - global_max);
    const float scaled_sum1 = row_sum1 * std::exp(row_max1 - global_max);
    const float global_sum = scaled_sum0 + scaled_sum1;
    const float expected_x = partial0_x * (scaled_sum0 / global_sum) + partial1_x * (scaled_sum1 / global_sum);
    const float expected_y = partial0_y * (scaled_sum0 / global_sum) + partial1_y * (scaled_sum1 / global_sum);

    if (!approx_eq(completion.global_output[0], expected_x, 5e-3f)
        || !approx_eq(completion.global_output[1], expected_y, 5e-3f)) {
        std::fprintf(stderr, "unexpected reduced context: got=(%f,%f) expected=(%f,%f)\n",
            completion.global_output[0], completion.global_output[1], expected_x, expected_y);
        clover_attention_reduce_engine_release_slot(&engine, completion.slot_id);
        clover_attention_reduce_engine_destroy(&engine);
        client.free_slot(slot1, nullptr);
        client.free_slot(slot0, nullptr);
        return 9;
    }

    if (clover_attention_reduce_engine_release_slot(&engine, completion.slot_id) != 0) {
        std::fprintf(stderr, "release slot failed\n");
        clover_attention_reduce_engine_destroy(&engine);
        client.free_slot(slot1, nullptr);
        client.free_slot(slot0, nullptr);
        return 10;
    }

    clover_attention_reduce_engine_destroy(&engine);
    client.free_slot(slot1, nullptr);
    client.free_slot(slot0, nullptr);
    client.close();
    std::puts("kvslot reduce bridge smoke ok");
    return 0;
}
