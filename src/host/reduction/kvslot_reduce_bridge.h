#ifndef CLOVER_KVSLOT_REDUCE_BRIDGE_H
#define CLOVER_KVSLOT_REDUCE_BRIDGE_H

#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

#include "src/host/reduction/attention_reducer.h"

struct CloverKvslotPartialBatchItem {
    uint32_t slot_id = 0;
    uint32_t window = 0;
    float score_scale = 1.0f;
    std::vector<uint32_t> local_head_indices;
    std::vector<float> queries;  // row-major, one row per local head.
};

struct CloverKvslotPartialBatchResult {
    uint32_t slot_id = 0;
    uint32_t group_heads = 0;
    uint32_t head_dim = 0;
    std::vector<float> context;
    std::vector<float> row_max;
    std::vector<float> row_sum;
};

class CloverKvslotHelperClient {
public:
    CloverKvslotHelperClient(std::string helper_path, uint32_t num_dpus);
    ~CloverKvslotHelperClient();

    CloverKvslotHelperClient(const CloverKvslotHelperClient &) = delete;
    CloverKvslotHelperClient &operator=(const CloverKvslotHelperClient &) = delete;

    bool start(std::string *error = nullptr);
    void close();
    bool running() const;

    bool allocate_slot(
        uint32_t slot_id,
        uint32_t capacity,
        uint32_t seq_len,
        uint32_t group_heads,
        uint32_t head_dim,
        const float *k,
        const float *v,
        std::string *error = nullptr);

    bool free_slot(uint32_t slot_id, std::string *error = nullptr);

    bool qk_softmax_av_partial_batch(
        const std::vector<CloverKvslotPartialBatchItem> &items,
        std::vector<CloverKvslotPartialBatchResult> *results,
        std::string *error = nullptr);

    const std::string &helper_path() const;
    uint32_t num_dpus() const;

private:
    std::string helper_path_;
    uint32_t num_dpus_ = 0;
    pid_t child_pid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    mutable std::mutex io_mu_;

    bool write_all(const void *data, size_t size, std::string *error);
    bool read_all(void *data, size_t size, std::string *error);
    bool spawn_process(std::string *error);
    void reset_process_state();
};

struct CloverKvslotReducerRequest {
    std::vector<uint32_t> slot_ids_by_dpu;
    uint32_t window = 0;
    float score_scale = 1.0f;
    std::vector<uint32_t> local_head_indices;
    std::vector<float> queries;  // row-major, one row per local head.
};

struct CloverKvslotReducerFetchContext {
    CloverKvslotHelperClient *client = nullptr;
    std::vector<CloverKvslotReducerRequest> requests;
};

int clover_attention_reduce_fetch_from_kvslot_partial(
    void *user_ctx,
    uint32_t micro_batch_id,
    uint32_t num_dpus,
    uint32_t num_requests,
    uint32_t d_head,
    float *local_max_out,
    float *local_sum_out,
    float *local_output_out);

#endif
