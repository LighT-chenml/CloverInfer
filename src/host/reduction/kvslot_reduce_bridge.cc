#include "kvslot_reduce_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t KVSLOT_MAGIC = 0x4B56534CU;
constexpr uint32_t KVSLOT_CMD_ALLOCATE = 1u;
constexpr uint32_t KVSLOT_CMD_FREE = 4u;
constexpr uint32_t KVSLOT_CMD_QK_SOFTMAX_AV_PARTIAL_BATCH = 12u;

struct KvslotIoHeader {
    uint32_t magic;
    uint32_t command;
    uint32_t slot_id;
    uint32_t reserved;
};

struct KvslotSlotArgs {
    uint32_t capacity;
    uint32_t seq_len;
    uint32_t group_heads;
    uint32_t head_dim;
    uint32_t dtype_code;
    float k_scale;
    float v_scale;
    uint32_t reserved;
};

struct KvslotAvBatchArgs {
    uint32_t num_slots;
    uint32_t reserved[3];
};

struct KvslotQkSoftmaxAvBatchItemArgs {
    uint32_t num_heads;
    uint32_t window;
    uint32_t head_dim;
    float score_scale;
};

static bool append_bytes(std::string *buffer, const void *data, size_t size)
{
    if (buffer == nullptr) {
        return false;
    }
    buffer->append(reinterpret_cast<const char *>(data), size);
    return true;
}

static void set_error(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

}  // namespace

CloverKvslotHelperClient::CloverKvslotHelperClient(std::string helper_path, uint32_t num_dpus)
    : helper_path_(std::move(helper_path)), num_dpus_(num_dpus)
{
}

CloverKvslotHelperClient::~CloverKvslotHelperClient()
{
    close();
}

bool CloverKvslotHelperClient::start(std::string *error)
{
    std::lock_guard<std::mutex> lock(io_mu_);
    if (running()) {
        return true;
    }
    return spawn_process(error);
}

void CloverKvslotHelperClient::close()
{
    std::lock_guard<std::mutex> lock(io_mu_);
    if (stdin_fd_ >= 0) {
        ::close(stdin_fd_);
        stdin_fd_ = -1;
    }
    if (stdout_fd_ >= 0) {
        ::close(stdout_fd_);
        stdout_fd_ = -1;
    }
    if (child_pid_ > 0) {
        ::kill(child_pid_, SIGTERM);
        int status = 0;
        (void)::waitpid(child_pid_, &status, 0);
        child_pid_ = -1;
    }
    reset_process_state();
}

bool CloverKvslotHelperClient::running() const
{
    return child_pid_ > 0 && stdin_fd_ >= 0 && stdout_fd_ >= 0;
}

bool CloverKvslotHelperClient::spawn_process(std::string *error)
{
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) {
        set_error(error, "pipe() failed");
        if (to_child[0] >= 0) {
            ::close(to_child[0]);
            ::close(to_child[1]);
        }
        if (from_child[0] >= 0) {
            ::close(from_child[0]);
            ::close(from_child[1]);
        }
        return false;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        set_error(error, "fork() failed");
        ::close(to_child[0]);
        ::close(to_child[1]);
        ::close(from_child[0]);
        ::close(from_child[1]);
        return false;
    }
    if (pid == 0) {
        ::dup2(to_child[0], STDIN_FILENO);
        ::dup2(from_child[1], STDOUT_FILENO);
        ::close(to_child[0]);
        ::close(to_child[1]);
        ::close(from_child[0]);
        ::close(from_child[1]);
        const std::string num_dpus_str = std::to_string(num_dpus_);
        execl(helper_path_.c_str(), helper_path_.c_str(), "--stdio", "--num-dpus", num_dpus_str.c_str(), (char *)nullptr);
        _exit(127);
    }

    ::close(to_child[0]);
    ::close(from_child[1]);
    child_pid_ = pid;
    stdin_fd_ = to_child[1];
    stdout_fd_ = from_child[0];
    return true;
}

void CloverKvslotHelperClient::reset_process_state()
{
    child_pid_ = -1;
    stdin_fd_ = -1;
    stdout_fd_ = -1;
}

bool CloverKvslotHelperClient::write_all(const void *data, size_t size, std::string *error)
{
    const uint8_t *cursor = reinterpret_cast<const uint8_t *>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t written = ::write(stdin_fd_, cursor, remaining);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error(error, "write() to kvslot helper failed");
            return false;
        }
        cursor += written;
        remaining -= (size_t)written;
    }
    return true;
}

bool CloverKvslotHelperClient::read_all(void *data, size_t size, std::string *error)
{
    uint8_t *cursor = reinterpret_cast<uint8_t *>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t nread = ::read(stdout_fd_, cursor, remaining);
        if (nread < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error(error, "read() from kvslot helper failed");
            return false;
        }
        if (nread == 0) {
            set_error(error, "kvslot helper closed pipe unexpectedly");
            return false;
        }
        cursor += nread;
        remaining -= (size_t)nread;
    }
    return true;
}

bool CloverKvslotHelperClient::allocate_slot(
    uint32_t slot_id,
    uint32_t capacity,
    uint32_t seq_len,
    uint32_t group_heads,
    uint32_t head_dim,
    const float *k,
    const float *v,
    std::string *error)
{
    std::lock_guard<std::mutex> lock(io_mu_);
    if (!running() && !spawn_process(error)) {
        return false;
    }

    const size_t elem_count = (size_t)seq_len * (size_t)group_heads * (size_t)head_dim;
    KvslotIoHeader header = {KVSLOT_MAGIC, KVSLOT_CMD_ALLOCATE, slot_id, 0u};
    KvslotSlotArgs args = {capacity, seq_len, group_heads, head_dim, 0u, 1.0f, 1.0f, 0u};
    if (!write_all(&header, sizeof(header), error) || !write_all(&args, sizeof(args), error)) {
        return false;
    }
    if (elem_count > 0) {
        if (!write_all(k, elem_count * sizeof(float), error) || !write_all(v, elem_count * sizeof(float), error)) {
            return false;
        }
    }
    KvslotSlotArgs out = {};
    return read_all(&out, sizeof(out), error);
}

bool CloverKvslotHelperClient::free_slot(uint32_t slot_id, std::string *error)
{
    std::lock_guard<std::mutex> lock(io_mu_);
    if (!running() && !spawn_process(error)) {
        return false;
    }
    KvslotIoHeader header = {KVSLOT_MAGIC, KVSLOT_CMD_FREE, slot_id, 0u};
    KvslotSlotArgs out = {};
    return write_all(&header, sizeof(header), error) && read_all(&out, sizeof(out), error);
}

bool CloverKvslotHelperClient::qk_softmax_av_partial_batch(
    const std::vector<CloverKvslotPartialBatchItem> &items,
    std::vector<CloverKvslotPartialBatchResult> *results,
    std::string *error)
{
    if (results == nullptr) {
        set_error(error, "results must not be null");
        return false;
    }
    std::lock_guard<std::mutex> lock(io_mu_);
    if (!running() && !spawn_process(error)) {
        return false;
    }
    if (items.empty()) {
        results->clear();
        return true;
    }

    KvslotIoHeader header = {KVSLOT_MAGIC, KVSLOT_CMD_QK_SOFTMAX_AV_PARTIAL_BATCH, 0u, 0u};
    KvslotAvBatchArgs batch_args = {};
    batch_args.num_slots = (uint32_t)items.size();
    if (!write_all(&header, sizeof(header), error) || !write_all(&batch_args, sizeof(batch_args), error)) {
        return false;
    }

    for (const CloverKvslotPartialBatchItem &item : items) {
        KvslotQkSoftmaxAvBatchItemArgs item_args = {};
        item_args.num_heads = (uint32_t)item.local_head_indices.size();
        item_args.window = item.window;
        item_args.head_dim = item_args.num_heads == 0u ? 0u : (uint32_t)(item.queries.size() / item_args.num_heads);
        item_args.score_scale = item.score_scale;
        if (!write_all(&item.slot_id, sizeof(item.slot_id), error)
            || !write_all(&item_args, sizeof(item_args), error)
            || !write_all(item.local_head_indices.data(), item.local_head_indices.size() * sizeof(uint32_t), error)
            || !write_all(item.queries.data(), item.queries.size() * sizeof(float), error)) {
            return false;
        }
    }

    KvslotAvBatchArgs out_args = {};
    if (!read_all(&out_args, sizeof(out_args), error)) {
        return false;
    }
    if (out_args.num_slots != items.size()) {
        set_error(error, "partial batch response size mismatch");
        return false;
    }

    results->clear();
    results->reserve(items.size());
    for (const CloverKvslotPartialBatchItem &item : items) {
        KvslotSlotArgs out = {};
        if (!read_all(&out, sizeof(out), error)) {
            return false;
        }
        CloverKvslotPartialBatchResult result;
        result.slot_id = item.slot_id;
        result.group_heads = out.group_heads;
        result.head_dim = out.head_dim;
        result.context.resize((size_t)out.group_heads * (size_t)out.head_dim);
        result.row_max.resize((size_t)out.group_heads);
        result.row_sum.resize((size_t)out.group_heads);
        if (!read_all(result.context.data(), result.context.size() * sizeof(float), error)
            || !read_all(result.row_max.data(), result.row_max.size() * sizeof(float), error)
            || !read_all(result.row_sum.data(), result.row_sum.size() * sizeof(float), error)) {
            return false;
        }
        results->push_back(std::move(result));
    }
    return true;
}

const std::string &CloverKvslotHelperClient::helper_path() const
{
    return helper_path_;
}

uint32_t CloverKvslotHelperClient::num_dpus() const
{
    return num_dpus_;
}

int clover_attention_reduce_fetch_from_kvslot_partial(
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
    CloverKvslotReducerFetchContext *ctx = reinterpret_cast<CloverKvslotReducerFetchContext *>(user_ctx);
    if (ctx == nullptr || ctx->client == nullptr) {
        return 1;
    }
    if (num_requests == 0u || d_head == 0u) {
        return 2;
    }
    if (ctx->requests.size() != (size_t)num_requests) {
        return 3;
    }

    std::vector<CloverKvslotPartialBatchItem> batch_items;
    batch_items.reserve((size_t)num_requests * (size_t)num_dpus);
    for (uint32_t req_idx = 0; req_idx < num_requests; ++req_idx) {
        const CloverKvslotReducerRequest &request = ctx->requests[req_idx];
        if (request.slot_ids_by_dpu.size() != (size_t)num_dpus) {
            return 4;
        }
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            CloverKvslotPartialBatchItem item;
            item.slot_id = request.slot_ids_by_dpu[dpu_idx];
            item.window = request.window;
            item.score_scale = request.score_scale;
            item.local_head_indices = request.local_head_indices;
            item.queries = request.queries;
            batch_items.push_back(std::move(item));
        }
    }

    std::vector<CloverKvslotPartialBatchResult> results;
    std::string error;
    if (!ctx->client->qk_softmax_av_partial_batch(batch_items, &results, &error)) {
        return 5;
    }
    if (results.size() != batch_items.size()) {
        return 6;
    }

    const size_t dpu_request_stride = (size_t)num_requests;
    const size_t dpu_output_stride = (size_t)num_requests * (size_t)d_head;
    std::fill(local_max_out, local_max_out + ((size_t)num_dpus * dpu_request_stride), -INFINITY);
    std::fill(local_sum_out, local_sum_out + ((size_t)num_dpus * dpu_request_stride), 0.0f);
    std::fill(local_output_out, local_output_out + ((size_t)num_dpus * dpu_output_stride), 0.0f);

    for (uint32_t req_idx = 0; req_idx < num_requests; ++req_idx) {
        const CloverKvslotReducerRequest &request = ctx->requests[req_idx];
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            const size_t result_idx = (size_t)req_idx * (size_t)num_dpus + (size_t)dpu_idx;
            const CloverKvslotPartialBatchResult &result = results[result_idx];
            const uint32_t physical_dpu = result.slot_id % num_dpus;
            if (physical_dpu != dpu_idx) {
                return 7;
            }
            if (result.group_heads == 0u || result.head_dim != d_head) {
                return 8;
            }
            if (result.row_max.size() != (size_t)result.group_heads
                || result.row_sum.size() != (size_t)result.group_heads
                || result.context.size() != (size_t)result.group_heads * (size_t)result.head_dim) {
                return 9;
            }
            if (result.group_heads != 1u) {
                return 10;
            }
            /*
             * The kvslot partial helper returns:
             * - row_max: the unscaled maximum raw QK score
             * - row_sum: sum(exp(score * scale - scaled_row_max))
             * - context: the local unnormalized numerator
             *
             * The reducer consumes per-DPU partials in "numerator + scaled max"
             * form:
             * - local_max := scaled row max
             * - local_sum := unchanged local denominator
             * - local_output := unchanged local numerator
             */
            const float scaled_row_max = result.row_max[0] * request.score_scale;
            const float local_row_sum = result.row_sum[0];
            local_max_out[(size_t)physical_dpu * dpu_request_stride + req_idx] = scaled_row_max;
            local_sum_out[(size_t)physical_dpu * dpu_request_stride + req_idx] = local_row_sum;
            for (uint32_t dim = 0; dim < d_head; ++dim) {
                local_output_out[(size_t)physical_dpu * dpu_output_stride + (size_t)req_idx * (size_t)d_head + dim] =
                    result.context[dim];
            }
        }
    }
    return 0;
}
