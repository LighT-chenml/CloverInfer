#include <torch/extension.h>

#include "attention_reducer.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PartialReduceGroup {
    int64_t logical_idx = 0;
    uint32_t num_dpus = 0;
    uint32_t num_requests = 0;
    uint32_t d_head = 0;
    std::vector<torch::Tensor> local_max_parts;
    std::vector<torch::Tensor> local_sum_parts;
    std::vector<torch::Tensor> local_output_parts;
};

torch::Tensor ensure_cpu_fp32_1d(const torch::Tensor &tensor, const char *name)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    torch::Tensor out = tensor;
    if (out.device().type() != torch::kCPU) {
        out = out.cpu();
    }
    if (out.scalar_type() != torch::kFloat32) {
        out = out.to(torch::kFloat32);
    }
    if (out.dim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be 1D");
    }
    return out.contiguous();
}

torch::Tensor ensure_cpu_fp32_2d(const torch::Tensor &tensor, const char *name)
{
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    torch::Tensor out = tensor;
    if (out.device().type() != torch::kCPU) {
        out = out.cpu();
    }
    if (out.scalar_type() != torch::kFloat32) {
        out = out.to(torch::kFloat32);
    }
    if (out.dim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be 2D");
    }
    return out.contiguous();
}

PartialReduceGroup build_partial_reduce_group(
    int64_t logical_idx,
    const std::vector<size_t> &part_indices,
    const std::vector<torch::Tensor> &local_max_parts,
    const std::vector<torch::Tensor> &local_sum_parts,
    const std::vector<torch::Tensor> &local_output_parts)
{
    PartialReduceGroup group;
    group.logical_idx = logical_idx;
    group.num_dpus = (uint32_t)part_indices.size();

    int64_t num_requests = -1;
    int64_t d_head = -1;
    for (size_t part_idx : part_indices) {
        torch::Tensor local_max = ensure_cpu_fp32_1d(local_max_parts[part_idx], "local_max");
        torch::Tensor local_sum = ensure_cpu_fp32_1d(local_sum_parts[part_idx], "local_sum");
        torch::Tensor local_output = ensure_cpu_fp32_2d(local_output_parts[part_idx], "local_output");
        if (local_max.size(0) != local_sum.size(0)) {
            throw std::invalid_argument("local_max/local_sum shape mismatch");
        }
        if (local_output.size(0) != local_max.size(0)) {
            throw std::invalid_argument("local_output/local_max shape mismatch");
        }
        if (num_requests < 0) {
            num_requests = local_max.size(0);
            d_head = local_output.size(1);
        } else if (num_requests != local_max.size(0) || d_head != local_output.size(1)) {
            throw std::invalid_argument("incompatible partial shapes for same logical_idx");
        }
        group.local_max_parts.push_back(local_max);
        group.local_sum_parts.push_back(local_sum);
        group.local_output_parts.push_back(local_output);
    }

    if (num_requests <= 0 || d_head <= 0) {
        throw std::invalid_argument("partial reduce group must contain at least one non-empty part");
    }
    group.num_requests = (uint32_t)num_requests;
    group.d_head = (uint32_t)d_head;
    return group;
}

int fetch_group_partials(
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
    const PartialReduceGroup *group = reinterpret_cast<const PartialReduceGroup *>(user_ctx);
    if (group == nullptr) {
        return 11;
    }
    if (group->num_dpus != num_dpus || group->num_requests != num_requests || group->d_head != d_head) {
        return 12;
    }

    const size_t max_count = (size_t)num_requests;
    const size_t output_count = (size_t)num_requests * (size_t)d_head;
    for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
        const float *local_max_ptr = group->local_max_parts[dpu_idx].data_ptr<float>();
        const float *local_sum_ptr = group->local_sum_parts[dpu_idx].data_ptr<float>();
        const float *local_output_ptr = group->local_output_parts[dpu_idx].data_ptr<float>();
        std::memcpy(local_max_out + ((size_t)dpu_idx * max_count), local_max_ptr, max_count * sizeof(float));
        std::memcpy(local_sum_out + ((size_t)dpu_idx * max_count), local_sum_ptr, max_count * sizeof(float));
        std::memcpy(
            local_output_out + ((size_t)dpu_idx * output_count),
            local_output_ptr,
            output_count * sizeof(float));
    }
    return 0;
}

torch::Tensor reduce_group_sync(const PartialReduceGroup &group)
{
    std::vector<float> local_max_buffer((size_t)group.num_dpus * (size_t)group.num_requests, 0.0f);
    std::vector<float> local_sum_buffer((size_t)group.num_dpus * (size_t)group.num_requests, 0.0f);
    std::vector<float> local_output_buffer(
        (size_t)group.num_dpus * (size_t)group.num_requests * (size_t)group.d_head,
        0.0f);
    const int fetch_status = fetch_group_partials(
        const_cast<PartialReduceGroup *>(&group),
        0u,
        group.num_dpus,
        group.num_requests,
        group.d_head,
        local_max_buffer.data(),
        local_sum_buffer.data(),
        local_output_buffer.data());
    if (fetch_status != 0) {
        throw std::runtime_error(
            "fetch_group_partials failed with status=" + std::to_string(fetch_status));
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor global_output = torch::empty(
        {(int64_t)group.num_requests, (int64_t)group.d_head},
        options);
    torch::Tensor global_max = torch::empty({(int64_t)group.num_requests}, options);
    torch::Tensor global_sum = torch::empty({(int64_t)group.num_requests}, options);

    clover_attention_reduce_input_t input;
    input.num_requests = group.num_requests;
    input.d_head = group.d_head;
    input.local_max = local_max_buffer.data();
    input.local_sum = local_sum_buffer.data();
    input.local_output = local_output_buffer.data();

    clover_attention_reduce_output_t output;
    output.num_requests = group.num_requests;
    output.d_head = group.d_head;
    output.global_output = global_output.data_ptr<float>();
    output.global_max = global_max.data_ptr<float>();
    output.global_sum = global_sum.data_ptr<float>();

    const int status = clover_attention_reduce_group(group.num_dpus, &input, &output);
    if (status != 0) {
        throw std::runtime_error(
            "clover_attention_reduce_group failed for logical_idx="
            + std::to_string(group.logical_idx));
    }
    return global_output;
}

}  // namespace

std::unordered_map<int64_t, torch::Tensor> merge_partial_contexts(
    const std::vector<int64_t> &logical_indices,
    const std::vector<torch::Tensor> &local_max_parts,
    const std::vector<torch::Tensor> &local_sum_parts,
    const std::vector<torch::Tensor> &local_output_parts)
{
    const size_t count = logical_indices.size();
    if (local_max_parts.size() != count || local_sum_parts.size() != count || local_output_parts.size() != count) {
        throw std::invalid_argument("merge_partial_contexts input size mismatch");
    }

    std::unordered_map<int64_t, std::vector<size_t>> grouped_indices;
    std::vector<int64_t> logical_order;
    for (size_t idx = 0; idx < count; ++idx) {
        auto insert = grouped_indices.emplace(logical_indices[idx], std::vector<size_t>());
        if (insert.second) {
            logical_order.push_back(logical_indices[idx]);
        }
        insert.first->second.push_back(idx);
    }

    std::unordered_map<int64_t, torch::Tensor> results;
    results.reserve(grouped_indices.size());
    std::vector<PartialReduceGroup> groups;
    groups.reserve(logical_order.size());
    for (int64_t logical_idx : logical_order) {
        const std::vector<size_t> &part_indices = grouped_indices.at(logical_idx);
        if (part_indices.empty()) {
            continue;
        }
        groups.push_back(
            build_partial_reduce_group(
                logical_idx,
                part_indices,
                local_max_parts,
                local_sum_parts,
                local_output_parts));
    }
    if (groups.empty()) {
        return results;
    }

    clover_attention_reduce_engine_t engine;
    clover_attention_reduce_engine_init(&engine);
    if (engine.impl_handle == 0u) {
        for (const PartialReduceGroup &group : groups) {
            results.emplace(group.logical_idx, reduce_group_sync(group));
        }
        return results;
    }

    const PartialReduceGroup *slot_groups[2] = {nullptr, nullptr};
    size_t next_group_idx = 0;
    size_t completed_groups = 0;
    try {
        while (completed_groups < groups.size()) {
            while (next_group_idx < groups.size()) {
                uint32_t slot_id = 0u;
                const PartialReduceGroup *group = &groups[next_group_idx];
                const int submit_status = clover_attention_reduce_engine_submit(
                    &engine,
                    (uint32_t)next_group_idx,
                    group->num_dpus,
                    group->num_requests,
                    group->d_head,
                    fetch_group_partials,
                    const_cast<PartialReduceGroup *>(group),
                    &slot_id);
                if (submit_status == 2) {
                    break;
                }
                if (submit_status != 0) {
                    throw std::runtime_error(
                        "clover_attention_reduce_engine_submit failed with status="
                        + std::to_string(submit_status));
                }
                slot_groups[slot_id] = group;
                next_group_idx += 1;
            }

            clover_attention_reduce_completion_t completion;
            const int wait_status = clover_attention_reduce_engine_wait_ready(
                &engine,
                60000u,
                &completion);
            if (wait_status != 0) {
                throw std::runtime_error(
                    "clover_attention_reduce_engine_wait_ready failed with status="
                    + std::to_string(wait_status));
            }
            if (completion.slot_id >= 2u || slot_groups[completion.slot_id] == nullptr) {
                throw std::runtime_error("attention reduce completion referenced an unknown slot");
            }
            const PartialReduceGroup *group = slot_groups[completion.slot_id];
            if (completion.status_code != 0 || completion.global_output == nullptr) {
                throw std::runtime_error(
                    "attention reduce worker failed for logical_idx="
                    + std::to_string(group->logical_idx)
                    + " status=" + std::to_string(completion.status_code));
            }

            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            torch::Tensor global_output = torch::empty(
                {(int64_t)group->num_requests, (int64_t)group->d_head},
                options);
            std::memcpy(
                global_output.data_ptr<float>(),
                completion.global_output,
                (size_t)group->num_requests * (size_t)group->d_head * sizeof(float));
            results.emplace(group->logical_idx, global_output);

            const int release_status = clover_attention_reduce_engine_release_slot(
                &engine,
                completion.slot_id);
            if (release_status != 0) {
                throw std::runtime_error(
                    "clover_attention_reduce_engine_release_slot failed with status="
                    + std::to_string(release_status));
            }
            slot_groups[completion.slot_id] = nullptr;
            completed_groups += 1;
        }
    } catch (...) {
        clover_attention_reduce_engine_destroy(&engine);
        throw;
    }
    clover_attention_reduce_engine_destroy(&engine);
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "merge_partial_contexts",
        &merge_partial_contexts,
        "Merge segmented partial softmax contexts on host");
}
