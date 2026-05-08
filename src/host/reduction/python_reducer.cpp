#include <torch/extension.h>

#include "attention_reducer.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

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
    for (int64_t logical_idx : logical_order) {
        const std::vector<size_t> &part_indices = grouped_indices.at(logical_idx);
        if (part_indices.empty()) {
            continue;
        }

        std::vector<torch::Tensor> local_max_tensors;
        std::vector<torch::Tensor> local_sum_tensors;
        std::vector<torch::Tensor> local_output_tensors;
        local_max_tensors.reserve(part_indices.size());
        local_sum_tensors.reserve(part_indices.size());
        local_output_tensors.reserve(part_indices.size());

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
            local_max_tensors.push_back(local_max);
            local_sum_tensors.push_back(local_sum);
            local_output_tensors.push_back(local_output);
        }

        std::vector<float> local_max_buffer((size_t)part_indices.size() * (size_t)num_requests, 0.0f);
        std::vector<float> local_sum_buffer((size_t)part_indices.size() * (size_t)num_requests, 0.0f);
        std::vector<float> local_output_buffer(
            (size_t)part_indices.size() * (size_t)num_requests * (size_t)d_head,
            0.0f);
        for (size_t seg_idx = 0; seg_idx < part_indices.size(); ++seg_idx) {
            const float *local_max_ptr = local_max_tensors[seg_idx].data_ptr<float>();
            const float *local_sum_ptr = local_sum_tensors[seg_idx].data_ptr<float>();
            const float *local_output_ptr = local_output_tensors[seg_idx].data_ptr<float>();
            for (int64_t req_idx = 0; req_idx < num_requests; ++req_idx) {
                local_max_buffer[seg_idx * (size_t)num_requests + (size_t)req_idx] = local_max_ptr[req_idx];
                local_sum_buffer[seg_idx * (size_t)num_requests + (size_t)req_idx] = local_sum_ptr[req_idx];
                for (int64_t dim = 0; dim < d_head; ++dim) {
                    local_output_buffer[
                        seg_idx * (size_t)num_requests * (size_t)d_head
                        + (size_t)req_idx * (size_t)d_head
                        + (size_t)dim] = local_output_ptr[req_idx * d_head + dim];
                }
            }
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor global_output = torch::empty({num_requests, d_head}, options);
        torch::Tensor global_max = torch::empty({num_requests}, options);
        torch::Tensor global_sum = torch::empty({num_requests}, options);

        clover_attention_reduce_input_t input;
        input.num_requests = (uint32_t)num_requests;
        input.d_head = (uint32_t)d_head;
        input.local_max = local_max_buffer.data();
        input.local_sum = local_sum_buffer.data();
        input.local_output = local_output_buffer.data();

        clover_attention_reduce_output_t output;
        output.num_requests = (uint32_t)num_requests;
        output.d_head = (uint32_t)d_head;
        output.global_output = global_output.data_ptr<float>();
        output.global_max = global_max.data_ptr<float>();
        output.global_sum = global_sum.data_ptr<float>();

        const int status = clover_attention_reduce_group((uint32_t)part_indices.size(), &input, &output);
        if (status != 0) {
            throw std::runtime_error(
                "clover_attention_reduce_group failed for logical_idx=" + std::to_string(logical_idx));
        }
        results.emplace(logical_idx, global_output);
    }
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "merge_partial_contexts",
        &merge_partial_contexts,
        "Merge segmented partial softmax contexts on host");
}
