#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__device__ float to_float(T val) {
    return static_cast<float>(val);
}

// Specialization for Half if needed, but implicit cast usually works in CUDA logic with accessors
// Actually standard casting works for scalar_t in torch.

template<typename scalar_t>
__global__ void paged_attention_kernel(
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, block_size, num_heads, head_dim]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    int block_size,
    int max_context_len,
    int head_dim,
    int max_blocks_per_seq
) {
    // Grid: (num_seqs, num_heads)
    // Block: (1) - For extreme simplicity validation. 
    
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    // Q vector pointer
    const scalar_t* q_vec = q + (seq_idx * gridDim.y * head_dim) + (head_idx * head_dim);
    
    int context_len = context_lens[seq_idx];
    
    float max_score = -1e20f;
    
    // Pass 1: Find Max
    for (int t = 0; t < context_len; ++t) {
        int block_idx = t / block_size;
        int block_offset = t % block_size;
        int physical_block_number = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        long stride_b = block_size * gridDim.y * head_dim;
        long stride_o = gridDim.y * head_dim;
        long stride_h = head_dim;
        
        const scalar_t* k_vec = k_cache + 
            physical_block_number * stride_b +
            block_offset * stride_o +
            head_idx * stride_h;
            
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
        }
        score *= (1.0f / sqrtf((float)head_dim)); 
        
        if (score > max_score) max_score = score;
    }
    
    // Pass 2: Softmax & Aggregation
    float sum_exp = 0.0f;
    
    // Reset output reg 
    // We cannot easily hold array in registers with dynamic loop, will write to global temp or just accumulate in float vars if we could?
    // But head_dim is dynamic. 
    // We'll write to global `out` as accumulator (init to 0)
    
    scalar_t* out_vec = out + (seq_idx * gridDim.y * head_dim) + (head_idx * head_dim);
    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = static_cast<scalar_t>(0.0f);
    }
    
    for (int t = 0; t < context_len; ++t) {
        int block_idx = t / block_size;
        int block_offset = t % block_size;
        int physical_block_number = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        long stride_b = block_size * gridDim.y * head_dim;
        long stride_o = gridDim.y * head_dim;
        long stride_h = head_dim;
        
        const scalar_t* k_vec = k_cache + 
            physical_block_number * stride_b + block_offset * stride_o + head_idx * stride_h;
            
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
        }
        score *= (1.0f / sqrtf((float)head_dim));
        
        float weight = expf(score - max_score);
        sum_exp += weight;
        
        const scalar_t* v_vec = v_cache + 
            physical_block_number * stride_b + block_offset * stride_o + head_idx * stride_h;
            
        for (int d = 0; d < head_dim; ++d) {
            float val = static_cast<float>(out_vec[d]);
            val += weight * static_cast<float>(v_vec[d]);
            out_vec[d] = static_cast<scalar_t>(val);
        }
    }
    
    // Finalize
    for (int d = 0; d < head_dim; ++d) {
        float val = static_cast<float>(out_vec[d]);
        val /= sum_exp;
        out_vec[d] = static_cast<scalar_t>(val);
    }
}

// Launcher
void paged_attention_kernel_launch(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_num_blocks_per_seq
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(1); 
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "paged_attention_kernel", ([&] {
        paged_attention_kernel<scalar_t><<<grid, block, 0>>>(
            out.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            block_size,
            max_context_len,
            head_dim,
            max_num_blocks_per_seq
        );
    }));
}
