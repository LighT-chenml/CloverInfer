#include <torch/extension.h>

// Forward declaration of CUDA function
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
);

void paged_attention(
    torch::Tensor& out,
    torch::Tensor& q,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len
) {
    // Basic checks
    int num_seqs = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int max_num_blocks_per_seq = block_tables.size(1);
    
    paged_attention_kernel_launch(
        out, q, k_cache, v_cache, block_tables, context_lens,
        block_size, max_context_len, num_seqs, num_heads, head_dim, max_num_blocks_per_seq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", &paged_attention, "Paged Attention (CUDA)");
}
