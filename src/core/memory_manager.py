import torch
from typing import Dict, List, Optional
import math

# Try importing the custom kernel
try:
    import clover_kernels
except ImportError:
    clover_kernels = None

    print("Warning: clover_kernels not installed. PagedAttention will fail.")

if clover_kernels:
    print("Success: clover_kernels installed. Using Optimized CUDA Kernels.")
else:
    print("Info: Using Python Reference Implementation for PagedAttention.")

class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype=torch.float32,
        device="cuda"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Physical Cache
        # Shape: [num_blocks, block_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Free stack
        self.free_blocks = list(range(num_blocks))
        
        # Mapping: request_id -> [block_indices]
        self.block_tables: Dict[str, List[int]] = {}
        self.context_lens: Dict[str, int] = {}
        
    def allocate(self, request_id: str, seq_len: int = 0):
        if request_id in self.block_tables:
            raise ValueError(f"Request {request_id} already exists")
            
        required_blocks = math.ceil((seq_len + 1) / self.block_size) 
        # +1 for safety or potential next token? Usually exact size.
        # If seq_len=0, allocate 1 block.
        if required_blocks == 0: required_blocks = 1
        
        if len(self.free_blocks) < required_blocks:
            raise RuntimeError("OOM: Not enough blocks")
            
        blocks = []
        for _ in range(required_blocks):
            blocks.append(self.free_blocks.pop())
            
        self.block_tables[request_id] = blocks
        self.context_lens[request_id] = seq_len

    def load_initial_kv(self, request_id: str, k_data: torch.Tensor, v_data: torch.Tensor):
        """
        Allocates blocks and fills them with initial KV data (from prefill).
        k_data, v_data: [SeqLen, NumHeads, HeadDim]
        """
        seq_len = k_data.shape[0]
        if request_id not in self.block_tables:
            self.allocate(request_id, seq_len)
            
        blocks = self.block_tables[request_id]
        cursor = 0
        for i, block_idx in enumerate(blocks):
            if cursor >= seq_len: break
            chunk_len = min(self.block_size, seq_len - cursor)
            k_chunk = k_data[cursor : cursor + chunk_len]
            v_chunk = v_data[cursor : cursor + chunk_len]
            self.k_cache[block_idx, :chunk_len, :, :] = k_chunk.to(self.device)
            self.v_cache[block_idx, :chunk_len, :, :] = v_chunk.to(self.device)
            cursor += chunk_len
        self.context_lens[request_id] = seq_len
        
    def append_slot(self, request_id: str):
        """
        Increments context length. Allocates new block if needed.
        Returns the (block_number, block_offset) for the new token.
        """
        current_len = self.context_lens[request_id]
        blocks = self.block_tables[request_id]
        
        # Current capacity
        capacity = len(blocks) * self.block_size
        
        if current_len >= capacity:
            # Need new block
            if not self.free_blocks:
                raise RuntimeError("OOM during append")
            new_block = self.free_blocks.pop()
            blocks.append(new_block)
        
        # Update length
        self.context_lens[request_id] += 1
        
        # Return physical location for the NEW token (at index current_len)
        block_idx = current_len // self.block_size
        block_offset = current_len % self.block_size
        physical_block = blocks[block_idx]
        
        return physical_block, block_offset

    def write_new_kv(self, request_id: str, k: torch.Tensor, v: torch.Tensor):
        """
        Writes a single new token's KV to the cache.
        k, v: [NumHeads, HeadDim] or [1, NumHeads, HeadDim]
        """
        if k.dim() == 3: k = k.squeeze(0)
        if v.dim() == 3: v = v.squeeze(0)
        block_idx, offset = self.append_slot(request_id)
        self.k_cache[block_idx, offset, :, :] = k
        self.v_cache[block_idx, offset, :, :] = v

    def free(self, request_id: str):
        if request_id not in self.block_tables:
            return
        
        blocks = self.block_tables[request_id]
        for b in blocks:
            self.free_blocks.append(b)
        
        del self.block_tables[request_id]
        del self.context_lens[request_id]

    def get_kernel_inputs(self, request_ids: List[str]):
        """
        Prepares tensors for the PagedAttention kernel.
        Returns: block_tables_tensor, context_lens_tensor, max_context_len
        """
        max_len = 0
        max_blocks = 0
        
        seq_lens = []
        tables = []
        
        for rid in request_ids:
            sl = self.context_lens[rid]
            blks = self.block_tables[rid]
            seq_lens.append(sl)
            tables.append(blks)
            
            if sl > max_len: max_len = sl
            if len(blks) > max_blocks: max_blocks = len(blks)
            
        # Pad Block Tables
        # block_tables: [num_seqs, max_blocks]
        padded_tables = torch.full((len(request_ids), max_blocks), -1, dtype=torch.int32, device=self.device)
        for i, blks in enumerate(tables):
            padded_tables[i, :len(blks)] = torch.tensor(blks, dtype=torch.int32, device=self.device)
            
        ctx_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        
        return padded_tables, ctx_lens_tensor, max_len

    def attention(self, q: torch.Tensor, request_ids: List[str]):
        """
        Runs PagedAttention.
        q: [num_seqs, num_heads, head_dim]
        """
        if clover_kernels is None or self.device == "cpu":
            # Fallback Reference Implementation
            return self._paged_attention_ref(q, request_ids)
            
        block_tables, ctx_lens, max_len = self.get_kernel_inputs(request_ids)
        out = torch.empty_like(q)
        clover_kernels.paged_attention(
            out, q, self.k_cache, self.v_cache, block_tables, ctx_lens,
            self.block_size, max_len
        )
        return out

    def _paged_attention_ref(self, q: torch.Tensor, request_ids: List[str]):
        """
        Slow Reference Implementation in PyTorch.
        """
        # q: [num_seqs, num_heads, head_dim]
        num_seqs, num_heads, head_dim = q.shape
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(head_dim)
        
        for i, rid in enumerate(request_ids):
            # Gather K, V for this sequence
            # This is slow copy
            blocks = self.block_tables[rid]
            context_len = self.context_lens[rid]
            
            # Construct flattened K, V [context_len, num_heads, head_dim]
            # Naive: Iterate tokens
            # Optimized Reference: Gather blocks
            
            # Let's just gather all blocks fully then slice
            phy_blocks = torch.tensor(blocks, device=self.device) # [num_blocks]
            
            # K_cache: [total_blocks, block_size, heads, dim]
            # Index select
            k_blks = self.k_cache[phy_blocks] # [n_blks, blk_size, heads, dim]
            v_blks = self.v_cache[phy_blocks]
            
            # Reshape to [n_blks * blk_size, heads, dim]
            k_flat = k_blks.view(-1, num_heads, head_dim)[:context_len]
            v_flat = v_blks.view(-1, num_heads, head_dim)[:context_len]
            
            # Attention: Q[i] (1, H, D) x K_flat.T (D, H, L) -> (1, H, L)
            # permute K_flat to [H, L, D]
            
            curr_q = q[i].unsqueeze(1) # [H, 1, D]
            
            curr_k = k_flat.permute(1, 0, 2) # [H, L, D]
            curr_v = v_flat.permute(1, 0, 2) # [H, L, D]
            
            # Scores: [H, 1, D] @ [H, D, L] -> [H, 1, L]
            scores = torch.matmul(curr_q, curr_k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            
            # Out: [H, 1, L] @ [H, L, D] -> [H, 1, D]
            o = torch.matmul(attn, curr_v)
            out[i] = o.squeeze(1)
            
        return out
