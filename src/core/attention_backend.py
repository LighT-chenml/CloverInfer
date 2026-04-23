from __future__ import annotations

import math
from typing import Dict, List

import torch


class CpuAttentionBackend:
    """Reference attention backend for the CPU/PIM node.

    The dense node sends already-scaled OPT queries, so this backend does not
    apply another 1/sqrt(head_dim) factor.
    """

    def __init__(self):
        self.k_cache: Dict[str, List[torch.Tensor]] = {}
        self.v_cache: Dict[str, List[torch.Tensor]] = {}
        self.context_lens: Dict[str, int] = {}

    def init_request(self, request_id: str, initial_kv: List[Dict[str, torch.Tensor]]) -> int:
        if request_id in self.k_cache:
            raise ValueError(f"Request {request_id} already exists")

        self.k_cache[request_id] = [layer["key"].detach().cpu().contiguous() for layer in initial_kv]
        self.v_cache[request_id] = [layer["value"].detach().cpu().contiguous() for layer in initial_kv]

        if not self.k_cache[request_id]:
            raise ValueError("initial_kv must contain at least one layer")

        seq_len = int(self.k_cache[request_id][0].shape[0])
        self.context_lens[request_id] = seq_len
        return seq_len

    def decode_layer(
        self,
        request_id: str,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if request_id not in self.k_cache:
            raise KeyError(f"Unknown request {request_id}")

        q = query.detach().cpu().contiguous()
        k_new = key.detach().cpu().contiguous()
        v_new = value.detach().cpu().contiguous()

        if q.dim() == 3:
            q = q.squeeze(0)
        if k_new.dim() == 3:
            k_new = k_new.squeeze(0)
        if v_new.dim() == 3:
            v_new = v_new.squeeze(0)

        self.k_cache[request_id][layer_idx] = torch.cat(
            [self.k_cache[request_id][layer_idx], k_new.unsqueeze(0)], dim=0
        )
        self.v_cache[request_id][layer_idx] = torch.cat(
            [self.v_cache[request_id][layer_idx], v_new.unsqueeze(0)], dim=0
        )

        keys = self.k_cache[request_id][layer_idx]
        values = self.v_cache[request_id][layer_idx]

        # q: [heads, dim], keys/values: [seq, heads, dim]
        scores = torch.einsum("hd,lhd->hl", q.float(), keys.float())
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)

        if layer_idx == len(self.k_cache[request_id]) - 1:
            self.context_lens[request_id] += 1

        return context.unsqueeze(0)

    def get_context_len(self, request_id: str) -> int:
        return self.context_lens[request_id]

    def free_request(self, request_id: str) -> None:
        self.k_cache.pop(request_id, None)
        self.v_cache.pop(request_id, None)
        self.context_lens.pop(request_id, None)
