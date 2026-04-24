from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PrefillResult:
    request_id: str
    prompt_len: int
    first_token_id: int
    initial_kv: List[Dict[str, torch.Tensor]]


@dataclass
class AttentionRequest:
    request_id: str
    layer_idx: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class GenerationMetrics:
    ttft: float
    tpot: float
    latency: float
    throughput: float
    total_tokens: int
    stage_timing: Dict[str, object]
