from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import pstdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


RequestLike = Mapping[str, object]


@dataclass(frozen=True)
class TokenRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))


@dataclass(frozen=True)
class RequestShard:
    request_id: str
    head_id: int
    dpu_id: int
    token_range: TokenRange

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "request_id": str(self.request_id),
            "head_id": int(self.head_id),
            "dpu_id": int(self.dpu_id),
            "token_range_start": int(self.token_range.start),
            "token_range_end": int(self.token_range.end),
            "token_count": int(self.token_range.length),
        }


@dataclass
class ShardingPlan:
    num_dpus: int
    num_heads: int
    dpu_groups: Dict[int, List[int]]
    shards_by_head: Dict[int, Dict[str, List[RequestShard]]]
    dpu_loads: Dict[int, int]
    load_stddev: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_dpus": int(self.num_dpus),
            "num_heads": int(self.num_heads),
            "dpu_groups": {
                int(head_id): [int(dpu_id) for dpu_id in dpu_ids]
                for head_id, dpu_ids in self.dpu_groups.items()
            },
            "per_head_shards": {
                int(head_id): {
                    str(request_id): [shard.to_dict() for shard in shards]
                    for request_id, shards in request_map.items()
                }
                for head_id, request_map in self.shards_by_head.items()
            },
            "dpu_loads": {int(dpu_id): int(load) for dpu_id, load in self.dpu_loads.items()},
            "load_stddev": float(self.load_stddev),
            "metadata": dict(self.metadata),
        }


def _normalize_request(request: RequestLike) -> Dict[str, object]:
    if "request_id" not in request:
        raise KeyError("request is missing request_id")
    if "seq_len" not in request and "kv_size" not in request:
        raise KeyError("request must contain seq_len or kv_size")
    seq_len = int(request.get("seq_len", request.get("kv_size", 0)) or 0)
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    return {
        "request_id": str(request["request_id"]),
        "seq_len": seq_len,
        "kv_size": int(request.get("kv_size", seq_len) or seq_len),
    }


def _build_head_groups(num_dpus: int, num_heads: int) -> Dict[int, List[int]]:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_dpus <= 0:
        raise ValueError("num_dpus must be positive")
    if num_dpus < num_heads:
        raise ValueError(
            f"num_dpus ({num_dpus}) must be >= num_heads ({num_heads}) for head-group sharding"
        )
    if num_dpus % num_heads != 0:
        raise ValueError(
            f"num_dpus ({num_dpus}) must be divisible by num_heads ({num_heads})"
        )
    group_size = num_dpus // num_heads
    return {
        head_id: list(range(head_id * group_size, (head_id + 1) * group_size))
        for head_id in range(num_heads)
    }


def _slice_tokens_balanced(
    seq_len: int,
    dpu_ids: Sequence[int],
    dpu_loads: MutableMapping[int, int],
) -> List[Tuple[int, TokenRange]]:
    if seq_len <= 0:
        return []
    if not dpu_ids:
        raise ValueError("dpu_ids must not be empty")

    # Keep token order contiguous, but bias larger chunks toward lighter-loaded DPUs.
    sorted_dpus = sorted(dpu_ids, key=lambda dpu_id: (int(dpu_loads.get(dpu_id, 0)), int(dpu_id)))
    remaining_tokens = int(seq_len)
    remaining_dpus = len(sorted_dpus)
    cursor = 0
    out: List[Tuple[int, TokenRange]] = []
    for dpu_id in sorted_dpus:
        if remaining_tokens <= 0:
            out.append((int(dpu_id), TokenRange(cursor, cursor)))
            remaining_dpus -= 1
            continue
        chunk = int(math.ceil(float(remaining_tokens) / float(max(1, remaining_dpus))))
        start = cursor
        end = min(seq_len, cursor + chunk)
        out.append((int(dpu_id), TokenRange(start, end)))
        dpu_loads[int(dpu_id)] = int(dpu_loads.get(int(dpu_id), 0)) + (end - start)
        cursor = end
        remaining_tokens -= end - start
        remaining_dpus -= 1
    return out


def _empty_shards_by_head(num_heads: int) -> Dict[int, Dict[str, List[RequestShard]]]:
    return {head_id: {} for head_id in range(num_heads)}


def _plan_for_requests(
    requests: Sequence[RequestLike],
    num_dpus: int,
    num_heads: int,
) -> ShardingPlan:
    normalized = [_normalize_request(request) for request in requests]
    dpu_groups = _build_head_groups(num_dpus, num_heads)
    dpu_loads = {dpu_id: 0 for dpu_id in range(num_dpus)}
    shards_by_head = _empty_shards_by_head(num_heads)

    # Place longer requests first so the balancing heuristic has more room.
    sorted_requests = sorted(
        normalized,
        key=lambda request: (-int(request["seq_len"]), str(request["request_id"])),
    )
    for request in sorted_requests:
        request_id = str(request["request_id"])
        seq_len = int(request["seq_len"])
        for head_id, group_dpus in dpu_groups.items():
            slices = _slice_tokens_balanced(seq_len, group_dpus, dpu_loads)
            shards_by_head[head_id][request_id] = [
                RequestShard(
                    request_id=request_id,
                    head_id=int(head_id),
                    dpu_id=int(dpu_id),
                    token_range=token_range,
                )
                for dpu_id, token_range in slices
                if token_range.length > 0
            ]

    load_stddev = pstdev(list(dpu_loads.values())) if dpu_loads else 0.0
    return ShardingPlan(
        num_dpus=int(num_dpus),
        num_heads=int(num_heads),
        dpu_groups=dpu_groups,
        shards_by_head=shards_by_head,
        dpu_loads=dpu_loads,
        load_stddev=float(load_stddev),
        metadata={
            "request_count": len(normalized),
            "planner": "balanced_contiguous_token_sharding",
        },
    )


def plan_sharding(requests: Sequence[RequestLike], D: int, H: int) -> Dict[str, object]:
    """
    Build a head-group + intra-head token sharding plan.

    The current planner intentionally chooses contiguous token ranges per DPU,
    because that matches the existing resident-KV blocked layout much better
    than arbitrary token scattering and is easier to materialize on UPMEM.
    """

    return _plan_for_requests(requests, int(D), int(H)).to_dict()


def update_sharding(
    existing_plan: Mapping[str, object],
    new_requests: Sequence[RequestLike],
) -> Dict[str, object]:
    """
    Incrementally extend an existing plan.

    Current implementation preserves existing requests as metadata and
    re-plans globally for determinism. This is a deliberate first step:
    the repository already contains several local placement heuristics, but
    not yet a stable planner contract. Global re-planning keeps behavior
    simple and measurable until we add migration-cost-aware updates.
    """

    num_dpus = int(existing_plan["num_dpus"])
    num_heads = int(existing_plan["num_heads"])
    prior_requests: Dict[str, Dict[str, object]] = {}
    per_head = dict(existing_plan.get("per_head_shards", {}) or {})
    for request_map in per_head.values():
        for request_id, shards in dict(request_map or {}).items():
            if request_id in prior_requests:
                continue
            seq_len = 0
            for shard in list(shards or []):
                seq_len = max(seq_len, int(shard.get("token_range_end", 0)))
            prior_requests[str(request_id)] = {
                "request_id": str(request_id),
                "seq_len": int(seq_len),
            }

    all_requests = list(prior_requests.values()) + [_normalize_request(request) for request in new_requests]
    plan = _plan_for_requests(all_requests, num_dpus, num_heads)
    plan.metadata["updated_from_existing"] = True
    plan.metadata["new_request_count"] = len(list(new_requests))
    return plan.to_dict()

