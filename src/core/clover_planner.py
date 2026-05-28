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
class HeadRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def to_dict(self) -> Dict[str, int]:
        return {
            "head_start": int(self.start),
            "head_end": int(self.end),
            "group_heads": int(self.length),
        }


@dataclass(frozen=True)
class RequestShard:
    request_id: str
    head_id: int
    head_range: HeadRange
    dpu_id: int
    token_range: TokenRange

    def to_dict(self) -> Dict[str, int | str]:
        weighted_token_count = int(self.token_range.length) * int(self.head_range.length)
        return {
            "request_id": str(self.request_id),
            "head_id": int(self.head_id),
            "head_start": int(self.head_range.start),
            "head_end": int(self.head_range.end),
            "group_heads": int(self.head_range.length),
            "dpu_id": int(self.dpu_id),
            "token_range_start": int(self.token_range.start),
            "token_range_end": int(self.token_range.end),
            "token_count": int(self.token_range.length),
            "weighted_token_count": int(weighted_token_count),
        }


@dataclass
class ShardingPlan:
    num_dpus: int
    num_heads: int
    head_group_ranges: Dict[int, HeadRange]
    dpu_groups: Dict[int, List[int]]
    shards_by_head: Dict[int, Dict[str, List[RequestShard]]]
    dpu_loads: Dict[int, int]
    load_stddev: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_dpus": int(self.num_dpus),
            "num_heads": int(self.num_heads),
            "head_group_ranges": {
                int(head_id): head_range.to_dict()
                for head_id, head_range in self.head_group_ranges.items()
            },
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


@dataclass
class DynamicTokenMetadata:
    """Host-side token-range metadata for PIM-native sharding.

    The table is deliberately small and explicit: each entry says which
    contiguous token range of one request/head group lives on which DPU.  It is
    used by the scheduler and attention backend as the synchronization point
    instead of requiring DPU-to-DPU coordination.
    """

    num_dpus: int
    num_heads: int
    dpu_groups: Dict[int, List[int]]
    head_group_ranges: Dict[int, HeadRange]
    range_table: Dict[int, Dict[str, List[RequestShard]]] = field(default_factory=dict)
    dpu_token_counts: Dict[int, int] = field(default_factory=dict)
    rebalance_threshold: float = 0.3
    migrations: List[Dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_plan(
        cls,
        plan: Mapping[str, object],
        *,
        rebalance_threshold: float = 0.3,
    ) -> "DynamicTokenMetadata":
        num_dpus = int(plan.get("num_dpus", 0) or 0)
        num_heads = int(plan.get("num_heads", 0) or 0)
        head_group_ranges: Dict[int, HeadRange] = {}
        for head_id, head_range in dict(plan.get("head_group_ranges", {}) or {}).items():
            item = dict(head_range or {})
            head_group_ranges[int(head_id)] = HeadRange(
                start=int(item.get("head_start", 0)),
                end=int(item.get("head_end", 0)),
            )
        dpu_groups = {
            int(head_id): [int(dpu_id) for dpu_id in list(group_dpus or [])]
            for head_id, group_dpus in dict(plan.get("dpu_groups", {}) or {}).items()
        }
        range_table: Dict[int, Dict[str, List[RequestShard]]] = {}
        for head_id, request_map in dict(plan.get("per_head_shards", {}) or {}).items():
            head_id_int = int(head_id)
            head_range = head_group_ranges.get(head_id_int)
            if head_range is None:
                continue
            range_table[head_id_int] = {}
            for request_id, shards in dict(request_map or {}).items():
                range_table[head_id_int][str(request_id)] = [
                    RequestShard(
                        request_id=str(shard.get("request_id", request_id)),
                        head_id=head_id_int,
                        head_range=head_range,
                        dpu_id=int(shard.get("dpu_id", 0)),
                        token_range=TokenRange(
                            int(shard.get("token_range_start", 0)),
                            int(shard.get("token_range_end", 0)),
                        ),
                    )
                    for shard in list(shards or [])
                    if int(shard.get("token_range_end", 0)) > int(shard.get("token_range_start", 0))
                ]
        counts = {int(dpu_id): 0 for dpu_id in range(max(0, num_dpus))}
        metadata = cls(
            num_dpus=num_dpus,
            num_heads=num_heads,
            dpu_groups=dpu_groups,
            head_group_ranges=head_group_ranges,
            range_table=range_table,
            dpu_token_counts=counts,
            rebalance_threshold=float(rebalance_threshold),
        )
        metadata.recompute_counts()
        return metadata

    def recompute_counts(self) -> Dict[int, int]:
        counts = {int(dpu_id): 0 for dpu_id in range(max(0, self.num_dpus))}
        for request_map in self.range_table.values():
            for shards in request_map.values():
                for shard in shards:
                    counts[int(shard.dpu_id)] = counts.get(int(shard.dpu_id), 0) + int(shard.token_range.length)
        self.dpu_token_counts = counts
        return counts

    def group_counts(self, head_id: int) -> Dict[int, int]:
        group_dpus = [int(dpu_id) for dpu_id in self.dpu_groups.get(int(head_id), [])]
        counts = {int(dpu_id): 0 for dpu_id in group_dpus}
        for shards in self.range_table.get(int(head_id), {}).values():
            for shard in shards:
                if int(shard.dpu_id) in counts:
                    counts[int(shard.dpu_id)] += int(shard.token_range.length)
        return counts

    def append_decode_token(self, request_id: str, head_id: int, *, token_idx: int | None = None) -> RequestShard:
        head_id = int(head_id)
        request_id = str(request_id)
        request_shards = self.range_table.setdefault(head_id, {}).setdefault(request_id, [])
        head_range = self.head_group_ranges.get(head_id, HeadRange(head_id, head_id + 1))
        group_dpus = [int(dpu_id) for dpu_id in self.dpu_groups.get(head_id, [])]
        if not group_dpus:
            raise ValueError(f"head group {head_id} has no DPU group")
        current_end = max((int(shard.token_range.end) for shard in request_shards), default=0)
        start = int(current_end if token_idx is None else token_idx)
        preferred_dpu = int(request_shards[-1].dpu_id) if request_shards else min(
            group_dpus,
            key=lambda dpu_id: (int(self.dpu_token_counts.get(int(dpu_id), 0)), int(dpu_id)),
        )
        if preferred_dpu not in group_dpus:
            preferred_dpu = group_dpus[0]
        least_loaded = min(
            group_dpus,
            key=lambda dpu_id: (int(self.dpu_token_counts.get(int(dpu_id), 0)), int(dpu_id)),
        )
        preferred_load = int(self.dpu_token_counts.get(preferred_dpu, 0))
        least_load = int(self.dpu_token_counts.get(least_loaded, 0))
        target_dpu = least_loaded if least_load + 1 < preferred_load else preferred_dpu
        shard = RequestShard(
            request_id=request_id,
            head_id=head_id,
            head_range=head_range,
            dpu_id=int(target_dpu),
            token_range=TokenRange(start, start + 1),
        )
        if request_shards and int(request_shards[-1].dpu_id) == int(target_dpu) and int(request_shards[-1].token_range.end) == start:
            previous = request_shards[-1]
            request_shards[-1] = RequestShard(
                request_id=previous.request_id,
                head_id=previous.head_id,
                head_range=previous.head_range,
                dpu_id=previous.dpu_id,
                token_range=TokenRange(previous.token_range.start, start + 1),
            )
            shard = request_shards[-1]
        else:
            request_shards.append(shard)
        self.dpu_token_counts[int(target_dpu)] = int(self.dpu_token_counts.get(int(target_dpu), 0)) + 1
        return shard

    def release_request(self, request_id: str) -> None:
        request_id = str(request_id)
        for request_map in self.range_table.values():
            request_map.pop(request_id, None)
        self.recompute_counts()

    def imbalance_cv(self, head_id: int) -> float:
        counts = list(self.group_counts(int(head_id)).values())
        if not counts:
            return 0.0
        mean = sum(counts) / float(len(counts))
        if mean <= 0.0:
            return 0.0
        variance = sum((float(value) - mean) ** 2.0 for value in counts) / float(len(counts))
        return math.sqrt(variance) / mean

    def rebalance_once(self, head_id: int, *, threshold: float | None = None) -> Dict[str, object]:
        head_id = int(head_id)
        counts = self.group_counts(head_id)
        if not counts:
            return {"migrated": False, "reason": "empty_group", "head_id": head_id}
        cv = self.imbalance_cv(head_id)
        threshold_value = self.rebalance_threshold if threshold is None else float(threshold)
        if cv <= threshold_value:
            return {
                "migrated": False,
                "reason": "below_threshold",
                "head_id": head_id,
                "cv": float(cv),
                "threshold": float(threshold_value),
            }
        src_dpu = max(counts, key=lambda dpu_id: (int(counts[dpu_id]), -int(dpu_id)))
        dst_dpu = min(counts, key=lambda dpu_id: (int(counts[dpu_id]), int(dpu_id)))
        move_tokens = max(1, (int(counts[src_dpu]) - int(counts[dst_dpu])) // 2)
        request_map = self.range_table.get(head_id, {})
        candidate: tuple[str, int, RequestShard] | None = None
        for request_id, shards in request_map.items():
            for shard_idx, shard in enumerate(shards):
                if int(shard.dpu_id) == int(src_dpu) and int(shard.token_range.length) > 1:
                    candidate = (str(request_id), int(shard_idx), shard)
                    break
            if candidate is not None:
                break
        if candidate is None:
            return {"migrated": False, "reason": "no_splittable_range", "head_id": head_id, "cv": float(cv)}
        request_id, shard_idx, shard = candidate
        take = min(move_tokens, int(shard.token_range.length) - 1)
        moved_start = int(shard.token_range.end) - int(take)
        kept = RequestShard(
            request_id=shard.request_id,
            head_id=shard.head_id,
            head_range=shard.head_range,
            dpu_id=shard.dpu_id,
            token_range=TokenRange(shard.token_range.start, moved_start),
        )
        moved = RequestShard(
            request_id=shard.request_id,
            head_id=shard.head_id,
            head_range=shard.head_range,
            dpu_id=int(dst_dpu),
            token_range=TokenRange(moved_start, shard.token_range.end),
        )
        shards = request_map[request_id]
        shards[shard_idx:shard_idx + 1] = [kept, moved]
        self.recompute_counts()
        migration = {
            "migrated": True,
            "head_id": head_id,
            "request_id": request_id,
            "from_dpu": int(src_dpu),
            "to_dpu": int(dst_dpu),
            "token_range_start": int(moved.token_range.start),
            "token_range_end": int(moved.token_range.end),
            "cv_before": float(cv),
            "cv_after": float(self.imbalance_cv(head_id)),
            "threshold": float(threshold_value),
        }
        self.migrations.append(migration)
        return migration

    def to_dict(self) -> Dict[str, object]:
        self.recompute_counts()
        return {
            "num_dpus": int(self.num_dpus),
            "num_heads": int(self.num_heads),
            "dpu_groups": {int(k): [int(v) for v in values] for k, values in self.dpu_groups.items()},
            "head_group_ranges": {int(k): value.to_dict() for k, value in self.head_group_ranges.items()},
            "range_table": {
                int(head_id): {
                    str(request_id): [shard.to_dict() for shard in shards]
                    for request_id, shards in request_map.items()
                }
                for head_id, request_map in self.range_table.items()
            },
            "dpu_token_counts": {int(k): int(v) for k, v in self.dpu_token_counts.items()},
            "rebalance_threshold": float(self.rebalance_threshold),
            "migrations": [dict(item) for item in self.migrations],
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


def _partition_contiguous(total_items: int, num_parts: int) -> List[Tuple[int, int]]:
    if total_items < 0:
        raise ValueError(f"total_items must be non-negative, got {total_items}")
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, got {num_parts}")
    if total_items < num_parts:
        raise ValueError(
            f"cannot partition {total_items} items into {num_parts} non-empty parts"
        )

    base = total_items // num_parts
    extra = total_items % num_parts
    cursor = 0
    spans: List[Tuple[int, int]] = []
    for part_idx in range(num_parts):
        width = base + (1 if part_idx < extra else 0)
        spans.append((cursor, cursor + width))
        cursor += width
    return spans


def _build_head_groups(num_dpus: int, num_heads: int) -> Tuple[Dict[int, HeadRange], Dict[int, List[int]]]:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_dpus <= 0:
        raise ValueError("num_dpus must be positive")
    effective_group_count = min(num_dpus, num_heads)
    head_ranges = _partition_contiguous(num_heads, effective_group_count)
    dpu_ranges = _partition_contiguous(num_dpus, effective_group_count)
    return (
        {
            head_id: HeadRange(start=int(start), end=int(end))
            for head_id, (start, end) in enumerate(head_ranges)
        },
        {
            head_id: list(range(int(start), int(end)))
            for head_id, (start, end) in enumerate(dpu_ranges)
        },
    )


def _slice_tokens_balanced(
    seq_len: int,
    dpu_ids: Sequence[int],
    dpu_loads: MutableMapping[int, int],
    load_scale: int = 1,
    dpu_free_capacity: Mapping[int, int] | None = None,
) -> List[Tuple[int, TokenRange]]:
    if seq_len <= 0:
        return []
    if not dpu_ids:
        raise ValueError("dpu_ids must not be empty")

    # Keep token order contiguous, but bias larger chunks toward lighter-loaded
    # or higher-free-capacity DPUs.  When live allocator stats are supplied, this
    # implements the prefill tier of the hybrid sharding design.
    if dpu_free_capacity:
        sorted_dpus = sorted(
            dpu_ids,
            key=lambda dpu_id: (
                -int(dpu_free_capacity.get(int(dpu_id), 0)),
                int(dpu_loads.get(dpu_id, 0)),
                int(dpu_id),
            ),
        )
        free_weights = {
            int(dpu_id): max(0, int(dpu_free_capacity.get(int(dpu_id), 0)))
            for dpu_id in sorted_dpus
        }
        if sum(free_weights.values()) <= 0:
            free_weights = {}
    else:
        sorted_dpus = sorted(dpu_ids, key=lambda dpu_id: (int(dpu_loads.get(dpu_id, 0)), int(dpu_id)))
        free_weights = {}
    remaining_tokens = int(seq_len)
    remaining_dpus = len(sorted_dpus)
    remaining_weight = sum(free_weights.values())
    cursor = 0
    out: List[Tuple[int, TokenRange]] = []
    for dpu_id in sorted_dpus:
        if remaining_tokens <= 0:
            out.append((int(dpu_id), TokenRange(cursor, cursor)))
            remaining_dpus -= 1
            continue
        if free_weights and remaining_weight > 0:
            weight = max(0, int(free_weights.get(int(dpu_id), 0)))
            chunk = int(math.ceil(float(remaining_tokens) * float(weight) / float(remaining_weight)))
            chunk = max(1, min(remaining_tokens, chunk))
            remaining_weight -= weight
        else:
            chunk = int(math.ceil(float(remaining_tokens) / float(max(1, remaining_dpus))))
        start = cursor
        end = min(seq_len, cursor + chunk)
        out.append((int(dpu_id), TokenRange(start, end)))
        dpu_loads[int(dpu_id)] = int(dpu_loads.get(int(dpu_id), 0)) + ((end - start) * max(1, int(load_scale)))
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
    dpu_free_capacity: Mapping[int, int] | None = None,
) -> ShardingPlan:
    normalized = [_normalize_request(request) for request in requests]
    head_group_ranges, dpu_groups = _build_head_groups(num_dpus, num_heads)
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
            head_range = head_group_ranges[int(head_id)]
            slices = _slice_tokens_balanced(
                seq_len,
                group_dpus,
                dpu_loads,
                load_scale=head_range.length,
                dpu_free_capacity=dpu_free_capacity,
            )
            shards_by_head[head_id][request_id] = [
                RequestShard(
                    request_id=request_id,
                    head_id=int(head_id),
                    head_range=head_range,
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
        head_group_ranges=head_group_ranges,
        dpu_groups=dpu_groups,
        shards_by_head=shards_by_head,
        dpu_loads=dpu_loads,
        load_stddev=float(load_stddev),
        metadata={
            "request_count": len(normalized),
            "planner": "balanced_contiguous_token_sharding",
            "capacity_aware_prefill": bool(dpu_free_capacity),
            "effective_group_count": int(len(dpu_groups)),
            "planner_mode": (
                "multi_dpu_per_head_group"
                if num_dpus > num_heads
                else "single_dpu_multi_head_group"
                if num_dpus < num_heads
                else "one_dpu_per_head_group"
            ),
        },
    )


def plan_sharding(
    requests: Sequence[RequestLike],
    D: int,
    H: int,
    dpu_free_capacity: Mapping[int, int] | None = None,
) -> Dict[str, object]:
    """
    Build a head-group + intra-head token sharding plan.

    The current planner intentionally chooses contiguous token ranges per DPU,
    because that matches the existing resident-KV blocked layout much better
    than arbitrary token scattering and is easier to materialize on UPMEM.
    """

    return _plan_for_requests(requests, int(D), int(H), dpu_free_capacity=dpu_free_capacity).to_dict()


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
