from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable, Dict, Iterable, List, Mapping, MutableSequence, Sequence


RequestLike = Mapping[str, object]
ShardingPlanLike = Mapping[str, object]
PredictPimTime = Callable[[Sequence[RequestLike], ShardingPlanLike], float]
PredictHostTime = Callable[[int], float]
CapacityChecker = Callable[[Sequence[RequestLike], ShardingPlanLike, int], Mapping[str, object]]
Planner = Callable[[Sequence[RequestLike], int, int], ShardingPlanLike]
AllocatorStatsProvider = Callable[[], Sequence[Mapping[str, object]]]


@dataclass
class MicroBatch:
    requests: List[Dict[str, object]]
    predicted_pim_time: float
    predicted_host_time: float
    time_gap: float
    capacity_ok: bool
    capacity_usage_ratio: float
    sharding_plan: Dict[str, object]
    selection_reason: str

    @property
    def total_tokens(self) -> int:
        return sum(int(request.get("num_new_tokens", 1) or 1) for request in self.requests)

    def to_dict(self) -> Dict[str, object]:
        return {
            "requests": [dict(request) for request in self.requests],
            "predicted_pim_time": float(self.predicted_pim_time),
            "predicted_host_time": float(self.predicted_host_time),
            "time_gap": float(self.time_gap),
            "capacity_ok": bool(self.capacity_ok),
            "capacity_usage_ratio": float(self.capacity_usage_ratio),
            "sharding_plan": dict(self.sharding_plan),
            "selection_reason": str(self.selection_reason),
        }


@dataclass
class SchedulerDecision:
    micro_batch: MicroBatch
    log_line: str


def default_predict_pim_time(batch_requests: Sequence[RequestLike], sharding_plan: ShardingPlanLike, a: float = 1.0, b: float = 0.0) -> float:
    dpu_loads = dict(sharding_plan.get("dpu_loads", {}) or {})
    max_token_per_dpu = max((int(load) for load in dpu_loads.values()), default=0)
    return float(a * max_token_per_dpu + b)


def default_predict_host_time(total_tokens: int, c: float = 1.0) -> float:
    return float(c * max(0, int(total_tokens)))


def default_capacity_checker(
    batch_requests: Sequence[RequestLike],
    sharding_plan: ShardingPlanLike,
    max_capacity_per_dpu: int,
) -> Dict[str, object]:
    dpu_loads = dict(sharding_plan.get("dpu_loads", {}) or {})
    peak = max((int(load) for load in dpu_loads.values()), default=0)
    usage_ratio = 0.0 if max_capacity_per_dpu <= 0 else min(1.0, float(peak) / float(max_capacity_per_dpu))
    return {
        "ok": peak <= int(max_capacity_per_dpu),
        "peak_load": int(peak),
        "usage_ratio": float(usage_ratio),
    }


def allocator_aware_capacity_checker(
    allocator_stats_provider: AllocatorStatsProvider,
    *,
    bytes_per_token: int = 1,
    capacity_field: str = "total_free_elems",
    require_slot_headroom: bool = False,
) -> CapacityChecker:
    """
    Build a capacity checker that combines planner-estimated incremental load
    with live allocator stats from the resident KV runtime.

    The provider is expected to return one dict per DPU containing at least:
    - `dpu_id`
    - one free-capacity field, defaulting to `total_free_elems`

    The planner's `dpu_loads` remain an approximation of additional demand for
    the candidate batch, but the admission decision is anchored to real helper
    state instead of a static max-token threshold alone.  Slot-table headroom is
    reported separately and can be made strict for allocator tests, but the
    online scheduler keeps it soft because decode appends often fit in existing
    resident blocks even when a physical DPU's 64-slot table is full.
    """

    normalized_bytes_per_token = max(1, int(bytes_per_token))

    def check_capacity(
        batch_requests: Sequence[RequestLike],
        sharding_plan: ShardingPlanLike,
        max_capacity_per_dpu: int,
    ) -> Dict[str, object]:
        del batch_requests
        stats = [dict(item) for item in allocator_stats_provider()]
        dpu_loads = {
            int(dpu_id): int(load)
            for dpu_id, load in dict(sharding_plan.get("dpu_loads", {}) or {}).items()
        }
        if not dpu_loads:
            return {
                "ok": True,
                "peak_load": 0,
                "usage_ratio": 0.0,
                "capacity_source": "allocator_stats",
                "peak_projected_bytes": 0,
                "min_remaining_bytes": int(max_capacity_per_dpu),
            }

        has_allocator_stats = bool(stats)
        free_by_dpu: Dict[int, int] = {}
        free_slots_by_dpu: Dict[int, int] = {}
        for item in stats:
            dpu_id = int(item.get("dpu_id", -1))
            if dpu_id < 0:
                continue
            free_bytes = int(item.get(capacity_field, item.get("largest_free_range", max_capacity_per_dpu)) or 0)
            free_by_dpu[dpu_id] = free_bytes
            if "live_slot_count" in item:
                free_slots_by_dpu[dpu_id] = max(0, 64 - int(item.get("live_slot_count", 0) or 0))

        dpu_groups = {
            int(group_id): [int(dpu_id) for dpu_id in list(group_dpus or [])]
            for group_id, group_dpus in dict(sharding_plan.get("dpu_groups", {}) or {}).items()
        }
        dpu_to_group: Dict[int, int] = {}
        for group_id, group_dpus in dpu_groups.items():
            for dpu_id in group_dpus:
                dpu_to_group[int(dpu_id)] = int(group_id)

        projected_bytes_by_dpu: Dict[int, int] = {}
        remaining_bytes_by_dpu: Dict[int, int] = {}
        remaining_slots_by_group: Dict[int, int] = {}
        ok = True
        for dpu_id, token_load in dpu_loads.items():
            projected = int(token_load) * normalized_bytes_per_token
            free_bytes = int(free_by_dpu.get(int(dpu_id), max_capacity_per_dpu))
            projected_bytes_by_dpu[int(dpu_id)] = projected
            remaining_bytes_by_dpu[int(dpu_id)] = free_bytes - projected
            if projected > free_bytes:
                ok = False
            if free_slots_by_dpu:
                group_id = int(dpu_to_group.get(int(dpu_id), int(dpu_id)))
                group_dpus = dpu_groups.get(group_id, [int(dpu_id)])
                free_slots = sum(int(free_slots_by_dpu.get(int(group_dpu), 64)) for group_dpu in group_dpus)
                remaining_slots_by_group[group_id] = min(
                    int(remaining_slots_by_group.get(group_id, free_slots - 1)),
                    free_slots - 1,
                )
                if require_slot_headroom and free_slots <= 0:
                    ok = False

        peak_load = max(dpu_loads.values(), default=0)
        peak_projected_bytes = max(projected_bytes_by_dpu.values(), default=0)
        if has_allocator_stats:
            limiting_capacity = max(
                1,
                min(int(free_by_dpu.get(dpu_id, max_capacity_per_dpu)) for dpu_id in dpu_loads),
            )
        else:
            limiting_capacity = max(1, int(max_capacity_per_dpu))
        usage_ratio = min(1.0, float(peak_projected_bytes) / float(limiting_capacity))
        return {
            "ok": bool(ok),
            "peak_load": int(peak_load),
            "usage_ratio": float(usage_ratio),
            "capacity_source": "allocator_stats",
            "peak_projected_bytes": int(peak_projected_bytes),
            "min_remaining_bytes": min(remaining_bytes_by_dpu.values(), default=int(max_capacity_per_dpu)),
            "min_remaining_slots": min(remaining_slots_by_group.values(), default=64),
            "slot_headroom_ok": min(remaining_slots_by_group.values(), default=64) >= 0,
            "slot_headroom_strict": bool(require_slot_headroom),
        }

    return check_capacity


class CapacityAwareMicroBatchScheduler:
    def __init__(
        self,
        *,
        planner: Planner,
        predict_pim_time: PredictPimTime,
        predict_host_time: PredictHostTime,
        capacity_checker: CapacityChecker,
        num_dpus: int,
        num_heads: int,
        time_gap_threshold: float = 0.0,
        lookahead_window: int = 1,
    ) -> None:
        self.planner = planner
        self.predict_pim_time = predict_pim_time
        self.predict_host_time = predict_host_time
        self.capacity_checker = capacity_checker
        self.num_dpus = int(num_dpus)
        self.num_heads = int(num_heads)
        self.time_gap_threshold = float(time_gap_threshold)
        self.lookahead_window = max(1, int(lookahead_window))

    def _evaluate_batch(
        self,
        requests: Sequence[RequestLike],
        max_capacity_per_dpu: int,
        selection_reason: str,
    ) -> MicroBatch:
        request_list = [dict(request) for request in requests]
        sharding_plan = dict(self.planner(request_list, self.num_dpus, self.num_heads))
        total_tokens = sum(int(request.get("num_new_tokens", 1) or 1) for request in request_list)
        predicted_pim = float(self.predict_pim_time(request_list, sharding_plan))
        predicted_host = float(self.predict_host_time(total_tokens))
        capacity = dict(self.capacity_checker(request_list, sharding_plan, int(max_capacity_per_dpu)))
        return MicroBatch(
            requests=request_list,
            predicted_pim_time=predicted_pim,
            predicted_host_time=predicted_host,
            time_gap=abs(predicted_pim - predicted_host),
            capacity_ok=bool(capacity.get("ok", False)),
            capacity_usage_ratio=float(capacity.get("usage_ratio", 0.0)),
            sharding_plan=sharding_plan,
            selection_reason=selection_reason,
        )

    def _best_lookahead_prefix(
        self,
        queue: Sequence[RequestLike],
        max_capacity_per_dpu: int,
        max_batch_size: int | None = None,
    ) -> MicroBatch:
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")
        max_window_size = self.lookahead_window
        if max_batch_size is not None:
            max_window_size = max(max_window_size, int(max_batch_size))
        window = [dict(request) for request in queue[:max_window_size]]
        if not window:
            raise ValueError("queue must not be empty")

        max_combo_size = len(window) if max_batch_size is None else min(len(window), int(max_batch_size))
        for size in range(max_combo_size, 0, -1):
            best_batch: MicroBatch | None = None
            for combo in itertools.combinations(window, size):
                evaluated = self._evaluate_batch(combo, max_capacity_per_dpu, selection_reason="lookahead")
                if not evaluated.capacity_ok:
                    continue
                if best_batch is None:
                    best_batch = evaluated
                    continue
                if (evaluated.time_gap, -len(evaluated.requests)) < (
                        best_batch.time_gap,
                        -len(best_batch.requests),
                    ):
                        best_batch = evaluated
            if best_batch is not None:
                return best_batch
        return self._evaluate_batch([window[0]], max_capacity_per_dpu, selection_reason="forced_singleton")

    def build_micro_batch(
        self,
        queue: MutableSequence[RequestLike],
        max_capacity_per_dpu: int,
        max_batch_size: int | None = None,
    ) -> SchedulerDecision:
        if not queue:
            raise ValueError("queue must not be empty")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")

        if self.lookahead_window > 1:
            batch = self._best_lookahead_prefix(queue, max_capacity_per_dpu, max_batch_size=max_batch_size)
            picked_ids = {str(request["request_id"]) for request in batch.requests}
            retained = [request for request in queue if str(request["request_id"]) not in picked_ids]
            queue[:] = retained
            return SchedulerDecision(
                micro_batch=batch,
                log_line=self._format_log_line(batch),
            )

        selected: List[Dict[str, object]] = []
        current_best: MicroBatch | None = None
        previous_gap: float | None = None
        for request in list(queue):
            if max_batch_size is not None and len(selected) >= int(max_batch_size):
                break
            candidate = selected + [dict(request)]
            evaluated = self._evaluate_batch(candidate, max_capacity_per_dpu, selection_reason="greedy")
            if not evaluated.capacity_ok:
                break
            if previous_gap is not None and evaluated.time_gap > previous_gap + self.time_gap_threshold:
                break
            selected = candidate
            current_best = evaluated
            previous_gap = evaluated.time_gap

        if current_best is None:
            current_best = self._evaluate_batch([dict(queue[0])], max_capacity_per_dpu, selection_reason="forced_singleton")

        del queue[: len(current_best.requests)]
        return SchedulerDecision(
            micro_batch=current_best,
            log_line=self._format_log_line(current_best),
        )

    def _format_log_line(self, batch: MicroBatch) -> str:
        return (
            "micro_batch "
            f"size={len(batch.requests)} "
            f"pim={batch.predicted_pim_time:.4f}s "
            f"host={batch.predicted_host_time:.4f}s "
            f"gap={batch.time_gap:.4f}s "
            f"capacity={batch.capacity_usage_ratio:.2%} "
            f"reason={batch.selection_reason}"
        )
