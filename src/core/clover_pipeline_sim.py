from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Sequence

from .clover_scheduler_components import CapacityAwareMicroBatchScheduler, MicroBatch


@dataclass
class SimulationConfig:
    num_dpus: int
    num_heads: int
    max_capacity_per_dpu: int
    transfer_latency_s: float
    transfer_bandwidth_tokens_per_s: float
    dpu_compute_scale_s: float
    dpu_compute_bias_s: float
    fc_compute_per_token_s: float
    fixed_batch_size: int = 4


@dataclass
class StageWindow:
    name: str
    start: float
    end: float
    batch_id: int


@dataclass
class SimulationResult:
    batches: List[Dict[str, object]]
    stage_windows: List[StageWindow]
    tokens_per_s: float
    utilization: Dict[str, float]


class SimplePipelineSimulator:
    def __init__(self, config: SimulationConfig, scheduler: CapacityAwareMicroBatchScheduler) -> None:
        self.config = config
        self.scheduler = scheduler

    def _transfer_time(self, batch: MicroBatch) -> float:
        total_tokens = max(1, batch.total_tokens)
        return float(self.config.transfer_latency_s + (total_tokens / max(self.config.transfer_bandwidth_tokens_per_s, 1e-9)))

    def _fc_time(self, batch: MicroBatch) -> float:
        return float(self.config.fc_compute_per_token_s * max(1, batch.total_tokens))

    def _dpu_time(self, batch: MicroBatch) -> float:
        dpu_loads = dict(batch.sharding_plan.get("dpu_loads", {}) or {})
        max_load = max((int(load) for load in dpu_loads.values()), default=0)
        modeled = float(self.config.dpu_compute_bias_s + self.config.dpu_compute_scale_s * max_load)
        return max(float(batch.predicted_pim_time), modeled)

    def _schedule_window(
        self,
        stage_name: str,
        duration: float,
        earliest_start: float,
        stage_available_at: Dict[str, float],
        batch_id: int,
        windows: List[StageWindow],
    ) -> tuple[float, float]:
        start = max(float(earliest_start), float(stage_available_at.get(stage_name, 0.0)))
        end = start + max(0.0, float(duration))
        stage_available_at[stage_name] = end
        windows.append(StageWindow(name=stage_name, start=start, end=end, batch_id=batch_id))
        return start, end

    def run(self, requests: Sequence[Dict[str, object]], optimized: bool = True) -> SimulationResult:
        pending = [dict(request) for request in requests]
        stage_available_at = {
            "transfer": 0.0,
            "dpu": 0.0,
            "reduce": 0.0,
            "fc": 0.0,
        }
        windows: List[StageWindow] = []
        batch_logs: List[Dict[str, object]] = []
        total_tokens = 0

        batch_id = 0
        while pending:
            if optimized:
                decision = self.scheduler.build_micro_batch(pending, self.config.max_capacity_per_dpu)
                batch = decision.micro_batch
                log_line = decision.log_line
            else:
                chunk = pending[: self.config.fixed_batch_size]
                del pending[: self.config.fixed_batch_size]
                batch = self.scheduler._evaluate_batch(chunk, self.config.max_capacity_per_dpu, selection_reason="fixed")
                log_line = self.scheduler._format_log_line(batch)

            total_tokens += batch.total_tokens
            transfer_start, transfer_end = self._schedule_window(
                "transfer",
                self._transfer_time(batch),
                0.0,
                stage_available_at,
                batch_id,
                windows,
            )
            dpu_start, dpu_end = self._schedule_window(
                "dpu",
                self._dpu_time(batch),
                transfer_end,
                stage_available_at,
                batch_id,
                windows,
            )
            reduce_start, reduce_end = self._schedule_window(
                "reduce",
                max(0.0, batch.predicted_host_time - self._fc_time(batch)),
                dpu_end,
                stage_available_at,
                batch_id,
                windows,
            )
            fc_start, fc_end = self._schedule_window(
                "fc",
                self._fc_time(batch),
                reduce_end,
                stage_available_at,
                batch_id,
                windows,
            )
            batch_logs.append(
                {
                    "batch_id": int(batch_id),
                    "batch": batch.to_dict(),
                    "log": log_line,
                    "timeline": {
                        "transfer": [transfer_start, transfer_end],
                        "dpu": [dpu_start, dpu_end],
                        "reduce": [reduce_start, reduce_end],
                        "fc": [fc_start, fc_end],
                    },
                }
            )
            batch_id += 1

        makespan = max(stage_available_at.values()) if stage_available_at else 0.0
        utilization = {}
        for stage_name in stage_available_at:
            busy = sum(window.end - window.start for window in windows if window.name == stage_name)
            utilization[stage_name] = 0.0 if makespan <= 0 else busy / makespan
        throughput = 0.0 if makespan <= 0 else float(total_tokens) / makespan
        return SimulationResult(
            batches=batch_logs,
            stage_windows=windows,
            tokens_per_s=throughput,
            utilization=utilization,
        )
