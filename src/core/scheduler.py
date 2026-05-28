from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Dict, List

import ray

from .config import ClusterConfig, ModelConfig
from .clover_planner import plan_sharding
from .clover_scheduler_components import (
    CapacityAwareMicroBatchScheduler,
    allocator_aware_capacity_checker,
    default_capacity_checker,
    default_predict_host_time,
    default_predict_pim_time,
)
from .nodes import AttentionNode, DecodeDenseNode, PrefillNode


def _actor_options(resource_name: str | None, num_gpus: float = 0):
    options = {"num_gpus": num_gpus}
    if resource_name:
        options["resources"] = {resource_name: 0.01}
    return options


def _empty_stage_timing() -> Dict[str, object]:
    return {
        "scheduler": {
            "prefill_rpc_s": 0.0,
            "attention_init_rpc_s": 0.0,
            "start_token_rpc_s": 0.0,
            "prepare_attention_rpc_s": 0.0,
            "attention_decode_rpc_s": 0.0,
            "finish_layer_rpc_s": 0.0,
            "sample_next_token_rpc_s": 0.0,
            "decode_tokens_rpc_s": 0.0,
            "free_request_rpc_s": 0.0,
        },
        "actors": {
            "prefill_compute_s": 0.0,
            "attention_init_compute_s": 0.0,
            "dense_start_token_compute_s": 0.0,
            "dense_prepare_attention_compute_s": 0.0,
            "attention_decode_compute_s": 0.0,
            "dense_finish_layer_compute_s": 0.0,
            "dense_sample_next_token_compute_s": 0.0,
            "dense_decode_tokens_compute_s": 0.0,
        },
        "counts": {
            "decode_steps": 0,
            "decode_layers": 0,
        },
    }


@ray.remote
class GlobalScheduler:
    def __init__(self, cluster_config: ClusterConfig, model_config: ModelConfig):
        self.cluster_config = cluster_config
        self.model_config = model_config
        self.prefill_nodes = []
        self.attention_nodes = []
        self.decode_dense_nodes = []
        self.runtime_model_spec = {
            "num_layers": int(model_config.num_layers),
            "hidden_size": int(model_config.hidden_size),
            "num_heads": int(model_config.num_heads),
            "num_key_value_heads": int(getattr(model_config, "num_key_value_heads", model_config.num_heads)),
            "vocab_size": 0,
        }
        self.decode_step_sync_window_s = max(0.0, float(cluster_config.decode_step_sync_window_s))
        self.decode_step_sync_max_size = max(1, int(cluster_config.decode_step_sync_max_size))
        self.attention_decode_wave_persist_enabled = bool(
            getattr(cluster_config, "attention_decode_wave_persist_enabled", False)
        )
        self.decode_step_sync_flushes = 0
        self.decode_step_sync_total_items = 0
        self.decode_step_sync_max_observed = 0
        self._decode_step_sync_next_cohort_id = 1
        self._decode_step_sync_batches: dict[int, list[tuple[asyncio.Future, str]]] = {}
        self._decode_step_sync_tasks: dict[int, asyncio.Task] = {}
        self._inflight_request_count = 0
        self.attention_layer_barrier_window_s = max(
            0.0, float(cluster_config.attention_layer_barrier_window_s)
        )
        self.attention_layer_barrier_max_size = max(
            1, int(cluster_config.attention_layer_barrier_max_size)
        )
        self.attention_layer_barrier_flushes = 0
        self.attention_layer_barrier_total_items = 0
        self.attention_layer_barrier_max_observed = 0
        self._attention_layer_barrier_batches: dict[tuple[int, int], list[asyncio.Future]] = {}
        self._attention_layer_barrier_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self.attention_batch_window_s = max(0.0, float(cluster_config.attention_rpc_batch_window_s))
        self.attention_batch_max_size = max(1, int(cluster_config.attention_rpc_batch_max_size))
        self.attention_rpc_cross_key_batch_enabled = bool(
            getattr(cluster_config, "attention_rpc_cross_key_batch_enabled", False)
        )
        self.attention_actor_side_batching_enabled = bool(
            getattr(cluster_config, "attention_actor_side_batching_enabled", False)
        )
        self.attention_batch_flushes = 0
        self.attention_batch_total_items = 0
        self.attention_batch_max_observed = 0
        self.attention_batch_multi_key_flushes = 0
        self.attention_batch_total_keys = 0
        self.attention_batch_max_keys_observed = 0
        self._attention_wavefront_batches: dict[tuple[int, int], list[tuple[dict, asyncio.Future, object]]] = {}
        self._attention_wavefront_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._attention_wavefront_expected_sizes: dict[tuple[int, int], int] = {}
        self._active_decode_requests = 0
        self.decode_continuous_batch_window_s = max(
            0.0, float(getattr(cluster_config, "decode_continuous_batch_window_s", 0.0))
        )
        self.decode_continuous_batch_max_size = max(
            1, int(getattr(cluster_config, "decode_continuous_batch_max_size", 8))
        )
        self.decode_continuous_batch_inflight_target_enabled = bool(
            getattr(cluster_config, "decode_continuous_batch_inflight_target_enabled", False)
        )
        self.decode_continuous_batch_startup_grace_s = max(
            0.0, float(getattr(cluster_config, "decode_continuous_batch_startup_grace_s", 0.0))
        )
        self.decode_continuous_batch_flushes = 0
        self.decode_continuous_batch_total_items = 0
        self.decode_continuous_batch_max_observed = 0
        self.decode_continuous_batch_size_histogram: Dict[int, int] = {}
        self.decode_continuous_batch_target_flushes = 0
        self.decode_continuous_batch_window_flushes = 0
        self.decode_continuous_batch_immediate_flushes = 0
        self.decode_continuous_batch_wait_s = 0.0
        self.decode_continuous_batch_reordered_flushes = 0
        self.decode_continuous_batch_same_rank_flushes = 0
        self.decode_continuous_batch_mixed_rank_flushes = 0
        self.decode_continuous_batch_total_context_span = 0
        self.clover_pim_disjoint_decode_stripe_packing_enabled = bool(
            getattr(cluster_config, "clover_pim_disjoint_decode_stripe_packing_enabled", False)
        ) and str(cluster_config.attention_backend) == "cloverinfer"
        self.decode_continuous_batch_disjoint_stripe_decisions = 0
        self.decode_continuous_batch_stripe_overlap_total = 0
        self.decode_continuous_batch_stripe_overlap_max = 0
        self.decode_continuous_batch_stripe_overlap_flushes = 0
        self.decode_continuous_batch_last_stripe_overlap: Dict[str, object] = {}
        self._decode_pending_queue = deque()
        self._decode_driver_task: asyncio.Task | None = None
        self._background_completion_tasks: set[asyncio.Task] = set()
        self.clover_predictive_scheduling_enabled = bool(
            getattr(cluster_config, "clover_predictive_scheduling_enabled", False)
        ) and str(cluster_config.attention_backend) == "cloverinfer"
        self.clover_predictive_scheduling_alpha = min(
            1.0,
            max(0.0, float(getattr(cluster_config, "clover_predictive_scheduling_alpha", 0.2))),
        )
        self.clover_predictive_scheduling_min_samples = max(
            1, int(getattr(cluster_config, "clover_predictive_scheduling_min_samples", 4))
        )
        self.clover_predictive_scheduling_context_bucket_tokens = max(
            1, int(getattr(cluster_config, "clover_predictive_scheduling_context_bucket_tokens", 256))
        )
        self._predictive_models: dict[str, dict[tuple[str, int, int], dict[str, float]]] = {
            "dense": {},
            "attention": {},
        }
        self.predictive_batch_decisions = 0
        self.predictive_batch_fallbacks = 0
        self.predictive_model_updates = 0
        self.predictive_ready_requests = 0
        self.predictive_unready_requests = 0
        self.predictive_last_batch: Dict[str, object] = {}
        self.clover_capacity_aware_batching_enabled = bool(
            getattr(cluster_config, "clover_capacity_aware_batching_enabled", False)
        ) and str(cluster_config.attention_backend) == "cloverinfer"
        self.clover_capacity_aware_time_gap_threshold = float(
            getattr(cluster_config, "clover_capacity_aware_time_gap_threshold", 0.0)
        )
        self.clover_capacity_aware_lookahead_window = max(
            1, int(getattr(cluster_config, "clover_capacity_aware_lookahead_window", 1))
        )
        self.clover_capacity_aware_pim_a = float(
            getattr(cluster_config, "clover_capacity_aware_pim_a", 1.0)
        )
        self.clover_capacity_aware_pim_b = float(
            getattr(cluster_config, "clover_capacity_aware_pim_b", 0.0)
        )
        self.clover_capacity_aware_host_c = float(
            getattr(cluster_config, "clover_capacity_aware_host_c", 1.0)
        )
        self.clover_capacity_aware_max_tokens_per_dpu = max(
            0, int(getattr(cluster_config, "clover_capacity_aware_max_tokens_per_dpu", 0))
        )
        self.clover_capacity_aware_require_slot_headroom = bool(
            getattr(cluster_config, "clover_capacity_aware_require_slot_headroom", False)
        )
        self.capacity_aware_batch_decisions = 0
        self.capacity_aware_batch_fallbacks = 0
        self.capacity_aware_last_batch: Dict[str, object] = {}
        self._capacity_aware_allocator_stats_cache: List[Dict[str, object]] = []
        self._capacity_aware_allocator_stats_refreshes = 0
        self._capacity_aware_scheduler: CapacityAwareMicroBatchScheduler | None = None
        if self.clover_capacity_aware_batching_enabled:
            capacity_checker = default_capacity_checker
            if (
                str(cluster_config.attention_backend) == "cloverinfer"
                and int(cluster_config.pim_num_dpus) > 0
            ):
                head_dim = max(1, int(getattr(model_config, "hidden_size", 0)) // max(1, int(model_config.num_heads)))
                capacity_checker = allocator_aware_capacity_checker(
                    lambda: list(self._capacity_aware_allocator_stats_cache),
                    # Map token pressure into the allocator's element units with
                    # a simple KV footprint proxy: one K row + one V row.
                    bytes_per_token=max(1, 2 * head_dim),
                    capacity_field="total_free_elems",
                    require_slot_headroom=self.clover_capacity_aware_require_slot_headroom,
                )
            self._capacity_aware_scheduler = CapacityAwareMicroBatchScheduler(
                planner=plan_sharding,
                predict_pim_time=lambda reqs, plan: default_predict_pim_time(
                    reqs,
                    plan,
                    a=self.clover_capacity_aware_pim_a,
                    b=self.clover_capacity_aware_pim_b,
                ),
                predict_host_time=lambda total_tokens: default_predict_host_time(
                    total_tokens,
                    c=self.clover_capacity_aware_host_c,
                ),
                capacity_checker=capacity_checker,
                num_dpus=int(cluster_config.pim_num_dpus),
                num_heads=int(model_config.num_heads),
                time_gap_threshold=self.clover_capacity_aware_time_gap_threshold,
                lookahead_window=self.clover_capacity_aware_lookahead_window,
            )
        self.clover_rankset_overlap_enabled = bool(
            getattr(cluster_config, "clover_rankset_overlap_enabled", False)
        ) and str(cluster_config.attention_backend) == "cloverinfer"
        self.clover_rankset_overlap_max_ranksets_per_batch = max(
            0, int(getattr(cluster_config, "clover_rankset_overlap_max_ranksets_per_batch", 0))
        )
        self.clover_rankset_overlap_transfer_granularity = str(
            getattr(cluster_config, "clover_rankset_overlap_transfer_granularity", "stripe")
        )
        self.rankset_overlap_plan_batches = 0
        self.rankset_overlap_plan_items = 0
        self.rankset_overlap_plan_ranksets = 0
        self.rankset_overlap_plan_max_ranksets = 0
        self.rankset_overlap_plan_same_rankset_batches = 0
        self.rankset_overlap_plan_mixed_rankset_batches = 0
        self.rankset_overlap_task_graph_layers = 0
        self.rankset_overlap_task_graph_work_items = 0
        self.rankset_overlap_task_graph_max_work_items = 0
        self.rankset_overlap_partial_ready_reports = 0
        self.rankset_overlap_partial_ready_work_items = 0
        self.rankset_overlap_partial_ready_max_work_items = 0
        self.rankset_overlap_first_work_item_completion_s_total = 0.0
        self.rankset_overlap_first_work_item_completion_s_count = 0
        self.rankset_overlap_last_work_item_completion_s_total = 0.0
        self.rankset_overlap_last_work_item_completion_s_count = 0
        self.rankset_overlap_first_transfer_ready_s_total = 0.0
        self.rankset_overlap_first_transfer_ready_s_count = 0
        self.rankset_overlap_last_transfer_ready_s_total = 0.0
        self.rankset_overlap_last_transfer_ready_s_count = 0
        self.rankset_overlap_transfer_span_s_total = 0.0
        self.rankset_overlap_transfer_span_s_count = 0
        self.rankset_overlap_transfer_duration_s_total = 0.0
        self.rankset_overlap_transfer_duration_s_count = 0
        self.rankset_overlap_work_item_span_s_total = 0.0
        self.rankset_overlap_work_item_span_s_count = 0
        self.rankset_overlap_work_item_duration_s_total = 0.0
        self.rankset_overlap_work_item_duration_s_count = 0
        self.rankset_overlap_last_batch: Dict[str, object] = {}
        self.rankset_overlap_last_task_graph: Dict[str, object] = {}
        self.rankset_overlap_last_execution_summary: Dict[str, object] = {}

    def _attention_batch_target_size(self) -> int:
        return max(1, min(self._active_decode_requests, self.attention_batch_max_size))

    def _decode_step_sync_target_size(self) -> int:
        return max(1, min(self._inflight_request_count, self.decode_step_sync_max_size))

    def _attention_layer_barrier_target_size(self) -> int:
        return max(1, min(self._active_decode_requests, self.attention_layer_barrier_max_size))

    def _decode_continuous_batch_target_size(self) -> int:
        if self.decode_continuous_batch_inflight_target_enabled:
            active_or_expected = max(
                int(self._active_decode_requests),
                int(self._inflight_request_count),
            )
            return max(1, min(active_or_expected, self.decode_continuous_batch_max_size))
        return max(1, min(self._active_decode_requests, self.decode_continuous_batch_max_size))

    def _track_background_completion(self, coro):
        task = asyncio.create_task(coro)
        self._background_completion_tasks.add(task)

        def _done_callback(done_task: asyncio.Task):
            self._background_completion_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                # The completion coroutine is responsible for surfacing request-level
                # failures to the waiting future. Swallow here so the decode loop can continue.
                pass

        task.add_done_callback(_done_callback)
        return task

    def _new_decode_state(
        self,
        *,
        request_id: str,
        prompt_len: int,
        first_token: int,
        first_token_time: float,
        max_tokens: int,
        request_start: float,
        return_metrics: bool,
        packing_hint: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        packing_hint = packing_hint or {}
        preferred_stripe = [
            int(physical_dpu)
            for physical_dpu in packing_hint.get("preferred_dpu_stripe", []) or []
        ]
        inferred_rank_hint = packing_hint.get("rank_index")
        inferred_rankset_id = packing_hint.get("rankset_id")
        if inferred_rankset_id in (None, "") and preferred_stripe:
            stripe_width = int(packing_hint.get("stripe_width", len(preferred_stripe) or 0))
            if inferred_rank_hint is None:
                inferred_rankset_id = f"stripe:w{max(1, stripe_width)}"
            else:
                inferred_rankset_id = f"rank{int(inferred_rank_hint)}:w{max(1, stripe_width)}"
        inferred_rankset_plan = list(packing_hint.get("rankset_plan", []) or [])
        if not inferred_rankset_plan and preferred_stripe:
            inferred_rankset_plan = [
                {
                    "rankset_id": str(inferred_rankset_id or "rank-unknown"),
                    "rank_index": None if inferred_rank_hint is None else int(inferred_rank_hint),
                    "physical_dpus": list(preferred_stripe),
                    "stripe_width": int(packing_hint.get("stripe_width", len(preferred_stripe) or 0)),
                    "transfer_granularity": str(self.clover_rankset_overlap_transfer_granularity),
                }
            ]
        return {
            "request_id": request_id,
            "prompt_len": int(prompt_len),
            "current_token": int(first_token),
            "generated_ids": [int(first_token)],
            "max_tokens": int(max_tokens),
            "step": 1,
            "request_start": float(request_start),
            "first_token_time": float(first_token_time),
            "return_metrics": bool(return_metrics),
            "stage_timing": _empty_stage_timing(),
            "pending_free": False,
            "done": False,
            "completion_future": None,
            "packing_rank_hint": inferred_rank_hint,
            "packing_stripe_width": int(packing_hint.get("stripe_width", len(preferred_stripe) or 0)),
            "packing_preferred_dpu_stripe": preferred_stripe,
            "packing_rankset_id": inferred_rankset_id,
            "packing_rankset_count": int(
                packing_hint.get("rankset_count", len(inferred_rankset_plan)) or len(inferred_rankset_plan)
            ),
            "packing_rankset_plan": inferred_rankset_plan,
            "packing_sharding_plan": dict(packing_hint.get("sharding_plan", {}) or {}),
            "packing_planner_mode": str(packing_hint.get("planner_mode", "") or ""),
        }

    def _decode_state_context_len(self, state: Dict[str, object]) -> int:
        return int(state["prompt_len"]) + int(state["step"])

    def _decode_state_rank_hint(self, state: Dict[str, object]) -> int | None:
        rank_hint = state.get("packing_rank_hint")
        if rank_hint is None:
            return None
        return int(rank_hint)

    def _decode_state_stripe(self, state: Dict[str, object]) -> tuple[int, ...]:
        stripe = state.get("packing_preferred_dpu_stripe", [])
        return tuple(int(physical_dpu) for physical_dpu in stripe)

    def _decode_state_rankset_id(self, state: Dict[str, object]) -> str | None:
        rankset_id = state.get("packing_rankset_id")
        if rankset_id in (None, ""):
            return None
        return str(rankset_id)

    def _record_decode_batch_shape(self, batch: List[Dict[str, object]]) -> None:
        context_lens = [self._decode_state_context_len(state) for state in batch]
        if context_lens:
            self.decode_continuous_batch_total_context_span += max(context_lens) - min(context_lens)
        known_rank_hints = [
            rank_hint
            for rank_hint in (self._decode_state_rank_hint(state) for state in batch)
            if rank_hint is not None
        ]
        if len(known_rank_hints) >= 2:
            if len(set(known_rank_hints)) == 1:
                self.decode_continuous_batch_same_rank_flushes += 1
            else:
                self.decode_continuous_batch_mixed_rank_flushes += 1
        stripes = [set(self._decode_state_stripe(state)) for state in batch]
        pair_overlaps: list[int] = []
        dpu_counts: Dict[int, int] = {}
        for stripe in stripes:
            for physical_dpu in stripe:
                dpu_counts[int(physical_dpu)] = dpu_counts.get(int(physical_dpu), 0) + 1
        for left_idx, left_stripe in enumerate(stripes):
            if not left_stripe:
                continue
            for right_stripe in stripes[left_idx + 1 :]:
                if right_stripe:
                    pair_overlaps.append(len(left_stripe.intersection(right_stripe)))
        overlap_total = int(sum(pair_overlaps))
        overlap_max = int(max(pair_overlaps, default=0))
        if len(batch) >= 2 and pair_overlaps:
            self.decode_continuous_batch_stripe_overlap_flushes += 1
            self.decode_continuous_batch_stripe_overlap_total += overlap_total
            self.decode_continuous_batch_stripe_overlap_max = max(
                self.decode_continuous_batch_stripe_overlap_max,
                overlap_max,
            )
        max_dpu_multiplicity = int(max(dpu_counts.values(), default=0))
        self.decode_continuous_batch_last_stripe_overlap = {
            "enabled": bool(self.clover_pim_disjoint_decode_stripe_packing_enabled),
            "request_ids": [str(state["request_id"]) for state in batch],
            "stripe_widths": [len(stripe) for stripe in stripes],
            "pair_overlap_total": overlap_total,
            "pair_overlap_max": overlap_max,
            "max_dpu_multiplicity": max_dpu_multiplicity,
            "shared_dpu_count": int(sum(1 for count in dpu_counts.values() if count > 1)),
        }

    def _plan_rankset_overlap_batch(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        rankset_ids = [
            rankset_id
            for rankset_id in (self._decode_state_rankset_id(state) for state in batch)
            if rankset_id is not None
        ]
        unique_rankset_ids = list(dict.fromkeys(rankset_ids))
        planned_rankset_count = len(unique_rankset_ids)
        self.rankset_overlap_plan_batches += 1
        self.rankset_overlap_plan_items += len(batch)
        self.rankset_overlap_plan_ranksets += planned_rankset_count
        self.rankset_overlap_plan_max_ranksets = max(
            self.rankset_overlap_plan_max_ranksets,
            planned_rankset_count,
        )
        if planned_rankset_count <= 1:
            self.rankset_overlap_plan_same_rankset_batches += 1
        else:
            self.rankset_overlap_plan_mixed_rankset_batches += 1
        batch_summary = {
            "enabled": bool(self.clover_rankset_overlap_enabled),
            "transfer_granularity": str(self.clover_rankset_overlap_transfer_granularity),
            "request_ids": [str(state["request_id"]) for state in batch],
            "rankset_ids": unique_rankset_ids,
            "rankset_count": int(planned_rankset_count),
            "per_request_rankset_count": [int(state.get("packing_rankset_count", 0) or 0) for state in batch],
            "planned_rankset_cap": int(self.clover_rankset_overlap_max_ranksets_per_batch),
        }
        self.rankset_overlap_last_batch = batch_summary
        return batch_summary

    def _build_rankset_task_graph(
        self,
        batch: List[Dict[str, object]],
        layer_idx: int,
        batch_plan: Dict[str, object],
    ) -> Dict[str, object]:
        grouped: Dict[str, Dict[str, object]] = {}
        fallback_rankset_id = "rankset-unknown"
        for state in batch:
            request_id = str(state["request_id"])
            request_ranksets = list(state.get("packing_rankset_plan", []) or [])
            if not request_ranksets:
                request_ranksets = [
                    {
                        "rankset_id": str(self._decode_state_rankset_id(state) or fallback_rankset_id),
                        "rank_index": self._decode_state_rank_hint(state),
                        "physical_dpus": list(self._decode_state_stripe(state)),
                        "stripe_width": int(state.get("packing_stripe_width", 0) or 0),
                        "transfer_granularity": str(self.clover_rankset_overlap_transfer_granularity),
                    }
                ]
            for request_rankset in request_ranksets:
                rankset_id = str(request_rankset.get("rankset_id", fallback_rankset_id))
                layer_group_map = dict(request_rankset.get("layer_group_map", {}) or {})
                layer_groups = list(layer_group_map.get(str(layer_idx), []) or [])
                if self.clover_rankset_overlap_transfer_granularity == "rankset" and not layer_groups:
                    continue
                transfer_granularity = str(
                    request_rankset.get(
                        "transfer_granularity",
                        self.clover_rankset_overlap_transfer_granularity,
                    )
                )
                work_item = grouped.get(rankset_id)
                if work_item is None:
                    work_item = {
                        "work_item_id": f"layer{int(layer_idx)}:{rankset_id}",
                        "layer_idx": int(layer_idx),
                        "rankset_id": rankset_id,
                        "rank_index": request_rankset.get("rank_index"),
                        "physical_dpus": list(request_rankset.get("physical_dpus", []) or []),
                        "stripe_width": int(request_rankset.get("stripe_width", 0) or 0),
                        "transfer_granularity": transfer_granularity,
                        "request_ids": [],
                        "request_group_slices": {},
                        "status": "planned",
                    }
                    grouped[rankset_id] = work_item
                work_item["request_ids"].append(request_id)
                if layer_groups:
                    work_item["request_group_slices"][request_id] = [
                        {
                            "head_start": int(group.get("head_start", 0)),
                            "head_end": int(group.get("head_end", 0)),
                            "group_heads": int(group.get("group_heads", 0)),
                            "physical_dpu": int(group.get("physical_dpu", 0)),
                            "k_slot": str(group.get("k_slot", "")),
                            "v_slot": str(group.get("v_slot", "")),
                        }
                        for group in layer_groups
                    ]

        work_items = sorted(grouped.values(), key=lambda item: str(item["work_item_id"]))
        if self.clover_rankset_overlap_max_ranksets_per_batch > 0:
            work_items = work_items[: self.clover_rankset_overlap_max_ranksets_per_batch]
        self.rankset_overlap_task_graph_layers += 1
        self.rankset_overlap_task_graph_work_items += len(work_items)
        self.rankset_overlap_task_graph_max_work_items = max(
            self.rankset_overlap_task_graph_max_work_items,
            len(work_items),
        )
        task_graph = {
            "enabled": bool(self.clover_rankset_overlap_enabled),
            "layer_idx": int(layer_idx),
            "batch_request_ids": [str(state["request_id"]) for state in batch],
            "transfer_granularity": str(self.clover_rankset_overlap_transfer_granularity),
            "batch_rankset_count": int(batch_plan.get("rankset_count", 0)),
            "work_item_count": int(len(work_items)),
            "work_items": work_items,
            "execution_mode": (
                "rankset_task_graph_async_dispatch_serial_compute"
                if bool(getattr(self.cluster_config, "clover_rankset_overlap_async_dispatch_enabled", False))
                else "scaffold_serial_attention"
            ),
        }
        self.rankset_overlap_last_task_graph = task_graph
        return task_graph

    def _record_rankset_execution(self, state: Dict[str, object], execution: Dict[str, object]) -> None:
        execution = dict(execution or {})
        work_item_events = list(execution.get("work_item_events", []) or [])
        execution["work_item_events"] = work_item_events
        state["last_rankset_execution"] = execution
        if not work_item_events:
            self.rankset_overlap_last_execution_summary = {
                "request_id": str(state.get("request_id", "")),
                "execution_mode": str(execution.get("execution_mode", "")),
                "fallback_to_full_batch": bool(execution.get("fallback_to_full_batch", False)),
                "fallback_reason": str(execution.get("fallback_reason", "")),
                "executed_work_item_count": int(execution.get("executed_work_item_count", 0)),
                "partial_ready_work_items": 0,
                "first_work_item_completion_s": 0.0,
                "last_work_item_completion_s": 0.0,
                "work_item_completion_span_s": 0.0,
                "work_item_duration_sum_s": 0.0,
            }
            return

        self.rankset_overlap_partial_ready_reports += 1
        self.rankset_overlap_partial_ready_work_items += len(work_item_events)
        self.rankset_overlap_partial_ready_max_work_items = max(
            self.rankset_overlap_partial_ready_max_work_items,
            len(work_item_events),
        )
        timeline_start_at = min(float(event.get("started_at", 0.0)) for event in work_item_events)
        first_finished_at = min(float(event.get("finished_at", timeline_start_at)) for event in work_item_events)
        last_finished_at = max(float(event.get("finished_at", timeline_start_at)) for event in work_item_events)
        first_transfer_ready_at = min(
            float(event.get("transfer_finished_at", timeline_start_at)) for event in work_item_events
        )
        last_transfer_ready_at = max(
            float(event.get("transfer_finished_at", timeline_start_at)) for event in work_item_events
        )
        completion_span_s = max(0.0, last_finished_at - first_finished_at)
        first_completion_s = max(0.0, first_finished_at - timeline_start_at)
        last_completion_s = max(0.0, last_finished_at - timeline_start_at)
        first_transfer_ready_s = max(0.0, first_transfer_ready_at - timeline_start_at)
        last_transfer_ready_s = max(0.0, last_transfer_ready_at - timeline_start_at)
        transfer_span_s = max(0.0, last_transfer_ready_at - first_transfer_ready_at)
        transfer_duration_sum_s = sum(
            max(0.0, float(event.get("transfer_duration_s", 0.0))) for event in work_item_events
        )
        duration_sum_s = sum(max(0.0, float(event.get("duration_s", 0.0))) for event in work_item_events)
        self.rankset_overlap_first_work_item_completion_s_total += first_completion_s
        self.rankset_overlap_first_work_item_completion_s_count += 1
        self.rankset_overlap_last_work_item_completion_s_total += last_completion_s
        self.rankset_overlap_last_work_item_completion_s_count += 1
        self.rankset_overlap_first_transfer_ready_s_total += first_transfer_ready_s
        self.rankset_overlap_first_transfer_ready_s_count += 1
        self.rankset_overlap_last_transfer_ready_s_total += last_transfer_ready_s
        self.rankset_overlap_last_transfer_ready_s_count += 1
        self.rankset_overlap_transfer_span_s_total += transfer_span_s
        self.rankset_overlap_transfer_span_s_count += 1
        self.rankset_overlap_transfer_duration_s_total += transfer_duration_sum_s
        self.rankset_overlap_transfer_duration_s_count += len(work_item_events)
        self.rankset_overlap_work_item_span_s_total += completion_span_s
        self.rankset_overlap_work_item_span_s_count += 1
        self.rankset_overlap_work_item_duration_s_total += duration_sum_s
        self.rankset_overlap_work_item_duration_s_count += len(work_item_events)
        self.rankset_overlap_last_execution_summary = {
            "request_id": str(state.get("request_id", "")),
            "execution_mode": str(execution.get("execution_mode", "")),
            "fallback_to_full_batch": bool(execution.get("fallback_to_full_batch", False)),
            "fallback_reason": str(execution.get("fallback_reason", "")),
            "executed_work_item_count": int(execution.get("executed_work_item_count", 0)),
            "partial_ready_work_items": int(len(work_item_events)),
            "timeline_start_at": float(timeline_start_at),
            "first_transfer_ready_s": float(first_transfer_ready_s),
            "last_transfer_ready_s": float(last_transfer_ready_s),
            "transfer_ready_span_s": float(transfer_span_s),
            "transfer_duration_sum_s": float(transfer_duration_sum_s),
            "first_work_item_completion_s": float(first_completion_s),
            "last_work_item_completion_s": float(last_completion_s),
            "work_item_completion_span_s": float(completion_span_s),
            "work_item_duration_sum_s": float(duration_sum_s),
            "work_item_ids": [str(event.get("work_item_id", "")) for event in work_item_events],
        }

    def _predictive_context_bucket(self, context_len: int) -> int:
        bucket = int(self.clover_predictive_scheduling_context_bucket_tokens)
        context_len = max(1, int(context_len))
        return ((context_len + bucket - 1) // bucket) * bucket

    def _predictive_feature_keys(self, state: Dict[str, object]) -> list[tuple[str, int, int]]:
        context_bucket = self._predictive_context_bucket(self._decode_state_context_len(state))
        stripe_width = max(1, int(state.get("packing_stripe_width", 0) or 1))
        return [
            ("exact", context_bucket, stripe_width),
            ("context", context_bucket, 0),
            ("global", 0, 0),
        ]

    def _update_predictive_component(
        self,
        component: str,
        state: Dict[str, object],
        observed_s: float,
    ) -> None:
        observed_s = max(0.0, float(observed_s))
        model = self._predictive_models[component]
        alpha = float(self.clover_predictive_scheduling_alpha)
        for key in self._predictive_feature_keys(state):
            entry = model.get(key)
            if entry is None:
                model[key] = {
                    "avg_s": observed_s,
                    "count": 1.0,
                }
                continue
            entry["avg_s"] = (1.0 - alpha) * float(entry.get("avg_s", observed_s)) + alpha * observed_s
            entry["count"] = float(entry.get("count", 0.0)) + 1.0

    def _lookup_predictive_component(
        self,
        component: str,
        state: Dict[str, object],
    ) -> tuple[float | None, int, str]:
        model = self._predictive_models[component]
        for level, context_bucket, stripe_width in self._predictive_feature_keys(state):
            entry = model.get((level, context_bucket, stripe_width))
            if entry is None:
                continue
            return (
                float(entry.get("avg_s", 0.0)),
                int(entry.get("count", 0.0)),
                str(level),
            )
        return (None, 0, "missing")

    def _predictive_state_costs(self, state: Dict[str, object]) -> Dict[str, object] | None:
        dense_avg, dense_count, dense_level = self._lookup_predictive_component("dense", state)
        attention_avg, attention_count, attention_level = self._lookup_predictive_component("attention", state)
        if dense_avg is None or attention_avg is None:
            return None
        ready = (
            dense_count >= self.clover_predictive_scheduling_min_samples
            and attention_count >= self.clover_predictive_scheduling_min_samples
        )
        if ready:
            self.predictive_ready_requests += 1
        else:
            self.predictive_unready_requests += 1
        return {
            "dense_s": float(dense_avg),
            "attention_s": float(attention_avg),
            "dense_count": int(dense_count),
            "attention_count": int(attention_count),
            "dense_level": dense_level,
            "attention_level": attention_level,
            "ready": bool(ready),
        }

    def _predictive_batch_candidate_score(
        self,
        seed_state: Dict[str, object],
        selected_states: List[Dict[str, object]],
        candidate_state: Dict[str, object],
        queue_index: int,
    ) -> tuple[float, float, float, int, int, int, int] | None:
        predicted_items = []
        for state in [*selected_states, candidate_state]:
            predicted = self._predictive_state_costs(state)
            if predicted is None or not bool(predicted["ready"]):
                return None
            predicted_items.append(predicted)

        dense_total = sum(float(item["dense_s"]) for item in predicted_items)
        attention_total = sum(float(item["attention_s"]) for item in predicted_items)
        device_bubble = abs(dense_total - attention_total)
        device_diffs = [float(item["dense_s"]) - float(item["attention_s"]) for item in predicted_items]
        diff_spread = max(device_diffs) - min(device_diffs) if device_diffs else 0.0
        rank_penalty, stripe_penalty, context_gap, width_gap, _ = self._decode_batch_candidate_score(
            seed_state,
            candidate_state,
            queue_index,
        )
        predicted_batch_cost = max(dense_total, attention_total)
        return (
            float(predicted_batch_cost),
            float(device_bubble),
            float(diff_spread),
            int(rank_penalty),
            int(stripe_penalty),
            int(context_gap + width_gap),
            int(queue_index),
        )

    def _record_predictive_batch(self, batch: List[Dict[str, object]], predictive_used: bool) -> None:
        batch_summary = {
            "enabled": bool(self.clover_predictive_scheduling_enabled),
            "used": bool(predictive_used),
            "request_ids": [str(state["request_id"]) for state in batch],
            "context_lens": [int(self._decode_state_context_len(state)) for state in batch],
            "rank_hints": [
                None if self._decode_state_rank_hint(state) is None else int(self._decode_state_rank_hint(state))
                for state in batch
            ],
            "stripe_widths": [int(state.get("packing_stripe_width", 0)) for state in batch],
        }
        if predictive_used:
            predicted_states = [self._predictive_state_costs(state) for state in batch]
            if all(item is not None for item in predicted_states):
                dense_total = sum(float(item["dense_s"]) for item in predicted_states if item is not None)
                attention_total = sum(float(item["attention_s"]) for item in predicted_states if item is not None)
                batch_summary["predicted_dense_s"] = float(dense_total)
                batch_summary["predicted_attention_s"] = float(attention_total)
                batch_summary["predicted_bubble_s"] = float(abs(dense_total - attention_total))
                batch_summary["prediction_levels"] = [
                    {
                        "dense": str(item["dense_level"]),
                        "attention": str(item["attention_level"]),
                    }
                    for item in predicted_states
                    if item is not None
                ]
        self.predictive_last_batch = batch_summary

    def _capacity_aware_request_view(self, state: Dict[str, object]) -> Dict[str, object]:
        return {
            "request_id": str(state["request_id"]),
            "seq_len": int(self._decode_state_context_len(state)),
            "num_new_tokens": 1,
            "_state": state,
        }

    async def _refresh_capacity_aware_allocator_stats(self) -> None:
        if not self.attention_nodes:
            return
        try:
            attention_info = await self.attention_nodes[0].get_info.remote()
        except Exception:
            return
        backend_debug = dict(attention_info.get("backend_debug", {}) or {})
        resident_store_debug = dict(backend_debug.get("resident_store_debug", {}) or {})
        allocator_stats = list(resident_store_debug.get("allocator_stats", []) or [])
        if allocator_stats:
            self._capacity_aware_allocator_stats_cache = [dict(item) for item in allocator_stats]
            self._capacity_aware_allocator_stats_refreshes += 1

    def _record_capacity_aware_batch(self, request_views: List[Dict[str, object]], batch_meta: Dict[str, object]) -> None:
        self.capacity_aware_last_batch = {
            "enabled": bool(self.clover_capacity_aware_batching_enabled),
            "request_ids": [str(item["request_id"]) for item in request_views],
            "context_lens": [int(item["seq_len"]) for item in request_views],
            "predicted_pim_time": float(batch_meta.get("predicted_pim_time", 0.0)),
            "predicted_host_time": float(batch_meta.get("predicted_host_time", 0.0)),
            "time_gap": float(batch_meta.get("time_gap", 0.0)),
            "capacity_ok": bool(batch_meta.get("capacity_ok", False)),
            "capacity_usage_ratio": float(batch_meta.get("capacity_usage_ratio", 0.0)),
            "selection_reason": str(batch_meta.get("selection_reason", "")),
            "require_slot_headroom": bool(self.clover_capacity_aware_require_slot_headroom),
            "allocator_stats_cached": int(len(self._capacity_aware_allocator_stats_cache)),
            "allocator_stats_refreshes": int(self._capacity_aware_allocator_stats_refreshes),
        }

    def _capacity_aware_effective_max_tokens_per_dpu(self) -> int:
        configured = int(self.clover_capacity_aware_max_tokens_per_dpu)
        if configured > 0:
            return configured
        if not self._capacity_aware_allocator_stats_cache:
            return 0
        free_values = [
            int(item.get("total_free_elems", item.get("largest_free_range", 0)) or 0)
            for item in self._capacity_aware_allocator_stats_cache
            if int(item.get("dpu_id", -1)) >= 0
        ]
        positive_free = [value for value in free_values if value > 0]
        if not positive_free:
            return 0
        head_dim = max(1, int(getattr(self.model_config, "hidden_size", 0)) // max(1, int(self.model_config.num_heads)))
        elems_per_token = max(1, 2 * int(head_dim))
        return max(1, min(positive_free) // elems_per_token)

    def _decode_batch_candidate_score(
        self,
        seed_state: Dict[str, object],
        candidate_state: Dict[str, object],
        queue_index: int,
    ) -> tuple[int, int, int, int, int]:
        seed_rank = self._decode_state_rank_hint(seed_state)
        candidate_rank = self._decode_state_rank_hint(candidate_state)
        rank_penalty = 1
        if seed_rank is not None and candidate_rank is not None:
            rank_penalty = 0 if seed_rank == candidate_rank else 2

        seed_stripe = set(self._decode_state_stripe(seed_state))
        candidate_stripe = set(self._decode_state_stripe(candidate_state))
        stripe_penalty = 1
        if seed_stripe and candidate_stripe:
            stripe_penalty = 0 if seed_stripe.intersection(candidate_stripe) else 2

        context_gap = abs(self._decode_state_context_len(seed_state) - self._decode_state_context_len(candidate_state))
        width_gap = abs(int(seed_state.get("packing_stripe_width", 0)) - int(candidate_state.get("packing_stripe_width", 0)))
        return (rank_penalty, stripe_penalty, context_gap, width_gap, int(queue_index))

    def _decode_batch_disjoint_candidate_score(
        self,
        seed_state: Dict[str, object],
        selected_states: List[Dict[str, object]],
        candidate_state: Dict[str, object],
        queue_index: int,
    ) -> tuple[int, int, int, int, int, int]:
        selected_stripes = [set(self._decode_state_stripe(state)) for state in selected_states]
        candidate_stripe = set(self._decode_state_stripe(candidate_state))
        overlap_total = 0
        overlap_max = 0
        if candidate_stripe and selected_stripes:
            overlaps = [len(candidate_stripe.intersection(stripe)) for stripe in selected_stripes if stripe]
            overlap_total = sum(overlaps)
            overlap_max = max(overlaps, default=0)

        seed_rank = self._decode_state_rank_hint(seed_state)
        candidate_rank = self._decode_state_rank_hint(candidate_state)
        rank_penalty = 1
        if seed_rank is not None and candidate_rank is not None:
            rank_penalty = 0 if seed_rank == candidate_rank else 2

        context_gap = abs(self._decode_state_context_len(seed_state) - self._decode_state_context_len(candidate_state))
        width_gap = abs(int(seed_state.get("packing_stripe_width", 0)) - int(candidate_state.get("packing_stripe_width", 0)))
        return (int(overlap_total), int(overlap_max), int(rank_penalty), int(context_gap), int(width_gap), int(queue_index))

    def _take_decode_batch(self, batch_size: int) -> List[Dict[str, object]]:
        if batch_size <= 0 or not self._decode_pending_queue:
            return []
        if self.clover_capacity_aware_batching_enabled and self._capacity_aware_scheduler is not None:
            queue_items = list(self._decode_pending_queue)
            request_views = [self._capacity_aware_request_view(state) for state in queue_items]
            max_capacity = self._capacity_aware_effective_max_tokens_per_dpu()
            if max_capacity > 0:
                try:
                    decision = self._capacity_aware_scheduler.build_micro_batch(
                        request_views,
                        max_capacity,
                        max_batch_size=batch_size,
                    )
                    chosen_ids = {str(item["request_id"]) for item in decision.micro_batch.requests}
                    batch = [
                        state
                        for state in queue_items
                        if str(state["request_id"]) in chosen_ids
                    ]
                    self._decode_pending_queue = deque(
                        state
                        for state in queue_items
                        if str(state["request_id"]) not in chosen_ids
                    )
                    self.capacity_aware_batch_decisions += 1
                    self._record_capacity_aware_batch(
                        decision.micro_batch.requests,
                        decision.micro_batch.to_dict(),
                    )
                    self._record_decode_batch_shape(batch)
                    return batch
                except Exception:
                    self.capacity_aware_batch_fallbacks += 1
        if batch_size >= len(self._decode_pending_queue):
            batch = list(self._decode_pending_queue)
            self._decode_pending_queue.clear()
            self._record_predictive_batch(batch, predictive_used=False)
            self._record_decode_batch_shape(batch)
            return batch

        queue_items = list(self._decode_pending_queue)
        seed_state = queue_items[0]
        selected_indices = [0]
        selected_states = [seed_state]
        predictive_used = False
        while len(selected_indices) < batch_size:
            predictive_candidates = []
            heuristic_candidates = []
            selected_index_set = set(selected_indices)
            for queue_index, candidate_state in enumerate(queue_items[1:], start=1):
                if queue_index in selected_index_set:
                    continue
                if self.clover_pim_disjoint_decode_stripe_packing_enabled:
                    heuristic_score = self._decode_batch_disjoint_candidate_score(
                        seed_state,
                        selected_states,
                        candidate_state,
                        queue_index,
                    )
                else:
                    heuristic_score = self._decode_batch_candidate_score(seed_state, candidate_state, queue_index)
                heuristic_candidates.append((heuristic_score, queue_index))
                if self.clover_predictive_scheduling_enabled:
                    predictive_score = self._predictive_batch_candidate_score(
                        seed_state,
                        selected_states,
                        candidate_state,
                        queue_index,
                    )
                    if predictive_score is not None:
                        predictive_candidates.append((predictive_score, queue_index))

            if self.clover_predictive_scheduling_enabled and predictive_candidates:
                predictive_candidates.sort(key=lambda item: item[0])
                chosen_index = int(predictive_candidates[0][1])
                predictive_used = True
            else:
                heuristic_candidates.sort(key=lambda item: item[0])
                chosen_index = int(heuristic_candidates[0][1])
                if self.clover_pim_disjoint_decode_stripe_packing_enabled:
                    self.decode_continuous_batch_disjoint_stripe_decisions += 1
            selected_indices.append(chosen_index)
            selected_states.append(queue_items[chosen_index])

        selected_index_set = set(selected_indices)
        batch = [queue_items[idx] for idx in selected_indices]
        self._decode_pending_queue = deque(
            item
            for idx, item in enumerate(queue_items)
            if idx not in selected_index_set
        )
        if self.clover_predictive_scheduling_enabled and predictive_used:
            self.predictive_batch_decisions += 1
        elif self.clover_predictive_scheduling_enabled:
            self.predictive_batch_fallbacks += 1
        self._record_predictive_batch(batch, predictive_used=predictive_used)

        if any(idx != expected for expected, idx in enumerate(selected_indices)):
            self.decode_continuous_batch_reordered_flushes += 1
        self._record_decode_batch_shape(batch)
        return batch

    def _ensure_decode_driver(self):
        if self._decode_driver_task is None or self._decode_driver_task.done():
            self._decode_driver_task = asyncio.create_task(self._decode_driver_loop())

    async def _decode_driver_loop(self):
        try:
            while self._decode_pending_queue:
                if self.clover_capacity_aware_batching_enabled and self._capacity_aware_scheduler is not None:
                    await self._refresh_capacity_aware_allocator_stats()
                target_size = self._decode_continuous_batch_target_size()
                flush_reason = "immediate"
                waited_s = 0.0
                if (
                    self.decode_continuous_batch_window_s > 0
                    and len(self._decode_pending_queue) < target_size
                    and (
                        self._inflight_request_count
                        if self.decode_continuous_batch_inflight_target_enabled
                        else self._active_decode_requests
                    )
                    > 1
                ):
                    wait_started = time.perf_counter()
                    deadline = wait_started + self.decode_continuous_batch_window_s
                    while (
                        len(self._decode_pending_queue) < target_size
                        and (
                            self._inflight_request_count
                            if self.decode_continuous_batch_inflight_target_enabled
                            else self._active_decode_requests
                        )
                        > len(self._decode_pending_queue)
                    ):
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        await asyncio.sleep(min(remaining, 0.001))
                    waited_s = time.perf_counter() - wait_started
                    self.decode_continuous_batch_wait_s += waited_s
                    if len(self._decode_pending_queue) >= target_size:
                        flush_reason = "target"
                    elif waited_s > 0:
                        flush_reason = "window"
                elif len(self._decode_pending_queue) >= target_size:
                    flush_reason = "target"

                if (
                    self.decode_continuous_batch_startup_grace_s > 0
                    and self.decode_continuous_batch_total_items == 0
                    and len(self._decode_pending_queue) < self.decode_continuous_batch_max_size
                    and self._inflight_request_count > len(self._decode_pending_queue)
                ):
                    grace_started = time.perf_counter()
                    grace_deadline = grace_started + self.decode_continuous_batch_startup_grace_s
                    while (
                        len(self._decode_pending_queue) < self.decode_continuous_batch_max_size
                        and self._inflight_request_count > len(self._decode_pending_queue)
                    ):
                        remaining = grace_deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        await asyncio.sleep(min(remaining, 0.001))
                    grace_waited_s = time.perf_counter() - grace_started
                    waited_s += grace_waited_s
                    self.decode_continuous_batch_wait_s += grace_waited_s
                    if len(self._decode_pending_queue) >= self.decode_continuous_batch_max_size:
                        flush_reason = "target"
                    elif grace_waited_s > 0 and flush_reason == "immediate":
                        flush_reason = "window"

                batch_size = min(len(self._decode_pending_queue), self.decode_continuous_batch_max_size)
                batch = self._take_decode_batch(batch_size)
                self.decode_continuous_batch_flushes += 1
                self.decode_continuous_batch_total_items += len(batch)
                self.decode_continuous_batch_max_observed = max(
                    self.decode_continuous_batch_max_observed,
                    len(batch),
                )
                self.decode_continuous_batch_size_histogram[len(batch)] = (
                    self.decode_continuous_batch_size_histogram.get(len(batch), 0) + 1
                )
                if flush_reason == "target":
                    self.decode_continuous_batch_target_flushes += 1
                elif flush_reason == "window":
                    self.decode_continuous_batch_window_flushes += 1
                else:
                    self.decode_continuous_batch_immediate_flushes += 1
                try:
                    await self._run_decode_wave(batch)
                except Exception as exc:
                    for state in batch:
                        await self._fail_decode_state(state, exc)
        finally:
            self._decode_driver_task = None
            if self._decode_pending_queue:
                self._ensure_decode_driver()

    async def _flush_decode_step_sync(self, step: int):
        try:
            if self.decode_step_sync_window_s > 0:
                await asyncio.sleep(self.decode_step_sync_window_s)
            await self._execute_decode_step_sync(step)
        except asyncio.CancelledError:
            return

    async def _execute_decode_step_sync(self, step: int):
        batch = self._decode_step_sync_batches.pop(step, [])
        self._decode_step_sync_tasks.pop(step, None)
        if not batch:
            return
        group_size = len(batch)
        cohort = None
        if self.attention_decode_wave_persist_enabled:
            cohort_id = f"step{int(step)}_cohort{self._decode_step_sync_next_cohort_id}"
            self._decode_step_sync_next_cohort_id += 1
            cohort = {
                "cohort_id": cohort_id,
                "group_size": int(group_size),
            }
        self.decode_step_sync_flushes += 1
        self.decode_step_sync_total_items += group_size
        self.decode_step_sync_max_observed = max(self.decode_step_sync_max_observed, group_size)
        for future, request_id in batch:
            if not future.done():
                if cohort is None:
                    future.set_result(None)
                else:
                    result = dict(cohort)
                    result["request_id"] = request_id
                    future.set_result(result)

    async def _maybe_flush_decode_step_syncs(self):
        target_size = self._decode_step_sync_target_size()
        ready_steps = [
            step
            for step, batch in self._decode_step_sync_batches.items()
            if len(batch) >= target_size
        ]
        for step in ready_steps:
            task = self._decode_step_sync_tasks.pop(step, None)
            if task is not None:
                task.cancel()
            await self._execute_decode_step_sync(step)

    async def _synchronize_decode_step(self, step: int, request_id: str):
        if self.decode_step_sync_window_s <= 0:
            if not self.attention_decode_wave_persist_enabled:
                return None
            return {
                "cohort_id": f"step{int(step)}_solo",
                "group_size": 1,
                "request_id": request_id,
            }
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        batch = self._decode_step_sync_batches.setdefault(int(step), [])
        batch.append((future, request_id))
        target_size = self._decode_step_sync_target_size()
        if len(batch) >= target_size:
            task = self._decode_step_sync_tasks.pop(int(step), None)
            if task is not None:
                task.cancel()
            await self._execute_decode_step_sync(int(step))
        elif int(step) not in self._decode_step_sync_tasks:
            self._decode_step_sync_tasks[int(step)] = asyncio.create_task(
                self._flush_decode_step_sync(int(step))
            )
        return await future

    async def _flush_attention_layer_barrier_key(self, key: tuple[int, int]):
        try:
            if self.attention_layer_barrier_window_s > 0:
                await asyncio.sleep(self.attention_layer_barrier_window_s)
            await self._execute_attention_layer_barrier_key(key)
        except asyncio.CancelledError:
            return

    async def _execute_attention_layer_barrier_key(self, key: tuple[int, int]):
        batch = self._attention_layer_barrier_batches.pop(key, [])
        self._attention_layer_barrier_tasks.pop(key, None)
        if not batch:
            return
        group_size = len(batch)
        self.attention_layer_barrier_flushes += 1
        self.attention_layer_barrier_total_items += group_size
        self.attention_layer_barrier_max_observed = max(
            self.attention_layer_barrier_max_observed, group_size
        )
        for future in batch:
            if not future.done():
                future.set_result(group_size)

    async def _maybe_flush_attention_layer_barriers(self):
        target_size = self._attention_layer_barrier_target_size()
        ready_keys = [
            key
            for key, batch in self._attention_layer_barrier_batches.items()
            if len(batch) >= target_size
        ]
        for key in ready_keys:
            task = self._attention_layer_barrier_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_layer_barrier_key(key)

    async def _synchronize_attention_layer(self, key: tuple[int, int]) -> int | None:
        if self.attention_layer_barrier_window_s <= 0:
            return None
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        batch = self._attention_layer_barrier_batches.setdefault(key, [])
        batch.append(future)
        target_size = self._attention_layer_barrier_target_size()
        if len(batch) >= target_size:
            task = self._attention_layer_barrier_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_layer_barrier_key(key)
        elif key not in self._attention_layer_barrier_tasks:
            self._attention_layer_barrier_tasks[key] = asyncio.create_task(
                self._flush_attention_layer_barrier_key(key)
            )
        return await future

    async def _flush_attention_wavefront_key(self, key: tuple[int, int]):
        try:
            if self.attention_batch_window_s > 0:
                await asyncio.sleep(self.attention_batch_window_s)
            await self._execute_attention_wavefront_key(key)
        except asyncio.CancelledError:
            return

    def _pop_attention_wavefront_bundle(
        self, seed_key: tuple[int, int] | tuple[int, int, str]
    ) -> list[tuple[tuple[int, int] | tuple[int, int, str], list[tuple[dict, asyncio.Future, object]]]]:
        seed_batch = self._attention_wavefront_batches.pop(seed_key, [])
        self._attention_wavefront_tasks.pop(seed_key, None)
        self._attention_wavefront_expected_sizes.pop(seed_key, None)
        if not seed_batch:
            return []

        bundle = [(seed_key, seed_batch)]
        if not self.attention_rpc_cross_key_batch_enabled:
            return bundle

        attention = seed_batch[0][2]
        merge_keys = []
        for key, batch in self._attention_wavefront_batches.items():
            if batch and batch[0][2] is attention:
                merge_keys.append(key)

        for key in merge_keys:
            batch = self._attention_wavefront_batches.pop(key, [])
            self._attention_wavefront_tasks.pop(key, None)
            self._attention_wavefront_expected_sizes.pop(key, None)
            if batch:
                bundle.append((key, batch))
        return bundle

    async def _execute_attention_wavefront_key(self, key: tuple[int, int]):
        bundle = self._pop_attention_wavefront_bundle(key)
        if not bundle:
            return
        payloads = []
        futures = []
        attention = bundle[0][1][0][2]
        for _, batch in bundle:
            payloads.extend(item[0] for item in batch)
            futures.extend(item[1] for item in batch)
        self.attention_batch_flushes += 1
        self.attention_batch_total_items += len(payloads)
        self.attention_batch_max_observed = max(self.attention_batch_max_observed, len(payloads))
        self.attention_batch_total_keys += len(bundle)
        self.attention_batch_max_keys_observed = max(
            self.attention_batch_max_keys_observed,
            len(bundle),
        )
        if len(bundle) > 1:
            self.attention_batch_multi_key_flushes += 1
        try:
            results = await attention.decode_layer_batch.remote(payloads)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)

    async def _maybe_flush_attention_wavefronts(self):
        ready_keys = []
        default_target_size = self._attention_batch_target_size()
        for key, batch in self._attention_wavefront_batches.items():
            target_size = self._attention_wavefront_expected_sizes.get(key, default_target_size)
            if len(batch) >= target_size:
                ready_keys.append(key)
        for key in ready_keys:
            task = self._attention_wavefront_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_wavefront_key(key)

    async def _batched_attention_decode(
        self,
        attention,
        prepared,
        decode_step: int,
        decode_wave: dict[str, object] | None = None,
    ):
        if self.attention_actor_side_batching_enabled:
            key: tuple[int, int] | tuple[int, int, str]
            if self.attention_decode_wave_persist_enabled and decode_wave is not None:
                cohort_id = str(decode_wave.get("cohort_id", "default"))
                key = (int(decode_step), int(prepared["layer_idx"]), cohort_id)
            else:
                key = (int(decode_step), int(prepared["layer_idx"]))
            await self._synchronize_attention_layer(key)
            return await attention.decode_layer.remote(prepared)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        key: tuple[int, int] | tuple[int, int, str]
        cohort_size = None
        if self.attention_decode_wave_persist_enabled and decode_wave is not None:
            cohort_id = str(decode_wave.get("cohort_id", "default"))
            if "group_size" in decode_wave:
                cohort_size = max(1, int(decode_wave["group_size"]))
            key = (int(decode_step), int(prepared["layer_idx"]), cohort_id)
        else:
            key = (int(decode_step), int(prepared["layer_idx"]))
        barrier_group_size = await self._synchronize_attention_layer(key)
        batch = self._attention_wavefront_batches.setdefault(key, [])
        batch.append((prepared, future, attention))
        if cohort_size is not None:
            current_expected = self._attention_wavefront_expected_sizes.get(key, 1)
            self._attention_wavefront_expected_sizes[key] = max(current_expected, cohort_size)
        if barrier_group_size is not None:
            current_expected = self._attention_wavefront_expected_sizes.get(key, 1)
            self._attention_wavefront_expected_sizes[key] = max(current_expected, int(barrier_group_size))
        target_size = self._attention_wavefront_expected_sizes.get(
            key, self._attention_batch_target_size()
        )
        if len(batch) >= target_size:
            task = self._attention_wavefront_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_wavefront_key(key)
        elif key not in self._attention_wavefront_tasks:
                self._attention_wavefront_tasks[key] = asyncio.create_task(
                    self._flush_attention_wavefront_key(key)
                )
        return await future

    async def _run_decode_wave(self, batch: List[Dict[str, object]]):
        if not batch:
            return

        attention = self.attention_nodes[0]
        dense = self.decode_dense_nodes[0]
        rankset_overlap_plan = self._plan_rankset_overlap_batch(batch)
        positions = [int(state["prompt_len"]) + int(state["step"]) - 1 for state in batch]
        token_ids = [int(state["current_token"]) for state in batch]
        request_ids = [str(state["request_id"]) for state in batch]

        rpc_started = time.perf_counter()
        start_token_result = await dense.start_token_batch.remote(token_ids, positions)
        start_rpc_s = time.perf_counter() - rpc_started
        hidden_states = [
            start_token_result["hidden"][idx : idx + 1]
            for idx in range(len(batch))
        ]
        start_compute_s = float(start_token_result.get("profile", {}).get("compute_s", 0.0))
        per_item_start_compute_s = start_compute_s / max(len(batch), 1)
        for state in batch:
            state["stage_timing"]["counts"]["decode_steps"] += 1
            state["stage_timing"]["scheduler"]["start_token_rpc_s"] += start_rpc_s / max(len(batch), 1)
            state["stage_timing"]["actors"]["dense_start_token_compute_s"] += per_item_start_compute_s

        for layer_idx in range(self.runtime_model_spec["num_layers"]):
            context_lens = [int(state["prompt_len"]) + int(state["step"]) for state in batch]
            rankset_task_graph = self._build_rankset_task_graph(batch, layer_idx, rankset_overlap_plan)

            rpc_started = time.perf_counter()
            prepared_items = await dense.prepare_attention_batch.remote(
                hidden_states,
                layer_idx,
                request_ids,
                context_lens,
            )
            prepare_rpc_s = time.perf_counter() - rpc_started
            for state, prepared in zip(batch, prepared_items):
                prepared["rankset_overlap_plan"] = dict(rankset_overlap_plan)
                prepared["rankset_task_graph"] = dict(rankset_task_graph)
                prepared["request_rankset_plan"] = list(state.get("packing_rankset_plan", []) or [])
                state["stage_timing"]["counts"]["decode_layers"] += 1
                state["stage_timing"]["scheduler"]["prepare_attention_rpc_s"] += (
                    prepare_rpc_s / max(len(batch), 1)
                )
                state["stage_timing"]["actors"]["dense_prepare_attention_compute_s"] += float(
                    prepared.get("profile", {}).get("compute_s", 0.0)
                )

            rpc_started = time.perf_counter()
            attention_results = await attention.decode_layer_batch.remote(prepared_items)
            attention_rpc_s = time.perf_counter() - rpc_started
            contexts = []
            for state, result in zip(batch, attention_results):
                state["stage_timing"]["scheduler"]["attention_decode_rpc_s"] += (
                    attention_rpc_s / max(len(batch), 1)
                )
                state["stage_timing"]["actors"]["attention_decode_compute_s"] += float(
                    result.get("profile", {}).get("compute_s", 0.0)
                )
                self._record_rankset_execution(
                    state,
                    dict(result.get("rankset_execution", {}) or {}),
                )
                contexts.append(result["context"])

            residuals = [prepared["residual"] for prepared in prepared_items]
            rpc_started = time.perf_counter()
            finish_results = await dense.finish_layer_batch.remote(residuals, contexts, layer_idx)
            finish_rpc_s = time.perf_counter() - rpc_started
            next_hidden_states = []
            for state, result in zip(batch, finish_results):
                state["stage_timing"]["scheduler"]["finish_layer_rpc_s"] += (
                    finish_rpc_s / max(len(batch), 1)
                )
                state["stage_timing"]["actors"]["dense_finish_layer_compute_s"] += float(
                    result.get("profile", {}).get("compute_s", 0.0)
                )
                hidden = result["hidden"]
                if isinstance(hidden, list):
                    if len(hidden) != 1:
                        raise ValueError(
                            f"finish_layer_batch returned unexpected hidden list length={len(hidden)}"
                        )
                    hidden = hidden[0]
                next_hidden_states.append(hidden)
            hidden_states = next_hidden_states

        rpc_started = time.perf_counter()
        sample_results = await dense.sample_next_token_batch.remote(hidden_states)
        sample_rpc_s = time.perf_counter() - rpc_started

        for state, result in zip(batch, sample_results):
            state["stage_timing"]["scheduler"]["sample_next_token_rpc_s"] += (
                sample_rpc_s / max(len(batch), 1)
            )
            state["stage_timing"]["actors"]["dense_sample_next_token_compute_s"] += float(
                result.get("profile", {}).get("compute_s", 0.0)
            )
            next_token = int(result["token_id"])
            state["generated_ids"].append(next_token)
            state["current_token"] = next_token
            state["step"] = int(state["step"]) + 1

        requeue_states = []
        for state in batch:
            next_token = int(state["current_token"])
            max_tokens = int(state["max_tokens"])
            total_tokens = len(state["generated_ids"])
            dense_step_s = (
                float(state["stage_timing"]["scheduler"]["start_token_rpc_s"])
                + float(state["stage_timing"]["actors"]["dense_start_token_compute_s"])
                + float(state["stage_timing"]["scheduler"]["prepare_attention_rpc_s"])
                + float(state["stage_timing"]["actors"]["dense_prepare_attention_compute_s"])
                + float(state["stage_timing"]["scheduler"]["finish_layer_rpc_s"])
                + float(state["stage_timing"]["actors"]["dense_finish_layer_compute_s"])
                + float(state["stage_timing"]["scheduler"]["sample_next_token_rpc_s"])
                + float(state["stage_timing"]["actors"]["dense_sample_next_token_compute_s"])
            ) - float(state.get("_predictive_dense_accum_s", 0.0))
            attention_step_s = (
                float(state["stage_timing"]["scheduler"]["attention_decode_rpc_s"])
                + float(state["stage_timing"]["actors"]["attention_decode_compute_s"])
            ) - float(state.get("_predictive_attention_accum_s", 0.0))
            state["_predictive_dense_accum_s"] = (
                float(state.get("_predictive_dense_accum_s", 0.0)) + max(0.0, dense_step_s)
            )
            state["_predictive_attention_accum_s"] = (
                float(state.get("_predictive_attention_accum_s", 0.0)) + max(0.0, attention_step_s)
            )
            if self.clover_predictive_scheduling_enabled:
                self._update_predictive_component("dense", state, dense_step_s)
                self._update_predictive_component("attention", state, attention_step_s)
                self.predictive_model_updates += 1
            if next_token == 2 or total_tokens >= max_tokens:
                self._track_background_completion(self._complete_decode_state(state))
            else:
                requeue_states.append(state)

        for state in requeue_states:
            self._decode_pending_queue.append(state)

    async def _fail_decode_state(self, state: Dict[str, object], exc: Exception):
        if state.get("pending_free", False):
            return
        state["pending_free"] = True
        try:
            await self.attention_nodes[0].free_request.remote(str(state["request_id"]))
        finally:
            self._inflight_request_count = max(0, self._inflight_request_count - 1)
            self._active_decode_requests = max(0, self._active_decode_requests - 1)
            future = state.get("completion_future")
            if future is not None and not future.done():
                future.set_exception(exc)

    async def _complete_decode_state(self, state: Dict[str, object]):
        if state.get("pending_free", False):
            return
        state["pending_free"] = True

        attention = self.attention_nodes[0]
        dense = self.decode_dense_nodes[0]
        request_id = str(state["request_id"])
        stage_timing = state["stage_timing"]
        return_metrics = bool(state["return_metrics"])

        attention_debug_before_free_ref = None
        attention_debug_after_free_ref = None
        attention_free_ref = None
        if return_metrics:
            attention_debug_before_free_ref = attention.get_info.remote()
            free_started = time.perf_counter()
            attention_free_ref = attention.free_request.remote(request_id)
            attention_debug_after_free_ref = attention.get_info.remote()
        else:
            free_started = time.perf_counter()
            attention_free_ref = attention.free_request.remote(request_id)

        decode_started = time.perf_counter()
        decode_result_ref = dense.decode_tokens.remote(state["generated_ids"])

        attention_debug_before_free = None
        if attention_debug_before_free_ref is not None:
            attention_debug_before_free = await attention_debug_before_free_ref

        await attention_free_ref
        stage_timing["scheduler"]["free_request_rpc_s"] += time.perf_counter() - free_started

        decode_result = await decode_result_ref
        stage_timing["scheduler"]["decode_tokens_rpc_s"] += time.perf_counter() - decode_started
        stage_timing["actors"]["dense_decode_tokens_compute_s"] += float(
            decode_result.get("profile", {}).get("compute_s", 0.0)
        )
        generated_text = decode_result["text"]

        self._inflight_request_count = max(0, self._inflight_request_count - 1)
        self._active_decode_requests = max(0, self._active_decode_requests - 1)
        await self._maybe_flush_decode_step_syncs()
        await self._maybe_flush_attention_layer_barriers()
        await self._maybe_flush_attention_wavefronts()

        future = state.get("completion_future")
        if future is None or future.done():
            return

        if return_metrics:
            request_end = time.time()
            latency = request_end - float(state["request_start"])
            total_tokens = len(state["generated_ids"])
            ttft = float(state["first_token_time"]) - float(state["request_start"])
            tpot = (latency - ttft) / max(total_tokens - 1, 1)
            throughput = total_tokens / latency if latency > 0 else 0.0
            metrics = {
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "throughput": throughput,
                "total_tokens": total_tokens,
            }
            scheduler_rpc_total = sum(stage_timing["scheduler"].values())
            actor_compute_total = sum(stage_timing["actors"].values())
            metrics["stage_timing"] = stage_timing
            metrics["stage_timing"]["scheduler"]["total_rpc_s"] = scheduler_rpc_total
            metrics["stage_timing"]["actors"]["total_compute_s"] = actor_compute_total
            metrics["stage_timing"]["scheduler_overhead_s"] = max(0.0, float(latency - scheduler_rpc_total))
            metrics["scheduler_attention_batching"] = {
                "window_s": float(self.attention_batch_window_s),
                "max_size": int(self.attention_batch_max_size),
                "cross_key_batch_enabled": bool(self.attention_rpc_cross_key_batch_enabled),
                "actor_side_batching_enabled": bool(self.attention_actor_side_batching_enabled),
                "flushes": int(self.attention_batch_flushes),
                "total_items": int(self.attention_batch_total_items),
                "max_observed_size": int(self.attention_batch_max_observed),
                "multi_key_flushes": int(self.attention_batch_multi_key_flushes),
                "total_keys": int(self.attention_batch_total_keys),
                "max_keys_observed": int(self.attention_batch_max_keys_observed),
                "pending": sum(len(batch) for batch in self._attention_wavefront_batches.values()),
            }
            metrics["scheduler_decode_step_sync"] = {
                "window_s": float(self.decode_step_sync_window_s),
                "max_size": int(self.decode_step_sync_max_size),
                "flushes": int(self.decode_step_sync_flushes),
                "total_items": int(self.decode_step_sync_total_items),
                "max_observed_size": int(self.decode_step_sync_max_observed),
                "pending": sum(len(batch) for batch in self._decode_step_sync_batches.values()),
            }
            metrics["scheduler_attention_layer_barrier"] = {
                "window_s": float(self.attention_layer_barrier_window_s),
                "max_size": int(self.attention_layer_barrier_max_size),
                "flushes": int(self.attention_layer_barrier_flushes),
                "total_items": int(self.attention_layer_barrier_total_items),
                "max_observed_size": int(self.attention_layer_barrier_max_observed),
                "pending": sum(len(batch) for batch in self._attention_layer_barrier_batches.values()),
            }
            metrics["scheduler_dense_continuous_batching"] = {
                "window_s": float(self.decode_continuous_batch_window_s),
                "startup_grace_s": float(self.decode_continuous_batch_startup_grace_s),
                "max_size": int(self.decode_continuous_batch_max_size),
                "flushes": int(self.decode_continuous_batch_flushes),
                "total_items": int(self.decode_continuous_batch_total_items),
                "max_observed_size": int(self.decode_continuous_batch_max_observed),
                "size_histogram": {
                    str(size): int(count)
                    for size, count in sorted(self.decode_continuous_batch_size_histogram.items())
                },
                "target_flushes": int(self.decode_continuous_batch_target_flushes),
                "window_flushes": int(self.decode_continuous_batch_window_flushes),
                "immediate_flushes": int(self.decode_continuous_batch_immediate_flushes),
                "wait_s": float(self.decode_continuous_batch_wait_s),
                "reordered_flushes": int(self.decode_continuous_batch_reordered_flushes),
                "same_rank_flushes": int(self.decode_continuous_batch_same_rank_flushes),
                "mixed_rank_flushes": int(self.decode_continuous_batch_mixed_rank_flushes),
                "avg_context_span": (
                    float(self.decode_continuous_batch_total_context_span) / float(self.decode_continuous_batch_flushes)
                    if self.decode_continuous_batch_flushes > 0
                    else 0.0
                ),
                "disjoint_stripe_packing_enabled": bool(
                    self.clover_pim_disjoint_decode_stripe_packing_enabled
                ),
                "disjoint_stripe_decisions": int(
                    self.decode_continuous_batch_disjoint_stripe_decisions
                ),
                "stripe_overlap_flushes": int(self.decode_continuous_batch_stripe_overlap_flushes),
                "avg_stripe_pair_overlap": (
                    float(self.decode_continuous_batch_stripe_overlap_total)
                    / float(self.decode_continuous_batch_stripe_overlap_flushes)
                    if self.decode_continuous_batch_stripe_overlap_flushes > 0
                    else 0.0
                ),
                "max_stripe_pair_overlap": int(self.decode_continuous_batch_stripe_overlap_max),
                "last_stripe_overlap": dict(self.decode_continuous_batch_last_stripe_overlap),
                "target_size": int(self._decode_continuous_batch_target_size()),
                "pending": len(self._decode_pending_queue),
                "inflight_target_enabled": bool(self.decode_continuous_batch_inflight_target_enabled),
                "predictive_enabled": bool(self.clover_predictive_scheduling_enabled),
                "predictive_batch_decisions": int(self.predictive_batch_decisions),
                "predictive_batch_fallbacks": int(self.predictive_batch_fallbacks),
                "predictive_model_updates": int(self.predictive_model_updates),
                "predictive_ready_requests": int(self.predictive_ready_requests),
                "predictive_unready_requests": int(self.predictive_unready_requests),
                "predictive_context_bucket_tokens": int(self.clover_predictive_scheduling_context_bucket_tokens),
                "predictive_min_samples": int(self.clover_predictive_scheduling_min_samples),
                "predictive_alpha": float(self.clover_predictive_scheduling_alpha),
                "predictive_last_batch": dict(self.predictive_last_batch),
                "capacity_aware_enabled": bool(self.clover_capacity_aware_batching_enabled),
                "capacity_aware_decisions": int(self.capacity_aware_batch_decisions),
                "capacity_aware_fallbacks": int(self.capacity_aware_batch_fallbacks),
                "capacity_aware_time_gap_threshold": float(self.clover_capacity_aware_time_gap_threshold),
                "capacity_aware_lookahead_window": int(self.clover_capacity_aware_lookahead_window),
                "capacity_aware_max_tokens_per_dpu": int(self.clover_capacity_aware_max_tokens_per_dpu),
                "capacity_aware_effective_max_tokens_per_dpu": int(self._capacity_aware_effective_max_tokens_per_dpu()),
                "capacity_aware_require_slot_headroom": bool(self.clover_capacity_aware_require_slot_headroom),
                "capacity_aware_allocator_stats_cached": int(len(self._capacity_aware_allocator_stats_cache)),
                "capacity_aware_allocator_stats_refreshes": int(self._capacity_aware_allocator_stats_refreshes),
                "capacity_aware_last_batch": dict(self.capacity_aware_last_batch),
                "planner_modes": sorted(
                    {
                        str(state.get("packing_planner_mode", "") or "")
                        for state in list(self._decode_pending_queue)
                        if str(state.get("packing_planner_mode", "") or "")
                    }
                ),
            }
            metrics["scheduler_rankset_overlap"] = {
                "enabled": bool(self.clover_rankset_overlap_enabled),
                "transfer_granularity": str(self.clover_rankset_overlap_transfer_granularity),
                "max_ranksets_per_batch": int(self.clover_rankset_overlap_max_ranksets_per_batch),
                "plan_batches": int(self.rankset_overlap_plan_batches),
                "plan_items": int(self.rankset_overlap_plan_items),
                "plan_ranksets": int(self.rankset_overlap_plan_ranksets),
                "plan_max_ranksets": int(self.rankset_overlap_plan_max_ranksets),
                "same_rankset_batches": int(self.rankset_overlap_plan_same_rankset_batches),
                "mixed_rankset_batches": int(self.rankset_overlap_plan_mixed_rankset_batches),
                "task_graph_layers": int(self.rankset_overlap_task_graph_layers),
                "task_graph_work_items": int(self.rankset_overlap_task_graph_work_items),
                "task_graph_max_work_items": int(self.rankset_overlap_task_graph_max_work_items),
                "partial_ready_reports": int(self.rankset_overlap_partial_ready_reports),
                "partial_ready_work_items": int(self.rankset_overlap_partial_ready_work_items),
                "partial_ready_max_work_items": int(self.rankset_overlap_partial_ready_max_work_items),
                "avg_first_transfer_ready_s": (
                    float(self.rankset_overlap_first_transfer_ready_s_total)
                    / float(self.rankset_overlap_first_transfer_ready_s_count)
                    if self.rankset_overlap_first_transfer_ready_s_count > 0
                    else 0.0
                ),
                "avg_last_transfer_ready_s": (
                    float(self.rankset_overlap_last_transfer_ready_s_total)
                    / float(self.rankset_overlap_last_transfer_ready_s_count)
                    if self.rankset_overlap_last_transfer_ready_s_count > 0
                    else 0.0
                ),
                "avg_transfer_ready_span_s": (
                    float(self.rankset_overlap_transfer_span_s_total)
                    / float(self.rankset_overlap_transfer_span_s_count)
                    if self.rankset_overlap_transfer_span_s_count > 0
                    else 0.0
                ),
                "avg_transfer_duration_s": (
                    float(self.rankset_overlap_transfer_duration_s_total)
                    / float(self.rankset_overlap_transfer_duration_s_count)
                    if self.rankset_overlap_transfer_duration_s_count > 0
                    else 0.0
                ),
                "avg_first_work_item_completion_s": (
                    float(self.rankset_overlap_first_work_item_completion_s_total)
                    / float(self.rankset_overlap_first_work_item_completion_s_count)
                    if self.rankset_overlap_first_work_item_completion_s_count > 0
                    else 0.0
                ),
                "avg_last_work_item_completion_s": (
                    float(self.rankset_overlap_last_work_item_completion_s_total)
                    / float(self.rankset_overlap_last_work_item_completion_s_count)
                    if self.rankset_overlap_last_work_item_completion_s_count > 0
                    else 0.0
                ),
                "avg_work_item_completion_span_s": (
                    float(self.rankset_overlap_work_item_span_s_total)
                    / float(self.rankset_overlap_work_item_span_s_count)
                    if self.rankset_overlap_work_item_span_s_count > 0
                    else 0.0
                ),
                "avg_work_item_duration_s": (
                    float(self.rankset_overlap_work_item_duration_s_total)
                    / float(self.rankset_overlap_work_item_duration_s_count)
                    if self.rankset_overlap_work_item_duration_s_count > 0
                    else 0.0
                ),
                "avg_ranksets_per_batch": (
                    float(self.rankset_overlap_plan_ranksets) / float(self.rankset_overlap_plan_batches)
                    if self.rankset_overlap_plan_batches > 0
                    else 0.0
                ),
                "last_batch": dict(self.rankset_overlap_last_batch),
                "last_task_graph": dict(self.rankset_overlap_last_task_graph),
                "last_execution_summary": dict(self.rankset_overlap_last_execution_summary),
                "last_execution": dict(state.get("last_rankset_execution", {}) or {}),
            }
            metrics["attention_backend_before_free"] = attention_debug_before_free
            metrics["attention_backend"] = (
                await attention_debug_after_free_ref
                if attention_debug_after_free_ref is not None
                else await attention.get_info.remote()
            )
            future.set_result((generated_text, metrics))
            return

        future.set_result(generated_text)

    async def initialize_cluster(self):
        gpu_prefill = (
            float(self.cluster_config.prefill_gpu_fraction)
            if self.cluster_config.use_gpu_for_prefill
            else 0.0
        )
        gpu_dense = (
            float(self.cluster_config.decode_dense_gpu_fraction)
            if self.cluster_config.use_gpu_for_decode_dense
            else 0.0
        )
        gpu_attention = (
            float(getattr(self.cluster_config, "attention_gpu_fraction", 0.0))
            if getattr(self.cluster_config, "use_gpu_for_attention", False)
            else 0.0
        )
        if gpu_prefill < 0 or gpu_dense < 0 or gpu_attention < 0:
            raise ValueError("GPU fractions must be non-negative")
        cluster_resources = ray.cluster_resources()
        available_gpus = float(cluster_resources.get("GPU", 0.0))
        requested_gpus = (
            gpu_prefill * int(self.cluster_config.num_prefill_workers)
            + gpu_dense * int(self.cluster_config.num_decode_dense_nodes)
            + gpu_attention * int(self.cluster_config.num_attention_nodes)
        )
        if requested_gpus > available_gpus + 1e-6:
            raise RuntimeError(
                "Requested GPU actors exceed visible Ray GPU capacity: "
                f"requested={requested_gpus}, available={available_gpus}. "
                "Adjust --prefill-gpu-fraction / --decode-dense-gpu-fraction or disable one side's GPU."
            )
        attention_backend_kwargs = {
            "attention_sparse_window": int(getattr(self.cluster_config, "attention_sparse_window", 0)),
        }
        if self.cluster_config.attention_backend in {"pim_naive", "cloverinfer"}:
            attention_backend_kwargs.update({
                "num_dpus": int(self.cluster_config.pim_num_dpus),
                "length": int(self.cluster_config.pim_length),
                "block_tokens": int(self.cluster_config.pim_block_tokens),
                "resident_store_backend": str(self.cluster_config.pim_resident_store_backend),
                "max_resident_groups_per_layer": int(self.cluster_config.pim_max_resident_groups_per_layer),
                "head_grouping_policy": str(self.cluster_config.pim_head_grouping_policy),
                "dpu_placement_policy": str(self.cluster_config.pim_dpu_placement_policy),
                "resident_kv_dtype": str(self.cluster_config.pim_resident_kv_dtype),
                "qk_full_enabled": bool(self.cluster_config.pim_qk_full_enabled),
                "qk_full_shadow_check": bool(self.cluster_config.pim_qk_full_shadow_check),
                "softmax_av_fused_enabled": bool(self.cluster_config.pim_softmax_av_fused_enabled),
                "softmax_av_shadow_check": bool(self.cluster_config.pim_softmax_av_shadow_check),
                "qk_mixed_enabled": bool(self.cluster_config.pim_qk_mixed_enabled),
                "qk_mixed_heads": int(self.cluster_config.pim_qk_mixed_heads),
                "qk_mixed_window": int(self.cluster_config.pim_qk_mixed_window),
                "decode_batch_window_s": float(self.cluster_config.attention_actor_batch_window_s),
                "decode_batch_max_size": int(self.cluster_config.attention_actor_batch_max_size),
                "expected_decode_batch_max_size": int(self.cluster_config.decode_continuous_batch_max_size),
            })
            if self.cluster_config.attention_backend == "cloverinfer":
                attention_backend_kwargs.update(
                    {
                        "cpu_shadow_enabled": bool(self.cluster_config.clover_cpu_shadow_enabled),
                        "shadow_checks_enabled": bool(self.cluster_config.clover_shadow_checks_enabled),
                        "op_profiling_enabled": bool(self.cluster_config.clover_op_profiling_enabled),
                        "cpu_fast_path_max_context_tokens": int(
                            self.cluster_config.clover_cpu_fast_path_max_context_tokens
                        ),
                        "adaptive_routing_enabled": bool(
                            getattr(self.cluster_config, "clover_adaptive_routing_enabled", False)
                        ),
                        "adaptive_route_compressed_kv_to_cpu": bool(
                            getattr(
                                self.cluster_config,
                                "clover_adaptive_route_compressed_kv_to_cpu",
                                True,
                            )
                        ),
                        "adaptive_route_sparse_window_max": int(
                            getattr(self.cluster_config, "clover_adaptive_route_sparse_window_max", 0)
                        ),
                        "adaptive_route_context_len_max": int(
                            getattr(self.cluster_config, "clover_adaptive_route_context_len_max", 0)
                        ),
                        "adaptive_probe_enabled": bool(
                            getattr(self.cluster_config, "clover_adaptive_probe_enabled", False)
                        ),
                        "shadow_check_token_interval": int(self.cluster_config.clover_shadow_check_token_interval),
                        "shadow_check_layer_interval": int(self.cluster_config.clover_shadow_check_layer_interval),
                        "host_qk_mixed_enabled": bool(self.cluster_config.clover_host_qk_mixed_enabled),
                        "pim_attention_enabled": bool(self.cluster_config.clover_pim_attention_enabled),
                        "pim_context_fused_experimental_enabled": bool(
                            self.cluster_config.clover_pim_context_fused_experimental_enabled
                        ),
                        "pim_qk_only_host_av_experimental_enabled": bool(
                            getattr(
                                self.cluster_config,
                                "clover_pim_qk_only_host_av_experimental_enabled",
                                False,
                            )
                        ),
                        "pim_rank_spread_alloc_experimental_enabled": bool(
                            self.cluster_config.clover_pim_rank_spread_alloc_experimental_enabled
                        ),
                        "pim_cross_rank_stripe_experimental_enabled": bool(
                            self.cluster_config.clover_pim_cross_rank_stripe_experimental_enabled
                        ),
                        "pim_rank_spread_multi_rank_batch_experimental_enabled": bool(
                            self.cluster_config.clover_pim_rank_spread_multi_rank_batch_experimental_enabled
                        ),
                        "pim_layer_rank_rotation_experimental_enabled": bool(
                            self.cluster_config.clover_pim_layer_rank_rotation_experimental_enabled
                        ),
                        "pim_slot_spill_alloc_experimental_enabled": bool(
                            self.cluster_config.clover_pim_slot_spill_alloc_experimental_enabled
                        ),
                        "pim_slot_pressure_aware_alloc_experimental_enabled": bool(
                            self.cluster_config.clover_pim_slot_pressure_aware_alloc_experimental_enabled
                        ),
                        "pim_emergency_slot_spill_experimental_enabled": bool(
                            self.cluster_config.clover_pim_emergency_slot_spill_experimental_enabled
                        ),
                        "pim_reserve_segment_tail_capacity_experimental_enabled": bool(
                            self.cluster_config.clover_pim_reserve_segment_tail_capacity_experimental_enabled
                        ),
                        "pim_reserve_segment_tail_capacity_tokens": int(
                            getattr(
                                self.cluster_config,
                                "clover_pim_reserve_segment_tail_capacity_tokens",
                                0,
                            )
                        ),
                        "pim_perf_guard_enabled": bool(
                            getattr(self.cluster_config, "clover_pim_perf_guard_enabled", False)
                        ),
                        "pim_perf_guard_force_cpu_for_compressed_kv": bool(
                            getattr(
                                self.cluster_config,
                                "clover_pim_perf_guard_force_cpu_for_compressed_kv",
                                True,
                            )
                        ),
                        "pim_perf_guard_min_decode_items": int(
                            getattr(self.cluster_config, "clover_pim_perf_guard_min_decode_items", 1)
                        ),
                        "pim_perf_guard_slowdown_threshold": float(
                            getattr(self.cluster_config, "clover_pim_perf_guard_slowdown_threshold", 1.2)
                        ),
                        "compact_short_segments_enabled": bool(
                            self.cluster_config.clover_compact_short_segments_enabled
                        ),
                        "compact_short_segment_min_tokens": int(
                            self.cluster_config.clover_compact_short_segment_min_tokens
                        ),
                        "fine_head_grouping_experimental_enabled": bool(
                            self.cluster_config.clover_fine_head_grouping_experimental_enabled
                        ),
                        "target_heads_per_group_experimental": int(
                            self.cluster_config.clover_target_heads_per_group_experimental
                        ),
                        "host_partial_reduce_enabled": bool(
                            getattr(self.cluster_config, "clover_host_partial_reduce_enabled", True)
                        ),
                        "rankset_overlap_async_dispatch_enabled": bool(
                            getattr(self.cluster_config, "clover_rankset_overlap_async_dispatch_enabled", False)
                        ),
                        "rankset_overlap_transfer_latency_s": float(
                            getattr(self.cluster_config, "clover_rankset_overlap_transfer_latency_s", 0.0)
                        ),
                    }
                )

        self.prefill_nodes = [
            PrefillNode.options(
                **_actor_options(self.cluster_config.prefill_resource, gpu_prefill)
            ).remote(i, self.model_config, self.cluster_config.use_gpu_for_prefill)
            for i in range(self.cluster_config.num_prefill_workers)
        ]

        self.attention_nodes = [
            AttentionNode.options(
                **_actor_options(self.cluster_config.attention_resource, gpu_attention)
            ).remote(
                i,
                self.model_config,
                self.cluster_config.attention_backend,
                attention_backend_kwargs,
                bool(getattr(self.cluster_config, "use_gpu_for_attention", False)),
            )
            for i in range(self.cluster_config.num_attention_nodes)
        ]

        self.decode_dense_nodes = [
            DecodeDenseNode.options(
                **_actor_options(self.cluster_config.decode_dense_resource, gpu_dense)
            ).remote(i, self.model_config, self.cluster_config.use_gpu_for_decode_dense)
            for i in range(self.cluster_config.num_decode_dense_nodes)
        ]

        infos = {
            "prefill": await self.prefill_nodes[0].get_info.remote(),
            "attention": await self.attention_nodes[0].get_info.remote(),
            "decode_dense": await self.decode_dense_nodes[0].get_info.remote(),
        }
        self.runtime_model_spec = await self.decode_dense_nodes[0].get_model_spec.remote()
        print(f"Cluster initialized: {infos}")
        return infos

    async def submit_request(self, prompt: str, return_metrics: bool = False, max_new_tokens: int | None = None):
        self._inflight_request_count += 1
        request_start = time.time()
        prefill = self.prefill_nodes[0]
        attention = self.attention_nodes[0]

        rpc_started = time.perf_counter()
        prefill_out = await prefill.process_prompt.remote(prompt)
        prefill_rpc_s = time.perf_counter() - rpc_started
        first_token_time = time.time()
        request_id = prefill_out["request_id"]
        prompt_len = int(prefill_out["prompt_len"])
        first_token = int(prefill_out["first_token_id"])
        max_tokens = int(max_new_tokens or self.model_config.max_new_tokens)
        expected_decode_batch_size = max(
            1,
            min(
                max(int(self._inflight_request_count), int(self._active_decode_requests) + 1),
                int(self.decode_continuous_batch_max_size),
            ),
        )
        rpc_started = time.perf_counter()
        init_result = await attention.init_request.remote(
            request_id,
            prefill_out["initial_kv"],
            max_tokens,
            expected_decode_batch_size,
        )
        decode_state = self._new_decode_state(
            request_id=request_id,
            prompt_len=prompt_len,
            first_token=first_token,
            first_token_time=first_token_time,
            max_tokens=max_tokens,
            request_start=request_start,
            return_metrics=return_metrics,
            packing_hint=dict(init_result.get("packing_hint", {}) or {}),
        )
        decode_state["stage_timing"]["scheduler"]["prefill_rpc_s"] += prefill_rpc_s
        decode_state["stage_timing"]["actors"]["prefill_compute_s"] += float(
            prefill_out.get("profile", {}).get("compute_s", 0.0)
        )
        self._active_decode_requests += 1
        decode_state["stage_timing"]["scheduler"]["attention_init_rpc_s"] += time.perf_counter() - rpc_started
        decode_state["stage_timing"]["actors"]["attention_init_compute_s"] += float(
            init_result.get("profile", {}).get("compute_s", 0.0)
        )
        if max_tokens <= 1:
            completion_future = asyncio.get_running_loop().create_future()
            decode_state["completion_future"] = completion_future
            await self._complete_decode_state(decode_state)
            return await completion_future

        completion_future = asyncio.get_running_loop().create_future()
        decode_state["completion_future"] = completion_future
        self._decode_pending_queue.append(decode_state)
        self._ensure_decode_driver()
        return await completion_future
