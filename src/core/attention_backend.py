from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
import struct
import subprocess
from typing import Dict, List, Sequence, Tuple

import torch

from .clover_planner import plan_sharding
from .resident_kv_store import HostResidentKVStore, UpmemKVSlotStore


@dataclass
class HeadGroupState:
    dpu_id: int
    head_start: int
    head_end: int
    seq_len: int
    capacity: int
    head_dim: int
    k_slot: str
    v_slot: str
    physical_dpus: List[int] | None = None
    token_segments: List[Dict[str, int]] | None = None

    @property
    def group_heads(self) -> int:
        return int(self.head_end - self.head_start)

    @property
    def live_elems(self) -> int:
        return int(self.seq_len) * self.group_heads * int(self.head_dim)

    @property
    def capacity_elems(self) -> int:
        return int(self.capacity) * self.group_heads * int(self.head_dim)


@dataclass
class LayerState:
    layer_idx: int
    num_heads: int
    head_dim: int
    head_groups: List[HeadGroupState]


@dataclass
class RequestState:
    request_id: str
    context_len: int
    num_layers: int
    layer_states: List[LayerState]
    preferred_dpu_stripe: List[int]
    sharding_plan: Dict[str, object] | None = None
    stripe_version: int = 0
    stripe_expand_count: int = 0
    last_stripe_update_reason: str = ""
    last_stripe_width: int = 0


class CpuAttentionBackend:
    """Reference attention backend for the CPU/PIM node.

    The dense node sends already-scaled OPT queries, so this backend does not
    apply another 1/sqrt(head_dim) factor.
    """

    def __init__(self):
        self.k_cache: Dict[str, List[torch.Tensor]] = {}
        self.v_cache: Dict[str, List[torch.Tensor]] = {}
        self.context_lens: Dict[str, int] = {}

    def init_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> int:
        del decode_reserve_tokens
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
        score_scale: float = 1.0,
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
        scores = torch.einsum("hd,lhd->hl", q.float(), keys.float()) * float(score_scale)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)

        if layer_idx == len(self.k_cache[request_id]) - 1:
            self.context_lens[request_id] += 1

        return context.unsqueeze(0)

    def decode_layer_batch(self, items: List[Dict[str, object]]) -> List[torch.Tensor]:
        return [
            self.decode_layer(
                str(item["request_id"]),
                int(item["layer_idx"]),
                item["query"],
                item["key"],
                item["value"],
                float(item.get("score_scale", 1.0)),
            )
            for item in items
        ]

    def get_context_len(self, request_id: str) -> int:
        return self.context_lens[request_id]

    def free_request(self, request_id: str) -> None:
        self.k_cache.pop(request_id, None)
        self.v_cache.pop(request_id, None)
        self.context_lens.pop(request_id, None)


class GpuAttentionBackend(CpuAttentionBackend):
    """GPU attention backend for AFD-style disaggregated execution.

    This keeps the RPC/workflow separation of the attention node, but executes
    the attention math on CUDA so it can share the decode GPU on the dense node
    machine.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("GpuAttentionBackend requires CUDA, but no GPU is available")
        super().__init__()
        self.device = torch.device("cuda")

    def init_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> int:
        del decode_reserve_tokens
        if request_id in self.k_cache:
            raise ValueError(f"Request {request_id} already exists")

        self.k_cache[request_id] = [layer["key"].detach().to(self.device).contiguous() for layer in initial_kv]
        self.v_cache[request_id] = [layer["value"].detach().to(self.device).contiguous() for layer in initial_kv]

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
        score_scale: float = 1.0,
    ) -> torch.Tensor:
        if request_id not in self.k_cache:
            raise KeyError(f"Unknown request {request_id}")

        q = query.detach().to(self.device).contiguous()
        k_new = key.detach().to(self.device).contiguous()
        v_new = value.detach().to(self.device).contiguous()

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
        scores = torch.einsum("hd,lhd->hl", q.float(), keys.float()) * float(score_scale)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)

        if layer_idx == len(self.k_cache[request_id]) - 1:
            self.context_lens[request_id] += 1

        return context.unsqueeze(0).cpu()


class PimNaiveAttentionBackend:
    """UPMEM-backed backend skeleton.

    Current behavior:
    - validates the UPMEM toolchain path via a smoke test on initialization
    - reuses the CPU reference path for attention correctness

    This keeps the scheduler and node contracts stable while we incrementally
    replace the internals with actual PIM kernels.
    """

    def __init__(
        self,
        repo_root: str | None = None,
        num_dpus: int = 4,
        length: int = 128,
        block_tokens: int = 256,
        resident_store_backend: str = "host",
        max_resident_groups_per_layer: int = 0,
        head_grouping_policy: str = "balanced",
        dpu_placement_policy: str = "rotated",
        resident_kv_dtype: str = "fp32",
        qk_check_interval: int = 1,
        qk_check_limit: int = 1,
        qk_full_enabled: bool = False,
        qk_full_shadow_check: bool = True,
        softmax_av_fused_enabled: bool = False,
        softmax_av_shadow_check: bool = True,
        qk_mixed_enabled: bool = True,
        qk_mixed_heads: int = 2,
        qk_mixed_window: int = 128,
        host_partial_reduce_enabled: bool = True,
        compact_short_segments_enabled: bool = False,
        compact_short_segment_min_tokens: int = 16,
    ):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.num_dpus = num_dpus
        self.length = length
        self.block_tokens = max(1, int(block_tokens))
        self.resident_store_backend = resident_store_backend
        self.max_resident_groups_per_layer = max(0, int(max_resident_groups_per_layer))
        self.head_grouping_policy = "balanced" if str(head_grouping_policy) == "auto" else str(head_grouping_policy)
        self.dpu_placement_policy = "rotated" if str(dpu_placement_policy) == "auto" else str(dpu_placement_policy)
        self.resident_kv_dtype = str(resident_kv_dtype)
        self.qk_check_interval = qk_check_interval
        self.qk_check_limit = qk_check_limit
        if self.head_grouping_policy not in {"legacy", "balanced", "coarse", "segment_aware"}:
            raise ValueError(f"Unsupported head_grouping_policy: {self.head_grouping_policy}")
        if self.dpu_placement_policy not in {"identity", "rotated", "rank_spread", "load_aware"}:
            raise ValueError(f"Unsupported dpu_placement_policy: {self.dpu_placement_policy}")
        if self.resident_kv_dtype not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported resident_kv_dtype: {self.resident_kv_dtype}")
        self.cpu_backend = CpuAttentionBackend()
        self.smoke_test_ok = False
        self.smoke_test_output = ""
        self.qk_check_count = 0
        self.qk_check_failures = 0
        self.qk_last_output = ""
        self.qk_shadow_max_abs_diff = 0
        self.qk_shadow_last_scores = []
        self.qk_full_enabled = bool(qk_full_enabled)
        self.qk_full_shadow_check = bool(qk_full_shadow_check)
        self.qk_full_count = 0
        self.qk_full_batch_calls = 0
        self.qk_full_shadow_checks = 0
        self.qk_full_shadow_max_abs_diff = 0.0
        self.qk_full_shadow_last_max_abs_diff = 0.0
        self.softmax_av_fused_enabled = bool(softmax_av_fused_enabled)
        self.softmax_av_shadow_check = bool(softmax_av_shadow_check)
        self.softmax_av_fused_ops = 0
        self.softmax_av_fused_batch_calls = 0
        self.softmax_av_fused_shadow_max_abs_diff = 0.0
        self.qk_mixed_enabled = bool(qk_mixed_enabled)
        self.qk_mixed_heads = max(0, int(qk_mixed_heads))
        self.qk_mixed_window = max(1, int(qk_mixed_window))
        self.host_partial_reduce_enabled = bool(host_partial_reduce_enabled)
        self.compact_short_segments_enabled = bool(compact_short_segments_enabled)
        self.compact_short_segment_min_tokens = max(1, int(compact_short_segment_min_tokens))
        self.planner_segment_plan_count = 0
        self.planner_segment_materialized_count = 0
        self.planner_segment_compacted_count = 0
        self.planner_segment_last_decision = ""
        self.qk_mixed_count = 0
        self.qk_mixed_last_max_abs_diff = 0.0
        self.qk_mixed_last_head_diffs = []
        self.qk_mixed_last_diag: Dict[str, object] = {}
        self.qk_mixed_last_diag_path = ""
        self.qk_batch_calls = 0
        self.decode_batch_calls = 0
        self.decode_batch_items = 0
        self.qk_helper_started = False
        self.qk_helper_restarts = 0
        self.qk_helper: subprocess.Popen | None = None
        self.resident_metadata_enabled = True
        self.resident_compute_enabled = True
        if resident_store_backend == "upmem_kvslot":
            self.resident_store = UpmemKVSlotStore(
                self.repo_root,
                self.num_dpus,
                kv_dtype=self.resident_kv_dtype,
                block_tokens=self.block_tokens,
                placement_policy=self.dpu_placement_policy,
                host_partial_reduce_enabled=self.host_partial_reduce_enabled,
            )
            self.resident_store.set_experimental_flags(shape_rounds_enabled=True)
        elif resident_store_backend == "host":
            self.resident_store = HostResidentKVStore()
        else:
            raise ValueError(f"Unsupported resident store backend: {resident_store_backend}")
        self.request_states: Dict[str, RequestState] = {}
        self.last_freed_request_id = ""
        self.resident_append_ops = 0
        self.resident_materialize_ops = 0
        self.resident_shadow_max_abs_diff = 0.0
        self.resident_av_enabled = isinstance(self.resident_store, UpmemKVSlotStore)
        self.resident_av_ops = 0
        self.resident_av_batch_calls = 0
        self.resident_av_shadow_max_abs_diff = 0.0
        self.init_rank_locality_reuse_count = 0
        self.init_rank_hash_fallback_count = 0
        self.init_rank_last_reason = ""
        self._run_dot_smoke_test()

    def _shares_persistent_dpu_owner(self) -> bool:
        return isinstance(self.resident_store, UpmemKVSlotStore)

    def _group_footprint_summary(self, group: HeadGroupState) -> Dict[str, object]:
        return {
            "dpu_id": int(group.dpu_id),
            "physical_dpus": [int(physical_dpu) for physical_dpu in list(group.physical_dpus or [group.dpu_id])],
            "heads": [int(group.head_start), int(group.head_end)],
            "group_heads": int(group.group_heads),
            "seq_len": int(group.seq_len),
            "capacity": int(group.capacity),
            "head_dim": int(group.head_dim),
            "live_elems": int(group.live_elems),
            "capacity_elems": int(group.capacity_elems),
            "token_segments": [
                {
                    "physical_dpu": int(segment.get("physical_dpu", group.dpu_id)),
                    "token_range_start": int(segment.get("token_range_start", 0)),
                    "token_range_end": int(segment.get("token_range_end", 0)),
                }
                for segment in list(group.token_segments or [])
            ],
            "resident_slot": self.resident_store.slot_debug(group.k_slot, group.v_slot),
        }

    def _layer_footprint_summary(self, layer_state: LayerState) -> Dict[str, object]:
        live_elems = sum(group.live_elems for group in layer_state.head_groups)
        capacity_elems = sum(group.capacity_elems for group in layer_state.head_groups)
        return {
            "layer_idx": int(layer_state.layer_idx),
            "num_heads": int(layer_state.num_heads),
            "head_dim": int(layer_state.head_dim),
            "group_count": len(layer_state.head_groups),
            "live_elems": int(live_elems),
            "capacity_elems": int(capacity_elems),
            "groups": [self._group_footprint_summary(group) for group in layer_state.head_groups],
        }

    def _request_footprint_summary(self, request_state: RequestState) -> Dict[str, object]:
        layer_summaries = [self._layer_footprint_summary(layer_state) for layer_state in request_state.layer_states]
        live_elems = sum(int(layer["live_elems"]) for layer in layer_summaries)
        capacity_elems = sum(int(layer["capacity_elems"]) for layer in layer_summaries)
        per_dpu_live_elems = [0 for _ in range(self.num_dpus)]
        per_dpu_capacity_elems = [0 for _ in range(self.num_dpus)]
        for layer_state in request_state.layer_states:
            for group in layer_state.head_groups:
                group_physical_dpus = [int(physical_dpu) % max(self.num_dpus, 1) for physical_dpu in list(group.physical_dpus or [group.dpu_id])]
                if not group_physical_dpus:
                    group_physical_dpus = [int(group.dpu_id) % max(self.num_dpus, 1)]
                share_live = int(group.live_elems) / float(len(group_physical_dpus))
                share_capacity = int(group.capacity_elems) / float(len(group_physical_dpus))
                for physical_dpu in group_physical_dpus:
                    per_dpu_live_elems[physical_dpu] += int(round(share_live))
                    per_dpu_capacity_elems[physical_dpu] += int(round(share_capacity))
        return {
            "request_id": request_state.request_id,
            "context_len": int(request_state.context_len),
            "num_layers": int(request_state.num_layers),
            "preferred_dpu_stripe": [int(physical_dpu) for physical_dpu in request_state.preferred_dpu_stripe],
            "stripe_width": len(request_state.preferred_dpu_stripe),
            "stripe_version": int(request_state.stripe_version),
            "stripe_expand_count": int(request_state.stripe_expand_count),
            "last_stripe_update_reason": request_state.last_stripe_update_reason,
            "last_stripe_width": int(request_state.last_stripe_width),
            "live_elems": int(live_elems),
            "capacity_elems": int(capacity_elems),
            "per_dpu_live_elems": per_dpu_live_elems,
            "per_dpu_capacity_elems": per_dpu_capacity_elems,
            "layers": layer_summaries,
        }

    def _request_packing_hint(self, request_state: RequestState) -> Dict[str, object]:
        stripe = [int(physical_dpu) for physical_dpu in request_state.preferred_dpu_stripe]
        rank_index = None
        if stripe and hasattr(self.resident_store, "_ensure_topology_cache") and hasattr(self.resident_store, "_topology_rank_index"):
            try:
                self.resident_store._ensure_topology_cache()
                rank_index = self.resident_store._topology_rank_index(int(stripe[0]))
            except Exception:
                rank_index = None
        rankset_id = None if rank_index is None else f"rank{int(rank_index)}:w{len(stripe)}"
        rankset_entries: Dict[str, Dict[str, object]] = {}
        for layer_state in request_state.layer_states:
            for group in layer_state.head_groups:
                group_physical_dpus = [int(physical_dpu) for physical_dpu in list(group.physical_dpus or [group.dpu_id])]
                if not group_physical_dpus:
                    group_physical_dpus = [int(group.dpu_id)]
                group_rank_index = self._stripe_rank_index(group_physical_dpus)
                if group_rank_index is None:
                    group_rankset_id = rankset_id or "rank-unknown"
                else:
                    group_rankset_id = f"rank{int(group_rank_index)}"
                entry = rankset_entries.get(group_rankset_id)
                if entry is None:
                    entry = {
                        "rankset_id": str(group_rankset_id),
                        "rank_index": None if group_rank_index is None else int(group_rank_index),
                        "physical_dpus": set(),
                        "stripe_width": 0,
                        "transfer_granularity": "rankset",
                        "layer_group_map": {},
                    }
                    rankset_entries[group_rankset_id] = entry
                for group_physical_dpu in group_physical_dpus:
                    entry["physical_dpus"].add(int(group_physical_dpu))
                layer_groups = entry["layer_group_map"].setdefault(str(layer_state.layer_idx), [])
                layer_groups.append(
                    {
                        "head_start": int(group.head_start),
                        "head_end": int(group.head_end),
                        "group_heads": int(group.group_heads),
                        "physical_dpu": int(group.dpu_id),
                        "physical_dpus": [int(physical_dpu) for physical_dpu in group_physical_dpus],
                        "token_segments": [
                            {
                                "physical_dpu": int(segment.get("physical_dpu", group.dpu_id)),
                                "token_range_start": int(segment.get("token_range_start", 0)),
                                "token_range_end": int(segment.get("token_range_end", 0)),
                            }
                            for segment in list(group.token_segments or [])
                        ],
                        "k_slot": str(group.k_slot),
                        "v_slot": str(group.v_slot),
                    }
                )

        rankset_plan = []
        for entry in sorted(
            rankset_entries.values(),
            key=lambda item: (
                -1 if item.get("rank_index") is None else int(item.get("rank_index", 0)),
                str(item.get("rankset_id", "")),
            ),
        ):
            physical_dpus = sorted(int(physical_dpu) for physical_dpu in entry.get("physical_dpus", set()))
            layer_group_map = {}
            for layer_key, groups in dict(entry.get("layer_group_map", {})).items():
                layer_group_map[str(layer_key)] = sorted(
                    [
                        {
                            "head_start": int(group["head_start"]),
                            "head_end": int(group["head_end"]),
                            "group_heads": int(group["group_heads"]),
                            "physical_dpu": int(group["physical_dpu"]),
                            "physical_dpus": [int(physical_dpu) for physical_dpu in list(group.get("physical_dpus", []) or [])],
                            "token_segments": [
                                {
                                    "physical_dpu": int(segment.get("physical_dpu", group["physical_dpu"])),
                                    "token_range_start": int(segment.get("token_range_start", 0)),
                                    "token_range_end": int(segment.get("token_range_end", 0)),
                                }
                                for segment in list(group.get("token_segments", []) or [])
                            ],
                            "k_slot": str(group["k_slot"]),
                            "v_slot": str(group["v_slot"]),
                        }
                        for group in groups
                    ],
                    key=lambda group: (int(group["head_start"]), int(group["head_end"])),
                )
            rankset_plan.append(
                {
                    "rankset_id": str(entry.get("rankset_id", "rank-unknown")),
                    "rank_index": entry.get("rank_index"),
                    "physical_dpus": physical_dpus,
                    "stripe_width": int(len(physical_dpus)),
                    "transfer_granularity": "rankset",
                    "layer_group_map": layer_group_map,
                }
            )
        if not rankset_plan and stripe:
            rankset_plan.append(
                {
                    "rankset_id": rankset_id or "rank-unknown",
                    "rank_index": None if rank_index is None else int(rank_index),
                    "physical_dpus": list(stripe),
                    "stripe_width": len(stripe),
                    "transfer_granularity": "stripe",
                    "layer_group_map": {},
                }
            )
        hint = {
            "context_len": int(request_state.context_len),
            "preferred_dpu_stripe": stripe,
            "stripe_width": len(stripe),
            "rank_index": None if rank_index is None else int(rank_index),
            "rankset_id": rankset_id,
            "rankset_count": len(rankset_plan),
            "rankset_plan": rankset_plan,
        }
        if request_state.sharding_plan:
            hint["sharding_plan"] = dict(request_state.sharding_plan)
            hint["planner_mode"] = str(
                dict(request_state.sharding_plan.get("metadata", {}) or {}).get("planner_mode", "")
            )
        return hint

    def _request_hash(self, request_id: str) -> int:
        return sum(ord(ch) for ch in str(request_id))

    def _planner_metadata(self, sharding_plan: Dict[str, object] | None) -> Dict[str, object]:
        return dict((sharding_plan or {}).get("metadata", {}) or {})

    def _normalize_physical_dpu_list(self, physical_dpus: Sequence[int] | None) -> List[int]:
        if self.num_dpus <= 0:
            return [0]
        normalized: List[int] = []
        seen = set()
        for physical_dpu in list(physical_dpus or []):
            value = int(physical_dpu) % self.num_dpus
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _coarsen_planner_group_specs(
        self,
        group_specs: Sequence[Tuple[int, int, List[int]]],
        target_group_count: int,
    ) -> List[Tuple[int, int, List[int]]]:
        if target_group_count <= 0 or target_group_count >= len(group_specs):
            return [(head_start, head_end, list(logical_dpus)) for head_start, head_end, logical_dpus in group_specs]

        total_groups = len(group_specs)
        base_width = total_groups // target_group_count
        extra_groups = total_groups % target_group_count
        merged_specs: List[Tuple[int, int, List[int]]] = []
        cursor = 0
        for merged_idx in range(target_group_count):
            width = base_width + (1 if merged_idx < extra_groups else 0)
            chunk = list(group_specs[cursor : cursor + width])
            cursor += width
            if not chunk:
                continue
            merged_logical_dpus: List[int] = []
            seen = set()
            for _, _, logical_dpus in chunk:
                for logical_dpu in logical_dpus:
                    normalized = int(logical_dpu)
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    merged_logical_dpus.append(normalized)
            merged_specs.append((int(chunk[0][0]), int(chunk[-1][1]), merged_logical_dpus))
        return merged_specs

    def _planner_group_specs(
        self,
        sharding_plan: Dict[str, object] | None,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> List[Tuple[int, int, List[int]]]:
        if not sharding_plan:
            return []

        head_group_ranges = dict(sharding_plan.get("head_group_ranges", {}) or {})
        dpu_groups = dict(sharding_plan.get("dpu_groups", {}) or {})
        if not head_group_ranges or not dpu_groups:
            return []

        group_specs: List[Tuple[int, int, List[int]]] = []
        for head_group_id in sorted(head_group_ranges.keys(), key=lambda item: int(item)):
            head_range = dict(head_group_ranges.get(head_group_id, {}) or {})
            head_start = int(head_range.get("head_start", 0))
            head_end = int(head_range.get("head_end", 0))
            logical_dpus = [int(dpu_id) for dpu_id in list(dpu_groups.get(head_group_id, []) or [])]
            if head_start < 0 or head_end > int(num_heads) or head_start >= head_end or not logical_dpus:
                return []
            group_specs.append((head_start, head_end, logical_dpus))

        if not group_specs:
            return []

        next_expected_head = 0
        for head_start, head_end, _ in group_specs:
            if int(head_start) != int(next_expected_head):
                return []
            next_expected_head = int(head_end)
        if int(next_expected_head) != int(num_heads):
            return []

        planner_mode = str(self._planner_metadata(sharding_plan).get("planner_mode", "") or "")
        if planner_mode == "single_dpu_multi_head_group":
            return group_specs

        target_group_count = self._effective_head_group_count(seq_len, num_heads, head_dim)
        return self._coarsen_planner_group_specs(group_specs, target_group_count)

    def _planner_segment_plan(
        self,
        sharding_plan: Dict[str, object] | None,
        *,
        request_id: str,
        seq_len: int,
        head_start: int,
        head_end: int,
        allowed_dpus: List[int],
    ) -> List[Dict[str, int]]:
        def _balanced_segments() -> List[Dict[str, int]]:
            if not allowed_dpus:
                return []
            cursor = 0
            remaining_tokens = int(seq_len)
            remaining_dpus = len(allowed_dpus)
            segments = []
            for physical_dpu in allowed_dpus:
                if remaining_tokens <= 0:
                    break
                chunk = int(math.ceil(float(remaining_tokens) / float(max(1, remaining_dpus))))
                token_end = min(int(seq_len), int(cursor + chunk))
                segments.append(
                    {
                        "dpu_id": int(physical_dpu),
                        "token_range_start": int(cursor),
                        "token_range_end": int(token_end),
                    }
                )
                remaining_tokens -= int(token_end - cursor)
                cursor = int(token_end)
                remaining_dpus -= 1
            return segments

        if not sharding_plan:
            return []

        per_head_shards = dict(sharding_plan.get("per_head_shards", {}) or {})
        target_request_map = None
        target_group_heads = max(0, int(head_end) - int(head_start))
        for head_group_id, request_map in per_head_shards.items():
            head_group_range = dict(dict(sharding_plan.get("head_group_ranges", {}) or {}).get(head_group_id, {}) or {})
            if int(head_group_range.get("head_start", -1)) != int(head_start):
                continue
            if int(head_group_range.get("head_end", -1)) != int(head_end):
                continue
            if int(head_group_range.get("group_heads", target_group_heads) or target_group_heads) != int(target_group_heads):
                continue
            target_request_map = dict(request_map or {})
            break
        if target_request_map is None:
            return _balanced_segments()

        request_shards = list(target_request_map.get(str(request_id), []) or [])
        if not request_shards:
            return _balanced_segments()

        if not allowed_dpus:
            return []
        logical_to_physical = {
            logical_idx: int(allowed_dpus[logical_idx % len(allowed_dpus)])
            for logical_idx in range(len(allowed_dpus))
        }
        segment_plan = []
        for shard in sorted(request_shards, key=lambda item: int(item.get("token_range_start", 0))):
            logical_dpu = int(shard.get("dpu_id", 0))
            segment_plan.append(
                {
                    "dpu_id": int(logical_to_physical.get(logical_dpu, allowed_dpus[logical_dpu % len(allowed_dpus)])),
                    "token_range_start": int(shard.get("token_range_start", 0)),
                    "token_range_end": int(shard.get("token_range_end", 0)),
                }
            )
        covered = sum(int(item["token_range_end"]) - int(item["token_range_start"]) for item in segment_plan)
        if covered != int(seq_len):
            return _balanced_segments()
        return segment_plan

    def _coarsen_short_segment_plan(
        self,
        segment_plan: Sequence[Dict[str, int]],
    ) -> tuple[List[Dict[str, int]], str]:
        if not self.compact_short_segments_enabled:
            return ([dict(item) for item in segment_plan], "")
        normalized = [
            {
                "dpu_id": int(item.get("dpu_id", item.get("physical_dpu", 0))),
                "token_range_start": int(item.get("token_range_start", 0)),
                "token_range_end": int(item.get("token_range_end", 0)),
            }
            for item in segment_plan
            if int(item.get("token_range_end", 0)) > int(item.get("token_range_start", 0))
        ]
        if len(normalized) <= 1:
            return (normalized, "")

        min_tokens = max(1, int(self.compact_short_segment_min_tokens))
        token_counts = [
            int(item["token_range_end"]) - int(item["token_range_start"])
            for item in normalized
        ]
        max_segment_tokens = max(token_counts, default=0)
        if max_segment_tokens >= min_tokens:
            return (normalized, "")

        coarsened: List[Dict[str, int]] = []
        chunk: List[Dict[str, int]] = []
        chunk_tokens = 0
        for item in normalized:
            chunk.append(item)
            chunk_tokens += int(item["token_range_end"]) - int(item["token_range_start"])
            if chunk_tokens < min_tokens:
                continue
            chosen_dpu = self._select_segment_dpu(chunk)
            coarsened.append(
                {
                    "dpu_id": int(chosen_dpu),
                    "token_range_start": int(chunk[0]["token_range_start"]),
                    "token_range_end": int(chunk[-1]["token_range_end"]),
                }
            )
            chunk = []
            chunk_tokens = 0

        if chunk:
            if coarsened:
                coarsened[-1]["token_range_end"] = int(chunk[-1]["token_range_end"])
            else:
                chosen_dpu = self._select_segment_dpu(chunk)
                coarsened.append(
                    {
                        "dpu_id": int(chosen_dpu),
                        "token_range_start": int(chunk[0]["token_range_start"]),
                        "token_range_end": int(chunk[-1]["token_range_end"]),
                    }
                )

        if len(coarsened) >= len(normalized):
            return (normalized, "")
        if len(coarsened) <= 1:
            reason = "short_segments_compacted"
        else:
            reason = "short_segments_coarsened"
        return (
            coarsened,
            f"{reason}:from={len(normalized)},to={len(coarsened)},"
            f"max_segment_tokens={max_segment_tokens},min_segment_tokens={min_tokens}",
        )

    def _segment_dpu_pressure(self, physical_dpu: int) -> tuple[int, int, int]:
        normalized_dpu = int(physical_dpu) % max(self.num_dpus, 1)
        slot_counts = getattr(self.resident_store, "dpu_live_slot_counts_by_dpu", None)
        elem_counts = getattr(self.resident_store, "dpu_live_elems_by_dpu", None)
        slot_pressure = 0
        elem_pressure = 0
        if isinstance(slot_counts, list) and normalized_dpu < len(slot_counts):
            slot_pressure = int(slot_counts[normalized_dpu])
        if isinstance(elem_counts, list) and normalized_dpu < len(elem_counts):
            elem_pressure = int(elem_counts[normalized_dpu])
        return (slot_pressure, elem_pressure, normalized_dpu)

    def _select_segment_dpu(
        self,
        segment_items: Sequence[Dict[str, int]],
        *,
        preferred_dpu: int | None = None,
    ) -> int:
        if not segment_items:
            return 0
        preferred = None if preferred_dpu is None else int(preferred_dpu) % max(self.num_dpus, 1)
        soft_limit = int(getattr(self.resident_store, "slot_pressure_soft_limit", 0))
        candidates: List[tuple[int, int, int, int, int]] = []
        for ordinal, item in enumerate(segment_items):
            dpu_id = int(item["dpu_id"]) % max(self.num_dpus, 1)
            token_count = int(item["token_range_end"]) - int(item["token_range_start"])
            slot_pressure, elem_pressure, normalized_dpu = self._segment_dpu_pressure(dpu_id)
            preferred_miss = 0 if preferred is not None and normalized_dpu == preferred else 1
            candidates.append(
                (
                    preferred_miss if preferred is not None else 0,
                    max(0, slot_pressure - soft_limit),
                    elem_pressure,
                    -int(token_count),
                    int(ordinal),
                    normalized_dpu,
                )
            )
        return int(min(candidates)[-1])

    def _planner_segment_materialization_decision(
        self,
        segment_plan: Sequence[Dict[str, int]],
    ) -> tuple[bool, str]:
        if not segment_plan:
            return (False, "no_segment_plan")

        unique_dpus = {
            int(item.get("dpu_id", item.get("physical_dpu", -1)))
            for item in segment_plan
        }
        if len(unique_dpus) <= 1:
            return (False, "single_dpu_segment_plan")

        if not self.compact_short_segments_enabled:
            return (True, "compact_short_segments_disabled")

        token_counts = [
            max(
                0,
                int(item.get("token_range_end", 0)) - int(item.get("token_range_start", 0)),
            )
            for item in segment_plan
        ]
        max_segment_tokens = max(token_counts, default=0)
        if max_segment_tokens < int(self.compact_short_segment_min_tokens):
            return (
                False,
                "short_segments_compacted:"
                f"max_segment_tokens={max_segment_tokens},"
                f"min_segment_tokens={int(self.compact_short_segment_min_tokens)}",
            )

        return (
            True,
            "segments_materialized:"
            f"max_segment_tokens={max_segment_tokens},"
            f"min_segment_tokens={int(self.compact_short_segment_min_tokens)}",
        )

    def _map_logical_dpus_to_physical(
        self,
        logical_dpu_ids: Sequence[int],
        preferred_dpu_stripe: List[int] | None = None,
        dpu_rotation: int = 0,
    ) -> List[int]:
        if self.num_dpus <= 0:
            return [0]

        physical_pool = self._normalize_physical_dpu_list(preferred_dpu_stripe)
        if not physical_pool:
            physical_pool = list(range(self.num_dpus))

        mapped: List[int] = []
        seen = set()
        for logical_dpu in logical_dpu_ids:
            physical_dpu = physical_pool[(int(logical_dpu) + int(dpu_rotation)) % len(physical_pool)]
            if physical_dpu in seen:
                continue
            seen.add(physical_dpu)
            mapped.append(int(physical_dpu))
        if mapped:
            return mapped
        return [int(physical_pool[int(dpu_rotation) % len(physical_pool)])]

    def _stripe_rank_index(self, stripe: List[int]) -> int | None:
        if (
            not stripe
            or not hasattr(self.resident_store, "_ensure_topology_cache")
            or not hasattr(self.resident_store, "_topology_rank_index")
        ):
            return None
        try:
            self.resident_store._ensure_topology_cache()
            return self.resident_store._topology_rank_index(int(stripe[0]))
        except Exception:
            return None

    def _active_request_rank_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for request_state in self.request_states.values():
            rank_index = self._stripe_rank_index(request_state.preferred_dpu_stripe)
            if rank_index is None:
                continue
            counts[int(rank_index)] = counts.get(int(rank_index), 0) + 1
        return counts

    def _rank_groups_with_indices(self) -> List[tuple[int, List[int]]]:
        if not hasattr(self.resident_store, "get_rank_groups"):
            return []
        rank_groups = self.resident_store.get_rank_groups()
        rank_groups_with_indices: List[tuple[int, List[int]]] = []
        for group in sorted(rank_groups, key=lambda item: (len(item), item[0] if item else 0)):
            if not group:
                continue
            rank_index = self._stripe_rank_index([int(group[0])])
            if rank_index is None:
                continue
            rank_groups_with_indices.append(
                (
                    int(rank_index),
                    [int(physical_dpu) for physical_dpu in group],
                )
            )
        return rank_groups_with_indices

    def _active_request_rank_stripes(self, target_rank_index: int) -> List[List[int]]:
        stripes: List[List[int]] = []
        for request_state in self.request_states.values():
            rank_index = self._stripe_rank_index(request_state.preferred_dpu_stripe)
            if rank_index is None or int(rank_index) != int(target_rank_index):
                continue
            stripe = [int(physical_dpu) for physical_dpu in request_state.preferred_dpu_stripe]
            if stripe:
                stripes.append(stripe)
        return stripes

    def _request_shape_targets(
        self,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> tuple[int, int, int]:
        total_live_elems = 0
        total_capacity_elems = 0
        max_layer_groups = 1
        reserve_tokens = max(0, int(decode_reserve_tokens))
        base_rollover_tokens = int(getattr(self.resident_store, "base_block_rollover_tokens", 0))
        growth_block_tokens = int(getattr(self.resident_store, "growth_block_tokens", 0))
        block_tokens = max(1, int(getattr(self, "block_tokens", 1)))
        for layer in initial_kv:
            layer_key = layer["key"]
            seq_len, num_heads, head_dim = (int(dim) for dim in layer_key.shape)
            group_count = self._effective_head_group_count(seq_len, num_heads, head_dim)
            max_layer_groups = max(max_layer_groups, group_count)
            total_live_elems += int(seq_len) * int(num_heads) * int(head_dim)
            logical_capacity = max(int(self.length), int(seq_len) + reserve_tokens)
            base_capacity = int(block_tokens * math.ceil(float(logical_capacity) / float(block_tokens)))
            effective_capacity = base_capacity
            tail_available = max(0, base_capacity - logical_capacity)
            # Mirror the blocked-store rollover behavior roughly enough for
            # stripe sizing: once the request is expected to decode beyond the
            # base resident length, a nearly-full tail block will spill into an
            # extra growth block instead of consuming the small remaining tail
            # in place. Keep shorter prompts below the base resident length on
            # the compact path so we do not over-widen Qwen/OPT stripes just to
            # reserve growth blocks they will not actually touch.
            if (
                reserve_tokens > 0
                and growth_block_tokens > 0
                and logical_capacity > int(self.length)
                and tail_available <= base_rollover_tokens
            ):
                effective_capacity += growth_block_tokens
            total_capacity_elems += effective_capacity * int(num_heads) * int(head_dim)
        # Host-backed resident storage has no per-DPU slot pool, so capacity
        # should not artificially widen the requested stripe there. Fall back to
        # a single logical stripe when the store does not expose DPU pool size.
        pool_capacity_elems = int(getattr(self.resident_store, "POOL_CAPACITY_ELEMS", 0))
        if pool_capacity_elems <= 0:
            min_dpus_by_capacity = 1
        else:
            min_dpus_by_capacity = max(
                1,
                math.ceil(float(total_capacity_elems) / float(pool_capacity_elems)),
            )
        return int(total_live_elems), int(max_layer_groups), int(min_dpus_by_capacity)

    def _choose_active_rank_for_request(
        self,
        request_id: str,
        stripe_width: int,
    ) -> tuple[int | None, str]:
        rank_groups = self._rank_groups_with_indices()
        if not rank_groups:
            return (None, "no_rank_groups")

        active_rank_counts = self._active_request_rank_counts()
        if not active_rank_counts:
            return (None, "no_active_rank")

        rank_live_elems: Dict[int, int] = {}
        dpu_live_elems = getattr(self.resident_store, "dpu_live_elems_by_dpu", None)
        if isinstance(dpu_live_elems, list):
            for rank_index, group in rank_groups:
                rank_live_elems[int(rank_index)] = sum(
                    int(dpu_live_elems[int(physical_dpu) % max(self.num_dpus, 1)])
                    for physical_dpu in group
                )

        request_hash = self._request_hash(request_id)
        candidate_scores = []
        for ordinal, (rank_index, group) in enumerate(rank_groups):
            active_count = int(active_rank_counts.get(int(rank_index), 0))
            if active_count <= 0:
                continue
            candidate_scores.append(
                (
                    -active_count,
                    int(rank_live_elems.get(int(rank_index), 0)),
                    abs(len(group) - int(stripe_width)),
                    (request_hash + ordinal) % max(len(rank_groups), 1),
                    int(rank_index),
                )
            )
        if not candidate_scores:
            return (None, "no_nonempty_active_rank")

        best_rank_index = min(candidate_scores)[-1]
        return (int(best_rank_index), "active_rank_reuse")

    def _effective_head_group_count(self, seq_len: int, num_heads: int, head_dim: int) -> int:
        max_groups = max(1, min(self.num_dpus, num_heads))
        if max_groups <= 1:
            return 1

        if self.head_grouping_policy == "coarse":
            # Coarse grouping is an opt-in policy for CloverInfer experiments:
            # keep each layer in as few groups as possible while respecting the
            # current UPMEM per-group head limit.
            max_heads_per_group = 32
            min_groups_by_shape = max(1, math.ceil(int(num_heads) / max_heads_per_group))
            return max(1, min(max_groups, min_groups_by_shape))

        if self.head_grouping_policy == "segment_aware":
            # For segmented logical slots, "balanced" often creates too many
            # tiny per-segment helper items, while "coarse" can make each item
            # too wide. This policy tries an intermediate grouping that keeps
            # roughly 4 heads together for longer-context decode.
            target_heads_per_group = 4 if int(seq_len) >= 192 else 2
            min_groups_by_shape = max(1, math.ceil(int(num_heads) / target_heads_per_group))
            return max(1, min(max_groups, min_groups_by_shape))

        # Small decode workloads regress when we spread a layer across too many
        # tiny resident groups. Keep more heads together until each group has a
        # meaningful amount of KV work to amortize helper and launch overheads.
        per_head_live_elems = max(1, int(seq_len) * int(head_dim))
        total_live_elems = per_head_live_elems * int(num_heads)
        if per_head_live_elems <= 8_192:
            min_heads_per_group = 4
        elif per_head_live_elems <= 32_768:
            min_heads_per_group = 2
        else:
            min_heads_per_group = 1

        max_groups_by_heads = max(1, math.ceil(int(num_heads) / min_heads_per_group))
        target_group_live_elems = 32_768
        max_groups_by_work = max(1, math.ceil(total_live_elems / target_group_live_elems))
        return max(1, min(max_groups, max_groups_by_heads, max_groups_by_work))

    def _choose_group_physical_dpu(
        self,
        *,
        group_idx: int,
        num_groups: int,
        dpu_rotation: int,
        preferred_dpu_stripe: List[int] | None = None,
    ) -> int:
        if self.num_dpus <= 0:
            return 0
        if preferred_dpu_stripe:
            normalized = [int(physical_dpu) % self.num_dpus for physical_dpu in preferred_dpu_stripe]
            if normalized:
                stripe_rotation = dpu_rotation % len(normalized)
                return normalized[(group_idx + stripe_rotation) % len(normalized)]
        if self.dpu_placement_policy == "rotated":
            return (group_idx + dpu_rotation) % self.num_dpus
        if self.dpu_placement_policy == "rank_spread":
            # Clover experiment: widen the stride between neighboring groups to
            # increase the chance that logical groups land on different ranks
            # when the runtime maps logical DPUs contiguously.
            stride = max(1, math.ceil(self.num_dpus / max(1, num_groups)))
            return (dpu_rotation + group_idx * stride) % self.num_dpus
        if self.dpu_placement_policy == "load_aware":
            return (group_idx + dpu_rotation) % self.num_dpus
        return group_idx % self.num_dpus

    def _build_head_groups(
        self,
        request_id: str,
        layer_idx: int,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
        decode_reserve_tokens: int = 0,
        preferred_dpu_stripe: List[int] | None = None,
        sharding_plan: Dict[str, object] | None = None,
    ) -> List[HeadGroupState]:
        seq_len, num_heads, head_dim = (int(dim) for dim in layer_key.shape)
        if num_heads <= 0:
            raise ValueError(f"layer {layer_idx} in request {request_id} has no attention heads")

        capacity = max(self.length, seq_len + max(0, int(decode_reserve_tokens)))
        head_groups = []
        request_hash = sum(ord(ch) for ch in request_id)
        dpu_rotation = (request_hash + int(layer_idx)) % max(self.num_dpus, 1)
        group_specs = self._planner_group_specs(sharding_plan, seq_len, num_heads, head_dim)
        if not group_specs:
            num_groups = self._effective_head_group_count(seq_len, num_heads, head_dim)
            group_ranges: List[tuple[int, int]] = []
            if self.head_grouping_policy == "legacy":
                heads_per_group = math.ceil(num_heads / num_groups)
                for group_idx in range(num_groups):
                    head_start = group_idx * heads_per_group
                    head_end = min(num_heads, head_start + heads_per_group)
                    if head_start >= head_end:
                        break
                    group_ranges.append((head_start, head_end))
            else:
                base_heads_per_group = num_heads // num_groups
                extra_head_groups = num_heads % num_groups
                head_start = 0
                for _ in range(num_groups):
                    group_heads = base_heads_per_group + (1 if len(group_ranges) < extra_head_groups else 0)
                    head_end = min(num_heads, head_start + group_heads)
                    if head_start >= head_end:
                        break
                    group_ranges.append((head_start, head_end))
                    head_start = head_end
            group_specs = [(head_start, head_end, []) for head_start, head_end in group_ranges]

        for group_idx, (head_start, head_end, logical_dpu_ids) in enumerate(group_specs):
            allowed_dpus = (
                self._map_logical_dpus_to_physical(
                    logical_dpu_ids,
                    preferred_dpu_stripe=preferred_dpu_stripe,
                    dpu_rotation=dpu_rotation,
                )
                if logical_dpu_ids
                else self._normalize_physical_dpu_list(preferred_dpu_stripe)
            )
            segment_plan = self._planner_segment_plan(
                sharding_plan,
                request_id=request_id,
                seq_len=seq_len,
                head_start=head_start,
                head_end=head_end,
                allowed_dpus=allowed_dpus,
            )
            use_segment_plan, segment_decision = self._planner_segment_materialization_decision(segment_plan)
            if segment_plan:
                self.planner_segment_plan_count += 1
                segment_plan, coarsen_decision = self._coarsen_short_segment_plan(segment_plan)
                use_segment_plan, segment_decision = self._planner_segment_materialization_decision(segment_plan)
                self.planner_segment_last_decision = coarsen_decision or segment_decision
                if use_segment_plan:
                    self.planner_segment_materialized_count += 1
                if coarsen_decision or (
                    not use_segment_plan
                    and len(
                        {int(item.get("dpu_id", item.get("physical_dpu", -1))) for item in segment_plan}
                    ) > 1
                ):
                    self.planner_segment_compacted_count += 1
            physical_dpu = self._choose_group_physical_dpu(
                group_idx=group_idx,
                num_groups=len(group_specs),
                dpu_rotation=dpu_rotation,
                preferred_dpu_stripe=allowed_dpus or preferred_dpu_stripe,
            )
            k_slot = f"{request_id}:layer{layer_idx}:group{group_idx}:k"
            v_slot = f"{request_id}:layer{layer_idx}:group{group_idx}:v"
            initial_k_group = layer_key[:, head_start:head_end, :].contiguous()
            initial_v_group = layer_value[:, head_start:head_end, :].contiguous()
            allocation_info = self.resident_store.allocate_group(
                k_slot,
                v_slot,
                initial_k_group,
                initial_v_group,
                capacity=capacity,
                preferred_dpu=physical_dpu,
                force_host_fallback=self.max_resident_groups_per_layer > 0 and group_idx >= self.max_resident_groups_per_layer,
                allowed_dpus=allowed_dpus or preferred_dpu_stripe,
                segment_plan=segment_plan if use_segment_plan else None,
            )
            actual_physical_dpu = allocation_info.get("physical_dpu", physical_dpu)
            allocation_segments = list(allocation_info.get("segments", []) or [])
            group_physical_dpus = sorted(
                {
                    int(segment.get("physical_dpu", actual_physical_dpu if actual_physical_dpu is not None else physical_dpu))
                    for segment in allocation_segments
                }
            )
            if not group_physical_dpus:
                group_physical_dpus = [physical_dpu if actual_physical_dpu is None else int(actual_physical_dpu)]
            head_groups.append(
                HeadGroupState(
                    dpu_id=physical_dpu if actual_physical_dpu is None else int(actual_physical_dpu),
                    head_start=head_start,
                    head_end=head_end,
                    seq_len=seq_len,
                    capacity=capacity,
                    head_dim=head_dim,
                    k_slot=k_slot,
                    v_slot=v_slot,
                    physical_dpus=group_physical_dpus,
                    token_segments=[
                        {
                            "physical_dpu": int(segment.get("physical_dpu", group_physical_dpus[0])),
                            "token_range_start": int(segment.get("token_start", segment.get("token_range_start", 0))),
                            "token_range_end": int(segment.get("token_end", segment.get("token_range_end", 0))),
                        }
                        for segment in allocation_segments
                    ],
                )
            )
        return head_groups

    def _preferred_dpu_stripe_for_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
        sharding_plan: Dict[str, object] | None = None,
    ) -> List[int]:
        if self.num_dpus <= 0:
            self.init_rank_last_reason = "no_dpus"
            return [0]
        if not initial_kv:
            self.init_rank_last_reason = "empty_initial_kv"
            return [0]
        _total_live_elems, max_layer_groups, min_dpus_by_capacity = self._request_shape_targets(
            initial_kv,
            decode_reserve_tokens=decode_reserve_tokens,
        )
        initial_context_len = int(initial_kv[0]["key"].shape[0]) if initial_kv else 0
        planner_metadata = self._planner_metadata(sharding_plan)
        planner_mode = str(planner_metadata.get("planner_mode", "") or "")
        planner_group_count = max(0, int(planner_metadata.get("effective_group_count", 0) or 0))
        if planner_mode == "single_dpu_multi_head_group":
            max_layer_groups = max(max_layer_groups, planner_group_count)

        request_hash = self._request_hash(request_id)
        # The last compact experiment proved that capacity-only shrinking makes
        # requests local, but can overload a tiny subset of DPUs and explode the
        # number of helper rounds. Aim for a medium-width stripe:
        # - narrow enough to keep batched rounds inside one rank when possible
        # - wide enough to spread same-layer groups across several DPUs
        if max_layer_groups <= 2:
            target_medium_width = 4
        elif max_layer_groups <= 4:
            target_medium_width = 8
        elif max_layer_groups <= 8:
            target_medium_width = 12
        else:
            target_medium_width = 16
        # If the prompt is already beyond the base resident length, delaying
        # stripe growth leaves most base groups pinned onto the narrower
        # initial subset. Front-load a moderate width increase so more same-rank
        # DPUs participate from request initialization without giving up
        # single-rank locality.
        if max_layer_groups <= 4 and initial_context_len >= int(self.length):
            target_medium_width = max(target_medium_width, 12)
            prompt_overshoot = max(0, initial_context_len - int(self.length))
            # Once the prompt is already beyond the base resident length, a
            # slightly wider initial stripe can activate more same-rank DPUs
            # before growth blocks appear. Keep this opt-in for small-group
            # requests where the wider stripe is still cheap.
            if prompt_overshoot >= max(8, self.block_tokens // 32):
                target_medium_width = max(target_medium_width, 16)
        elif max_layer_groups > 4 and initial_context_len >= int(self.length) + max(96, self.block_tokens // 2):
            target_medium_width = max(target_medium_width, 16)
        stripe_width = max(min_dpus_by_capacity, target_medium_width)
        planner_dpu_group_width = 1
        planner_dpu_groups = dict((sharding_plan or {}).get("dpu_groups", {}) or {})
        if planner_dpu_groups:
            planner_dpu_group_width = max(
                1,
                max(len(list(group_dpus or [])) for group_dpus in planner_dpu_groups.values()),
            )
        stripe_width = max(stripe_width, planner_dpu_group_width)
        if planner_mode == "single_dpu_multi_head_group":
            stripe_width = max(stripe_width, planner_group_count)
        stripe_width = min(self.num_dpus, max(1, stripe_width))

        target_rank_index, target_reason = self._choose_active_rank_for_request(
            request_id,
            stripe_width,
        )
        stripe = self._rank_local_stripe_for_request(
            request_id,
            stripe_width,
            target_rank_index=target_rank_index,
        )
        if target_rank_index is not None:
            self.init_rank_locality_reuse_count += 1
            self.init_rank_last_reason = target_reason
            return stripe

        self.init_rank_hash_fallback_count += 1
        self.init_rank_last_reason = target_reason
        rank_groups = self._rank_groups_with_indices()
        if rank_groups:
            fallback_rank_index = int(rank_groups[request_hash % len(rank_groups)][0])
            return self._rank_local_stripe_for_request(
                request_id,
                stripe_width,
                target_rank_index=fallback_rank_index,
            )
        base_dpu = request_hash % self.num_dpus
        return [(base_dpu + offset) % self.num_dpus for offset in range(stripe_width)]

    def _medium_stripe_target_width(self, max_layer_groups: int) -> int:
        if max_layer_groups <= 2:
            return 4
        if max_layer_groups <= 4:
            return 8
        if max_layer_groups <= 8:
            return 12
        return 16

    def _max_layer_group_count_for_request(self, request_state: RequestState) -> int:
        if not request_state.layer_states:
            return 1
        return max(1, max(len(layer_state.head_groups) for layer_state in request_state.layer_states))

    def _rank_local_stripe_for_request(
        self,
        request_id: str,
        stripe_width: int,
        target_rank_index: int | None = None,
        preferred_anchor_dpu: int | None = None,
        required_dpus: List[int] | None = None,
    ) -> List[int]:
        stripe_width = min(self.num_dpus, max(1, int(stripe_width)))
        request_hash = self._request_hash(request_id)
        rank_groups = self._rank_groups_with_indices()
        if rank_groups:
            chosen_group: List[int] | None = None
            for rank_index, group in rank_groups:
                if target_rank_index is not None and int(rank_index) == int(target_rank_index):
                    chosen_group = group
                    break
            if chosen_group is None and preferred_anchor_dpu is not None:
                anchor_rank = self._stripe_rank_index([int(preferred_anchor_dpu)])
                for rank_index, group in rank_groups:
                    if anchor_rank is not None and int(rank_index) == int(anchor_rank):
                        chosen_group = group
                        break
            if chosen_group is None:
                chosen_group = rank_groups[request_hash % len(rank_groups)][1]
            if chosen_group:
                width = min(stripe_width, len(chosen_group))
                if width >= len(chosen_group):
                    return [int(physical_dpu) for physical_dpu in chosen_group]

                group_index = {
                    int(physical_dpu): idx for idx, physical_dpu in enumerate(chosen_group)
                }
                max_start = len(chosen_group) - width
                hash_start = (request_hash // max(1, len(rank_groups))) % max(1, max_start + 1)
                existing_stripes = []
                if target_rank_index is not None:
                    existing_stripes = self._active_request_rank_stripes(int(target_rank_index))
                dpu_live_elems = getattr(self.resident_store, "dpu_live_elems_by_dpu", None)

                required_positions = []
                if required_dpus:
                    required_positions = sorted(
                        {
                            int(group_index[int(physical_dpu)])
                            for physical_dpu in required_dpus
                            if int(physical_dpu) in group_index
                        }
                    )

                start_min = 0
                start_max = max_start
                if required_positions:
                    req_lo = min(required_positions)
                    req_hi = max(required_positions)
                    if req_hi - req_lo + 1 > width:
                        required_positions = []
                    else:
                        start_min = max(0, req_hi - width + 1)
                        start_max = min(max_start, req_lo)

                candidate_scores = []
                for start_idx in range(start_min, start_max + 1):
                    candidate = [int(physical_dpu) for physical_dpu in chosen_group[start_idx : start_idx + width]]
                    candidate_set = set(candidate)
                    overlap = 0
                    min_center_distance = 0.0
                    if existing_stripes:
                        overlap = sum(
                            len(candidate_set.intersection({int(physical_dpu) for physical_dpu in existing}))
                            for existing in existing_stripes
                        )
                        candidate_center = float(start_idx) + (float(width - 1) / 2.0)
                        existing_centers = []
                        for existing in existing_stripes:
                            positions = sorted(
                                int(group_index[int(physical_dpu)])
                                for physical_dpu in existing
                                if int(physical_dpu) in group_index
                            )
                            if positions:
                                existing_centers.append(
                                    float(positions[0] + positions[-1]) / 2.0
                                )
                        if existing_centers:
                            min_center_distance = min(
                                abs(candidate_center - existing_center)
                                for existing_center in existing_centers
                            )
                    live_load = 0
                    if isinstance(dpu_live_elems, list):
                        live_load = sum(
                            int(dpu_live_elems[int(physical_dpu) % max(self.num_dpus, 1)])
                            for physical_dpu in candidate
                        )
                    anchor_distance = 0
                    if preferred_anchor_dpu is not None and int(preferred_anchor_dpu) in group_index:
                        anchor_idx = int(group_index[int(preferred_anchor_dpu)])
                        if anchor_idx < start_idx:
                            anchor_distance = start_idx - anchor_idx
                        elif anchor_idx >= start_idx + width:
                            anchor_distance = anchor_idx - (start_idx + width - 1)
                    edge_distance = min(start_idx, max_start - start_idx)
                    candidate_scores.append(
                        (
                            overlap,
                            live_load,
                            -float(min_center_distance),
                            anchor_distance,
                            -edge_distance,
                            abs(start_idx - hash_start),
                            start_idx,
                            candidate,
                        )
                    )
                if candidate_scores:
                    return min(candidate_scores)[-1]
        base_dpu = request_hash % self.num_dpus
        return [(base_dpu + offset) % self.num_dpus for offset in range(stripe_width)]

    def _maybe_expand_request_stripe(self, request_state: RequestState) -> None:
        if self.num_dpus <= 0 or not request_state.layer_states:
            return
        current_width = len(request_state.preferred_dpu_stripe)
        if current_width <= 0:
            return
        max_layer_groups = self._max_layer_group_count_for_request(request_state)
        target_width = current_width
        base_medium_width = self._medium_stripe_target_width(max_layer_groups)
        context_len = int(request_state.context_len)
        short_growth_threshold = int(self.length) + max(16, self.block_tokens // 8)
        medium_growth_threshold = int(self.length) + max(48, self.block_tokens // 4)
        long_growth_threshold = int(self.length) + max(96, self.block_tokens // 2)

        if current_width < base_medium_width and context_len >= short_growth_threshold:
            target_width = max(target_width, base_medium_width)
        if max_layer_groups <= 4:
            if context_len >= medium_growth_threshold:
                target_width = max(target_width, 12)
            if context_len >= long_growth_threshold:
                target_width = max(target_width, 16)
        else:
            growth_blocks = max(0, (context_len - int(self.length)) // max(1, self.block_tokens // 2))
            if growth_blocks > 0:
                target_width = max(target_width, base_medium_width + min(8, growth_blocks * 2))
        target_width = min(self.num_dpus, max(current_width, target_width))
        if target_width <= current_width:
            return
        new_stripe = self._rank_local_stripe_for_request(
            request_state.request_id,
            target_width,
            target_rank_index=self._stripe_rank_index(request_state.preferred_dpu_stripe),
            preferred_anchor_dpu=int(request_state.preferred_dpu_stripe[0]),
            required_dpus=[int(physical_dpu) for physical_dpu in request_state.preferred_dpu_stripe],
        )
        if len(new_stripe) <= current_width:
            return
        request_state.preferred_dpu_stripe = list(new_stripe)
        request_state.stripe_version += 1
        request_state.stripe_expand_count += 1
        request_state.last_stripe_update_reason = (
            f"context_len={context_len},max_layer_groups={max_layer_groups},from={current_width},to={len(new_stripe)}"
        )
        request_state.last_stripe_width = len(new_stripe)
        for layer_state in request_state.layer_states:
            for group in layer_state.head_groups:
                self.resident_store.update_group_allowed_dpus(
                    group.k_slot,
                    group.v_slot,
                    request_state.preferred_dpu_stripe,
                )

    def _build_request_state(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> RequestState:
        layer_states = []
        if not initial_kv:
            raise ValueError("initial_kv must contain at least one layer")

        first_layer_key = initial_kv[0]["key"].detach().cpu().contiguous()
        if first_layer_key.dim() != 3:
            raise ValueError(
                f"initial key for request {request_id} layer 0 must be 3D, "
                f"got shape {tuple(first_layer_key.shape)}"
            )

        context_len = int(first_layer_key.shape[0])
        sharding_plan = plan_sharding(
            [{"request_id": str(request_id), "seq_len": int(context_len)}],
            D=max(1, int(self.num_dpus)),
            H=max(1, int(first_layer_key.shape[1])),
        )
        preferred_dpu_stripe = self._preferred_dpu_stripe_for_request(
            request_id,
            initial_kv,
            decode_reserve_tokens=decode_reserve_tokens,
            sharding_plan=sharding_plan,
        )
        for layer_idx, layer in enumerate(initial_kv):
            layer_key = layer["key"].detach().cpu().contiguous()
            if layer_key.dim() != 3:
                raise ValueError(
                    f"initial key for request {request_id} layer {layer_idx} must be 3D, "
                    f"got shape {tuple(layer_key.shape)}"
                )

            seq_len, num_heads, head_dim = (int(dim) for dim in layer_key.shape)
            layer_states.append(
                LayerState(
                    layer_idx=layer_idx,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    head_groups=self._build_head_groups(
                        request_id,
                        layer_idx,
                        layer_key,
                        layer["value"].detach().cpu().contiguous(),
                        decode_reserve_tokens,
                        preferred_dpu_stripe=preferred_dpu_stripe,
                        sharding_plan=sharding_plan,
                    ),
                )
            )

        return RequestState(
            request_id=request_id,
            context_len=context_len,
            num_layers=len(layer_states),
            layer_states=layer_states,
            preferred_dpu_stripe=preferred_dpu_stripe,
            sharding_plan=sharding_plan,
            stripe_version=0,
            stripe_expand_count=0,
            last_stripe_update_reason="init",
            last_stripe_width=len(preferred_dpu_stripe),
        )

    def _append_resident_kv(
        self,
        request_state: RequestState,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        if layer_idx == 0:
            self._maybe_expand_request_stripe(request_state)
        layer_state = request_state.layer_states[layer_idx]
        expected_seq_len = request_state.context_len + 1

        def _refresh_group_metadata(group: HeadGroupState, append_info: Dict[str, int]) -> None:
            group.seq_len = int(append_info["seq_len"])
            group.capacity = int(append_info["capacity"])
            slot_debug = self.resident_store.slot_debug(group.k_slot, group.v_slot)
            slot_segments = list(slot_debug.get("segments", []) or [])
            if slot_segments:
                group.physical_dpus = sorted(
                    {
                        int(segment.get("physical_dpu", group.dpu_id))
                        for segment in slot_segments
                    }
                )
                group.token_segments = [
                    {
                        "physical_dpu": int(segment.get("physical_dpu", group.dpu_id)),
                        "token_range_start": int(segment.get("token_range_start", 0)),
                        "token_range_end": int(segment.get("token_range_end", 0)),
                    }
                    for segment in slot_segments
                ]

        for group in layer_state.head_groups:
            if group.seq_len != request_state.context_len:
                raise RuntimeError(
                    f"resident metadata seq_len mismatch before append for request={request_state.request_id} "
                    f"layer={layer_idx} dpu={group.dpu_id}: group_seq_len={group.seq_len} "
                    f"context_len={request_state.context_len}"
                )
            group_k_new = k_new[group.head_start:group.head_end, :].unsqueeze(0).contiguous()
            group_v_new = v_new[group.head_start:group.head_end, :].unsqueeze(0).contiguous()
            if group_k_new.shape[1] != group.head_end - group.head_start:
                raise RuntimeError(
                    f"resident k append shape mismatch for request={request_state.request_id} "
                    f"layer={layer_idx} dpu={group.dpu_id}: got={tuple(group_k_new.shape)}"
                )
            try:
                append_info = self.resident_store.append_group(
                    group.k_slot,
                    group.v_slot,
                    group_k_new,
                    group_v_new,
                )
            except RuntimeError:
                if not hasattr(self.resident_store, "migrate_group_to_host_fallback"):
                    raise
                self.resident_store.migrate_group_to_host_fallback(group.k_slot, group.v_slot)
                append_info = self.resident_store.append_group(
                    group.k_slot,
                    group.v_slot,
                    group_k_new,
                    group_v_new,
                )
            _refresh_group_metadata(group, append_info)
            if group.seq_len != expected_seq_len:
                raise RuntimeError(
                    f"resident store seq_len mismatch after append for request={request_state.request_id} "
                    f"layer={layer_idx} dpu={group.dpu_id}: group_seq_len={group.seq_len} "
                    f"expected={expected_seq_len}"
                )
            self.resident_append_ops += 1

        if layer_idx == request_state.num_layers - 1:
            request_state.context_len = expected_seq_len

    def _materialize_layer_kv(self, request_state: RequestState, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer_state = request_state.layer_states[layer_idx]
        if not layer_state.head_groups:
            raise RuntimeError(f"request {request_state.request_id} layer {layer_idx} has no resident head groups")

        seq_lens = {group.seq_len for group in layer_state.head_groups}
        if len(seq_lens) != 1:
            raise RuntimeError(
                f"inconsistent resident seq_lens for request={request_state.request_id} "
                f"layer={layer_idx}: {sorted(seq_lens)}"
            )

        materialized_groups = [
            self.resident_store.materialize_group(group.k_slot, group.v_slot)
            for group in layer_state.head_groups
        ]
        keys = torch.cat([pair[0] for pair in materialized_groups], dim=1).contiguous()
        values = torch.cat([pair[1] for pair in materialized_groups], dim=1).contiguous()
        if int(keys.shape[1]) != layer_state.num_heads or int(values.shape[1]) != layer_state.num_heads:
            raise RuntimeError(
                f"resident materialization head mismatch for request={request_state.request_id} "
                f"layer={layer_idx}: keys={tuple(keys.shape)} values={tuple(values.shape)} "
                f"expected_heads={layer_state.num_heads}"
            )
        self.resident_materialize_ops += 1
        return keys, values

    def _head_group_for_head(self, layer_state: LayerState, head_idx: int) -> HeadGroupState:
        for group in layer_state.head_groups:
            if group.head_start <= head_idx < group.head_end:
                return group
        raise IndexError(f"head_idx {head_idx} not covered by resident groups for layer {layer_state.layer_idx}")

    def _update_resident_shadow_diff(
        self,
        request_id: str,
        layer_idx: int,
        resident_keys: torch.Tensor,
        resident_values: torch.Tensor,
    ) -> None:
        cpu_keys = self.cpu_backend.k_cache[request_id][layer_idx]
        cpu_values = self.cpu_backend.v_cache[request_id][layer_idx]
        key_diff = float(torch.max(torch.abs(resident_keys.float() - cpu_keys.float())).item())
        value_diff = float(torch.max(torch.abs(resident_values.float() - cpu_values.float())).item())
        self.resident_shadow_max_abs_diff = max(self.resident_shadow_max_abs_diff, key_diff, value_diff)

    def _summarize_request_state(self, request_state: RequestState) -> Dict[str, object]:
        footprint = self._request_footprint_summary(request_state)
        footprint["layers_preview"] = footprint["layers"][: min(2, len(footprint["layers"]))]
        return footprint

    def _run_make_smoke(self, subdir: str, env_overrides: Dict[str, str]) -> str:
        smoke_dir = os.path.join(self.repo_root, "src", "pim", subdir)
        env = os.environ.copy()
        env.update(env_overrides)
        completed = subprocess.run(
            ["make", "clean", "run"],
            cwd=smoke_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise RuntimeError(f"UPMEM {subdir} smoke test failed:\n{output}")
        return output

    def _run_dot_smoke_test(self) -> None:
        if self._shares_persistent_dpu_owner():
            self.smoke_test_output = (
                "skipped upmem_dot smoke test: resident kvslot helper owns the persistent DPU set"
            )
            self.smoke_test_ok = True
            return
        self.smoke_test_output = self._run_make_smoke(
            "upmem_dot",
            {
                "NUM_DPUS": str(self.num_dpus),
                "LENGTH": str(self.length),
            },
        )
        self.smoke_test_ok = True

    def _run_qk_smoke_test(self, head_dim: int, keys_per_dpu: int) -> None:
        head_dim = max(2, min(128, int(head_dim)))
        if head_dim % 2 != 0:
            head_dim -= 1
        keys_per_dpu = max(1, min(128, int(keys_per_dpu)))
        self.qk_last_output = self._run_make_smoke(
            "upmem_qk",
            {
                "NUM_DPUS": str(min(self.num_dpus, 2)),
                "HEAD_DIM": str(head_dim),
                "KEYS_PER_DPU": str(keys_per_dpu),
            },
        )
        self.qk_check_count += 1

    def _ensure_qk_binary(self) -> str:
        qk_dir = os.path.join(self.repo_root, "src", "pim", "upmem_qk")
        binary_path = os.path.join(qk_dir, "build", "host_qk")
        source_paths = [
            os.path.join(qk_dir, "host_qk.c"),
            os.path.join(qk_dir, "dpu_qk.c"),
            os.path.join(qk_dir, "common.h"),
            os.path.join(qk_dir, "Makefile"),
        ]
        needs_build = not os.path.exists(binary_path)
        if not needs_build:
            binary_mtime = os.path.getmtime(binary_path)
            needs_build = any(os.path.getmtime(path) > binary_mtime for path in source_paths)

        if needs_build:
            build = subprocess.run(
                ["make", "all"],
                cwd=qk_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if build.returncode != 0:
                raise RuntimeError((build.stdout + build.stderr).strip())
        return binary_path

    def _ensure_qk_helper(self) -> subprocess.Popen:
        binary_path = self._ensure_qk_binary()
        qk_dir = os.path.join(self.repo_root, "src", "pim", "upmem_qk")
        if self.qk_helper is not None and self.qk_helper.poll() is None:
            return self.qk_helper

        self.qk_helper = subprocess.Popen(
            [binary_path, "--stdio", "--num-dpus", str(self.num_dpus)],
            cwd=qk_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.qk_helper_started = True
        self.qk_helper_restarts += 1
        return self.qk_helper

    def _restart_qk_helper(self) -> subprocess.Popen:
        if self.qk_helper is not None:
            try:
                self.qk_helper.kill()
            except Exception:
                pass
            self.qk_helper = None
        return self._ensure_qk_helper()

    def _run_qk_scores(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, float]:
        scores, scale = self._run_qk_scores_batch(query.unsqueeze(0), keys.unsqueeze(0))
        return scores[0], scale

    def _run_qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, float]:
        q = queries.detach().cpu().float().contiguous()
        k = keys.detach().cpu().float().contiguous()
        if q.dim() != 2:
            raise ValueError(f"queries must be 2D, got shape {tuple(q.shape)}")
        if k.dim() != 3:
            raise ValueError(f"keys must be 3D, got shape {tuple(k.shape)}")
        if q.shape[0] != k.shape[0]:
            raise ValueError(f"queries and keys batch mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")

        num_queries = int(q.shape[0])
        head_dim = min(128, int(q.shape[1]), int(k.shape[2]))
        if head_dim < 2:
            raise ValueError("head_dim must be at least 2 for UPMEM QK")
        if head_dim % 2 != 0:
            head_dim -= 1
        num_keys = min(128, int(k.shape[1]))

        scale = 1024.0
        q_i32 = torch.clamp(torch.round(q[:, :head_dim] * scale), -2**31, 2**31 - 1).to(torch.int32)
        k_i32 = torch.clamp(torch.round(k[:, :num_keys, :head_dim] * scale), -2**31, 2**31 - 1).to(torch.int32)
        expected = torch.einsum("qkd,qd->qk", k_i32.to(torch.int64), q_i32.to(torch.int64))

        # When resident KV already owns a persistent DPU set, route qk-mixed
        # through the same helper to avoid double DPU allocation.
        if isinstance(self.resident_store, UpmemKVSlotStore):
            actual = self.resident_store.qk_scores_batch(q_i32, k_i32)
            diff = torch.max(torch.abs(actual - expected)).item() if num_keys > 0 else 0
            self.qk_shadow_max_abs_diff = max(self.qk_shadow_max_abs_diff, int(diff))
            self.qk_shadow_last_scores = actual[0, : min(8, num_keys)].tolist()
            if diff != 0:
                raise RuntimeError(f"UPMEM qk score mismatch max_abs_diff={diff}")
            self.qk_check_count += num_queries
            self.qk_batch_calls += 1
            return actual.float() / (scale * scale), scale

        header = struct.pack("<IIII", 0x514B494F, head_dim, num_keys, num_queries)
        payload = header + q_i32.numpy().tobytes(order="C") + k_i32.numpy().tobytes(order="C")
        expected_bytes = 16 + (num_queries * num_keys * 8)

        helper = self._ensure_qk_helper()
        assert helper.stdin is not None
        assert helper.stdout is not None
        try:
            helper.stdin.write(payload)
            helper.stdin.flush()
            raw_output = helper.stdout.read(expected_bytes)
        except Exception:
            helper = self._restart_qk_helper()
            assert helper.stdin is not None
            assert helper.stdout is not None
            helper.stdin.write(payload)
            helper.stdin.flush()
            raw_output = helper.stdout.read(expected_bytes)

        if raw_output is None or len(raw_output) != expected_bytes:
            stderr_text = ""
            if helper.stderr is not None:
                try:
                    stderr_text = helper.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    stderr_text = ""
            self.qk_last_output = stderr_text.strip()
            raise RuntimeError(f"UPMEM qk helper returned incomplete output:\n{self.qk_last_output}")

        out_header = raw_output[:16]
        magic, out_head_dim, out_num_keys, out_num_queries = struct.unpack("<IIII", out_header)
        if magic != 0x514B494F or out_head_dim != head_dim or out_num_keys != num_keys or out_num_queries != num_queries:
            raise RuntimeError("UPMEM qk helper returned an invalid header")
        raw_scores = raw_output[16:]

        actual = torch.tensor(struct.unpack(f"<{num_queries * num_keys}q", raw_scores), dtype=torch.int64).view(num_queries, num_keys)
        diff = torch.max(torch.abs(actual - expected)).item() if num_keys > 0 else 0
        self.qk_shadow_max_abs_diff = max(self.qk_shadow_max_abs_diff, int(diff))
        self.qk_shadow_last_scores = actual[0, : min(8, num_keys)].tolist()
        if diff != 0:
            raise RuntimeError(f"UPMEM qk score mismatch max_abs_diff={diff}")
        self.qk_check_count += num_queries
        self.qk_batch_calls += 1
        return actual.float() / (scale * scale), scale

    def _run_smoke_test(self) -> None:
        # Backwards-compatible alias for older interactive sessions.
        smoke_dir = os.path.join(self.repo_root, "src", "pim", "upmem_dot")
        env = os.environ.copy()
        env["NUM_DPUS"] = str(self.num_dpus)
        env["LENGTH"] = str(self.length)
        completed = subprocess.run(
            ["make", "clean", "run"],
            cwd=smoke_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.smoke_test_output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "UPMEM smoke test failed for pim_naive backend:\n"
                f"{self.smoke_test_output}"
            )
        self.smoke_test_ok = True

    def init_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> int:
        seq_len = self.cpu_backend.init_request(request_id, initial_kv)
        self.request_states[request_id] = self._build_request_state(
            request_id,
            initial_kv,
            decode_reserve_tokens,
        )
        return seq_len

    def get_request_packing_hint(self, request_id: str) -> Dict[str, object]:
        request_state = self.request_states.get(str(request_id))
        if request_state is None:
            return {}
        return self._request_packing_hint(request_state)

    def _normalize_decode_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = query.detach().cpu().contiguous()
        k_new = key.detach().cpu().contiguous()
        v_new = value.detach().cpu().contiguous()

        if q.dim() == 3:
            q = q.squeeze(0)
        if k_new.dim() == 3:
            k_new = k_new.squeeze(0)
        if v_new.dim() == 3:
            v_new = v_new.squeeze(0)
        return q, k_new, v_new

    def _prepare_decode_record(self, item: Dict[str, object]) -> Dict[str, object]:
        request_id = str(item["request_id"])
        layer_idx = int(item["layer_idx"])
        score_scale = float(item.get("score_scale", 1.0))
        if request_id not in self.cpu_backend.k_cache:
            raise KeyError(f"Unknown request {request_id}")
        if request_id not in self.request_states:
            raise KeyError(f"Missing resident metadata for request {request_id}")

        q, k_new, v_new = self._normalize_decode_tensors(
            item["query"],
            item["key"],
            item["value"],
        )
        request_state = self.request_states[request_id]
        if layer_idx >= request_state.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for request {request_id} "
                f"with {request_state.num_layers} layers"
            )
        self._append_resident_kv(request_state, layer_idx, k_new, v_new)

        self.cpu_backend.k_cache[request_id][layer_idx] = torch.cat(
            [self.cpu_backend.k_cache[request_id][layer_idx], k_new.unsqueeze(0)], dim=0
        )
        self.cpu_backend.v_cache[request_id][layer_idx] = torch.cat(
            [self.cpu_backend.v_cache[request_id][layer_idx], v_new.unsqueeze(0)], dim=0
        )

        use_resident_av = self.resident_compute_enabled and self.resident_av_enabled
        if self.resident_compute_enabled and not use_resident_av:
            keys, values = self._materialize_layer_kv(request_state, layer_idx)
            self._update_resident_shadow_diff(request_id, layer_idx, keys, values)
        else:
            keys = self.cpu_backend.k_cache[request_id][layer_idx]
            values = self.cpu_backend.v_cache[request_id][layer_idx]

        q_fp32 = q if q.dtype == torch.float32 and q.is_contiguous() else q.to(torch.float32).contiguous()
        scores = None
        if not (self.resident_compute_enabled and self.qk_full_enabled):
            scores = torch.einsum("hd,lhd->hl", q_fp32, keys.float()) * score_scale
        return {
            "request_id": request_id,
            "request_state": request_state,
            "layer_idx": layer_idx,
            "query_dtype": q.dtype,
            "q_fp32": q_fp32,
            "keys": keys,
            "values": values,
            "scores": scores,
            "score_scale": score_scale,
            "use_resident_av": use_resident_av,
        }

    def _compute_host_scores(self, record: Dict[str, object]) -> torch.Tensor:
        return torch.einsum(
            "hd,lhd->hl",
            record["q_fp32"],
            self.cpu_backend.k_cache[record["request_id"]][record["layer_idx"]].float(),
        ) * float(record["score_scale"])

    def _finalize_ready_context_records(self, records: List[Dict[str, object]]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for record in records:
            if record["layer_idx"] == len(self.cpu_backend.k_cache[record["request_id"]]) - 1:
                self.cpu_backend.context_lens[record["request_id"]] += 1
            outputs.append(record["context"].unsqueeze(0))
        return outputs

    def _apply_qk_full_batch(self, records: List[Dict[str, object]]) -> None:
        if not self.qk_full_enabled:
            self.qk_full_shadow_last_max_abs_diff = 0.0
            return

        flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]] = []
        slot_query_refs: list[tuple[Dict[str, object], int, int]] = []

        for record in records:
            layer_state = record["request_state"].layer_states[record["layer_idx"]]
            record["full_qk_group_scores"] = []
            for group in layer_state.head_groups:
                flat_slot_queries.append(
                    (
                        group.k_slot,
                        group.v_slot,
                        list(range(group.group_heads)),
                        int(group.seq_len),
                        record["q_fp32"][group.head_start:group.head_end].contiguous(),
                    )
                )
                slot_query_refs.append((record, int(group.head_start), int(group.head_end)))

        if flat_slot_queries:
            slot_score_mats = self.resident_store.qk_slot_scores_batch(flat_slot_queries)
            self.qk_batch_calls += 1
            self.qk_full_batch_calls += 1
            for (record, head_start, head_end), score_mat in zip(slot_query_refs, slot_score_mats):
                record["full_qk_group_scores"].append((head_start, head_end, score_mat))

        self.qk_full_shadow_last_max_abs_diff = 0.0
        for record in records:
            group_scores = sorted(record.pop("full_qk_group_scores", []), key=lambda item: item[0])
            if not group_scores:
                if record["scores"] is None:
                    record["scores"] = self._compute_host_scores(record)
                continue

            scores = torch.cat([score_mat for _, _, score_mat in group_scores], dim=0).to(torch.float32)
            scores = scores * float(record["score_scale"])
            record["scores"] = scores
            self.qk_full_count += int(scores.shape[0])

            if self.qk_full_shadow_check:
                host_scores = self._compute_host_scores(record)
                diff = float(torch.max(torch.abs(scores - host_scores)).item()) if scores.numel() > 0 else 0.0
                self.qk_full_shadow_checks += 1
                self.qk_full_shadow_last_max_abs_diff = max(self.qk_full_shadow_last_max_abs_diff, diff)
                self.qk_full_shadow_max_abs_diff = max(self.qk_full_shadow_max_abs_diff, diff)

    def _apply_qk_context_fused_batch(self, records: List[Dict[str, object]]) -> None:
        flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]] = []
        slot_query_refs: list[tuple[Dict[str, object], int, int]] = []

        for record in records:
            layer_state = record["request_state"].layer_states[record["layer_idx"]]
            record["fused_group_contexts"] = []
            for group in layer_state.head_groups:
                local_head_indices = list(range(group.group_heads))
                flat_slot_queries.append(
                    (
                        group.k_slot,
                        group.v_slot,
                        local_head_indices,
                        int(group.seq_len),
                        record["q_fp32"][group.head_start:group.head_end].contiguous(),
                        float(record["score_scale"]),
                    )
                )
                slot_query_refs.append((record, int(group.head_start), int(group.head_end)))
                self.qk_full_count += int(group.group_heads)

        if flat_slot_queries:
            group_contexts = self.resident_store.qk_softmax_weighted_value_sum_batch(flat_slot_queries)
            self.qk_full_batch_calls += 1
            self.softmax_av_fused_batch_calls += 1
            for (record, head_start, head_end), context in zip(slot_query_refs, group_contexts):
                record["fused_group_contexts"].append((head_start, head_end, context))

        self.qk_full_shadow_last_max_abs_diff = 0.0
        need_qk_shadow = bool(self.qk_full_shadow_check)
        need_context_shadow = bool(self.softmax_av_shadow_check)

        if need_qk_shadow and flat_slot_queries:
            shadow_slot_queries = [
                (k_slot, v_slot, local_head_indices, window, queries)
                for k_slot, v_slot, local_head_indices, window, queries, _ in flat_slot_queries
            ]
            slot_score_mats = self.resident_store.qk_slot_scores_batch(shadow_slot_queries)
            self.qk_batch_calls += 1
            for (record, head_start, head_end), score_mat in zip(slot_query_refs, slot_score_mats):
                record.setdefault("shadow_group_scores", []).append((head_start, head_end, score_mat))

        for record in records:
            group_contexts = sorted(record.pop("fused_group_contexts", []), key=lambda item: item[0])
            if not group_contexts:
                record["context"] = torch.empty_like(record["q_fp32"])
                continue
            record["context"] = torch.cat([ctx for _, _, ctx in group_contexts], dim=0).to(record["query_dtype"])
            self.softmax_av_fused_ops += 1

            host_scores = None
            if need_qk_shadow or need_context_shadow:
                host_scores = self._compute_host_scores(record)

            if need_qk_shadow:
                group_scores = sorted(record.pop("shadow_group_scores", []), key=lambda item: item[0])
                if group_scores:
                    scores = torch.cat([score_mat for _, _, score_mat in group_scores], dim=0).to(torch.float32)
                    scores = scores * float(record["score_scale"])
                    diff = float(torch.max(torch.abs(scores - host_scores)).item()) if scores.numel() > 0 else 0.0
                    self.qk_full_shadow_checks += 1
                    self.qk_full_shadow_last_max_abs_diff = max(self.qk_full_shadow_last_max_abs_diff, diff)
                    self.qk_full_shadow_max_abs_diff = max(self.qk_full_shadow_max_abs_diff, diff)

            if need_context_shadow:
                weights = torch.softmax(host_scores, dim=-1)
                cpu_context = torch.einsum("hl,lhd->hd", weights, record["values"].float()).to(record["query_dtype"])
                av_diff = float(torch.max(torch.abs(record["context"].float() - cpu_context.float())).item())
                self.softmax_av_fused_shadow_max_abs_diff = max(self.softmax_av_fused_shadow_max_abs_diff, av_diff)

    def _apply_qk_mixed_batch(self, records: List[Dict[str, object]]) -> None:
        if not self.qk_mixed_enabled:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            return

        if len(records) > 1 or len(self.request_states) > 1:
            # Throughput-oriented concurrent serving often still reaches this
            # path with a singleton layer batch while multiple requests remain
            # active overall. In practice the mixed-QK overwrite pass is still
            # net-negative in that situation, so keep it for truly
            # single-request decode only.
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            return

        flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]] = []
        slot_query_refs: list[tuple[Dict[str, object], tuple[str, str]]] = []
        total_mixed_heads = 0

        for record in records:
            scores = record["scores"]
            keys = record["keys"]
            mixed_heads = min(self.qk_mixed_heads, int(scores.shape[0]))
            record["mixed_head_diffs"] = []
            record["head_to_slot_row"] = []
            record["slot_score_map"] = {}
            if mixed_heads <= 0:
                continue

            total_mixed_heads += mixed_heads
            layer_state = record["request_state"].layer_states[record["layer_idx"]]
            window = min(self.qk_mixed_window, int(keys.shape[0]))
            grouped_slot_queries: Dict[tuple[str, str], Dict[str, object]] = {}
            head_to_slot_row: list[tuple[tuple[str, str], int]] = []
            for head in range(mixed_heads):
                group = self._head_group_for_head(layer_state, head)
                slot_key = (group.k_slot, group.v_slot)
                if slot_key not in grouped_slot_queries:
                    grouped_slot_queries[slot_key] = {
                        "local_head_indices": [],
                        "head_rows": [],
                        "window": int(window),
                    }
                slot_entry = grouped_slot_queries[slot_key]
                slot_entry["local_head_indices"].append(int(head - group.head_start))
                slot_entry["head_rows"].append(int(head))
                head_to_slot_row.append((slot_key, len(slot_entry["local_head_indices"]) - 1))

            record["head_to_slot_row"] = head_to_slot_row
            for slot_key, entry in grouped_slot_queries.items():
                slot_query = (
                    slot_key[0],
                    slot_key[1],
                    list(entry["local_head_indices"]),
                    int(entry["window"]),
                    record["q_fp32"][list(entry["head_rows"])].contiguous(),
                )
                flat_slot_queries.append(slot_query)
                slot_query_refs.append((record, slot_key))

        if total_mixed_heads <= 0:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            return

        try:
            if flat_slot_queries:
                slot_score_mats = self.resident_store.qk_slot_scores_batch(flat_slot_queries)
                self.qk_batch_calls += 1
                for (record, slot_key), score_mat in zip(slot_query_refs, slot_score_mats):
                    record["slot_score_map"][slot_key] = score_mat.to(record["scores"].dtype) * float(record["score_scale"])

            head_diffs: list[float] = []
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            for record in records:
                for head, (slot_key, row_idx) in enumerate(record["head_to_slot_row"]):
                    head_scores = record["slot_score_map"][slot_key][row_idx]
                    cpu_window_scores = record["scores"][head, -int(head_scores.shape[0]) :].clone()
                    diff = float(torch.max(torch.abs(head_scores - cpu_window_scores)).item()) if head_scores.numel() > 0 else 0.0
                    record["mixed_head_diffs"].append(diff)
                    head_diffs.append(diff)
                    if not self.qk_mixed_last_diag and head_scores.numel() > 0:
                        dpu_has_nan = bool(torch.isnan(head_scores).any().item())
                        cpu_has_nan = bool(torch.isnan(cpu_window_scores).any().item())
                        dpu_has_inf = bool(torch.isinf(head_scores).any().item())
                        cpu_has_inf = bool(torch.isinf(cpu_window_scores).any().item())
                        if dpu_has_nan or cpu_has_nan or dpu_has_inf or cpu_has_inf:
                            self.qk_mixed_last_diag = {
                                "request_id": str(record["request_id"]),
                                "layer_idx": int(record["layer_idx"]),
                                "head": int(head),
                                "slot_key": [str(slot_key[0]), str(slot_key[1])],
                                "row_idx": int(row_idx),
                                "window": int(head_scores.shape[0]),
                                "dpu_has_nan": dpu_has_nan,
                                "cpu_has_nan": cpu_has_nan,
                                "dpu_has_inf": dpu_has_inf,
                                "cpu_has_inf": cpu_has_inf,
                                "dpu_preview": head_scores[: min(8, head_scores.shape[0])].detach().cpu().tolist(),
                                "cpu_preview": cpu_window_scores[: min(8, cpu_window_scores.shape[0])].detach().cpu().tolist(),
                            }
                    if head_scores.numel() > 0:
                        record["scores"][head, -int(head_scores.shape[0]) :] = head_scores

            self.qk_mixed_last_head_diffs = head_diffs
            self.qk_mixed_last_max_abs_diff = max(head_diffs) if head_diffs else 0.0
            self.qk_mixed_count += total_mixed_heads
        except Exception:
            self.qk_check_failures += 1
            raise

    def _finalize_decode_records(self, records: List[Dict[str, object]]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        flat_slot_weights: list[tuple[str, str, torch.Tensor]] = []
        slot_weight_refs: list[tuple[int, int]] = []
        flat_slot_scores: list[tuple[str, str, torch.Tensor]] = []
        slot_score_refs: list[tuple[int, int]] = []

        for record_idx, record in enumerate(records):
            use_fused_softmax_av = bool(record["use_resident_av"] and self.softmax_av_fused_enabled)
            if use_fused_softmax_av:
                layer_state = record["request_state"].layer_states[record["layer_idx"]]
                slot_scores = [
                    (
                        group.k_slot,
                        group.v_slot,
                        record["scores"][group.head_start:group.head_end, :].contiguous(),
                    )
                    for group in layer_state.head_groups
                ]
                flat_slot_scores.extend(slot_scores)
                slot_score_refs.append((record_idx, len(slot_scores)))
                if self.softmax_av_shadow_check:
                    weights = torch.softmax(record["scores"], dim=-1)
                    record["weights"] = weights
            else:
                weights = torch.softmax(record["scores"], dim=-1)
                record["weights"] = weights
                if record["use_resident_av"]:
                    layer_state = record["request_state"].layer_states[record["layer_idx"]]
                    slot_weights = [
                        (
                            group.k_slot,
                            group.v_slot,
                            weights[group.head_start:group.head_end, :].contiguous(),
                        )
                        for group in layer_state.head_groups
                    ]
                    flat_slot_weights.extend(slot_weights)
                    slot_weight_refs.append((record_idx, len(slot_weights)))
                else:
                    record["context"] = torch.einsum("hl,lhd->hd", weights, record["values"].float()).to(record["query_dtype"])

        if flat_slot_scores:
            group_contexts = self.resident_store.softmax_weighted_value_sum_batch(flat_slot_scores)
            self.softmax_av_fused_batch_calls += 1
            offset = 0
            for record_idx, group_count in slot_score_refs:
                record = records[record_idx]
                context = torch.cat(group_contexts[offset : offset + group_count], dim=0).to(record["query_dtype"])
                offset += group_count
                if self.softmax_av_shadow_check:
                    cpu_context = torch.einsum("hl,lhd->hd", record["weights"], record["values"].float()).to(record["query_dtype"])
                    av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                    self.softmax_av_fused_shadow_max_abs_diff = max(self.softmax_av_fused_shadow_max_abs_diff, av_diff)
                self.softmax_av_fused_ops += 1
                record["context"] = context

        if flat_slot_weights:
            group_contexts = self.resident_store.weighted_value_sum_batch(flat_slot_weights)
            self.resident_av_batch_calls += 1
            offset = 0
            for record_idx, group_count in slot_weight_refs:
                record = records[record_idx]
                context = torch.cat(group_contexts[offset : offset + group_count], dim=0).to(record["query_dtype"])
                offset += group_count
                cpu_context = torch.einsum("hl,lhd->hd", record["weights"], record["values"].float()).to(record["query_dtype"])
                av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                self.resident_av_shadow_max_abs_diff = max(self.resident_av_shadow_max_abs_diff, av_diff)
                self.resident_av_ops += 1
                record["context"] = context

        for record in records:
            if record["layer_idx"] == len(self.cpu_backend.k_cache[record["request_id"]]) - 1:
                self.cpu_backend.context_lens[record["request_id"]] += 1
            outputs.append(record["context"].unsqueeze(0))
        return outputs

    def decode_layer_batch(self, items: List[Dict[str, object]]) -> List[torch.Tensor]:
        if not items:
            return []
        self.decode_batch_calls += 1
        self.decode_batch_items += len(items)
        records = [self._prepare_decode_record(item) for item in items]
        use_qk_context_fused = (
            self.qk_full_enabled
            and self.resident_compute_enabled
            and self.resident_av_enabled
            and self.softmax_av_fused_enabled
        )
        if use_qk_context_fused:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            self._apply_qk_context_fused_batch(records)
            return self._finalize_ready_context_records(records)
        if self.qk_full_enabled and self.resident_compute_enabled:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            self._apply_qk_full_batch(records)
        else:
            self.qk_full_shadow_last_max_abs_diff = 0.0
            self._apply_qk_mixed_batch(records)
        return self._finalize_decode_records(records)

    def decode_layer(
        self,
        request_id: str,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.decode_layer_batch(
            [
                {
                    "request_id": request_id,
                    "layer_idx": layer_idx,
                    "query": query,
                    "key": key,
                    "value": value,
                    "score_scale": score_scale,
                }
            ]
        )[0]

    def get_context_len(self, request_id: str) -> int:
        return self.cpu_backend.get_context_len(request_id)

    def free_request(self, request_id: str) -> None:
        self.cpu_backend.free_request(request_id)
        request_state = self.request_states.pop(request_id, None)
        if request_state is not None:
            for layer_state in request_state.layer_states:
                for group in layer_state.head_groups:
                    self.resident_store.free_group(group.k_slot, group.v_slot)
            self.last_freed_request_id = request_id

    def get_debug_info(self) -> Dict[str, object]:
        request_preview = None
        request_footprints = []
        if self.request_states:
            preview_key = next(iter(self.request_states))
            request_preview = self._summarize_request_state(self.request_states[preview_key])
            for request_state in self.request_states.values():
                footprint = self._request_footprint_summary(request_state)
                request_footprints.append(
                    {
                        "request_id": request_state.request_id,
                        "context_len": int(request_state.context_len),
                        "num_layers": int(request_state.num_layers),
                        "preferred_dpu_stripe": [int(physical_dpu) for physical_dpu in request_state.preferred_dpu_stripe],
                        "stripe_width": len(request_state.preferred_dpu_stripe),
                        "planner_mode": str(
                            dict((request_state.sharding_plan or {}).get("metadata", {}) or {}).get(
                                "planner_mode",
                                "",
                            )
                        ),
                        "stripe_version": int(request_state.stripe_version),
                        "stripe_expand_count": int(request_state.stripe_expand_count),
                        "last_stripe_update_reason": request_state.last_stripe_update_reason,
                        "last_stripe_width": int(request_state.last_stripe_width),
                        "live_elems": int(footprint["live_elems"]),
                        "capacity_elems": int(footprint["capacity_elems"]),
                        "per_dpu_live_elems": footprint["per_dpu_live_elems"],
                        "per_dpu_capacity_elems": footprint["per_dpu_capacity_elems"],
                    }
                )
        total_live_elems = sum(int(item["live_elems"]) for item in request_footprints)
        total_capacity_elems = sum(int(item["capacity_elems"]) for item in request_footprints)
        return {
            "smoke_test_ok": self.smoke_test_ok,
            "num_dpus": self.num_dpus,
            "length": self.length,
            "block_tokens": self.block_tokens,
            "smoke_test_output": self.smoke_test_output,
            "qk_check_interval": self.qk_check_interval,
            "qk_check_limit": self.qk_check_limit,
            "qk_check_count": self.qk_check_count,
            "qk_check_failures": self.qk_check_failures,
            "qk_last_output": self.qk_last_output,
            "qk_shadow_max_abs_diff": self.qk_shadow_max_abs_diff,
            "qk_shadow_last_scores": self.qk_shadow_last_scores,
            "qk_full_enabled": self.qk_full_enabled,
            "qk_full_shadow_check": self.qk_full_shadow_check,
            "qk_full_count": self.qk_full_count,
            "qk_full_batch_calls": self.qk_full_batch_calls,
            "qk_full_shadow_checks": self.qk_full_shadow_checks,
            "qk_full_shadow_max_abs_diff": self.qk_full_shadow_max_abs_diff,
            "qk_full_shadow_last_max_abs_diff": self.qk_full_shadow_last_max_abs_diff,
            "softmax_av_fused_enabled": self.softmax_av_fused_enabled,
            "softmax_av_shadow_check": self.softmax_av_shadow_check,
            "softmax_av_fused_ops": self.softmax_av_fused_ops,
            "softmax_av_fused_batch_calls": self.softmax_av_fused_batch_calls,
            "softmax_av_fused_shadow_max_abs_diff": self.softmax_av_fused_shadow_max_abs_diff,
            "qk_mixed_enabled": self.qk_mixed_enabled,
            "qk_mixed_heads": self.qk_mixed_heads,
            "qk_mixed_window": self.qk_mixed_window,
            "host_partial_reduce_enabled": self.host_partial_reduce_enabled,
            "compact_short_segments_enabled": self.compact_short_segments_enabled,
            "compact_short_segment_min_tokens": self.compact_short_segment_min_tokens,
            "planner_segment_plan_count": self.planner_segment_plan_count,
            "planner_segment_materialized_count": self.planner_segment_materialized_count,
            "planner_segment_compacted_count": self.planner_segment_compacted_count,
            "planner_segment_last_decision": self.planner_segment_last_decision,
            "qk_mixed_count": self.qk_mixed_count,
            "qk_mixed_last_max_abs_diff": self.qk_mixed_last_max_abs_diff,
            "qk_mixed_last_head_diffs": self.qk_mixed_last_head_diffs,
            "qk_mixed_last_diag": self.qk_mixed_last_diag,
            "qk_mixed_last_diag_path": self.qk_mixed_last_diag_path,
            "qk_batch_calls": self.qk_batch_calls,
            "decode_batch_calls": self.decode_batch_calls,
            "decode_batch_items": self.decode_batch_items,
            "qk_helper_started": self.qk_helper_started,
            "qk_helper_restarts": self.qk_helper_restarts,
            "resident_metadata_enabled": self.resident_metadata_enabled,
            "resident_compute_enabled": self.resident_compute_enabled,
            "resident_store_backend": self.resident_store_backend,
            "resident_kv_dtype": self.resident_kv_dtype,
            "resident_av_enabled": self.resident_av_enabled,
            "max_resident_groups_per_layer": self.max_resident_groups_per_layer,
            "head_grouping_policy": self.head_grouping_policy,
            "dpu_placement_policy": self.dpu_placement_policy,
            "resident_request_count": len(self.request_states),
            "resident_last_freed_request_id": self.last_freed_request_id,
            "resident_append_ops": self.resident_append_ops,
            "resident_materialize_ops": self.resident_materialize_ops,
            "resident_shadow_max_abs_diff": self.resident_shadow_max_abs_diff,
            "resident_av_ops": self.resident_av_ops,
            "resident_av_batch_calls": self.resident_av_batch_calls,
            "resident_av_shadow_max_abs_diff": self.resident_av_shadow_max_abs_diff,
            "init_rank_locality_reuse_count": self.init_rank_locality_reuse_count,
            "init_rank_hash_fallback_count": self.init_rank_hash_fallback_count,
            "init_rank_last_reason": self.init_rank_last_reason,
            "resident_total_live_elems": total_live_elems,
            "resident_total_capacity_elems": total_capacity_elems,
            "resident_request_footprints": request_footprints,
            "resident_store_debug": self.resident_store.get_debug_info(),
            "resident_request_preview": request_preview,
        }
