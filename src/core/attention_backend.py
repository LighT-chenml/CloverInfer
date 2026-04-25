from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
import subprocess
from typing import Dict, List

import torch

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

    def get_context_len(self, request_id: str) -> int:
        return self.context_lens[request_id]

    def free_request(self, request_id: str) -> None:
        self.k_cache.pop(request_id, None)
        self.v_cache.pop(request_id, None)
        self.context_lens.pop(request_id, None)


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
        resident_store_backend: str = "host",
        max_resident_groups_per_layer: int = 0,
        head_grouping_policy: str = "balanced",
        dpu_placement_policy: str = "rotated",
        resident_kv_dtype: str = "fp32",
        qk_check_interval: int = 1,
        qk_check_limit: int = 1,
        qk_mixed_enabled: bool = True,
        qk_mixed_heads: int = 2,
        qk_mixed_window: int = 128,
    ):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.num_dpus = num_dpus
        self.length = length
        self.resident_store_backend = resident_store_backend
        self.max_resident_groups_per_layer = max(0, int(max_resident_groups_per_layer))
        self.head_grouping_policy = str(head_grouping_policy)
        self.dpu_placement_policy = str(dpu_placement_policy)
        self.resident_kv_dtype = str(resident_kv_dtype)
        self.qk_check_interval = qk_check_interval
        self.qk_check_limit = qk_check_limit
        if self.head_grouping_policy not in {"legacy", "balanced"}:
            raise ValueError(f"Unsupported head_grouping_policy: {self.head_grouping_policy}")
        if self.dpu_placement_policy not in {"identity", "rotated"}:
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
        self.qk_mixed_enabled = bool(qk_mixed_enabled)
        self.qk_mixed_heads = max(0, int(qk_mixed_heads))
        self.qk_mixed_window = max(1, int(qk_mixed_window))
        self.qk_mixed_count = 0
        self.qk_mixed_last_max_abs_diff = 0.0
        self.qk_mixed_last_head_diffs = []
        self.qk_batch_calls = 0
        self.qk_helper_started = False
        self.qk_helper_restarts = 0
        self.qk_helper: subprocess.Popen | None = None
        self.resident_metadata_enabled = True
        self.resident_compute_enabled = True
        if resident_store_backend == "upmem_kvslot":
            self.resident_store = UpmemKVSlotStore(self.repo_root, self.num_dpus, kv_dtype=self.resident_kv_dtype)
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
        self.resident_av_shadow_max_abs_diff = 0.0
        self._run_dot_smoke_test()

    def _shares_persistent_dpu_owner(self) -> bool:
        return isinstance(self.resident_store, UpmemKVSlotStore)

    def _group_footprint_summary(self, group: HeadGroupState) -> Dict[str, object]:
        return {
            "dpu_id": int(group.dpu_id),
            "heads": [int(group.head_start), int(group.head_end)],
            "group_heads": int(group.group_heads),
            "seq_len": int(group.seq_len),
            "capacity": int(group.capacity),
            "head_dim": int(group.head_dim),
            "live_elems": int(group.live_elems),
            "capacity_elems": int(group.capacity_elems),
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
                physical_dpu = int(group.dpu_id) % max(self.num_dpus, 1)
                per_dpu_live_elems[physical_dpu] += int(group.live_elems)
                per_dpu_capacity_elems[physical_dpu] += int(group.capacity_elems)
        return {
            "request_id": request_state.request_id,
            "context_len": int(request_state.context_len),
            "num_layers": int(request_state.num_layers),
            "live_elems": int(live_elems),
            "capacity_elems": int(capacity_elems),
            "per_dpu_live_elems": per_dpu_live_elems,
            "per_dpu_capacity_elems": per_dpu_capacity_elems,
            "layers": layer_summaries,
        }

    def _build_head_groups(
        self,
        request_id: str,
        layer_idx: int,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
        decode_reserve_tokens: int = 0,
    ) -> List[HeadGroupState]:
        seq_len, num_heads, head_dim = (int(dim) for dim in layer_key.shape)
        if num_heads <= 0:
            raise ValueError(f"layer {layer_idx} in request {request_id} has no attention heads")

        num_groups = max(1, min(self.num_dpus, num_heads))
        capacity = max(self.length, seq_len + max(0, int(decode_reserve_tokens)))
        head_groups = []
        request_hash = sum(ord(ch) for ch in request_id)
        dpu_rotation = (request_hash + int(layer_idx)) % max(self.num_dpus, 1)
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

        for group_idx, (head_start, head_end) in enumerate(group_ranges):
            if self.dpu_placement_policy == "rotated":
                physical_dpu = (group_idx + dpu_rotation) % max(self.num_dpus, 1)
            else:
                physical_dpu = group_idx % max(self.num_dpus, 1)
            k_slot = f"{request_id}:layer{layer_idx}:group{group_idx}:k"
            v_slot = f"{request_id}:layer{layer_idx}:group{group_idx}:v"
            initial_k_group = layer_key[:, head_start:head_end, :].contiguous()
            initial_v_group = layer_value[:, head_start:head_end, :].contiguous()
            self.resident_store.allocate_group(
                k_slot,
                v_slot,
                initial_k_group,
                initial_v_group,
                capacity=capacity,
                preferred_dpu=physical_dpu,
                force_host_fallback=self.max_resident_groups_per_layer > 0 and group_idx >= self.max_resident_groups_per_layer,
            )
            head_groups.append(
                HeadGroupState(
                    dpu_id=physical_dpu,
                    head_start=head_start,
                    head_end=head_end,
                    seq_len=seq_len,
                    capacity=capacity,
                    head_dim=head_dim,
                    k_slot=k_slot,
                    v_slot=v_slot,
                )
            )
        return head_groups

    def _build_request_state(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> RequestState:
        layer_states = []
        if not initial_kv:
            raise ValueError("initial_kv must contain at least one layer")

        context_len = int(initial_kv[0]["key"].shape[0])
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
                    ),
                )
            )

        return RequestState(
            request_id=request_id,
            context_len=context_len,
            num_layers=len(layer_states),
            layer_states=layer_states,
        )

    def _append_resident_kv(
        self,
        request_state: RequestState,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        layer_state = request_state.layer_states[layer_idx]
        expected_seq_len = request_state.context_len + 1
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
            append_info = self.resident_store.append_group(
                group.k_slot,
                group.v_slot,
                group_k_new,
                group_v_new,
            )
            group.seq_len = int(append_info["seq_len"])
            group.capacity = int(append_info["capacity"])
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

    def decode_layer(
        self,
        request_id: str,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_scale: float = 1.0,
    ) -> torch.Tensor:
        if request_id not in self.cpu_backend.k_cache:
            raise KeyError(f"Unknown request {request_id}")
        if request_id not in self.request_states:
            raise KeyError(f"Missing resident metadata for request {request_id}")

        q = query.detach().cpu().contiguous()
        k_new = key.detach().cpu().contiguous()
        v_new = value.detach().cpu().contiguous()

        if q.dim() == 3:
            q = q.squeeze(0)
        if k_new.dim() == 3:
            k_new = k_new.squeeze(0)
        if v_new.dim() == 3:
            v_new = v_new.squeeze(0)

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

        scores = torch.einsum("hd,lhd->hl", q.float(), keys.float()) * float(score_scale)
        if self.qk_mixed_enabled:
            try:
                window = min(self.qk_mixed_window, keys.shape[0])
                mixed_heads = min(self.qk_mixed_heads, scores.shape[0])
                head_diffs = []
                if mixed_heads > 0:
                    batched_scores, _ = self._run_qk_scores_batch(
                        q[:mixed_heads],
                        keys[-window:, :mixed_heads, :].permute(1, 0, 2).contiguous(),
                    )
                    batched_scores = batched_scores * float(score_scale)
                    for head in range(mixed_heads):
                        cpu_window_scores = scores[head, -window:].clone()
                        diff = float(torch.max(torch.abs(batched_scores[head] - cpu_window_scores)).item())
                        head_diffs.append(diff)
                        scores[head, -window:] = batched_scores[head].to(scores.dtype)
                self.qk_mixed_last_head_diffs = head_diffs
                self.qk_mixed_last_max_abs_diff = max(head_diffs) if head_diffs else 0.0
                self.qk_mixed_count += mixed_heads
            except Exception:
                self.qk_check_failures += 1
                raise

        weights = torch.softmax(scores, dim=-1)
        if use_resident_av:
            layer_state = request_state.layer_states[layer_idx]
            group_contexts = []
            for group in layer_state.head_groups:
                group_weights = weights[group.head_start:group.head_end, :].contiguous()
                group_contexts.append(
                    self.resident_store.weighted_value_sum(group.k_slot, group.v_slot, group_weights)
                )
            context = torch.cat(group_contexts, dim=0).to(query.dtype)
            cpu_context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)
            av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
            self.resident_av_shadow_max_abs_diff = max(self.resident_av_shadow_max_abs_diff, av_diff)
            self.resident_av_ops += 1
        else:
            context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)

        if layer_idx == len(self.cpu_backend.k_cache[request_id]) - 1:
            self.cpu_backend.context_lens[request_id] += 1

        return context.unsqueeze(0)

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
            "smoke_test_output": self.smoke_test_output,
            "qk_check_interval": self.qk_check_interval,
            "qk_check_limit": self.qk_check_limit,
            "qk_check_count": self.qk_check_count,
            "qk_check_failures": self.qk_check_failures,
            "qk_last_output": self.qk_last_output,
            "qk_shadow_max_abs_diff": self.qk_shadow_max_abs_diff,
            "qk_shadow_last_scores": self.qk_shadow_last_scores,
            "qk_mixed_enabled": self.qk_mixed_enabled,
            "qk_mixed_heads": self.qk_mixed_heads,
            "qk_mixed_window": self.qk_mixed_window,
            "qk_mixed_count": self.qk_mixed_count,
            "qk_mixed_last_max_abs_diff": self.qk_mixed_last_max_abs_diff,
            "qk_mixed_last_head_diffs": self.qk_mixed_last_head_diffs,
            "qk_batch_calls": self.qk_batch_calls,
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
            "resident_av_shadow_max_abs_diff": self.resident_av_shadow_max_abs_diff,
            "resident_total_live_elems": total_live_elems,
            "resident_total_capacity_elems": total_capacity_elems,
            "resident_request_footprints": request_footprints,
            "resident_store_debug": self.resident_store.get_debug_info(),
            "resident_request_preview": request_preview,
        }
