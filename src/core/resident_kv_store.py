from __future__ import annotations

from dataclasses import dataclass
import math
import os
import numpy as np
import struct
import subprocess
import shutil
import time
from typing import Dict

import torch


@dataclass
class _HostKVSlot:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int
    capacity: int
    group_heads: int
    head_dim: int


class ResidentKVStore:
    """Storage/runtime boundary for resident KV partitions.

    The first implementation is host-backed, but the attention backend talks to
    it through this interface so the storage path can later move toward a
    DPU-managed implementation without rewriting request lifecycle logic.
    """

    backend_name = "unknown"

    def allocate_group(
        self,
        k_slot: str,
        v_slot: str,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
        preferred_dpu: int | None = None,
        force_host_fallback: bool = False,
    ) -> None:
        raise NotImplementedError

    def append_group(
        self,
        k_slot: str,
        v_slot: str,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Dict[str, int]:
        raise NotImplementedError

    def materialize_group(self, k_slot: str, v_slot: str) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def slot_debug(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        raise NotImplementedError

    def free_group(self, k_slot: str, v_slot: str) -> None:
        raise NotImplementedError

    def get_debug_info(self) -> Dict[str, object]:
        raise NotImplementedError

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    def weighted_value_sum(self, k_slot: str, v_slot: str, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weighted_value_sum_batch(self, slot_weights: list[tuple[str, str, torch.Tensor]]) -> list[torch.Tensor]:
        raise NotImplementedError

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    def qk_softmax_weighted_value_sum_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]],
    ) -> list[torch.Tensor]:
        raise NotImplementedError


class _KVSlotHelperClient:
    MAGIC = 0x4B56534C
    CMD_ALLOCATE = 1
    CMD_APPEND = 2
    CMD_READBACK = 3
    CMD_FREE = 4
    CMD_GET_STATS = 5
    CMD_QK_BATCH = 6
    CMD_AV = 7
    CMD_AV_BATCH = 8
    CMD_QK_SLOT_BATCH = 9
    CMD_SOFTMAX_AV_BATCH = 10
    CMD_QK_SOFTMAX_AV_BATCH = 11
    CMD_QK_SOFTMAX_AV_PARTIAL_BATCH = 12
    CMD_GET_PROFILE = 13
    CMD_GET_TOPOLOGY = 14
    MAX_SLOTS_PER_DPU = 256
    MAX_BATCH_ITEMS = 32

    def __init__(self, binary_path: str, num_dpus: int, cwd: str, kv_dtype: str = "fp32"):
        self.binary_path = binary_path
        self.num_dpus = num_dpus
        self.cwd = cwd
        self.kv_dtype = str(kv_dtype)
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.persistent_state_active = False
        self.helper_env: Dict[str, str] = {}
        # Host-side micro profiling to understand overheads (dtype conversion,
        # serialization, pipe I/O) when running fp16 resident KV experiments.
        self.host_profile_totals_s: Dict[str, float] = {
            "allocate_group_write": 0.0,
            "allocate_group_read": 0.0,
            "append_group_write": 0.0,
            "append_group_read": 0.0,
            "materialize_group_read": 0.0,
        }
        self.host_profile_counts: Dict[str, int] = {key: 0 for key in self.host_profile_totals_s}

    def set_env_flag(self, name: str, enabled: bool) -> None:
        value = "1" if enabled else "0"
        if self.helper_env.get(name) == value:
            return
        self.helper_env[name] = value
        if self.proc is not None and self.proc.poll() is None:
            if self.persistent_state_active:
                raise RuntimeError(
                    f"cannot reconfigure kvslot helper flag {name} while persistent DPU state is active"
                )
            self.close()

    def _ensure_proc(self) -> subprocess.Popen:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        if self.proc is not None and self.proc.poll() is not None:
            stderr_text = ""
            if self.proc.stderr is not None:
                try:
                    stderr_text = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    stderr_text = ""
            if self.persistent_state_active:
                # In long-running experiments the helper can crash (e.g. transient
                # driver issues). Raise, but include a hint to help debugging.
                raise RuntimeError(
                    "kvslot helper exited while persistent DPU state was active: "
                    f"{stderr_text.strip()}"
                )
        env = os.environ.copy()
        env.update(self.helper_env)
        self.proc = subprocess.Popen(
            [self.binary_path, "--stdio", "--num-dpus", str(self.num_dpus)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.restarts += 1
        return self.proc

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        self.persistent_state_active = False
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _dtype_code(self) -> int:
        return 1 if self.kv_dtype == "fp16" else 0

    def _elem_bytes(self) -> int:
        return 2 if self.kv_dtype == "fp16" else 4

    def _write(self, payload: bytes) -> None:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.flush()

    def _write_parts(self, parts: list[bytes | memoryview]) -> None:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        for part in parts:
            proc.stdin.write(part)
        proc.stdin.flush()

    def _read_exact(self, n: int) -> bytes:
        proc = self._ensure_proc()
        assert proc.stdout is not None
        data = proc.stdout.read(n)
        if data is None or len(data) != n:
            stderr_text = ""
            if proc.stderr is not None:
                try:
                    stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    stderr_text = ""
            raise RuntimeError(f"kvslot helper returned incomplete output: {stderr_text.strip()}")
        return data

    def allocate_group(self, slot_id: int, capacity: int, initial_k: torch.Tensor, initial_v: torch.Tensor) -> Dict[str, int]:
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        header = struct.pack("<IIII", self.MAGIC, self.CMD_ALLOCATE, slot_id, 0)
        args = struct.pack("<IIIII", int(capacity), seq_len, group_heads, head_dim, self._dtype_code())
        # Avoid building a single large bytes object (extra copy). The tensors
        # are already CPU-contiguous from the resident store encode path.
        started = time.perf_counter()
        k_np = initial_k.numpy()
        v_np = initial_v.numpy()
        self._write_parts([header, args, memoryview(k_np).cast("B"), memoryview(v_np).cast("B")])
        self.host_profile_totals_s["allocate_group_write"] += float(time.perf_counter() - started)
        self.host_profile_counts["allocate_group_write"] += 1
        started = time.perf_counter()
        out = struct.unpack("<IIIII", self._read_exact(20))
        self.host_profile_totals_s["allocate_group_read"] += float(time.perf_counter() - started)
        self.host_profile_counts["allocate_group_read"] += 1
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
        }

    def append_group(self, slot_id: int, k_new: torch.Tensor, v_new: torch.Tensor) -> Dict[str, int]:
        append_len, group_heads, head_dim = (int(dim) for dim in k_new.shape)
        header = struct.pack("<IIII", self.MAGIC, self.CMD_APPEND, slot_id, 0)
        args = struct.pack("<IIIII", 0, append_len, group_heads, head_dim, self._dtype_code())
        started = time.perf_counter()
        k_np = k_new.numpy()
        v_np = v_new.numpy()
        self._write_parts([header, args, memoryview(k_np).cast("B"), memoryview(v_np).cast("B")])
        self.host_profile_totals_s["append_group_write"] += float(time.perf_counter() - started)
        self.host_profile_counts["append_group_write"] += 1
        started = time.perf_counter()
        out = struct.unpack("<IIIII", self._read_exact(20))
        self.host_profile_totals_s["append_group_read"] += float(time.perf_counter() - started)
        self.host_profile_counts["append_group_read"] += 1
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
        }

    def materialize_group(self, slot_id: int) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_READBACK, slot_id, 0)
        self._write(header)
        out = struct.unpack("<IIIII", self._read_exact(20))
        capacity, seq_len, group_heads, head_dim = (int(item) for item in out[:4])
        elems = seq_len * group_heads * head_dim
        elem_bytes = self._elem_bytes()
        if elems == 0:
            dtype = torch.int16 if elem_bytes == 2 else torch.int32
            k = torch.empty((0, group_heads, head_dim), dtype=dtype)
            v = torch.empty((0, group_heads, head_dim), dtype=dtype)
        else:
            started = time.perf_counter()
            k_bytes = self._read_exact(elems * elem_bytes)
            v_bytes = self._read_exact(elems * elem_bytes)
            # struct.unpack is very slow for large payloads; use frombuffer.
            if elem_bytes == 2:
                k = torch.from_numpy(np.frombuffer(k_bytes, dtype="<i2").copy()).view(seq_len, group_heads, head_dim)
                v = torch.from_numpy(np.frombuffer(v_bytes, dtype="<i2").copy()).view(seq_len, group_heads, head_dim)
            else:
                k = torch.from_numpy(np.frombuffer(k_bytes, dtype="<i4").copy()).view(seq_len, group_heads, head_dim)
                v = torch.from_numpy(np.frombuffer(v_bytes, dtype="<i4").copy()).view(seq_len, group_heads, head_dim)
            self.host_profile_totals_s["materialize_group_read"] += float(time.perf_counter() - started)
            self.host_profile_counts["materialize_group_read"] += 1
        return k, v, {
            "capacity": capacity,
            "seq_len": seq_len,
            "group_heads": group_heads,
            "head_dim": head_dim,
        }

    def free_group(self, slot_id: int) -> None:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_FREE, slot_id, 0)
        self._write(header)
        self._read_exact(20)

    def get_allocator_stats(self) -> list[Dict[str, int]]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_GET_STATS, 0, 0)
        self._write(header)
        stats = []
        for dpu_id in range(self.num_dpus):
            out = struct.unpack("<IIIIII", self._read_exact(24))
            stats.append(
                {
                    "dpu_id": dpu_id,
                    "next_free_elem": int(out[0]),
                    "free_range_count": int(out[1]),
                    "free_elems_total": int(out[2]),
                    "largest_free_range": int(out[3]),
                    "live_slot_count": int(out[4]),
                    "live_elems_total": int(out[5]),
                }
            )
        return stats

    def get_profile_stats(self) -> Dict[str, int]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_GET_PROFILE, 0, 0)
        self._write(header)
        out = struct.unpack("<32Q", self._read_exact(32 * 8))
        keys = [
            "qk_rounds_total",
            "qk_batched_rounds",
            "qk_fallback_rounds",
            "qk_round_items_total",
            "qk_batched_items_total",
            "qk_active_ranks_total",
            "qk_max_round_size",
            "qk_max_active_ranks",
            "av_rounds_total",
            "av_batched_rounds",
            "av_fallback_rounds",
            "av_round_items_total",
            "av_batched_items_total",
            "av_active_ranks_total",
            "av_max_round_size",
            "av_max_active_ranks",
            "qk_batched_round_total_ns",
            "qk_batched_xfer_to_ns",
            "qk_batched_launch_ns",
            "qk_batched_xfer_from_ns",
            "qk_fallback_round_total_ns",
            "qk_fallback_launch_ns",
            "qk_fallback_sync_ns",
            "qk_fallback_xfer_from_ns",
            "av_batched_round_total_ns",
            "av_batched_xfer_to_ns",
            "av_batched_launch_ns",
            "av_batched_xfer_from_ns",
            "av_fallback_round_total_ns",
            "av_fallback_launch_ns",
            "av_fallback_sync_ns",
            "av_fallback_xfer_from_ns",
        ]
        return {key: int(value) for key, value in zip(keys, out)}

    def get_topology(self) -> Dict[str, object]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_GET_TOPOLOGY, 0, 0)
        self._write(header)
        raw_header = struct.unpack("<IIII", self._read_exact(16))
        nr_dpus = int(raw_header[0])
        nr_ranks = int(raw_header[1])
        items = []
        for _ in range(nr_dpus):
            logical_dpu_id, rank_index, rank_id, _reserved = struct.unpack("<IIII", self._read_exact(16))
            items.append(
                {
                    "logical_dpu_id": int(logical_dpu_id),
                    "rank_index": int(rank_index),
                    "rank_id": int(rank_id),
                }
            )
        return {
            "nr_dpus": nr_dpus,
            "nr_ranks": nr_ranks,
            "items": items,
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if queries.dim() != 2:
            raise ValueError(f"queries must be 2D, got shape {tuple(queries.shape)}")
        if keys.dim() != 3:
            raise ValueError(f"keys must be 3D, got shape {tuple(keys.shape)}")
        if int(queries.shape[0]) != int(keys.shape[0]):
            raise ValueError(f"queries and keys batch mismatch: {tuple(queries.shape)} vs {tuple(keys.shape)}")

        num_queries = int(queries.shape[0])
        head_dim = int(queries.shape[1])
        num_keys = int(keys.shape[1])
        if int(keys.shape[2]) != head_dim:
            raise ValueError(f"key head_dim mismatch: queries={head_dim} keys={int(keys.shape[2])}")

        q_i32 = queries.detach().cpu().to(torch.int32).contiguous()
        k_i32 = keys.detach().cpu().to(torch.int32).contiguous()
        q_np = q_i32.numpy()
        k_np = k_i32.numpy()
        self._write_parts(
            [
                struct.pack("<IIII", self.MAGIC, self.CMD_QK_BATCH, 0, 0),
                struct.pack("<IIII", head_dim, num_keys, num_queries, 0),
                memoryview(q_np).cast("B"),
                memoryview(k_np).cast("B"),
            ]
        )
        out_header = struct.unpack("<IIII", self._read_exact(16))
        out_head_dim, out_num_keys, out_num_queries = (int(out_header[0]), int(out_header[1]), int(out_header[2]))
        if out_head_dim != head_dim or out_num_keys != num_keys or out_num_queries != num_queries:
            raise RuntimeError("kvslot helper returned an invalid qk batch header")
        raw_scores = self._read_exact(num_queries * num_keys * 8)
        scores = torch.from_numpy(np.frombuffer(raw_scores, dtype="<i8").copy()).view(num_queries, num_keys)
        return scores

    def qk_slot_scores_batch(self, slot_queries: list[tuple[int, list[int], int, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        if len(slot_queries) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(slot_queries), self.MAX_BATCH_ITEMS):
                outputs.extend(self.qk_slot_scores_batch(slot_queries[offset : offset + self.MAX_BATCH_ITEMS]))
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_QK_SLOT_BATCH, 0, 0),
            struct.pack("<IIII", int(len(slot_queries)), 0, 0, 0),
        ]
        expected_meta = []

        for slot_id, local_head_indices, window, queries in slot_queries:
            q = queries if queries.device.type == "cpu" and queries.dtype == torch.float32 and queries.is_contiguous() else queries.detach().cpu().to(torch.float32).contiguous()
            if q.dim() != 2:
                raise ValueError(f"slot queries must be 2D, got shape {tuple(q.shape)}")
            num_heads = int(q.shape[0])
            head_dim = int(q.shape[1])
            if num_heads != len(local_head_indices):
                raise ValueError(
                    f"slot query head count mismatch: queries={num_heads} local_head_indices={len(local_head_indices)}"
                )
            payload_parts.append(struct.pack("<I", int(slot_id)))
            payload_parts.append(
                struct.pack(
                    "<IIII",
                    num_heads,
                    int(window),
                    head_dim,
                    0,
                )
            )
            payload_parts.append(struct.pack(f"<{num_heads}I", *[int(idx) for idx in local_head_indices]))
            payload_parts.append(memoryview(q.numpy()).cast("B"))
            expected_meta.append((num_heads, int(window)))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_items = int(out_args[0])
        if out_num_items != len(slot_queries):
            raise RuntimeError(
                f"kvslot helper returned invalid qk slot batch header: expected={len(slot_queries)} actual={out_num_items}"
            )

        scores_per_item: list[torch.Tensor] = []
        for idx, (expected_heads, expected_window) in enumerate(expected_meta):
            out = struct.unpack("<IIII", self._read_exact(16))
            actual_heads = int(out[0])
            actual_window = int(out[1])
            if actual_heads != expected_heads or actual_window != expected_window:
                raise RuntimeError(
                    "kvslot helper returned invalid qk slot batch item header: "
                    f"index={idx} expected=({expected_heads}, {expected_window}) actual=({actual_heads}, {actual_window})"
                )
            raw_scores = self._read_exact(actual_heads * actual_window * 4)
            scores = torch.from_numpy(np.frombuffer(raw_scores, dtype="<f4").copy()).view(actual_heads, actual_window)
            scores_per_item.append(scores)
        return scores_per_item

    def weighted_value_sum(self, slot_id: int, weights: torch.Tensor) -> torch.Tensor:
        w = weights.detach().cpu().to(torch.float32).contiguous()
        if w.dim() != 2:
            raise ValueError(f"weights must be 2D, got shape {tuple(w.shape)}")

        w_np = w.numpy()
        self._write_parts(
            [
                struct.pack("<IIII", self.MAGIC, self.CMD_AV, slot_id, 0),
                memoryview(w_np).cast("B"),
            ]
        )

        out = struct.unpack("<IIIII", self._read_exact(20))
        _, seq_len, group_heads, head_dim, _ = (int(item) for item in out)
        if int(w.shape[0]) != group_heads or int(w.shape[1]) != seq_len:
            raise RuntimeError(
                "kvslot helper returned invalid av header: "
                f"weights={tuple(w.shape)} header=({seq_len}, {group_heads}, {head_dim})"
            )
        raw_context = self._read_exact(group_heads * head_dim * 4)
        context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
        return context

    def weighted_value_sum_batch(self, slot_weights: list[tuple[int, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_weights:
            return []
        if len(slot_weights) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(slot_weights), self.MAX_BATCH_ITEMS):
                outputs.extend(self.weighted_value_sum_batch(slot_weights[offset : offset + self.MAX_BATCH_ITEMS]))
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_AV_BATCH, 0, 0),
            struct.pack("<IIII", int(len(slot_weights)), 0, 0, 0),
        ]
        expected_meta = []

        for slot_id, weights in slot_weights:
            w = weights if weights.device.type == "cpu" and weights.dtype == torch.float32 and weights.is_contiguous() else weights.detach().cpu().to(torch.float32).contiguous()
            if w.dim() != 2:
                raise ValueError(f"weights must be 2D, got shape {tuple(w.shape)}")
            payload_parts.append(struct.pack("<IIII", self.MAGIC, self.CMD_AV, int(slot_id), 0))
            payload_parts.append(memoryview(w.numpy()).cast("B"))
            expected_meta.append((int(slot_id), int(w.shape[0]), int(w.shape[1])))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_slots = int(out_args[0])
        if out_num_slots != len(slot_weights):
            raise RuntimeError(
                f"kvslot helper returned invalid av batch header: expected={len(slot_weights)} actual={out_num_slots}"
            )

        contexts: list[torch.Tensor] = []
        for idx, (_, expected_heads, expected_seq_len) in enumerate(expected_meta):
            out = struct.unpack("<IIIII", self._read_exact(20))
            _, seq_len, group_heads, head_dim, _ = (int(item) for item in out)
            if group_heads != expected_heads or seq_len != expected_seq_len:
                raise RuntimeError(
                    "kvslot helper returned invalid av batch item header: "
                    f"index={idx} weights=({expected_heads}, {expected_seq_len}) header=({seq_len}, {group_heads}, {head_dim})"
                )
            raw_context = self._read_exact(group_heads * head_dim * 4)
            context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
            contexts.append(context)
        return contexts

    def softmax_weighted_value_sum_batch(self, slot_scores: list[tuple[int, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_scores:
            return []
        if len(slot_scores) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(slot_scores), self.MAX_BATCH_ITEMS):
                outputs.extend(
                    self.softmax_weighted_value_sum_batch(slot_scores[offset : offset + self.MAX_BATCH_ITEMS])
                )
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_SOFTMAX_AV_BATCH, 0, 0),
            struct.pack("<IIII", int(len(slot_scores)), 0, 0, 0),
        ]
        expected_meta = []

        for slot_id, scores in slot_scores:
            s = (
                scores
                if scores.device.type == "cpu" and scores.dtype == torch.float32 and scores.is_contiguous()
                else scores.detach().cpu().to(torch.float32).contiguous()
            )
            if s.dim() != 2:
                raise ValueError(f"scores must be 2D, got shape {tuple(s.shape)}")
            payload_parts.append(struct.pack("<IIII", self.MAGIC, self.CMD_SOFTMAX_AV_BATCH, int(slot_id), 0))
            payload_parts.append(memoryview(s.numpy()).cast("B"))
            expected_meta.append((int(slot_id), int(s.shape[0]), int(s.shape[1])))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_slots = int(out_args[0])
        if out_num_slots != len(slot_scores):
            raise RuntimeError(
                "kvslot helper returned invalid softmax-av batch header: "
                f"expected={len(slot_scores)} actual={out_num_slots}"
            )

        contexts: list[torch.Tensor] = []
        for idx, (_, expected_heads, expected_seq_len) in enumerate(expected_meta):
            out = struct.unpack("<IIIII", self._read_exact(20))
            _, seq_len, group_heads, head_dim, _ = (int(item) for item in out)
            if group_heads != expected_heads or seq_len != expected_seq_len:
                raise RuntimeError(
                    "kvslot helper returned invalid softmax-av batch item header: "
                    f"index={idx} scores=({expected_heads}, {expected_seq_len}) header=({seq_len}, {group_heads}, {head_dim})"
                )
            raw_context = self._read_exact(group_heads * head_dim * 4)
            context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
            contexts.append(context)
        return contexts

    def qk_softmax_weighted_value_sum_batch(
        self,
        slot_queries: list[tuple[int, list[int], int, torch.Tensor, float]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        if len(slot_queries) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(slot_queries), self.MAX_BATCH_ITEMS):
                outputs.extend(
                    self.qk_softmax_weighted_value_sum_batch(
                        slot_queries[offset : offset + self.MAX_BATCH_ITEMS]
                    )
                )
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_QK_SOFTMAX_AV_BATCH, 0, 0),
            struct.pack("<IIII", int(len(slot_queries)), 0, 0, 0),
        ]
        expected_meta = []

        for slot_id, local_head_indices, window, queries, score_scale in slot_queries:
            q = (
                queries
                if queries.device.type == "cpu" and queries.dtype == torch.float32 and queries.is_contiguous()
                else queries.detach().cpu().to(torch.float32).contiguous()
            )
            if q.dim() != 2:
                raise ValueError(f"slot queries must be 2D, got shape {tuple(q.shape)}")
            num_heads = int(q.shape[0])
            head_dim = int(q.shape[1])
            if num_heads != len(local_head_indices):
                raise ValueError(
                    f"slot query head count mismatch: queries={num_heads} local_head_indices={len(local_head_indices)}"
                )
            payload_parts.append(struct.pack("<I", int(slot_id)))
            payload_parts.append(struct.pack("<IIIf", num_heads, int(window), head_dim, float(score_scale)))
            payload_parts.append(struct.pack(f"<{num_heads}I", *[int(idx) for idx in local_head_indices]))
            payload_parts.append(memoryview(q.numpy()).cast("B"))
            expected_meta.append((num_heads, head_dim))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_items = int(out_args[0])
        if out_num_items != len(slot_queries):
            raise RuntimeError(
                "kvslot helper returned invalid qk-softmax-av batch header: "
                f"expected={len(slot_queries)} actual={out_num_items}"
            )

        contexts: list[torch.Tensor] = []
        for idx, (expected_heads, expected_head_dim) in enumerate(expected_meta):
            out = struct.unpack("<IIIII", self._read_exact(20))
            _, _, group_heads, head_dim, _ = (int(item) for item in out)
            if group_heads != expected_heads or head_dim != expected_head_dim:
                raise RuntimeError(
                    "kvslot helper returned invalid qk-softmax-av batch item header: "
                    f"index={idx} expected=({expected_heads}, {expected_head_dim}) actual=({group_heads}, {head_dim})"
                )
            raw_context = self._read_exact(group_heads * head_dim * 4)
            context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
            contexts.append(context)
        return contexts

    def qk_softmax_weighted_value_sum_partial_batch(
        self,
        slot_queries: list[tuple[int, list[int], int, torch.Tensor, float]],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not slot_queries:
            return []
        if len(slot_queries) > self.MAX_BATCH_ITEMS:
            outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for offset in range(0, len(slot_queries), self.MAX_BATCH_ITEMS):
                outputs.extend(
                    self.qk_softmax_weighted_value_sum_partial_batch(
                        slot_queries[offset : offset + self.MAX_BATCH_ITEMS]
                    )
                )
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_QK_SOFTMAX_AV_PARTIAL_BATCH, 0, 0),
            struct.pack("<IIII", int(len(slot_queries)), 0, 0, 0),
        ]
        expected_meta = []

        for slot_id, local_head_indices, window, queries, score_scale in slot_queries:
            q = (
                queries
                if queries.device.type == "cpu" and queries.dtype == torch.float32 and queries.is_contiguous()
                else queries.detach().cpu().to(torch.float32).contiguous()
            )
            if q.dim() != 2:
                raise ValueError(f"slot queries must be 2D, got shape {tuple(q.shape)}")
            num_heads = int(q.shape[0])
            head_dim = int(q.shape[1])
            if num_heads != len(local_head_indices):
                raise ValueError(
                    f"slot query head count mismatch: queries={num_heads} local_head_indices={len(local_head_indices)}"
                )
            payload_parts.append(struct.pack("<I", int(slot_id)))
            payload_parts.append(struct.pack("<IIIf", num_heads, int(window), head_dim, float(score_scale)))
            payload_parts.append(struct.pack(f"<{num_heads}I", *[int(idx) for idx in local_head_indices]))
            payload_parts.append(memoryview(q.numpy()).cast("B"))
            expected_meta.append((num_heads, head_dim))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_items = int(out_args[0])
        if out_num_items != len(slot_queries):
            raise RuntimeError(
                "kvslot helper returned invalid qk-softmax-av-partial batch header: "
                f"expected={len(slot_queries)} actual={out_num_items}"
            )

        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx, (expected_heads, expected_head_dim) in enumerate(expected_meta):
            out = struct.unpack("<IIIII", self._read_exact(20))
            _, _, group_heads, head_dim, _ = (int(item) for item in out)
            if group_heads != expected_heads or head_dim != expected_head_dim:
                raise RuntimeError(
                    "kvslot helper returned invalid qk-softmax-av-partial batch item header: "
                    f"index={idx} expected=({expected_heads}, {expected_head_dim}) actual=({group_heads}, {head_dim})"
                )
            raw_context = self._read_exact(group_heads * head_dim * 4)
            raw_row_max = self._read_exact(group_heads * 4)
            raw_row_sum = self._read_exact(group_heads * 4)
            context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
            row_max = torch.from_numpy(np.frombuffer(raw_row_max, dtype="<f4").copy())
            row_sum = torch.from_numpy(np.frombuffer(raw_row_sum, dtype="<f4").copy())
            outputs.append((context, row_max, row_sum))
        return outputs


class HostResidentKVStore(ResidentKVStore):
    backend_name = "host_slot_store"

    def __init__(self):
        self.groups: Dict[tuple[str, str], _HostKVSlot] = {}
        self.total_allocations = 0
        self.live_slots = 0
        self.grow_ops = 0
        self.append_ops = 0
        self.materialize_ops = 0
        self.current_allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.op_timing_totals_s: Dict[str, float] = {
            "allocate_group": 0.0,
            "append_group": 0.0,
            "materialize_group": 0.0,
            "free_group": 0.0,
            "qk_slot_scores_batch": 0.0,
            "weighted_value_sum_batch": 0.0,
            "softmax_weighted_value_sum_batch": 0.0,
            "qk_softmax_weighted_value_sum_batch": 0.0,
        }
        self.op_timing_counts: Dict[str, int] = {key: 0 for key in self.op_timing_totals_s}
        self.batch_item_totals: Dict[str, int] = {
            "qk_slot_scores_batch": 0,
            "weighted_value_sum_batch": 0,
            "softmax_weighted_value_sum_batch": 0,
            "qk_softmax_weighted_value_sum_batch": 0,
        }

    def _record_timing(self, name: str, started_at: float, batch_items: int | None = None) -> None:
        self.op_timing_totals_s[name] += float(time.perf_counter() - started_at)
        self.op_timing_counts[name] += 1
        if batch_items is not None and name in self.batch_item_totals:
            self.batch_item_totals[name] += int(batch_items)

    def _slot_key(self, k_slot: str, v_slot: str) -> tuple[str, str]:
        return (k_slot, v_slot)

    def _slot_bytes(self, slot: _HostKVSlot) -> int:
        return int(slot.k_cache.numel() * slot.k_cache.element_size() + slot.v_cache.numel() * slot.v_cache.element_size())

    def _adjust_allocated_bytes(self, old_bytes: int, new_bytes: int) -> None:
        self.current_allocated_bytes += new_bytes - old_bytes
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.current_allocated_bytes)

    def allocate_group(
        self,
        k_slot: str,
        v_slot: str,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
        preferred_dpu: int | None = None,
        force_host_fallback: bool = False,
    ) -> None:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        if key in self.groups:
            raise ValueError(f"KV slot already exists: {key}")
        if initial_k.shape != initial_v.shape:
            raise ValueError(f"initial K/V shape mismatch for slot {key}: {tuple(initial_k.shape)} vs {tuple(initial_v.shape)}")
        if initial_k.dim() != 3:
            raise ValueError(f"initial K/V for slot {key} must be 3D, got {tuple(initial_k.shape)}")

        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        capacity = max(int(capacity), seq_len, 1)
        k_cache = torch.zeros((capacity, group_heads, head_dim), dtype=initial_k.dtype)
        v_cache = torch.zeros((capacity, group_heads, head_dim), dtype=initial_v.dtype)
        k_cache[:seq_len] = initial_k.contiguous()
        v_cache[:seq_len] = initial_v.contiguous()

        slot = _HostKVSlot(
            k_cache=k_cache,
            v_cache=v_cache,
            seq_len=seq_len,
            capacity=capacity,
            group_heads=group_heads,
            head_dim=head_dim,
        )
        self.groups[key] = slot
        self.total_allocations += 1
        self.live_slots += 1
        self._adjust_allocated_bytes(0, self._slot_bytes(slot))
        self._record_timing("allocate_group", started_at)

    def _grow_slot(self, slot: _HostKVSlot, target_seq_len: int) -> None:
        old_bytes = self._slot_bytes(slot)
        new_capacity = max(slot.capacity * 2, target_seq_len, 1)
        new_k = torch.zeros((new_capacity, slot.group_heads, slot.head_dim), dtype=slot.k_cache.dtype)
        new_v = torch.zeros((new_capacity, slot.group_heads, slot.head_dim), dtype=slot.v_cache.dtype)
        if slot.seq_len > 0:
            new_k[:slot.seq_len] = slot.k_cache[:slot.seq_len]
            new_v[:slot.seq_len] = slot.v_cache[:slot.seq_len]
        slot.k_cache = new_k
        slot.v_cache = new_v
        slot.capacity = new_capacity
        self.grow_ops += 1
        self._adjust_allocated_bytes(old_bytes, self._slot_bytes(slot))

    def append_group(
        self,
        k_slot: str,
        v_slot: str,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Dict[str, int]:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        if key not in self.groups:
            raise KeyError(f"Unknown KV slot: {key}")
        slot = self.groups[key]

        if k_new.shape != v_new.shape:
            raise ValueError(f"append K/V shape mismatch for slot {key}: {tuple(k_new.shape)} vs {tuple(v_new.shape)}")
        if k_new.dim() != 3:
            raise ValueError(f"append K/V for slot {key} must be 3D, got {tuple(k_new.shape)}")
        if int(k_new.shape[1]) != slot.group_heads or int(k_new.shape[2]) != slot.head_dim:
            raise ValueError(
                f"append shape mismatch for slot {key}: got={tuple(k_new.shape)} "
                f"expected=(*, {slot.group_heads}, {slot.head_dim})"
            )

        append_len = int(k_new.shape[0])
        expected_seq_len = slot.seq_len + append_len
        if expected_seq_len > slot.capacity:
            self._grow_slot(slot, expected_seq_len)
        slot.k_cache[slot.seq_len : expected_seq_len] = k_new.contiguous()
        slot.v_cache[slot.seq_len : expected_seq_len] = v_new.contiguous()
        slot.seq_len = expected_seq_len
        self.append_ops += 1
        result = {
            "seq_len": slot.seq_len,
            "capacity": slot.capacity,
        }
        self._record_timing("append_group", started_at)
        return result

    def materialize_group(self, k_slot: str, v_slot: str) -> tuple[torch.Tensor, torch.Tensor]:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        if key not in self.groups:
            raise KeyError(f"Unknown KV slot: {key}")
        slot = self.groups[key]
        self.materialize_ops += 1
        result = (
            slot.k_cache[: slot.seq_len].contiguous(),
            slot.v_cache[: slot.seq_len].contiguous(),
        )
        self._record_timing("materialize_group", started_at)
        return result

    def slot_debug(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        key = self._slot_key(k_slot, v_slot)
        if key not in self.groups:
            raise KeyError(f"Unknown KV slot: {key}")
        slot = self.groups[key]
        return {
            "backend": self.backend_name,
            "shape": list(slot.k_cache.shape),
            "seq_len": slot.seq_len,
            "capacity": slot.capacity,
            "group_heads": slot.group_heads,
            "head_dim": slot.head_dim,
        }

    def free_group(self, k_slot: str, v_slot: str) -> None:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        slot = self.groups.pop(key, None)
        if slot is None:
            return
        self.live_slots -= 1
        self._adjust_allocated_bytes(self._slot_bytes(slot), 0)
        self._record_timing("free_group", started_at)

    def get_debug_info(self) -> Dict[str, object]:
        return {
            "backend": self.backend_name,
            "live_slots": self.live_slots,
            "total_allocations": self.total_allocations,
            "grow_ops": self.grow_ops,
            "append_ops": self.append_ops,
            "materialize_ops": self.materialize_ops,
            "current_allocated_bytes": self.current_allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "op_timing_totals_s": dict(self.op_timing_totals_s),
            "op_timing_counts": dict(self.op_timing_counts),
            "batch_item_totals": dict(self.batch_item_totals),
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = queries.detach().cpu().to(torch.int32).contiguous()
        k = keys.detach().cpu().to(torch.int32).contiguous()
        return torch.einsum("qkd,qd->qk", k.to(torch.int64), q.to(torch.int64))

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        started_at = time.perf_counter()
        outputs: list[torch.Tensor] = []
        for k_slot, v_slot, local_head_indices, window, queries in slot_queries:
            key = self._slot_key(k_slot, v_slot)
            if key not in self.groups:
                raise KeyError(f"Unknown KV slot: {key}")
            slot = self.groups[key]
            actual_window = min(int(window), int(slot.seq_len))
            q = queries.detach().cpu().to(torch.float32).contiguous()
            if q.dim() != 2:
                raise ValueError(f"slot queries must be 2D, got shape {tuple(q.shape)}")
            head_dim = min(int(q.shape[-1]), int(slot.head_dim))
            if actual_window <= 0:
                outputs.append(torch.empty((len(local_head_indices), 0), dtype=torch.float32))
                continue
            score_rows = []
            for row_idx, local_head_idx in enumerate(local_head_indices):
                if local_head_idx < 0 or local_head_idx >= slot.group_heads:
                    raise ValueError(
                        f"local_head_idx out of range for slot {key}: got={local_head_idx} group_heads={slot.group_heads}"
                    )
                query_vec = q[row_idx, :head_dim]
                keys = slot.k_cache[slot.seq_len - actual_window : slot.seq_len, local_head_idx, :head_dim].float().contiguous()
                score_rows.append(torch.einsum("ld,d->l", keys, query_vec).contiguous())
            outputs.append(torch.stack(score_rows, dim=0))
        self._record_timing("qk_slot_scores_batch", started_at, batch_items=len(slot_queries))
        return outputs

    def weighted_value_sum(self, k_slot: str, v_slot: str, weights: torch.Tensor) -> torch.Tensor:
        key = self._slot_key(k_slot, v_slot)
        if key not in self.groups:
            raise KeyError(f"Unknown KV slot: {key}")
        slot = self.groups[key]
        w = weights.detach().cpu().to(torch.float32).contiguous()
        if tuple(w.shape) != (slot.group_heads, slot.seq_len):
            raise ValueError(
                f"weight shape mismatch for slot {key}: got={tuple(w.shape)} "
                f"expected=({slot.group_heads}, {slot.seq_len})"
            )
        values = slot.v_cache[: slot.seq_len].float()
        return torch.einsum("hl,lhd->hd", w, values).contiguous()

    def weighted_value_sum_batch(self, slot_weights: list[tuple[str, str, torch.Tensor]]) -> list[torch.Tensor]:
        started_at = time.perf_counter()
        outputs = [self.weighted_value_sum(k_slot, v_slot, weights) for k_slot, v_slot, weights in slot_weights]
        self._record_timing("weighted_value_sum_batch", started_at, batch_items=len(slot_weights))
        return outputs

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        started_at = time.perf_counter()
        contexts: list[torch.Tensor] = []
        for k_slot, v_slot, scores in slot_scores:
            key = self._slot_key(k_slot, v_slot)
            if key not in self.groups:
                raise KeyError(f"Unknown KV slot: {key}")
            slot = self.groups[key]
            s = scores.detach().cpu().to(torch.float32).contiguous()
            if tuple(s.shape) != (slot.group_heads, slot.seq_len):
                raise ValueError(
                    f"score shape mismatch for slot {key}: got={tuple(s.shape)} "
                    f"expected=({slot.group_heads}, {slot.seq_len})"
                )
            values = slot.v_cache[: slot.seq_len].float()
            weights = torch.softmax(s, dim=-1)
            contexts.append(torch.einsum("hl,lhd->hd", weights, values).contiguous())
        self._record_timing("softmax_weighted_value_sum_batch", started_at, batch_items=len(slot_scores))
        return contexts

    def qk_softmax_weighted_value_sum_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]],
    ) -> list[torch.Tensor]:
        started_at = time.perf_counter()
        contexts: list[torch.Tensor] = []
        for k_slot, v_slot, local_head_indices, window, queries, score_scale in slot_queries:
            key = self._slot_key(k_slot, v_slot)
            if key not in self.groups:
                raise KeyError(f"Unknown KV slot: {key}")
            slot = self.groups[key]
            actual_window = min(int(window), int(slot.seq_len))
            q = queries.detach().cpu().to(torch.float32).contiguous()
            if q.dim() != 2:
                raise ValueError(f"slot queries must be 2D, got shape {tuple(q.shape)}")
            head_dim = min(int(q.shape[-1]), int(slot.head_dim))
            if actual_window <= 0:
                contexts.append(torch.empty((len(local_head_indices), slot.head_dim), dtype=torch.float32))
                continue
            values = slot.v_cache[slot.seq_len - actual_window : slot.seq_len].float()
            row_contexts = []
            for row_idx, local_head_idx in enumerate(local_head_indices):
                if local_head_idx < 0 or local_head_idx >= slot.group_heads:
                    raise ValueError(
                        f"local_head_idx out of range for slot {key}: got={local_head_idx} group_heads={slot.group_heads}"
                    )
                query_vec = q[row_idx, :head_dim]
                keys = slot.k_cache[slot.seq_len - actual_window : slot.seq_len, local_head_idx, :head_dim].float().contiguous()
                scores = torch.einsum("ld,d->l", keys, query_vec).contiguous() * float(score_scale)
                weights = torch.softmax(scores, dim=-1)
                row_contexts.append(torch.einsum("l,ld->d", weights, values[:, local_head_idx, :]).contiguous())
            contexts.append(torch.stack(row_contexts, dim=0))
        self._record_timing("qk_softmax_weighted_value_sum_batch", started_at, batch_items=len(slot_queries))
        return contexts


class UpmemKVSlotStore(ResidentKVStore):
    backend_name = "upmem_kvslot_store"
    POOL_CAPACITY_ELEMS = 256 * 32 * 128

    def __init__(
        self,
        repo_root: str,
        num_dpus: int,
        kv_dtype: str = "fp32",
        block_tokens: int = 256,
        tail_capacity_buckets: list[int] | None = None,
        dpu_placement_policy: str = "rotated",
    ):
        self.repo_root = repo_root
        self.num_dpus = num_dpus
        self.kv_dtype = str(kv_dtype)
        self.block_tokens = max(1, int(block_tokens))
        if tail_capacity_buckets is None:
            tail_capacity_buckets = [16, 32, 64, 128, 256]
        sanitized: list[int] = []
        for value in tail_capacity_buckets:
            try:
                bucket = int(value)
            except Exception:
                continue
            if bucket <= 0:
                continue
            if bucket > self.block_tokens:
                continue
            if bucket not in sanitized:
                sanitized.append(bucket)
        if not sanitized:
            sanitized = [min(16, self.block_tokens), self.block_tokens]
        sanitized.sort()
        self.tail_capacity_buckets = sanitized
        self.dpu_placement_policy = str(dpu_placement_policy)
        if self.kv_dtype not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported resident kv dtype: {self.kv_dtype}")
        if self.dpu_placement_policy not in {"identity", "rotated", "rank_spread"}:
            raise ValueError(f"Unsupported dpu placement policy: {self.dpu_placement_policy}")
        kvslot_dir, helper_binary_path = self._resolve_kvslot_helper_paths(repo_root)
        self.kvslot_dir = kvslot_dir
        self.helper_binary_path = helper_binary_path
        self.helper = _KVSlotHelperClient(
            binary_path=helper_binary_path,
            num_dpus=num_dpus,
            cwd=kvslot_dir,
            kv_dtype=self.kv_dtype,
        )
        self.host_fallback = HostResidentKVStore()
        self.slot_mapping: Dict[tuple[str, str], Dict[str, object]] = {}
        self._slot_id_map: Dict[tuple[str, str], int] = {}
        self._free_slot_ids_by_dpu: list[list[int]] = [[] for _ in range(num_dpus)]
        self._next_slot_seq_by_dpu = [0 for _ in range(num_dpus)]
        self._helper_topology_cache: Dict[int, Dict[str, int]] = {}
        self.dpu_allocations = 0
        self.fallback_allocations = 0
        self.dpu_free_ops = 0
        self.dpu_allocate_failures = 0
        self.dpu_live_slots = 0
        self.dpu_capacity_fallbacks = 0
        self.dpu_live_elems_by_dpu = [0 for _ in range(num_dpus)]
        self.last_allocate_error = ""
        self.last_allocate_error_stage = ""
        self.op_timing_totals_s: Dict[str, float] = {
            "allocate_group": 0.0,
            "append_group": 0.0,
            "materialize_group": 0.0,
            "free_group": 0.0,
            "qk_slot_scores_batch_total": 0.0,
            "qk_slot_scores_batch_dpu": 0.0,
            "qk_slot_scores_batch_host_fallback": 0.0,
            "weighted_value_sum_batch_total": 0.0,
            "weighted_value_sum_batch_dpu": 0.0,
            "weighted_value_sum_batch_host_fallback": 0.0,
            "softmax_weighted_value_sum_batch_total": 0.0,
            "softmax_weighted_value_sum_batch_dpu": 0.0,
            "softmax_weighted_value_sum_batch_host_fallback": 0.0,
            "qk_softmax_weighted_value_sum_batch_total": 0.0,
            "qk_softmax_weighted_value_sum_batch_dpu": 0.0,
            "qk_softmax_weighted_value_sum_batch_host_fallback": 0.0,
        }
        self.op_timing_counts: Dict[str, int] = {key: 0 for key in self.op_timing_totals_s}
        self.batch_item_totals: Dict[str, int] = {
            "qk_slot_scores_batch_total": 0,
            "qk_slot_scores_batch_blocked_logical_items": 0,
            "qk_slot_scores_batch_segmented_logical_items": 0,
            "qk_slot_scores_batch_dpu_items": 0,
            "qk_slot_scores_batch_host_fallback_items": 0,
            "weighted_value_sum_batch_total": 0,
            "weighted_value_sum_batch_blocked_logical_items": 0,
            "weighted_value_sum_batch_segmented_logical_items": 0,
            "weighted_value_sum_batch_dpu_items": 0,
            "weighted_value_sum_batch_host_fallback_items": 0,
            "softmax_weighted_value_sum_batch_total": 0,
            "softmax_weighted_value_sum_batch_blocked_logical_items": 0,
            "softmax_weighted_value_sum_batch_segmented_logical_items": 0,
            "softmax_weighted_value_sum_batch_dpu_items": 0,
            "softmax_weighted_value_sum_batch_host_fallback_items": 0,
            "qk_softmax_weighted_value_sum_batch_total": 0,
            "qk_softmax_weighted_value_sum_batch_blocked_logical_items": 0,
            "qk_softmax_weighted_value_sum_batch_segmented_logical_items": 0,
            "qk_softmax_weighted_value_sum_batch_dpu_items": 0,
            "qk_softmax_weighted_value_sum_batch_host_fallback_items": 0,
        }

    def _kvslot_dir_candidates(self, repo_root: str) -> list[str]:
        candidates = [os.path.join(repo_root, "src", "pim", "upmem_kvslot")]
        shared_repo_root = os.environ.get("CLOVER_SHARED_REPO_ROOT", "").strip()
        if shared_repo_root:
            candidates.append(os.path.join(shared_repo_root, "src", "pim", "upmem_kvslot"))
        shared_kvslot_dir = os.environ.get("CLOVER_SHARED_UPMEM_KVSLOT_DIR", "").strip()
        if shared_kvslot_dir:
            candidates.append(shared_kvslot_dir)

        deduped: list[str] = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _try_build_kvslot_helper(self, kvslot_dir: str) -> bool:
        host_binary_path = os.path.join(kvslot_dir, "build", "host_kvslot")
        # Ray `working_dir` often ships prebuilt artifacts under `build/`.
        # If we change sources (e.g. `host_kvslot.c`) but an old binary is present,
        # we must rebuild or we'll silently run stale code.
        force_rebuild = os.environ.get("CLOVER_FORCE_REBUILD_KVSLOT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if os.path.isfile(host_binary_path) and not force_rebuild:
            # If this directory contains sources, we still want to detect stale binaries.
            # If it doesn't, accept the prebuilt binary.
            has_sources = os.path.isfile(os.path.join(kvslot_dir, "Makefile")) and os.path.isfile(
                os.path.join(kvslot_dir, "host_kvslot.c")
            )
            if not has_sources:
                return True

        if not os.path.isfile(os.path.join(kvslot_dir, "Makefile")):
            return False
        if not os.path.isfile(os.path.join(kvslot_dir, "host_kvslot.c")):
            return False

        if os.path.isfile(host_binary_path) and not force_rebuild:
            # Rebuild when sources are newer than the binary.
            try:
                binary_mtime = os.path.getmtime(host_binary_path)
                source_paths = [
                    os.path.join(kvslot_dir, "Makefile"),
                    os.path.join(kvslot_dir, "common.h"),
                    os.path.join(kvslot_dir, "host_kvslot.c"),
                    os.path.join(kvslot_dir, "dpu_kvslot.c"),
                ]
                sources_mtime = max(
                    os.path.getmtime(path) for path in source_paths if os.path.isfile(path)
                )
                if sources_mtime <= binary_mtime:
                    return True
            except Exception:
                # If we can't stat, fall through to `make` and let it decide.
                pass

            # If we don't have the UPMEM toolchain available locally, avoid failing hard:
            # keep using the existing binary even if sources appear newer.
            if shutil.which("dpu-upmem-dpurte-clang") is None:
                return True

        try:
            subprocess.run(
                ["make", "-C", kvslot_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
        except Exception:
            return False
        return os.path.isfile(host_binary_path)

    def _resolve_kvslot_helper_paths(self, repo_root: str) -> tuple[str, str]:
        for kvslot_dir in self._kvslot_dir_candidates(repo_root):
            if not os.path.isdir(kvslot_dir):
                continue
            # Always go through `_try_build_kvslot_helper()` so mtime/force-rebuild logic is honored.
            if self._try_build_kvslot_helper(kvslot_dir):
                return kvslot_dir, os.path.join(kvslot_dir, "build", "host_kvslot")

        existing_dirs: list[str] = []
        for kvslot_dir in self._kvslot_dir_candidates(repo_root):
            if os.path.isdir(kvslot_dir):
                existing_dirs.append(kvslot_dir)
        fallback_dir = existing_dirs[0] if existing_dirs else self._kvslot_dir_candidates(repo_root)[0]
        return fallback_dir, os.path.join(fallback_dir, "build", "host_kvslot")

    def set_experimental_flags(
        self,
        *,
        context_fused_enabled: bool | None = None,
        shape_rounds_enabled: bool | None = None,
        best_round_seed_enabled: bool | None = None,
        rank_spread_alloc_enabled: bool | None = None,
    ) -> None:
        if context_fused_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_CONTEXT_FUSED", bool(context_fused_enabled))
        if shape_rounds_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_SHAPE_ROUNDS", bool(shape_rounds_enabled))
        if best_round_seed_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_BEST_ROUND_SEED", bool(best_round_seed_enabled))
        if rank_spread_alloc_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_RANK_SPREAD_ALLOC", bool(rank_spread_alloc_enabled))

    def _topology_rank_index(self, physical_dpu: int) -> int | None:
        item = self._helper_topology_cache.get(int(physical_dpu))
        if item is None:
            return None
        return int(item.get("rank_index", 0))

    def _topology_rank_id(self, physical_dpu: int) -> int | None:
        item = self._helper_topology_cache.get(int(physical_dpu))
        if item is None:
            return None
        return int(item.get("rank_id", 0))

    def _ensure_topology_cache(self) -> None:
        if self.num_dpus <= 0 or self._helper_topology_cache:
            return
        try:
            helper_topology = self.helper.get_topology()
        except Exception:
            return
        self._helper_topology_cache = {
            int(item["logical_dpu_id"]): {
                "rank_index": int(item["rank_index"]),
                "rank_id": int(item["rank_id"]),
            }
            for item in helper_topology.get("items", [])
        }

    def _placement_order(self, preferred_dpu: int | None = None) -> list[int]:
        if self.num_dpus <= 0:
            return [0]
        base_physical_dpu = self._normalize_preferred_dpu(preferred_dpu)
        if self.num_dpus == 1 or self.dpu_placement_policy in {"identity", "rotated"}:
            return [
                (base_physical_dpu + offset) % self.num_dpus
                for offset in range(self.num_dpus)
            ]

        self._ensure_topology_cache()
        if len(self._helper_topology_cache) < self.num_dpus:
            return [
                (base_physical_dpu + offset) % self.num_dpus
                for offset in range(self.num_dpus)
            ]

        base_rank = self._topology_rank_index(base_physical_dpu)
        if base_rank is None:
            return [
                (base_physical_dpu + offset) % self.num_dpus
                for offset in range(self.num_dpus)
            ]

        dpus_by_rank: Dict[int, list[int]] = {}
        for physical_dpu in range(self.num_dpus):
            rank_index = self._topology_rank_index(physical_dpu)
            if rank_index is None:
                continue
            dpus_by_rank.setdefault(int(rank_index), []).append(int(physical_dpu))
        if not dpus_by_rank:
            return [
                (base_physical_dpu + offset) % self.num_dpus
                for offset in range(self.num_dpus)
            ]

        rank_order = sorted(dpus_by_rank)
        if int(base_rank) in rank_order:
            base_pos = rank_order.index(int(base_rank))
            rank_order = rank_order[base_pos:] + rank_order[:base_pos]
        for rank_index, dpu_ids in dpus_by_rank.items():
            dpu_ids.sort()
            if rank_index == int(base_rank) and base_physical_dpu in dpu_ids:
                base_pos = dpu_ids.index(base_physical_dpu)
                dpus_by_rank[rank_index] = dpu_ids[base_pos:] + dpu_ids[:base_pos]

        placement: list[int] = []
        rank_pass = 0
        while len(placement) < self.num_dpus:
            added_this_pass = False
            for rank_index in rank_order:
                dpu_ids = dpus_by_rank.get(rank_index, [])
                if rank_pass >= len(dpu_ids):
                    continue
                placement.append(int(dpu_ids[rank_pass]))
                added_this_pass = True
            if not added_this_pass:
                break
            rank_pass += 1
        if len(placement) != self.num_dpus:
            return [
                (base_physical_dpu + offset) % self.num_dpus
                for offset in range(self.num_dpus)
            ]
        return placement

    def _placement_dpu_for_block(self, slot_info: Dict[str, object], block_idx: int) -> int:
        placement_order = slot_info.get("placement_order")
        if isinstance(placement_order, list) and placement_order:
            return int(placement_order[int(block_idx) % len(placement_order)])
        base_physical_dpu = int(slot_info.get("base_physical_dpu", slot_info.get("physical_dpu", 0)))
        return self._placement_order(base_physical_dpu)[int(block_idx) % max(self.num_dpus, 1)]

    def _record_timing(self, name: str, started_at: float) -> None:
        self.op_timing_totals_s[name] += float(time.perf_counter() - started_at)
        self.op_timing_counts[name] += 1

    def _slot_key(self, k_slot: str, v_slot: str) -> tuple[str, str]:
        return (k_slot, v_slot)

    def _block_slot_key(self, key: tuple[str, str], block_idx: int) -> tuple[str, str]:
        return (f"{key[0]}#blk{block_idx}", f"{key[1]}#blk{block_idx}")

    def _normalize_preferred_dpu(self, preferred_dpu: int | None) -> int:
        if self.num_dpus <= 0:
            return 0
        if preferred_dpu is None:
            return 0
        return int(preferred_dpu) % self.num_dpus

    def _assign_slot_id(self, key: tuple[str, str], preferred_dpu: int | None = None) -> int:
        if key in self._slot_id_map:
            return self._slot_id_map[key]
        physical_dpu = self._normalize_preferred_dpu(preferred_dpu)
        if self.num_dpus > 0 and self._free_slot_ids_by_dpu[physical_dpu]:
            slot_id = self._free_slot_ids_by_dpu[physical_dpu].pop()
        else:
            seq = self._next_slot_seq_by_dpu[physical_dpu]
            slot_id = physical_dpu + (seq * max(self.num_dpus, 1))
            self._next_slot_seq_by_dpu[physical_dpu] += 1
        self._slot_id_map[key] = slot_id
        return slot_id

    def _supports_dpu_slot(self, initial_k: torch.Tensor, capacity: int, slot_id: int | None) -> bool:
        if slot_id is None:
            return False
        max_slots = self.num_dpus * self.helper.MAX_SLOTS_PER_DPU
        if slot_id >= max_slots:
            return False
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        return capacity <= 256 and group_heads <= 32 and head_dim <= 128 and seq_len <= capacity

    def _encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.kv_dtype == "fp16":
            return tensor.detach().cpu().to(torch.float16).contiguous().view(torch.int16)
        return tensor.detach().cpu().float().contiguous().view(torch.int32)

    def _decode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.kv_dtype == "fp16":
            return tensor.view(torch.float16).to(torch.float32).contiguous()
        return tensor.view(torch.float32).contiguous()

    def _slot_elem_count(self, capacity: int, group_heads: int, head_dim: int) -> int:
        elem_count = int(capacity) * int(group_heads) * int(head_dim)
        if self.kv_dtype == "fp16":
            return max(1, (elem_count + 1) // 2)
        return elem_count

    def _build_block_layout(self, capacity: int, seq_len: int) -> list[tuple[int, int]]:
        remaining_capacity = int(capacity)
        remaining_seq_len = int(seq_len)
        layout: list[tuple[int, int]] = []
        while remaining_capacity > 0:
            block_capacity = min(self.block_tokens, remaining_capacity)
            block_seq_len = min(block_capacity, remaining_seq_len)
            layout.append((block_capacity, block_seq_len))
            remaining_capacity -= block_capacity
            remaining_seq_len -= block_seq_len
        return layout

    def _supports_dpu_shape(self, group_heads: int, head_dim: int) -> bool:
        return int(group_heads) <= 32 and int(head_dim) <= 128

    def _bucket_block_capacity(self, seq_len: int) -> int:
        """Pick a per-block capacity <= self.block_tokens to reduce MRAM waste.

        The helper only supports capacities up to 256 (KVSLOT_MAX_CAPACITY). When
        we always allocate blocks at `self.block_tokens`, tail blocks with just a
        few tokens still reserve the full 256-token slice, which quickly exhausts
        the per-DPU MRAM pool under concurrent long-context workloads.
        """
        want = max(1, int(seq_len))
        cap_limit = max(1, int(self.block_tokens))
        if want >= cap_limit:
            return cap_limit

        for bucket in self.tail_capacity_buckets:
            if want <= int(bucket):
                return int(bucket)
        return cap_limit

    def _release_slot_reservation(
        self,
        key: tuple[str, str],
        slot_id: int,
        physical_dpu: int,
    ) -> None:
        mapped_slot_id = self._slot_id_map.get(key)
        if mapped_slot_id == int(slot_id):
            self._slot_id_map.pop(key, None)
        normalized_dpu = int(physical_dpu) % max(self.num_dpus, 1)
        free_ids = self._free_slot_ids_by_dpu[normalized_dpu]
        if int(slot_id) not in free_ids:
            free_ids.append(int(slot_id))

    def _rollback_block_allocation(
        self,
        *,
        block_key: tuple[str, str],
        slot_id: int,
        physical_dpu: int,
        elem_count: int,
        helper_allocated: bool,
        counters_applied: bool,
    ) -> None:
        normalized_dpu = int(physical_dpu) % max(self.num_dpus, 1)
        freed_helper_state = False
        if helper_allocated:
            try:
                self.helper.free_group(int(slot_id))
                self.dpu_free_ops += 1
                freed_helper_state = True
            except Exception:
                pass
        if counters_applied:
            self.dpu_allocations = max(0, self.dpu_allocations - 1)
            self.dpu_live_slots = max(0, self.dpu_live_slots - 1)
            self.dpu_live_elems_by_dpu[normalized_dpu] = max(
                0, self.dpu_live_elems_by_dpu[normalized_dpu] - int(elem_count)
            )
        self._release_slot_reservation(block_key, int(slot_id), normalized_dpu)
        self.helper.persistent_state_active = self.dpu_live_slots > 0
        if self.dpu_live_slots == 0 and freed_helper_state:
            self.helper.close()

    def _free_block_infos(self, blocks: list[Dict[str, object]]) -> None:
        for block in reversed(blocks):
            slot_id = int(block["slot_id"])
            physical_dpu = int(block["physical_dpu"]) % max(self.num_dpus, 1)
            try:
                self.helper.free_group(slot_id)
            except Exception:
                pass
            self.dpu_free_ops += 1
            self.dpu_live_slots = max(0, self.dpu_live_slots - 1)
            elem_count = int(block.get("elem_count", 0))
            self.dpu_live_elems_by_dpu[physical_dpu] = max(
                0, self.dpu_live_elems_by_dpu[physical_dpu] - elem_count
            )
            block_key = tuple(block["block_key"])
            self._slot_id_map.pop(block_key, None)
            self._free_slot_ids_by_dpu[physical_dpu].append(slot_id)
        self.helper.persistent_state_active = self.dpu_live_slots > 0
        if self.dpu_live_slots == 0:
            self.helper.close()

    def _active_block_plan(
        self,
        slot_info: Dict[str, object],
        window: int | None = None,
    ) -> list[tuple[Dict[str, object], int]]:
        blocks = self._blocked_slot_blocks(slot_info)
        if not blocks:
            return []
        if window is None:
            return [(block, int(block["seq_len"])) for block in blocks if int(block["seq_len"]) > 0]
        remaining = min(int(window), int(slot_info["seq_len"]))
        plan_reversed: list[tuple[Dict[str, object], int]] = []
        for block in reversed(blocks):
            if remaining <= 0:
                break
            block_seq_len = int(block["seq_len"])
            if block_seq_len <= 0:
                continue
            take_len = min(block_seq_len, remaining)
            plan_reversed.append((block, take_len))
            remaining -= take_len
        return list(reversed(plan_reversed))

    def _blocked_slot_blocks(self, slot_info: Dict[str, object]) -> list[Dict[str, object]]:
        return list(slot_info.get("blocks", slot_info.get("segments", [])))

    def _is_blocked_slot(self, slot_info: Dict[str, object]) -> bool:
        return str(slot_info.get("backend", "")) == "dpu_blocked"

    def _allocate_block_append_only(
        self,
        *,
        key: tuple[str, str],
        slot_info: Dict[str, object],
        group_heads: int,
        head_dim: int,
        block_k: torch.Tensor,
        block_v: torch.Tensor,
    ) -> Dict[str, object]:
        blocks = self._blocked_slot_blocks(slot_info)
        block_idx = len(blocks)
        block_physical_dpu = self._placement_dpu_for_block(slot_info, block_idx) if self.num_dpus > 0 else 0
        # Use a bucketed capacity for tail blocks to avoid allocating the full
        # block_tokens (usually 256) when we only need a small tail.
        block_capacity = max(int(block_k.shape[0]), self._bucket_block_capacity(int(block_k.shape[0])))
        block_key = self._block_slot_key(key, block_idx)
        block_slot_id = self._assign_slot_id(block_key, preferred_dpu=block_physical_dpu)
        block_elem_count = self._slot_elem_count(block_capacity, group_heads, head_dim)
        helper_allocated = False
        counters_applied = False
        try:
            if not self._supports_dpu_slot(block_k, block_capacity, block_slot_id):
                raise RuntimeError(
                    f"Blocked DPU slot shape unsupported for {key}: "
                    f"shape={tuple(block_k.shape)} capacity={block_capacity} slot_id={block_slot_id}"
                )
            if self.dpu_live_elems_by_dpu[block_physical_dpu] + block_elem_count > self.POOL_CAPACITY_ELEMS:
                self.dpu_capacity_fallbacks += 1
                raise RuntimeError(
                    f"Blocked DPU pool capacity exceeded for {key} on physical_dpu={block_physical_dpu}"
                )

            info = self.helper.allocate_group(
                block_slot_id,
                block_capacity,
                self._encode_tensor(block_k),
                self._encode_tensor(block_v),
            )
            helper_allocated = True
            block = {
                "block_key": list(block_key),
                "slot_id": int(block_slot_id),
                "physical_dpu": int(block_physical_dpu),
                "block_index": int(block_idx),
                "elem_count": int(block_elem_count),
                "seq_len": int(info["seq_len"]),
                "capacity": int(info["capacity"]),
                "group_heads": int(info["group_heads"]),
                "head_dim": int(info["head_dim"]),
            }
            self.dpu_allocations += 1
            self.dpu_live_slots += 1
            self.dpu_live_elems_by_dpu[block_physical_dpu] += block_elem_count
            self.helper.persistent_state_active = True
            counters_applied = True
            return block
        except Exception as exc:
            self.last_allocate_error_stage = "blocked_allocate_block"
            self.last_allocate_error = repr(exc)
            self._rollback_block_allocation(
                block_key=block_key,
                slot_id=block_slot_id,
                physical_dpu=block_physical_dpu,
                elem_count=block_elem_count,
                helper_allocated=helper_allocated,
                counters_applied=counters_applied,
            )
            raise

    def _allocate_blocked_group(
        self,
        *,
        key: tuple[str, str],
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
        physical_dpu: int,
        group_heads: int,
        head_dim: int,
    ) -> Dict[str, object]:
        allocated_blocks: list[Dict[str, object]] = []
        seq_len = int(initial_k.shape[0])
        seq_offset = 0
        placement_order = self._placement_order(physical_dpu)
        slot_info = {
            "backend": "dpu_blocked",
            "blocks": allocated_blocks,
            "segments": allocated_blocks,
            "seq_len": 0,
            "capacity": int(capacity),
            "group_heads": int(group_heads),
            "head_dim": int(head_dim),
            "block_tokens": int(self.block_tokens),
            "base_physical_dpu": int(physical_dpu),
            "placement_order": placement_order,
        }
        try:
            while seq_offset < seq_len:
                block_seq_len = min(self.block_tokens, seq_len - seq_offset)
                block_initial_k = initial_k[seq_offset : seq_offset + block_seq_len].contiguous()
                block_initial_v = initial_v[seq_offset : seq_offset + block_seq_len].contiguous()
                block = self._allocate_block_append_only(
                    key=key,
                    slot_info=slot_info,
                    group_heads=group_heads,
                    head_dim=head_dim,
                    block_k=block_initial_k,
                    block_v=block_initial_v,
                )
                allocated_blocks.append(block)
                seq_offset += block_seq_len
            slot_info["seq_len"] = int(seq_len)
            slot_info["capacity"] = max(int(capacity), sum(int(block["capacity"]) for block in allocated_blocks))
            return slot_info
        except Exception:
            if allocated_blocks:
                self._free_block_infos(allocated_blocks)
            raise

    def allocate_group(
        self,
        k_slot: str,
        v_slot: str,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
        preferred_dpu: int | None = None,
        force_host_fallback: bool = False,
    ) -> None:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        physical_dpu = self._normalize_preferred_dpu(preferred_dpu) if self.num_dpus > 0 else 0
        elem_count = self._slot_elem_count(capacity, group_heads, head_dim)
        if not force_host_fallback and self._supports_dpu_shape(group_heads, head_dim):
            try:
                blocked_slot_info = self._allocate_blocked_group(
                    key=key,
                    initial_k=initial_k,
                    initial_v=initial_v,
                    capacity=capacity,
                    physical_dpu=physical_dpu,
                    group_heads=group_heads,
                    head_dim=head_dim,
                )
            except Exception as exc:
                self.dpu_allocate_failures += 1
                self.last_allocate_error_stage = "allocate_group_blocked"
                self.last_allocate_error = repr(exc)
                blocked_slot_info = None
            if blocked_slot_info is not None:
                self.slot_mapping[key] = blocked_slot_info
                self._record_timing("allocate_group", started_at)
                return
            slot_id = self._assign_slot_id(key, preferred_dpu=preferred_dpu)
        else:
            slot_id = self._assign_slot_id(key, preferred_dpu=preferred_dpu)

        if not force_host_fallback and self._supports_dpu_slot(initial_k, capacity, slot_id):
            if self.dpu_live_elems_by_dpu[physical_dpu] + elem_count > self.POOL_CAPACITY_ELEMS:
                self.dpu_capacity_fallbacks += 1
            else:
                try:
                    info = self.helper.allocate_group(
                        slot_id,
                        capacity,
                        self._encode_tensor(initial_k),
                        self._encode_tensor(initial_v),
                    )
                except Exception as exc:
                    self.dpu_allocate_failures += 1
                    self.last_allocate_error_stage = "allocate_group_regular"
                    self.last_allocate_error = repr(exc)
                    if self.dpu_live_slots > 0:
                        raise
                    info = None
                if info is not None:
                    self.slot_mapping[key] = {
                        "backend": "dpu",
                        "slot_id": slot_id,
                        "physical_dpu": physical_dpu,
                        "elem_count": elem_count,
                        "seq_len": int(info["seq_len"]),
                        "capacity": int(info["capacity"]),
                        "group_heads": int(info["group_heads"]),
                        "head_dim": int(info["head_dim"]),
                    }
                    self.dpu_allocations += 1
                    self.dpu_live_slots += 1
                    self.dpu_live_elems_by_dpu[physical_dpu] += elem_count
                    self.helper.persistent_state_active = True
                    self._record_timing("allocate_group", started_at)
                    return

        self.host_fallback.allocate_group(
            k_slot,
            v_slot,
            initial_k,
            initial_v,
            capacity,
            preferred_dpu=preferred_dpu,
        )
        self.slot_mapping[key] = {
            "backend": "host_fallback",
            "slot_id": slot_id,
            "physical_dpu": physical_dpu,
            "elem_count": elem_count,
        }
        self.fallback_allocations += 1
        self._record_timing("allocate_group", started_at)

    def append_group(
        self,
        k_slot: str,
        v_slot: str,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Dict[str, int]:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if self._is_blocked_slot(slot_info):
            append_total = int(k_new.shape[0])
            blocks = self._blocked_slot_blocks(slot_info)
            group_heads = int(slot_info["group_heads"])
            head_dim = int(slot_info["head_dim"])
            if append_total <= 0:
                out = {
                    "seq_len": int(slot_info["seq_len"]),
                    "capacity": int(slot_info["capacity"]),
                }
                self._record_timing("append_group", started_at)
                return out

            tail_block = blocks[-1] if blocks else None
            tail_available = 0 if tail_block is None else max(0, int(tail_block["capacity"]) - int(tail_block["seq_len"]))
            tail_take_len = min(tail_available, append_total)
            new_blocks: list[Dict[str, object]] = []
            append_offset = tail_take_len
            try:
                staged_blocks = list(blocks)
                while append_offset < append_total:
                    take_len = min(self.block_tokens, append_total - append_offset)
                    block_k = k_new[append_offset : append_offset + take_len].contiguous()
                    block_v = v_new[append_offset : append_offset + take_len].contiguous()
                    staged_slot_info = dict(slot_info)
                    staged_slot_info["blocks"] = staged_blocks
                    new_block = self._allocate_block_append_only(
                        key=key,
                        slot_info=staged_slot_info,
                        group_heads=group_heads,
                        head_dim=head_dim,
                        block_k=block_k,
                        block_v=block_v,
                    )
                    staged_blocks.append(new_block)
                    new_blocks.append(new_block)
                    append_offset += take_len

                if tail_take_len > 0 and tail_block is not None:
                    result = self.helper.append_group(
                        int(tail_block["slot_id"]),
                        self._encode_tensor(k_new[:tail_take_len].contiguous()),
                        self._encode_tensor(v_new[:tail_take_len].contiguous()),
                    )
                    tail_block["seq_len"] = int(result["seq_len"])
                    tail_block["capacity"] = int(result["capacity"])
            except Exception:
                if new_blocks:
                    self._free_block_infos(new_blocks)
                raise

            if new_blocks:
                blocks.extend(new_blocks)
            slot_info["seq_len"] = int(slot_info["seq_len"]) + append_total
            logical_capacity = 0
            for block in blocks:
                logical_capacity += int(block["capacity"])
            slot_info["capacity"] = max(int(slot_info.get("capacity", 0)), int(logical_capacity))
            out = {
                "seq_len": int(slot_info["seq_len"]),
                "capacity": int(slot_info["capacity"]),
            }
            self._record_timing("append_group", started_at)
            return out
        if slot_info["backend"] == "dpu":
            result = self.helper.append_group(
                int(slot_info["slot_id"]),
                self._encode_tensor(k_new),
                self._encode_tensor(v_new),
            )
            slot_info["seq_len"] = int(result["seq_len"])
            slot_info["capacity"] = int(result["capacity"])
            out = {
                "seq_len": int(result["seq_len"]),
                "capacity": int(result["capacity"]),
            }
            self._record_timing("append_group", started_at)
            return out
        out = self.host_fallback.append_group(k_slot, v_slot, k_new, v_new)
        self._record_timing("append_group", started_at)
        return out

    def materialize_group(self, k_slot: str, v_slot: str) -> tuple[torch.Tensor, torch.Tensor]:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if self._is_blocked_slot(slot_info):
            materialized_blocks = []
            for block in self._blocked_slot_blocks(slot_info):
                if int(block["seq_len"]) <= 0:
                    continue
                k, v, info = self.helper.materialize_group(int(block["slot_id"]))
                block["seq_len"] = int(info["seq_len"])
                block["capacity"] = int(info["capacity"])
                materialized_blocks.append((self._decode_tensor(k), self._decode_tensor(v)))
            if not materialized_blocks:
                empty = torch.empty(
                    (0, int(slot_info["group_heads"]), int(slot_info["head_dim"])),
                    dtype=torch.float32,
                )
                self._record_timing("materialize_group", started_at)
                return empty, empty.clone()
            out = (
                torch.cat([item[0] for item in materialized_blocks], dim=0).contiguous(),
                torch.cat([item[1] for item in materialized_blocks], dim=0).contiguous(),
            )
            self._record_timing("materialize_group", started_at)
            return out
        if slot_info["backend"] == "dpu":
            k, v, info = self.helper.materialize_group(int(slot_info["slot_id"]))
            slot_info["seq_len"] = int(info["seq_len"])
            slot_info["capacity"] = int(info["capacity"])
            out = self._decode_tensor(k), self._decode_tensor(v)
            self._record_timing("materialize_group", started_at)
            return out
        out = self.host_fallback.materialize_group(k_slot, v_slot)
        self._record_timing("materialize_group", started_at)
        return out

    def slot_debug(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if self._is_blocked_slot(slot_info):
            blocks = self._blocked_slot_blocks(slot_info)
            return {
                "backend": self.backend_name,
                "storage": "dpu_blocked",
                "seq_len": int(slot_info["seq_len"]),
                "capacity": int(slot_info["capacity"]),
                "group_heads": int(slot_info["group_heads"]),
                "head_dim": int(slot_info["head_dim"]),
                "block_tokens": int(slot_info.get("block_tokens", self.block_tokens)),
                "block_count": len(blocks),
                "blocks": [
                    {
                        "slot_id": int(block["slot_id"]),
                        "physical_dpu": int(block["physical_dpu"]),
                        "block_index": int(block.get("block_index", 0)),
                        "rank_index": self._topology_rank_index(int(block["physical_dpu"])),
                        "rank_id": self._topology_rank_id(int(block["physical_dpu"])),
                        "seq_len": int(block["seq_len"]),
                        "capacity": int(block["capacity"]),
                    }
                    for block in blocks
                ],
                "segment_count": len(blocks),
                "segments": [
                    {
                        "slot_id": int(block["slot_id"]),
                        "physical_dpu": int(block["physical_dpu"]),
                        "rank_index": self._topology_rank_index(int(block["physical_dpu"])),
                        "rank_id": self._topology_rank_id(int(block["physical_dpu"])),
                        "seq_len": int(block["seq_len"]),
                        "capacity": int(block["capacity"]),
                    }
                    for block in blocks
                ],
            }
        if slot_info["backend"] == "dpu":
            return {
                "backend": self.backend_name,
                "storage": "dpu",
                "slot_id": int(slot_info["slot_id"]),
                "physical_dpu": int(slot_info.get("physical_dpu", 0)),
                "rank_index": self._topology_rank_index(int(slot_info.get("physical_dpu", 0))),
                "rank_id": self._topology_rank_id(int(slot_info.get("physical_dpu", 0))),
                "seq_len": int(slot_info["seq_len"]),
                "capacity": int(slot_info["capacity"]),
                "group_heads": int(slot_info["group_heads"]),
                "head_dim": int(slot_info["head_dim"]),
            }
        debug = self.host_fallback.slot_debug(k_slot, v_slot)
        debug["backend"] = self.backend_name
        debug["storage"] = "host_fallback"
        return debug

    def free_group(self, k_slot: str, v_slot: str) -> None:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping.pop(key, None)
        if slot_info is None:
            return
        if self._is_blocked_slot(slot_info):
            self._free_block_infos(self._blocked_slot_blocks(slot_info))
            self._record_timing("free_group", started_at)
            return
        slot_id = self._slot_id_map.pop(key, None)
        if slot_info["backend"] == "dpu":
            self.helper.free_group(int(slot_info["slot_id"]))
            self.dpu_free_ops += 1
            self.dpu_live_slots = max(0, self.dpu_live_slots - 1)
            physical_dpu = int(slot_info.get("physical_dpu", 0))
            elem_count = int(slot_info.get("elem_count", 0))
            self.dpu_live_elems_by_dpu[physical_dpu] = max(0, self.dpu_live_elems_by_dpu[physical_dpu] - elem_count)
            self.helper.persistent_state_active = self.dpu_live_slots > 0
            if self.dpu_live_slots == 0:
                # Release the allocated DPU set once the resident store becomes
                # empty, so later experiments do not inherit a stale helper.
                self.helper.close()
        else:
            self.host_fallback.free_group(k_slot, v_slot)
        if slot_id is not None:
            physical_dpu = int(slot_info.get("physical_dpu", 0)) % max(self.num_dpus, 1)
            self._free_slot_ids_by_dpu[physical_dpu].append(int(slot_id))
        self._record_timing("free_group", started_at)

    def get_debug_info(self) -> Dict[str, object]:
        allocator_stats = []
        helper_profile = {}
        helper_topology = {}
        helper_live = self.helper.proc is not None and self.helper.proc.poll() is None
        if self.helper.persistent_state_active or helper_live:
            try:
                allocator_stats = self.helper.get_allocator_stats()
            except Exception:
                allocator_stats = []
            try:
                helper_profile = self.helper.get_profile_stats()
            except Exception:
                helper_profile = {}
            try:
                helper_topology = self.helper.get_topology()
                self._helper_topology_cache = {
                    int(item["logical_dpu_id"]): {
                        "rank_index": int(item["rank_index"]),
                        "rank_id": int(item["rank_id"]),
                    }
                    for item in helper_topology.get("items", [])
                }
            except Exception:
                helper_topology = {}
                self._helper_topology_cache = {}
        pool_capacity_elems = 256 * 32 * 128
        for stats in allocator_stats:
            tail_free = max(pool_capacity_elems - int(stats["next_free_elem"]), 0)
            total_free = int(stats["free_elems_total"]) + tail_free
            stats["tail_free_elems"] = tail_free
            stats["total_free_elems"] = total_free
            stats["pool_capacity_elems"] = pool_capacity_elems
            stats["used_elems_estimate"] = max(pool_capacity_elems - total_free, 0)
            stats["usage_ratio"] = float(stats["used_elems_estimate"]) / float(pool_capacity_elems) if pool_capacity_elems else 0.0
        return {
            "backend": self.backend_name,
            "num_dpus": self.num_dpus,
            "kv_dtype": self.kv_dtype,
            "dpu_placement_policy": self.dpu_placement_policy,
            "kvslot_dir": self.kvslot_dir,
            "helper_binary_path": self.helper_binary_path,
            "helper_env": dict(self.helper.helper_env),
            "helper_profile": helper_profile,
            "helper_host_profile_totals_s": dict(self.helper.host_profile_totals_s),
            "helper_host_profile_counts": dict(self.helper.host_profile_counts),
            "helper_topology": helper_topology,
            "live_slots": len(self.slot_mapping),
            "dpu_allocations": self.dpu_allocations,
            "dpu_free_ops": self.dpu_free_ops,
            "dpu_allocate_failures": self.dpu_allocate_failures,
            "dpu_live_slots": self.dpu_live_slots,
            "dpu_capacity_fallbacks": self.dpu_capacity_fallbacks,
            "last_allocate_error_stage": self.last_allocate_error_stage,
            "last_allocate_error": self.last_allocate_error,
            "dpu_live_elems_by_dpu": list(self.dpu_live_elems_by_dpu),
            "dpu_pool_capacity_elems": self.POOL_CAPACITY_ELEMS,
            "fallback_allocations": self.fallback_allocations,
            "helper_restarts": self.helper.restarts,
            "free_slot_ids": sum(len(ids) for ids in self._free_slot_ids_by_dpu),
            "next_slot_id": max(
                (dpu_id + (seq * max(self.num_dpus, 1)) for dpu_id, seq in enumerate(self._next_slot_seq_by_dpu)),
                default=0,
            ),
            "allocator_stats": allocator_stats,
            "host_fallback": self.host_fallback.get_debug_info(),
            "op_timing_totals_s": dict(self.op_timing_totals_s),
            "op_timing_counts": dict(self.op_timing_counts),
            "batch_item_totals": dict(self.batch_item_totals),
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return self.helper.qk_scores_batch(queries, keys)

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        total_started_at = time.perf_counter()

        outputs: list[torch.Tensor | None] = [None for _ in slot_queries]
        dpu_items: list[tuple[int, list[int], int, torch.Tensor]] = []
        dpu_refs: list[tuple[str, int]] = []
        host_fallback_queries: list[tuple[int, tuple[str, str, list[int], int, torch.Tensor]]] = []
        blocked_outputs: Dict[int, list[torch.Tensor]] = {}

        for idx, (k_slot, v_slot, local_head_indices, window, queries) in enumerate(slot_queries):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if self._is_blocked_slot(slot_info):
                self.batch_item_totals["qk_slot_scores_batch_blocked_logical_items"] += 1
                self.batch_item_totals["qk_slot_scores_batch_segmented_logical_items"] += 1
                actual_window = min(int(window), int(slot_info["seq_len"]))
                if actual_window <= 0:
                    outputs[idx] = torch.empty((len(local_head_indices), 0), dtype=torch.float32)
                    continue
                for block, take_len in self._active_block_plan(slot_info, actual_window):
                    dpu_items.append(
                        (int(block["slot_id"]), [int(v) for v in local_head_indices], int(take_len), queries)
                    )
                    dpu_refs.append(("blocked", idx))
            elif slot_info["backend"] == "dpu":
                actual_window = min(int(window), int(slot_info["seq_len"]))
                dpu_items.append((int(slot_info["slot_id"]), [int(v) for v in local_head_indices], actual_window, queries))
                dpu_refs.append(("regular", idx))
            else:
                host_fallback_queries.append(
                    (idx, (k_slot, v_slot, [int(v) for v in local_head_indices], window, queries))
                )

        if host_fallback_queries:
            host_started_at = time.perf_counter()
            host_outputs = self.host_fallback.qk_slot_scores_batch([item for _, item in host_fallback_queries])
            self._record_timing("qk_slot_scores_batch_host_fallback", host_started_at)
            self.batch_item_totals["qk_slot_scores_batch_host_fallback_items"] += len(host_fallback_queries)
            for (idx, _), output in zip(host_fallback_queries, host_outputs):
                outputs[idx] = output

        if dpu_items:
            dpu_started_at = time.perf_counter()
            dpu_outputs = self.helper.qk_slot_scores_batch(dpu_items)
            self._record_timing("qk_slot_scores_batch_dpu", dpu_started_at)
            self.batch_item_totals["qk_slot_scores_batch_dpu_items"] += len(dpu_items)
            for (ref_kind, logical_idx), scores in zip(dpu_refs, dpu_outputs):
                if ref_kind == "regular":
                    outputs[logical_idx] = scores
                else:
                    blocked_outputs.setdefault(logical_idx, []).append(scores)

        for logical_idx, score_parts in blocked_outputs.items():
            outputs[logical_idx] = torch.cat(score_parts, dim=1).contiguous()

        self._record_timing("qk_slot_scores_batch_total", total_started_at)
        self.batch_item_totals["qk_slot_scores_batch_total"] += len(slot_queries)
        return [output for output in outputs if output is not None]

    def weighted_value_sum(self, k_slot: str, v_slot: str, weights: torch.Tensor) -> torch.Tensor:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if self._is_blocked_slot(slot_info):
            context: torch.Tensor | None = None
            weight_offset = 0
            for block, block_len in self._active_block_plan(slot_info):
                block_weights = weights[:, weight_offset : weight_offset + int(block_len)].contiguous()
                weight_offset += int(block_len)
                block_context = self.helper.weighted_value_sum(int(block["slot_id"]), block_weights).to(torch.float32)
                context = block_context if context is None else (context + block_context)
            if context is None:
                return torch.empty((weights.shape[0], int(slot_info["head_dim"])), dtype=torch.float32)
            return context
        if slot_info["backend"] == "dpu":
            return self.helper.weighted_value_sum(int(slot_info["slot_id"]), weights)
        return self.host_fallback.weighted_value_sum(k_slot, v_slot, weights)

    def weighted_value_sum_batch(self, slot_weights: list[tuple[str, str, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_weights:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_weights]
        dpu_batch: list[tuple[int, torch.Tensor]] = []
        dpu_refs: list[tuple[str, int]] = []
        host_fallback_weights: list[tuple[int, tuple[str, str, torch.Tensor]]] = []
        blocked_contexts: Dict[int, torch.Tensor] = {}

        for idx, (k_slot, v_slot, weights) in enumerate(slot_weights):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if self._is_blocked_slot(slot_info):
                self.batch_item_totals["weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["weighted_value_sum_batch_segmented_logical_items"] += 1
                weight_offset = 0
                for block, block_len in self._active_block_plan(slot_info):
                    block_weights = weights[:, weight_offset : weight_offset + int(block_len)].contiguous()
                    weight_offset += int(block_len)
                    dpu_batch.append((int(block["slot_id"]), block_weights))
                    dpu_refs.append(("blocked", idx))
            elif slot_info["backend"] == "dpu":
                dpu_batch.append((int(slot_info["slot_id"]), weights))
                dpu_refs.append(("regular", idx))
            else:
                host_fallback_weights.append((idx, (k_slot, v_slot, weights)))

        if host_fallback_weights:
            host_started_at = time.perf_counter()
            host_contexts = self.host_fallback.weighted_value_sum_batch([item for _, item in host_fallback_weights])
            self._record_timing("weighted_value_sum_batch_host_fallback", host_started_at)
            self.batch_item_totals["weighted_value_sum_batch_host_fallback_items"] += len(host_fallback_weights)
            for (idx, _), context in zip(host_fallback_weights, host_contexts):
                contexts[idx] = context

        if dpu_batch:
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.weighted_value_sum_batch(dpu_batch)
            self._record_timing("weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["weighted_value_sum_batch_dpu_items"] += len(dpu_batch)
            for (ref_kind, idx), context in zip(dpu_refs, dpu_contexts):
                if ref_kind == "regular":
                    contexts[idx] = context
                else:
                    if idx not in blocked_contexts:
                        blocked_contexts[idx] = context
                    else:
                        blocked_contexts[idx] = blocked_contexts[idx] + context

        for idx, context in blocked_contexts.items():
            contexts[idx] = context

        self._record_timing("weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["weighted_value_sum_batch_total"] += len(slot_weights)
        return [context for context in contexts if context is not None]

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_scores:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_scores]
        dpu_batch: list[tuple[int, torch.Tensor]] = []
        dpu_indices: list[int] = []
        blocked_scores: list[tuple[int, tuple[str, str, torch.Tensor]]] = []
        host_fallback_scores: list[tuple[int, tuple[str, str, torch.Tensor]]] = []

        for idx, (k_slot, v_slot, scores) in enumerate(slot_scores):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if self._is_blocked_slot(slot_info):
                self.batch_item_totals["softmax_weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["softmax_weighted_value_sum_batch_segmented_logical_items"] += 1
                blocked_scores.append((idx, (k_slot, v_slot, scores)))
            elif slot_info["backend"] == "dpu":
                dpu_batch.append((int(slot_info["slot_id"]), scores))
                dpu_indices.append(idx)
            else:
                host_fallback_scores.append((idx, (k_slot, v_slot, scores)))

        if host_fallback_scores:
            host_started_at = time.perf_counter()
            host_contexts = self.host_fallback.softmax_weighted_value_sum_batch(
                [item for _, item in host_fallback_scores]
            )
            self._record_timing("softmax_weighted_value_sum_batch_host_fallback", host_started_at)
            self.batch_item_totals["softmax_weighted_value_sum_batch_host_fallback_items"] += len(
                host_fallback_scores
            )
            for (idx, _), context in zip(host_fallback_scores, host_contexts):
                contexts[idx] = context

        if dpu_batch:
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.softmax_weighted_value_sum_batch(dpu_batch)
            self._record_timing("softmax_weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["softmax_weighted_value_sum_batch_dpu_items"] += len(dpu_batch)
            for idx, context in zip(dpu_indices, dpu_contexts):
                contexts[idx] = context

        if blocked_scores:
            blocked_weights = []
            for _, (k_slot, v_slot, scores) in blocked_scores:
                normalized_scores = scores.detach().cpu().to(torch.float32).contiguous()
                blocked_weights.append(
                    (k_slot, v_slot, torch.softmax(normalized_scores, dim=-1))
                )
            blocked_contexts = self.weighted_value_sum_batch(blocked_weights)
            for (idx, _), context in zip(blocked_scores, blocked_contexts):
                contexts[idx] = context

        self._record_timing("softmax_weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["softmax_weighted_value_sum_batch_total"] += len(slot_scores)
        return [context for context in contexts if context is not None]

    def qk_softmax_weighted_value_sum_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_queries]
        dpu_batch: list[tuple[int, list[int], int, torch.Tensor, float]] = []
        dpu_indices: list[int] = []
        blocked_queries: list[
            tuple[int, tuple[str, str, list[int], int, torch.Tensor, float]]
        ] = []
        host_fallback_queries: list[
            tuple[int, tuple[str, str, list[int], int, torch.Tensor, float]]
        ] = []

        for idx, (k_slot, v_slot, local_head_indices, window, queries, score_scale) in enumerate(slot_queries):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if self._is_blocked_slot(slot_info):
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_segmented_logical_items"] += 1
                blocked_queries.append(
                    (
                        idx,
                        (k_slot, v_slot, [int(v) for v in local_head_indices], window, queries, float(score_scale)),
                    )
                )
            elif slot_info["backend"] == "dpu":
                actual_window = min(int(window), int(slot_info["seq_len"]))
                dpu_batch.append(
                    (
                        int(slot_info["slot_id"]),
                        [int(v) for v in local_head_indices],
                        actual_window,
                        queries,
                        float(score_scale),
                    )
                )
                dpu_indices.append(idx)
            else:
                host_fallback_queries.append(
                    (
                        idx,
                        (k_slot, v_slot, [int(v) for v in local_head_indices], window, queries, float(score_scale)),
                    )
                )

        if host_fallback_queries:
            host_started_at = time.perf_counter()
            host_contexts = self.host_fallback.qk_softmax_weighted_value_sum_batch(
                [item for _, item in host_fallback_queries]
            )
            self._record_timing("qk_softmax_weighted_value_sum_batch_host_fallback", host_started_at)
            self.batch_item_totals["qk_softmax_weighted_value_sum_batch_host_fallback_items"] += len(
                host_fallback_queries
            )
            for (idx, _), context in zip(host_fallback_queries, host_contexts):
                contexts[idx] = context

        if dpu_batch:
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.qk_softmax_weighted_value_sum_batch(dpu_batch)
            self._record_timing("qk_softmax_weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["qk_softmax_weighted_value_sum_batch_dpu_items"] += len(dpu_batch)
            for idx, context in zip(dpu_indices, dpu_contexts):
                contexts[idx] = context

        if blocked_queries:
            partial_batch: list[tuple[int, list[int], int, torch.Tensor, float]] = []
            partial_refs: list[tuple[int, int]] = []
            merged_contexts: Dict[int, torch.Tensor] = {}
            merged_row_max: Dict[int, torch.Tensor] = {}
            merged_row_sum: Dict[int, torch.Tensor] = {}

            for logical_idx, (k_slot, v_slot, local_head_indices, window, queries, score_scale) in blocked_queries:
                key = self._slot_key(k_slot, v_slot)
                slot_info = self.slot_mapping[key]
                actual_window = min(int(window), int(slot_info["seq_len"]))
                if actual_window <= 0:
                    contexts[logical_idx] = torch.empty((len(local_head_indices), int(slot_info["head_dim"])), dtype=torch.float32)
                    continue
                for block, take_len in self._active_block_plan(slot_info, actual_window):
                    partial_batch.append(
                        (
                            int(block["slot_id"]),
                            [int(v) for v in local_head_indices],
                            int(take_len),
                            queries,
                            float(score_scale),
                        )
                    )
                    partial_refs.append((logical_idx, len(local_head_indices)))

            if partial_batch:
                dpu_started_at = time.perf_counter()
                partial_outputs = self.helper.qk_softmax_weighted_value_sum_partial_batch(partial_batch)
                self._record_timing("qk_softmax_weighted_value_sum_batch_dpu", dpu_started_at)
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_dpu_items"] += len(partial_batch)

                for (logical_idx, _), (segment_context, segment_row_max, segment_row_sum) in zip(partial_refs, partial_outputs):
                    segment_context = segment_context.to(torch.float32)
                    segment_row_max = segment_row_max.to(torch.float32)
                    segment_row_sum = segment_row_sum.to(torch.float32)
                    if logical_idx not in merged_contexts:
                        merged_contexts[logical_idx] = segment_context
                        merged_row_max[logical_idx] = segment_row_max
                        merged_row_sum[logical_idx] = segment_row_sum
                        continue

                    prev_row_max = merged_row_max[logical_idx]
                    prev_row_sum = merged_row_sum[logical_idx]
                    prev_context = merged_contexts[logical_idx]
                    combined_row_max = torch.maximum(prev_row_max, segment_row_max)
                    prev_scale = torch.exp(prev_row_max - combined_row_max)
                    seg_scale = torch.exp(segment_row_max - combined_row_max)
                    combined_row_sum = prev_row_sum * prev_scale + segment_row_sum * seg_scale
                    safe_sum = torch.clamp(combined_row_sum, min=1e-12)
                    prev_weight = (prev_row_sum * prev_scale / safe_sum).unsqueeze(1)
                    seg_weight = (segment_row_sum * seg_scale / safe_sum).unsqueeze(1)
                    merged_contexts[logical_idx] = (prev_context * prev_weight) + (segment_context * seg_weight)
                    merged_row_max[logical_idx] = combined_row_max
                    merged_row_sum[logical_idx] = combined_row_sum

            for logical_idx, context in merged_contexts.items():
                contexts[logical_idx] = context

        self._record_timing("qk_softmax_weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["qk_softmax_weighted_value_sum_batch_total"] += len(slot_queries)
        return [context for context in contexts if context is not None]
