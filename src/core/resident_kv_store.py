from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import struct
import subprocess
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
    MAX_SLOTS_PER_DPU = 64

    def __init__(self, binary_path: str, num_dpus: int, cwd: str, kv_dtype: str = "fp32"):
        self.binary_path = binary_path
        self.num_dpus = num_dpus
        self.cwd = cwd
        self.kv_dtype = str(kv_dtype)
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.persistent_state_active = False

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
                raise RuntimeError(
                    "kvslot helper exited while persistent DPU state was active: "
                    f"{stderr_text.strip()}"
                )
        self.proc = subprocess.Popen(
            [self.binary_path, "--stdio", "--num-dpus", str(self.num_dpus)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        payload = header + args + initial_k.numpy().tobytes(order="C") + initial_v.numpy().tobytes(order="C")
        self._write(payload)
        out = struct.unpack("<IIIII", self._read_exact(20))
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
        payload = header + args + k_new.numpy().tobytes(order="C") + v_new.numpy().tobytes(order="C")
        self._write(payload)
        out = struct.unpack("<IIIII", self._read_exact(20))
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
            k_bytes = self._read_exact(elems * elem_bytes)
            v_bytes = self._read_exact(elems * elem_bytes)
            if elem_bytes == 2:
                k = torch.tensor(struct.unpack(f"<{elems}h", k_bytes), dtype=torch.int16).view(seq_len, group_heads, head_dim)
                v = torch.tensor(struct.unpack(f"<{elems}h", v_bytes), dtype=torch.int16).view(seq_len, group_heads, head_dim)
            else:
                k = torch.tensor(struct.unpack(f"<{elems}i", k_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
                v = torch.tensor(struct.unpack(f"<{elems}i", v_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
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
        return {
            "seq_len": slot.seq_len,
            "capacity": slot.capacity,
        }

    def materialize_group(self, k_slot: str, v_slot: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._slot_key(k_slot, v_slot)
        if key not in self.groups:
            raise KeyError(f"Unknown KV slot: {key}")
        slot = self.groups[key]
        self.materialize_ops += 1
        return (
            slot.k_cache[: slot.seq_len].contiguous(),
            slot.v_cache[: slot.seq_len].contiguous(),
        )

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
        key = self._slot_key(k_slot, v_slot)
        slot = self.groups.pop(key, None)
        if slot is None:
            return
        self.live_slots -= 1
        self._adjust_allocated_bytes(self._slot_bytes(slot), 0)

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
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = queries.detach().cpu().to(torch.int32).contiguous()
        k = keys.detach().cpu().to(torch.int32).contiguous()
        return torch.einsum("qkd,qd->qk", k.to(torch.int64), q.to(torch.int64))

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
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
        return [self.weighted_value_sum(k_slot, v_slot, weights) for k_slot, v_slot, weights in slot_weights]

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
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
        return contexts


class UpmemKVSlotStore(ResidentKVStore):
    backend_name = "upmem_kvslot_store"
    POOL_CAPACITY_ELEMS = 256 * 32 * 128

    def __init__(self, repo_root: str, num_dpus: int, kv_dtype: str = "fp32"):
        self.repo_root = repo_root
        self.num_dpus = num_dpus
        self.kv_dtype = str(kv_dtype)
        if self.kv_dtype not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported resident kv dtype: {self.kv_dtype}")
        kvslot_dir = os.path.join(repo_root, "src", "pim", "upmem_kvslot")
        self.helper = _KVSlotHelperClient(
            binary_path=os.path.join(kvslot_dir, "build", "host_kvslot"),
            num_dpus=num_dpus,
            cwd=kvslot_dir,
            kv_dtype=self.kv_dtype,
        )
        self.host_fallback = HostResidentKVStore()
        self.slot_mapping: Dict[tuple[str, str], Dict[str, object]] = {}
        self._slot_id_map: Dict[tuple[str, str], int] = {}
        self._free_slot_ids_by_dpu: list[list[int]] = [[] for _ in range(num_dpus)]
        self._next_slot_seq_by_dpu = [0 for _ in range(num_dpus)]
        self.dpu_allocations = 0
        self.fallback_allocations = 0
        self.dpu_free_ops = 0
        self.dpu_allocate_failures = 0
        self.dpu_live_slots = 0
        self.dpu_capacity_fallbacks = 0
        self.dpu_live_elems_by_dpu = [0 for _ in range(num_dpus)]

    def _slot_key(self, k_slot: str, v_slot: str) -> tuple[str, str]:
        return (k_slot, v_slot)

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
        key = self._slot_key(k_slot, v_slot)
        slot_id = self._assign_slot_id(key, preferred_dpu=preferred_dpu)
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        physical_dpu = self._normalize_preferred_dpu(preferred_dpu) if self.num_dpus > 0 else 0
        elem_count = self._slot_elem_count(capacity, group_heads, head_dim)
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
                except Exception:
                    self.dpu_allocate_failures += 1
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

    def append_group(
        self,
        k_slot: str,
        v_slot: str,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Dict[str, int]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "dpu":
            result = self.helper.append_group(
                int(slot_info["slot_id"]),
                self._encode_tensor(k_new),
                self._encode_tensor(v_new),
            )
            slot_info["seq_len"] = int(result["seq_len"])
            slot_info["capacity"] = int(result["capacity"])
            return {
                "seq_len": int(result["seq_len"]),
                "capacity": int(result["capacity"]),
            }
        return self.host_fallback.append_group(k_slot, v_slot, k_new, v_new)

    def materialize_group(self, k_slot: str, v_slot: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "dpu":
            k, v, info = self.helper.materialize_group(int(slot_info["slot_id"]))
            slot_info["seq_len"] = int(info["seq_len"])
            slot_info["capacity"] = int(info["capacity"])
            return self._decode_tensor(k), self._decode_tensor(v)
        return self.host_fallback.materialize_group(k_slot, v_slot)

    def slot_debug(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "dpu":
            return {
                "backend": self.backend_name,
                "storage": "dpu",
                "slot_id": int(slot_info["slot_id"]),
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
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping.pop(key, None)
        if slot_info is None:
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

    def get_debug_info(self) -> Dict[str, object]:
        allocator_stats = []
        helper_live = self.helper.proc is not None and self.helper.proc.poll() is None
        if self.helper.persistent_state_active or helper_live:
            try:
                allocator_stats = self.helper.get_allocator_stats()
            except Exception:
                allocator_stats = []
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
            "live_slots": len(self.slot_mapping),
            "dpu_allocations": self.dpu_allocations,
            "dpu_free_ops": self.dpu_free_ops,
            "dpu_allocate_failures": self.dpu_allocate_failures,
            "dpu_live_slots": self.dpu_live_slots,
            "dpu_capacity_fallbacks": self.dpu_capacity_fallbacks,
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
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return self.helper.qk_scores_batch(queries, keys)

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []

        outputs: list[torch.Tensor | None] = [None for _ in slot_queries]
        dpu_items: list[tuple[int, list[int], int, torch.Tensor]] = []
        dpu_indices: list[int] = []

        for idx, (k_slot, v_slot, local_head_indices, window, queries) in enumerate(slot_queries):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] == "dpu":
                actual_window = min(int(window), int(slot_info["seq_len"]))
                dpu_items.append((int(slot_info["slot_id"]), [int(v) for v in local_head_indices], actual_window, queries))
                dpu_indices.append(idx)
            else:
                outputs[idx] = self.host_fallback.qk_slot_scores_batch(
                    [(k_slot, v_slot, [int(v) for v in local_head_indices], window, queries)]
                )[0]

        if dpu_items:
            dpu_outputs = self.helper.qk_slot_scores_batch(dpu_items)
            for idx, scores in zip(dpu_indices, dpu_outputs):
                outputs[idx] = scores

        return [output for output in outputs if output is not None]

    def weighted_value_sum(self, k_slot: str, v_slot: str, weights: torch.Tensor) -> torch.Tensor:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "dpu":
            return self.helper.weighted_value_sum(int(slot_info["slot_id"]), weights)
        return self.host_fallback.weighted_value_sum(k_slot, v_slot, weights)

    def weighted_value_sum_batch(self, slot_weights: list[tuple[str, str, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_weights:
            return []

        contexts: list[torch.Tensor | None] = [None for _ in slot_weights]
        dpu_batch: list[tuple[int, torch.Tensor]] = []
        dpu_indices: list[int] = []

        for idx, (k_slot, v_slot, weights) in enumerate(slot_weights):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] == "dpu":
                dpu_batch.append((int(slot_info["slot_id"]), weights))
                dpu_indices.append(idx)
            else:
                contexts[idx] = self.host_fallback.weighted_value_sum(k_slot, v_slot, weights)

        if dpu_batch:
            dpu_contexts = self.helper.weighted_value_sum_batch(dpu_batch)
            for idx, context in zip(dpu_indices, dpu_contexts):
                contexts[idx] = context

        return [context for context in contexts if context is not None]

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_scores:
            return []

        contexts: list[torch.Tensor | None] = [None for _ in slot_scores]
        dpu_batch: list[tuple[int, torch.Tensor]] = []
        dpu_indices: list[int] = []

        for idx, (k_slot, v_slot, scores) in enumerate(slot_scores):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] == "dpu":
                dpu_batch.append((int(slot_info["slot_id"]), scores))
                dpu_indices.append(idx)
            else:
                contexts[idx] = self.host_fallback.softmax_weighted_value_sum_batch([(k_slot, v_slot, scores)])[0]

        if dpu_batch:
            dpu_contexts = self.helper.softmax_weighted_value_sum_batch(dpu_batch)
            for idx, context in zip(dpu_indices, dpu_contexts):
                contexts[idx] = context

        return [context for context in contexts if context is not None]
