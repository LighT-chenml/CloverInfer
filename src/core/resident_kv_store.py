from __future__ import annotations

from dataclasses import dataclass
import os
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


class _KVSlotHelperClient:
    MAGIC = 0x4B56534C
    CMD_ALLOCATE = 1
    CMD_APPEND = 2
    CMD_READBACK = 3

    def __init__(self, binary_path: str, num_dpus: int, cwd: str):
        self.binary_path = binary_path
        self.num_dpus = num_dpus
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None
        self.restarts = 0

    def _ensure_proc(self) -> subprocess.Popen:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        self.proc = subprocess.Popen(
            [self.binary_path, "--stdio", "--num-dpus", str(self.num_dpus)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.restarts += 1
        return self.proc

    def _write(self, payload: bytes) -> None:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        proc.stdin.write(payload)
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
        args = struct.pack("<IIII", int(capacity), seq_len, group_heads, head_dim)
        payload = header + args + initial_k.numpy().tobytes(order="C") + initial_v.numpy().tobytes(order="C")
        self._write(payload)
        out = struct.unpack("<IIII", self._read_exact(16))
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
        }

    def append_group(self, slot_id: int, k_new: torch.Tensor, v_new: torch.Tensor) -> Dict[str, int]:
        append_len, group_heads, head_dim = (int(dim) for dim in k_new.shape)
        header = struct.pack("<IIII", self.MAGIC, self.CMD_APPEND, slot_id, 0)
        args = struct.pack("<IIII", 0, append_len, group_heads, head_dim)
        payload = header + args + k_new.numpy().tobytes(order="C") + v_new.numpy().tobytes(order="C")
        self._write(payload)
        out = struct.unpack("<IIII", self._read_exact(16))
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
        }

    def materialize_group(self, slot_id: int) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_READBACK, slot_id, 0)
        self._write(header)
        out = struct.unpack("<IIII", self._read_exact(16))
        capacity, seq_len, group_heads, head_dim = (int(item) for item in out)
        elems = seq_len * group_heads * head_dim
        if elems == 0:
            k = torch.empty((0, group_heads, head_dim), dtype=torch.int32)
            v = torch.empty((0, group_heads, head_dim), dtype=torch.int32)
        else:
            k_bytes = self._read_exact(elems * 4)
            v_bytes = self._read_exact(elems * 4)
            k = torch.tensor(struct.unpack(f"<{elems}i", k_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
            v = torch.tensor(struct.unpack(f"<{elems}i", v_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
        return k, v, {
            "capacity": capacity,
            "seq_len": seq_len,
            "group_heads": group_heads,
            "head_dim": head_dim,
        }


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


class UpmemKVSlotStore(ResidentKVStore):
    backend_name = "upmem_kvslot_store"

    def __init__(self, repo_root: str, num_dpus: int):
        self.repo_root = repo_root
        self.num_dpus = num_dpus
        kvslot_dir = os.path.join(repo_root, "src", "pim", "upmem_kvslot")
        self.helper = _KVSlotHelperClient(
            binary_path=os.path.join(kvslot_dir, "build", "host_kvslot"),
            num_dpus=num_dpus,
            cwd=kvslot_dir,
        )
        self.host_fallback = HostResidentKVStore()
        self.slot_mapping: Dict[tuple[str, str], Dict[str, object]] = {}
        self._slot_id_map: Dict[tuple[str, str], int] = {}
        self._next_slot_id = 0
        self.dpu_allocations = 0
        self.fallback_allocations = 0

    def _slot_key(self, k_slot: str, v_slot: str) -> tuple[str, str]:
        return (k_slot, v_slot)

    def _assign_slot_id(self, key: tuple[str, str]) -> int:
        if key in self._slot_id_map:
            return self._slot_id_map[key]
        slot_id = self._next_slot_id
        self._slot_id_map[key] = slot_id
        self._next_slot_id += 1
        return slot_id

    def _supports_dpu_slot(self, initial_k: torch.Tensor, capacity: int, slot_id: int | None) -> bool:
        if slot_id is None or slot_id >= self.num_dpus:
            return False
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        return capacity <= 256 and group_heads <= 32 and head_dim <= 128 and seq_len <= capacity

    def _encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().float().contiguous().view(torch.int32)

    def _decode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(torch.float32).contiguous()

    def allocate_group(
        self,
        k_slot: str,
        v_slot: str,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
    ) -> None:
        key = self._slot_key(k_slot, v_slot)
        slot_id = self._assign_slot_id(key)
        if self._supports_dpu_slot(initial_k, capacity, slot_id):
            info = self.helper.allocate_group(
                slot_id,
                capacity,
                self._encode_tensor(initial_k),
                self._encode_tensor(initial_v),
            )
            self.slot_mapping[key] = {
                "backend": "dpu",
                "slot_id": slot_id,
                "seq_len": int(info["seq_len"]),
                "capacity": int(info["capacity"]),
                "group_heads": int(info["group_heads"]),
                "head_dim": int(info["head_dim"]),
            }
            self.dpu_allocations += 1
            return

        self.host_fallback.allocate_group(k_slot, v_slot, initial_k, initial_v, capacity)
        self.slot_mapping[key] = {
            "backend": "host_fallback",
            "slot_id": slot_id,
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
        self._slot_id_map.pop(key, None)
        if slot_info["backend"] == "host_fallback":
            self.host_fallback.free_group(k_slot, v_slot)

    def get_debug_info(self) -> Dict[str, object]:
        return {
            "backend": self.backend_name,
            "num_dpus": self.num_dpus,
            "live_slots": len(self.slot_mapping),
            "dpu_allocations": self.dpu_allocations,
            "fallback_allocations": self.fallback_allocations,
            "helper_restarts": self.helper.restarts,
            "host_fallback": self.host_fallback.get_debug_info(),
        }
