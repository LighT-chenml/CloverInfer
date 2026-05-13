from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
import os
import numpy as np
import shlex
import struct
import subprocess
import sys
import time
from typing import Dict, List

import torch


_HOST_REDUCTION_MODULE = None
_HOST_REDUCTION_IMPORT_ATTEMPTED = False

KVSLOT_DTYPE_FP32 = 0
KVSLOT_DTYPE_FP16 = 1
KVSLOT_DTYPE_INT8 = 2
MIXED_INT8_FP16_KV_DTYPE = "mixed_int8_fp16"
SUPPORTED_RESIDENT_KV_DTYPES = {"fp32", "fp16", "int8", MIXED_INT8_FP16_KV_DTYPE}


def normalize_resident_kv_dtype(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"int8_fp16", "k_int8_v_fp16", "int8k_fp16v"}:
        return MIXED_INT8_FP16_KV_DTYPE
    return normalized


def resident_kv_dtype_code(kv_dtype: str, part: str = "k") -> int:
    normalized = normalize_resident_kv_dtype(kv_dtype)
    if normalized == MIXED_INT8_FP16_KV_DTYPE:
        return KVSLOT_DTYPE_INT8 if str(part).lower().startswith("k") else KVSLOT_DTYPE_FP16
    if normalized == "int8":
        return KVSLOT_DTYPE_INT8
    if normalized == "fp16":
        return KVSLOT_DTYPE_FP16
    return KVSLOT_DTYPE_FP32


def kvslot_dtype_elem_bytes(dtype_code: int) -> int:
    if int(dtype_code) == KVSLOT_DTYPE_INT8:
        return 1
    if int(dtype_code) == KVSLOT_DTYPE_FP16:
        return 2
    return 4


def kvslot_packed_word_count(logical_elems: int, dtype_code: int) -> int:
    elems = max(1, int(logical_elems))
    if int(dtype_code) == KVSLOT_DTYPE_INT8:
        words = (elems + 3) // 4
    elif int(dtype_code) == KVSLOT_DTYPE_FP16:
        words = (elems + 1) // 2
    else:
        words = elems
    return (words + 1) & ~1


def _load_host_reduction_module():
    global _HOST_REDUCTION_MODULE
    global _HOST_REDUCTION_IMPORT_ATTEMPTED
    if _HOST_REDUCTION_MODULE is not None:
        return _HOST_REDUCTION_MODULE
    if _HOST_REDUCTION_IMPORT_ATTEMPTED:
        return None
    _HOST_REDUCTION_IMPORT_ATTEMPTED = True
    try:
        _HOST_REDUCTION_MODULE = importlib.import_module("clover_host_reduce")
    except Exception:
        module_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "host", "reduction")
        )
        try:
            for file_name in os.listdir(module_dir):
                if not file_name.startswith("clover_host_reduce") or not file_name.endswith(".so"):
                    continue
                module_path = os.path.join(module_dir, file_name)
                spec = importlib.util.spec_from_file_location("clover_host_reduce", module_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _HOST_REDUCTION_MODULE = module
                break
        except Exception:
            _HOST_REDUCTION_MODULE = None
    return _HOST_REDUCTION_MODULE


def _iter_candidate_kvslot_dirs(repo_root: str) -> List[str]:
    roots: List[str] = []
    for candidate in (
        repo_root,
        os.environ.get("CLOVER_REPO_ROOT"),
        os.environ.get("PROJECT_DIR"),
        os.getcwd(),
        os.path.expanduser("~/CloverInfer"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ):
        if not candidate:
            continue
        resolved = os.path.abspath(str(candidate))
        if resolved not in roots:
            roots.append(resolved)

    kvslot_dirs: List[str] = []
    for root in roots:
        if os.path.basename(root) == "src":
            candidate_dir = os.path.join(root, "pim", "upmem_kvslot")
        else:
            candidate_dir = os.path.join(root, "src", "pim", "upmem_kvslot")
        if candidate_dir not in kvslot_dirs:
            kvslot_dirs.append(candidate_dir)
    return kvslot_dirs


def _kvslot_limit_config() -> tuple[Dict[str, str], bool]:
    """Return Makefile limit overrides and whether the user explicitly set one."""
    config: Dict[str, str] = {}
    explicit = False
    for make_name, clover_name, default_value in (
        ("KVSLOT_MAX_CAPACITY", "CLOVER_KVSLOT_MAX_CAPACITY", "256"),
        ("KVSLOT_MAX_HEADS", "CLOVER_KVSLOT_MAX_HEADS", "32"),
    ):
        raw_value = os.environ.get(clover_name)
        if raw_value is None:
            raw_value = os.environ.get(make_name)
        else:
            explicit = True
        if raw_value is None:
            raw_value = default_value
        else:
            explicit = True
        try:
            value = max(1, int(str(raw_value).strip()))
        except Exception:
            value = int(default_value)
        config[make_name] = str(value)
    return config, explicit


def _kvslot_source_is_newer(kvslot_dir: str, helper_path: str) -> bool:
    if not os.path.exists(helper_path):
        return True
    try:
        helper_mtime = os.path.getmtime(helper_path)
    except OSError:
        return True
    for rel_path in ("Makefile", "common.h", "dpu_kvslot.c", "host_kvslot.c"):
        source_path = os.path.join(kvslot_dir, rel_path)
        try:
            if os.path.getmtime(source_path) > helper_mtime:
                return True
        except OSError:
            continue
    return False


def _read_kvslot_build_config(kvslot_dir: str) -> Dict[str, str]:
    stamp_path = os.path.join(kvslot_dir, "build", ".kvslot_build_config")
    config: Dict[str, str] = {}
    try:
        with open(stamp_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                if key:
                    config[key] = value
    except OSError:
        pass
    return config


def _write_kvslot_build_config(kvslot_dir: str, config: Dict[str, str]) -> None:
    stamp_path = os.path.join(kvslot_dir, "build", ".kvslot_build_config")
    try:
        os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
        with open(stamp_path, "w", encoding="utf-8") as handle:
            for key in sorted(config):
                handle.write(f"{key}={config[key]}\n")
    except OSError:
        pass


def _kvslot_helper_needs_build(
    kvslot_dir: str,
    helper_path: str,
    desired_config: Dict[str, str],
    limits_explicit: bool,
) -> tuple[bool, bool]:
    if not os.path.exists(helper_path):
        return True, False
    if _kvslot_source_is_newer(kvslot_dir, helper_path):
        return True, False
    recorded_config = _read_kvslot_build_config(kvslot_dir)
    if recorded_config:
        return recorded_config != desired_config, recorded_config != desired_config
    if limits_explicit:
        # A pre-existing helper has no recorded compile-time limits. Rebuild
        # when the caller asks for non-default limits so Python and DPU agree.
        return True, True
    return False, False


def _run_kvslot_build(
    kvslot_dir: str,
    desired_config: Dict[str, str],
    *,
    clean_first: bool,
) -> subprocess.CompletedProcess[str]:
    upmem_env_script = os.environ.get("CLOVER_UPMEM_ENV", "/usr/upmem_env.sh")
    build_target = "clean all" if clean_first else "all"
    make_vars = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(desired_config.items()))
    build_cmd = f"make {build_target} {make_vars}".strip()
    if os.path.exists(upmem_env_script):
        build_cmd = f". {shlex.quote(upmem_env_script)} >/dev/null 2>&1 && {build_cmd}"
    build_env = os.environ.copy()
    build_env.update(desired_config)
    return subprocess.run(
        build_cmd,
        cwd=kvslot_dir,
        shell=True,
        executable="/bin/bash",
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(os.environ.get("CLOVER_KVSLOT_AUTOBUILD_TIMEOUT_S", "120")),
    )


def _resolve_kvslot_helper_paths(repo_root: str) -> tuple[str, str]:
    helper_override = os.environ.get("CLOVER_KVSLOT_HELPER")
    kvslot_dir_override = os.environ.get("CLOVER_KVSLOT_DIR")
    searched_paths: List[str] = []
    build_failures: List[str] = []
    desired_config, limits_explicit = _kvslot_limit_config()

    if helper_override:
        helper_path = os.path.abspath(helper_override)
        if os.path.exists(helper_path):
            kvslot_dir = kvslot_dir_override
            if not kvslot_dir:
                kvslot_dir = os.path.abspath(os.path.join(os.path.dirname(helper_path), ".."))
            return os.path.abspath(kvslot_dir), helper_path
        searched_paths.append(helper_path)

    for kvslot_dir in _iter_candidate_kvslot_dirs(repo_root):
        helper_path = os.path.join(kvslot_dir, "build", "host_kvslot")
        searched_paths.append(helper_path)
        needs_build, clean_first = _kvslot_helper_needs_build(
            kvslot_dir,
            helper_path,
            desired_config,
            limits_explicit,
        )
        if os.path.exists(helper_path) and not needs_build:
            return kvslot_dir, helper_path

        if os.environ.get("CLOVER_KVSLOT_AUTOBUILD", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            makefile_path = os.path.join(kvslot_dir, "Makefile")
            if not os.path.exists(makefile_path):
                continue
            completed = _run_kvslot_build(
                kvslot_dir,
                desired_config,
                clean_first=clean_first,
            )
            if completed.returncode == 0 and os.path.exists(helper_path):
                _write_kvslot_build_config(kvslot_dir, desired_config)
                return kvslot_dir, helper_path
            build_output = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part and part.strip()
            )
            build_failures.append(
                f"{kvslot_dir}: exit={completed.returncode}"
                + (f"\n{build_output[-2000:]}" if build_output else "")
            )

    searched_desc = ", ".join(searched_paths)
    build_desc = ""
    if build_failures:
        build_desc = " Auto-build attempts failed:\n" + "\n".join(build_failures[-3:])
    raise FileNotFoundError(
        "UPMEM kvslot helper binary is missing. "
        f"Searched: {searched_desc}. "
        "Build it with `make -C <repo>/src/pim/upmem_kvslot all` "
        "after installing the UPMEM toolchain, or set CLOVER_KVSLOT_HELPER."
        f"{build_desc}"
    )


@dataclass
class _HostKVSlot:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int
    capacity: int
    group_heads: int
    head_dim: int
    segments: List[_TokenSegmentSpec]


@dataclass(frozen=True)
class _TokenSegmentSpec:
    physical_dpu: int
    token_start: int
    token_end: int

    @property
    def token_count(self) -> int:
        return max(0, int(self.token_end) - int(self.token_start))


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
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
    ) -> Dict[str, object]:
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

    def get_rank_groups(self) -> list[list[int]]:
        return []

    def update_group_allowed_dpus(
        self,
        k_slot: str,
        v_slot: str,
        allowed_dpus: list[int] | None,
    ) -> Dict[str, object]:
        return {
            "updated": False,
            "allowed_dpus": [] if allowed_dpus is None else [int(dpu) for dpu in allowed_dpus],
        }


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
    CMD_AV_GROUPED_BATCH = 15
    CMD_QK_SLOT_GROUPED_BATCH = 16
    MAX_SLOTS_PER_DPU = 64
    MAX_BATCH_ITEMS = 32
    MAX_GROUP_SEGMENTS = 8
    MAX_DPU_CAPACITY = max(1, int(os.environ.get("CLOVER_KVSLOT_MAX_CAPACITY", "256")))
    MAX_HEADS = max(1, int(os.environ.get("CLOVER_KVSLOT_MAX_HEADS", "32")))
    SLOT_ARGS_STRUCT = struct.Struct("<IIIIIffI")
    SLOT_ARGS_SIZE = SLOT_ARGS_STRUCT.size

    def __init__(self, binary_path: str, num_dpus: int, cwd: str, kv_dtype: str = "fp32"):
        self.binary_path = binary_path
        self.num_dpus = num_dpus
        self.cwd = cwd
        self.kv_dtype = normalize_resident_kv_dtype(kv_dtype)
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.persistent_state_active = False
        self.helper_env: Dict[str, str] = {}

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
        return resident_kv_dtype_code(self.kv_dtype, "k")

    def _v_dtype_code(self) -> int:
        return resident_kv_dtype_code(self.kv_dtype, "v")

    def _elem_bytes(self) -> int:
        return kvslot_dtype_elem_bytes(self._dtype_code())

    def _v_elem_bytes(self) -> int:
        return kvslot_dtype_elem_bytes(self._v_dtype_code())

    def _slot_arg_scales(
        self,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> tuple[float, float]:
        resolved_k_scale = float(k_scale if k_scale is not None else getattr(initial_k, "_clover_quant_scale", 1.0))
        resolved_v_scale = float(v_scale if v_scale is not None else getattr(initial_v, "_clover_quant_scale", 1.0))
        return resolved_k_scale, resolved_v_scale

    def _slot_args_pack(
        self,
        capacity: int,
        seq_len: int,
        group_heads: int,
        head_dim: int,
        *,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> bytes:
        return self.SLOT_ARGS_STRUCT.pack(
            int(capacity),
            int(seq_len),
            int(group_heads),
            int(head_dim),
            int(self._dtype_code()),
            float(k_scale),
            float(v_scale),
            int(self._v_dtype_code()),
        )

    def _read_slot_args(self) -> tuple[int, int, int, int, int, float, float, int]:
        return self.SLOT_ARGS_STRUCT.unpack(self._read_exact(self.SLOT_ARGS_SIZE))

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

    def allocate_group(
        self,
        slot_id: int,
        capacity: int,
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> Dict[str, int]:
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        header = struct.pack("<IIII", self.MAGIC, self.CMD_ALLOCATE, slot_id, 0)
        k_scale, v_scale = self._slot_arg_scales(initial_k, initial_v, k_scale=k_scale, v_scale=v_scale)
        args = self._slot_args_pack(
            int(capacity),
            seq_len,
            group_heads,
            head_dim,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        payload = header + args + initial_k.numpy().tobytes(order="C") + initial_v.numpy().tobytes(order="C")
        self._write(payload)
        out = self._read_slot_args()
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
            "dtype_code": int(out[4]),
            "k_scale": float(out[5]),
            "v_scale": float(out[6]),
            "v_dtype_code": int(out[7]),
        }

    def append_group(
        self,
        slot_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> Dict[str, int]:
        append_len, group_heads, head_dim = (int(dim) for dim in k_new.shape)
        header = struct.pack("<IIII", self.MAGIC, self.CMD_APPEND, slot_id, 0)
        k_scale, v_scale = self._slot_arg_scales(k_new, v_new, k_scale=k_scale, v_scale=v_scale)
        args = self._slot_args_pack(0, append_len, group_heads, head_dim, k_scale=k_scale, v_scale=v_scale)
        payload = header + args + k_new.numpy().tobytes(order="C") + v_new.numpy().tobytes(order="C")
        self._write(payload)
        out = self._read_slot_args()
        return {
            "capacity": int(out[0]),
            "seq_len": int(out[1]),
            "group_heads": int(out[2]),
            "head_dim": int(out[3]),
            "dtype_code": int(out[4]),
            "k_scale": float(out[5]),
            "v_scale": float(out[6]),
            "v_dtype_code": int(out[7]),
        }

    def materialize_group(self, slot_id: int) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_READBACK, slot_id, 0)
        self._write(header)
        out = self._read_slot_args()
        capacity, seq_len, group_heads, head_dim = (int(item) for item in out[:4])
        k_dtype_code = int(out[4])
        v_dtype_code = int(out[7])
        elems = seq_len * group_heads * head_dim
        k_elem_bytes = kvslot_dtype_elem_bytes(k_dtype_code)
        v_elem_bytes = kvslot_dtype_elem_bytes(v_dtype_code)
        if elems == 0:
            k_dtype = torch.int8 if k_elem_bytes == 1 else (torch.int16 if k_elem_bytes == 2 else torch.int32)
            v_dtype = torch.int8 if v_elem_bytes == 1 else (torch.int16 if v_elem_bytes == 2 else torch.int32)
            k = torch.empty((0, group_heads, head_dim), dtype=k_dtype)
            v = torch.empty((0, group_heads, head_dim), dtype=v_dtype)
        else:
            k_bytes = self._read_exact(elems * k_elem_bytes)
            v_bytes = self._read_exact(elems * v_elem_bytes)
            if k_elem_bytes == 1:
                k = torch.from_numpy(np.frombuffer(k_bytes, dtype=np.int8).copy()).view(seq_len, group_heads, head_dim)
            elif k_elem_bytes == 2:
                k = torch.tensor(struct.unpack(f"<{elems}h", k_bytes), dtype=torch.int16).view(seq_len, group_heads, head_dim)
            else:
                k = torch.tensor(struct.unpack(f"<{elems}i", k_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
            if v_elem_bytes == 1:
                v = torch.from_numpy(np.frombuffer(v_bytes, dtype=np.int8).copy()).view(seq_len, group_heads, head_dim)
            elif v_elem_bytes == 2:
                v = torch.tensor(struct.unpack(f"<{elems}h", v_bytes), dtype=torch.int16).view(seq_len, group_heads, head_dim)
            else:
                v = torch.tensor(struct.unpack(f"<{elems}i", v_bytes), dtype=torch.int32).view(seq_len, group_heads, head_dim)
        return k, v, {
            "capacity": capacity,
            "seq_len": seq_len,
            "group_heads": group_heads,
            "head_dim": head_dim,
            "dtype_code": k_dtype_code,
            "k_scale": float(out[5]),
            "v_scale": float(out[6]),
            "v_dtype_code": v_dtype_code,
        }

    def free_group(self, slot_id: int) -> None:
        header = struct.pack("<IIII", self.MAGIC, self.CMD_FREE, slot_id, 0)
        self._write(header)
        self._read_exact(self.SLOT_ARGS_SIZE)

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

    def qk_slot_scores_grouped_batch(
        self,
        grouped_slot_queries: list[list[tuple[int, int, list[int], torch.Tensor]]],
    ) -> list[torch.Tensor]:
        if not grouped_slot_queries:
            return []
        if len(grouped_slot_queries) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(grouped_slot_queries), self.MAX_BATCH_ITEMS):
                outputs.extend(
                    self.qk_slot_scores_grouped_batch(
                        grouped_slot_queries[offset : offset + self.MAX_BATCH_ITEMS]
                    )
                )
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_QK_SLOT_GROUPED_BATCH, 0, 0),
            struct.pack("<IIII", int(len(grouped_slot_queries)), 0, 0, 0),
        ]
        expected_meta: list[tuple[int, int]] = []

        for group in grouped_slot_queries:
            if not group or len(group) > self.MAX_GROUP_SEGMENTS:
                raise ValueError(f"invalid grouped qk slot query count: {len(group)}")
            segment_count = len(group)
            first_local_head_indices: list[int] | None = None
            first_num_heads: int | None = None
            first_head_dim: int | None = None
            total_window = 0

            payload_parts.append(struct.pack("<I", int(segment_count)))
            for slot_id, window, local_head_indices, queries in group:
                q = (
                    queries
                    if queries.device.type == "cpu" and queries.dtype == torch.float32 and queries.is_contiguous()
                    else queries.detach().cpu().to(torch.float32).contiguous()
                )
                if q.dim() != 2:
                    raise ValueError(f"grouped slot queries must be 2D, got shape {tuple(q.shape)}")
                num_heads = int(q.shape[0])
                head_dim = int(q.shape[1])
                normalized_local_head_indices = [int(idx) for idx in local_head_indices]
                if num_heads != len(normalized_local_head_indices):
                    raise ValueError(
                        "grouped slot query head count mismatch: "
                        f"queries={num_heads} local_head_indices={len(normalized_local_head_indices)}"
                    )
                if first_local_head_indices is None:
                    first_local_head_indices = normalized_local_head_indices
                    first_num_heads = num_heads
                    first_head_dim = head_dim
                else:
                    if normalized_local_head_indices != first_local_head_indices:
                        raise ValueError("grouped qk segments must share identical local_head_indices")
                    if num_heads != first_num_heads or head_dim != first_head_dim:
                        raise ValueError("grouped qk segments must share identical query shape")
                payload_parts.append(
                    struct.pack("<IIII", self.MAGIC, self.CMD_QK_SLOT_GROUPED_BATCH, int(slot_id), 0)
                )
                payload_parts.append(
                    struct.pack(
                        "<IIII",
                        num_heads,
                        int(window),
                        head_dim,
                        0,
                    )
                )
                payload_parts.append(struct.pack(f"<{num_heads}I", *normalized_local_head_indices))
                payload_parts.append(memoryview(q.numpy()).cast("B"))
                total_window += int(window)
            expected_meta.append((int(first_num_heads or 0), int(total_window)))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_items = int(out_args[0])
        if out_num_items != len(grouped_slot_queries):
            raise RuntimeError(
                "kvslot helper returned invalid grouped qk slot batch header: "
                f"expected={len(grouped_slot_queries)} actual={out_num_items}"
            )

        scores_per_item: list[torch.Tensor] = []
        for idx, (expected_heads, expected_window) in enumerate(expected_meta):
            out = struct.unpack("<IIII", self._read_exact(16))
            actual_heads = int(out[0])
            actual_window = int(out[1])
            if actual_heads != expected_heads or actual_window != expected_window:
                raise RuntimeError(
                    "kvslot helper returned invalid grouped qk slot batch item header: "
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

        out = self._read_slot_args()
        _, seq_len, group_heads, head_dim, _ = (int(item) for item in out[:5])
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
            out = self._read_slot_args()
            _, seq_len, group_heads, head_dim, _ = (int(item) for item in out[:5])
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
            out = self._read_slot_args()
            _, seq_len, group_heads, head_dim, _ = (int(item) for item in out[:5])
            if group_heads != expected_heads or seq_len != expected_seq_len:
                raise RuntimeError(
                    "kvslot helper returned invalid softmax-av batch item header: "
                    f"index={idx} scores=({expected_heads}, {expected_seq_len}) header=({seq_len}, {group_heads}, {head_dim})"
                )
            raw_context = self._read_exact(group_heads * head_dim * 4)
            context = torch.from_numpy(np.frombuffer(raw_context, dtype="<f4").copy()).view(group_heads, head_dim)
            contexts.append(context)
        return contexts

    def weighted_value_sum_grouped_batch(
        self,
        grouped_slot_weights: list[list[tuple[int, int, torch.Tensor]]],
    ) -> list[torch.Tensor]:
        if not grouped_slot_weights:
            return []
        max_group_segments = int(getattr(self, "MAX_GROUP_SEGMENTS", _KVSlotHelperClient.MAX_GROUP_SEGMENTS))
        max_dpu_capacity = int(getattr(self, "MAX_DPU_CAPACITY", _KVSlotHelperClient.MAX_DPU_CAPACITY))
        split_groups: list[list[tuple[int, int, torch.Tensor]]] = []
        split_owners: list[int] = []
        split_required = False
        for owner_idx, group in enumerate(grouped_slot_weights):
            if not group:
                raise ValueError(f"invalid grouped slot weight count: {len(group)}")
            current_group: list[tuple[int, int, torch.Tensor]] = []
            current_tokens = 0
            for item in group:
                segment_len = int(item[1])
                if segment_len <= 0:
                    raise ValueError(f"invalid grouped AV segment length: {segment_len}")
                if segment_len > max_dpu_capacity:
                    raise ValueError(
                        "grouped AV segment length exceeds DPU capacity: "
                        f"segment_len={segment_len} max_capacity={max_dpu_capacity}"
                    )
                would_exceed_segments = len(current_group) >= max_group_segments
                would_exceed_capacity = current_tokens + segment_len > max_dpu_capacity
                if current_group and (would_exceed_segments or would_exceed_capacity):
                    split_groups.append(current_group)
                    split_owners.append(int(owner_idx))
                    current_group = []
                    current_tokens = 0
                    split_required = True
                current_group.append(item)
                current_tokens += segment_len
            if current_group:
                split_groups.append(current_group)
                split_owners.append(int(owner_idx))
            if len(split_groups) > owner_idx + 1:
                split_required = True

        if split_required:
            split_outputs = self.weighted_value_sum_grouped_batch(split_groups)
            merged_outputs: list[torch.Tensor | None] = [None for _ in grouped_slot_weights]
            for owner_idx, context in zip(split_owners, split_outputs):
                if merged_outputs[owner_idx] is None:
                    merged_outputs[owner_idx] = context
                else:
                    merged_outputs[owner_idx] = merged_outputs[owner_idx] + context
            missing = [idx for idx, output in enumerate(merged_outputs) if output is None]
            if missing:
                raise RuntimeError(f"grouped AV split produced incomplete outputs: missing_indices={missing[:16]}")
            return [output for output in merged_outputs if output is not None]
        if len(grouped_slot_weights) > self.MAX_BATCH_ITEMS:
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(grouped_slot_weights), self.MAX_BATCH_ITEMS):
                outputs.extend(
                    self.weighted_value_sum_grouped_batch(
                        grouped_slot_weights[offset : offset + self.MAX_BATCH_ITEMS]
                    )
                )
            return outputs

        payload_parts: list[bytes | memoryview] = [
            struct.pack("<IIII", self.MAGIC, self.CMD_AV_GROUPED_BATCH, 0, 0),
            struct.pack("<IIII", int(len(grouped_slot_weights)), 0, 0, 0),
        ]
        expected_meta = []

        for group in grouped_slot_weights:
            if not group or len(group) > self.MAX_GROUP_SEGMENTS:
                raise ValueError(f"invalid grouped slot weight count: {len(group)}")
            group_heads = None
            total_seq_len = 0
            head_dim_hint = None
            payload_parts.append(struct.pack("<I", int(len(group))))
            weight_parts: list[torch.Tensor] = []
            for slot_id, segment_len, weights in group:
                w = (
                    weights
                    if weights.device.type == "cpu" and weights.dtype == torch.float32 and weights.is_contiguous()
                    else weights.detach().cpu().to(torch.float32).contiguous()
                )
                if w.dim() != 2:
                    raise ValueError(f"grouped weights must be 2D, got shape {tuple(w.shape)}")
                if int(w.shape[1]) != int(segment_len):
                    raise ValueError(
                        f"grouped weight segment mismatch: weights={tuple(w.shape)} segment_len={segment_len}"
                    )
                if group_heads is None:
                    group_heads = int(w.shape[0])
                elif int(w.shape[0]) != group_heads:
                    raise ValueError("grouped weights must have identical group_heads")
                payload_parts.append(struct.pack("<IIII", self.MAGIC, self.CMD_AV_GROUPED_BATCH, int(slot_id), 0))
                payload_parts.append(struct.pack("<I", int(segment_len)))
                weight_parts.append(w)
                total_seq_len += int(segment_len)
                head_dim_hint = 0 if head_dim_hint is None else head_dim_hint
            merged_weights = torch.cat(weight_parts, dim=1).contiguous()
            payload_parts.append(memoryview(merged_weights.numpy()).cast("B"))
            expected_meta.append((int(group_heads or 0), int(total_seq_len)))

        self._write_parts(payload_parts)

        out_args = struct.unpack("<IIII", self._read_exact(16))
        out_num_slots = int(out_args[0])
        if out_num_slots != len(grouped_slot_weights):
            raise RuntimeError(
                "kvslot helper returned invalid grouped av batch header: "
                f"expected={len(grouped_slot_weights)} actual={out_num_slots}"
            )

        contexts: list[torch.Tensor] = []
        for idx, (expected_heads, expected_seq_len) in enumerate(expected_meta):
            out = self._read_slot_args()
            _, seq_len, group_heads, head_dim, _ = (int(item) for item in out[:5])
            if group_heads != expected_heads or seq_len != expected_seq_len:
                raise RuntimeError(
                    "kvslot helper returned invalid grouped av batch item header: "
                    f"index={idx} weights=({expected_heads}, {expected_seq_len}) header=({seq_len}, {group_heads}, {head_dim})"
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
            out = self._read_slot_args()
            _, _, group_heads, head_dim, _ = (int(item) for item in out[:5])
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
            out = self._read_slot_args()
            _, _, group_heads, head_dim, _ = (int(item) for item in out[:5])
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

    def _segment_specs_from_plan(
        self,
        seq_len: int,
        preferred_dpu: int | None = None,
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
    ) -> List[_TokenSegmentSpec]:
        if not segment_plan:
            physical_dpu = 0 if preferred_dpu is None else int(preferred_dpu)
            return [
                _TokenSegmentSpec(
                    physical_dpu=int(physical_dpu),
                    token_start=0,
                    token_end=int(seq_len),
                )
            ]

        normalized_specs = [
            _TokenSegmentSpec(
                physical_dpu=int(item.get("dpu_id", preferred_dpu or 0)),
                token_start=int(item.get("token_range_start", 0)),
                token_end=int(item.get("token_range_end", 0)),
            )
            for item in segment_plan
        ]
        normalized_specs = [spec for spec in normalized_specs if spec.token_count > 0]
        if not normalized_specs:
            raise ValueError("segment_plan produced no non-empty token segments")

        normalized_specs = sorted(normalized_specs, key=lambda spec: (int(spec.token_start), int(spec.token_end)))
        cursor = 0
        allowed = None if allowed_dpus is None else {int(dpu) for dpu in allowed_dpus}
        for spec in normalized_specs:
            if int(spec.token_start) != int(cursor):
                raise ValueError(
                    f"segment_plan must be contiguous and non-overlapping: expected_start={cursor} got={spec.token_start}"
                )
            if int(spec.token_end) <= int(spec.token_start):
                raise ValueError(f"invalid token segment range: {spec}")
            if allowed is not None and int(spec.physical_dpu) not in allowed:
                raise ValueError(
                    f"segment_plan physical_dpu={spec.physical_dpu} not present in allowed_dpus={sorted(allowed)}"
                )
            cursor = int(spec.token_end)
        if int(cursor) != int(seq_len):
            raise ValueError(f"segment_plan coverage mismatch: expected_seq_len={seq_len} covered={cursor}")
        return normalized_specs

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
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
    ) -> Dict[str, object]:
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
        segment_specs = self._segment_specs_from_plan(
            seq_len=seq_len,
            preferred_dpu=preferred_dpu,
            allowed_dpus=allowed_dpus,
            segment_plan=segment_plan,
        )
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
            segments=list(segment_specs),
        )
        self.groups[key] = slot
        self.total_allocations += 1
        self.live_slots += 1
        self._adjust_allocated_bytes(0, self._slot_bytes(slot))
        self._record_timing("allocate_group", started_at)
        return {
            "backend": "host",
            "physical_dpu": None if preferred_dpu is None else int(preferred_dpu),
            "storage": "host",
            "segments": [
                {
                    "physical_dpu": int(spec.physical_dpu),
                    "token_start": int(spec.token_start),
                    "token_end": int(spec.token_end),
                    "seq_len": int(spec.token_count),
                    "capacity": int(spec.token_count),
                }
                for spec in segment_specs
            ],
        }

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
        if slot.segments:
            last_segment = slot.segments[-1]
            slot.segments[-1] = _TokenSegmentSpec(
                physical_dpu=int(last_segment.physical_dpu),
                token_start=int(last_segment.token_start),
                token_end=int(expected_seq_len),
            )
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
            "segments": [
                {
                    "physical_dpu": int(spec.physical_dpu),
                    "token_range_start": int(spec.token_start),
                    "token_range_end": int(spec.token_end),
                    "seq_len": int(spec.token_count),
                    "capacity": int(spec.token_count),
                }
                for spec in slot.segments
            ],
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

    def update_group_allowed_dpus(
        self,
        k_slot: str,
        v_slot: str,
        allowed_dpus: list[int] | None,
    ) -> Dict[str, object]:
        del k_slot, v_slot
        return {
            "updated": False,
            "allowed_dpus": [] if allowed_dpus is None else [int(dpu) for dpu in allowed_dpus],
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
        if w.dim() != 2 or int(w.shape[0]) != int(slot.group_heads) or int(w.shape[1]) > int(slot.seq_len):
            raise ValueError(
                f"weight shape mismatch for slot {key}: got={tuple(w.shape)} "
                f"expected=({slot.group_heads}, <= {slot.seq_len})"
            )
        weight_len = int(w.shape[1])
        values = slot.v_cache[slot.seq_len - weight_len : slot.seq_len].float()
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
            if s.dim() != 2 or int(s.shape[0]) != int(slot.group_heads) or int(s.shape[1]) > int(slot.seq_len):
                raise ValueError(
                    f"score shape mismatch for slot {key}: got={tuple(s.shape)} "
                    f"expected=({slot.group_heads}, <= {slot.seq_len})"
                )
            score_len = int(s.shape[1])
            values = slot.v_cache[slot.seq_len - score_len : slot.seq_len].float()
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
    DEFAULT_POOL_CAPACITY_ELEMS = 256 * 32 * 128
    POOL_CAPACITY_ELEMS = DEFAULT_POOL_CAPACITY_ELEMS

    def __init__(
        self,
        repo_root: str,
        num_dpus: int,
        kv_dtype: str = "fp32",
        block_tokens: int = 256,
        placement_policy: str = "rotated",
        host_partial_reduce_enabled: bool = True,
    ):
        self.repo_root = repo_root
        self.num_dpus = num_dpus
        self.kv_dtype = normalize_resident_kv_dtype(kv_dtype)
        self.block_tokens = max(1, int(block_tokens))
        # Decode-time growth blocks should stay small enough to react to
        # stripe expansion, but not so small that one logical request turns
        # into many helper AV/QK items and round explosions.
        self.growth_block_tokens = max(64, min(self.block_tokens, 128))
        self.base_block_rollover_tokens = max(32, min(self.block_tokens, 160))
        self.placement_policy = str(placement_policy)
        self.host_partial_reduce_enabled = bool(host_partial_reduce_enabled)
        if self.kv_dtype not in SUPPORTED_RESIDENT_KV_DTYPES:
            raise ValueError(f"Unsupported resident kv dtype: {self.kv_dtype}")
        kvslot_dir, helper_binary_path = _resolve_kvslot_helper_paths(repo_root)
        self.helper = _KVSlotHelperClient(
            binary_path=helper_binary_path,
            num_dpus=num_dpus,
            cwd=kvslot_dir,
            kv_dtype=self.kv_dtype,
        )
        self.max_dpu_capacity = max(1, int(getattr(self.helper, "MAX_DPU_CAPACITY", 256)))
        self.max_heads = max(1, int(getattr(self.helper, "MAX_HEADS", 32)))
        self.POOL_CAPACITY_ELEMS = int(self.max_dpu_capacity) * int(self.max_heads) * 128
        self.host_fallback = HostResidentKVStore()
        self.slot_mapping: Dict[tuple[str, str], Dict[str, object]] = {}
        self._slot_id_map: Dict[tuple[str, str], int] = {}
        self._free_slot_ids_by_dpu: list[list[int]] = [[] for _ in range(num_dpus)]
        self._next_slot_seq_by_dpu = [0 for _ in range(num_dpus)]
        self.dpu_live_slot_counts_by_dpu = [0 for _ in range(num_dpus)]
        self.slot_pressure_soft_limit = max(1, int(self.helper.MAX_SLOTS_PER_DPU) - max(8, int(self.helper.MAX_SLOTS_PER_DPU // 8)))
        self.slot_pressure_aware_alloc_enabled = False
        self._helper_topology_cache: Dict[int, Dict[str, int]] = {}
        self.dpu_allocations = 0
        self.fallback_allocations = 0
        self.dpu_free_ops = 0
        self.dpu_allocate_failures = 0
        self.dpu_allocate_failure_reasons: Dict[str, int] = {}
        self.dpu_allocate_last_failure: Dict[str, object] = {}
        self.dpu_live_slots = 0
        self.dpu_capacity_fallbacks = 0
        self.slot_spill_alloc_enabled = False
        self.slot_spill_allocations = 0
        self.emergency_slot_spill_enabled = False
        self.emergency_slot_spill_allocations = 0
        self.host_fallback_migrations = 0
        self.slot_capacity_reroutes = 0
        self.reserve_segment_tail_capacity_enabled = False
        self.reserve_segment_tail_capacity_tokens = 0
        self.dpu_live_elems_by_dpu = [0 for _ in range(num_dpus)]
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
            "qk_softmax_weighted_value_sum_batch_host_reduce": 0.0,
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
            "qk_softmax_weighted_value_sum_batch_host_reduce_items": 0,
        }

    def set_experimental_flags(
        self,
        *,
        context_fused_enabled: bool | None = None,
        shape_rounds_enabled: bool | None = None,
        rank_spread_alloc_enabled: bool | None = None,
        rank_spread_multi_rank_batch_enabled: bool | None = None,
        slot_spill_alloc_enabled: bool | None = None,
        slot_pressure_aware_alloc_enabled: bool | None = None,
        emergency_slot_spill_enabled: bool | None = None,
        host_partial_reduce_enabled: bool | None = None,
        reserve_segment_tail_capacity_enabled: bool | None = None,
        reserve_segment_tail_capacity_tokens: int | None = None,
    ) -> None:
        if context_fused_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_CONTEXT_FUSED", bool(context_fused_enabled))
        if shape_rounds_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_SHAPE_ROUNDS", bool(shape_rounds_enabled))
        if rank_spread_alloc_enabled is not None:
            self.helper.set_env_flag("CLOVER_KVSLOT_RANK_SPREAD_ALLOC", bool(rank_spread_alloc_enabled))
        if rank_spread_multi_rank_batch_enabled is not None:
            self.helper.set_env_flag(
                "CLOVER_KVSLOT_ALLOW_RANK_SPREAD_MULTI_RANK_BATCH",
                bool(rank_spread_multi_rank_batch_enabled),
            )
        if slot_spill_alloc_enabled is not None:
            self.slot_spill_alloc_enabled = bool(slot_spill_alloc_enabled)
        if slot_pressure_aware_alloc_enabled is not None:
            self.slot_pressure_aware_alloc_enabled = bool(slot_pressure_aware_alloc_enabled)
        if emergency_slot_spill_enabled is not None:
            self.emergency_slot_spill_enabled = bool(emergency_slot_spill_enabled)
        if host_partial_reduce_enabled is not None:
            self.host_partial_reduce_enabled = bool(host_partial_reduce_enabled)
        if reserve_segment_tail_capacity_enabled is not None:
            self.reserve_segment_tail_capacity_enabled = bool(reserve_segment_tail_capacity_enabled)
        if reserve_segment_tail_capacity_tokens is not None:
            self.reserve_segment_tail_capacity_tokens = max(0, int(reserve_segment_tail_capacity_tokens))

    def _merge_partial_contexts(
        self,
        partial_entries: list[Dict[str, object]],
        partial_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        host_reduce = _load_host_reduction_module()
        if host_reduce is not None:
            ordered_indices: list[int] = []
            local_max_parts: list[torch.Tensor] = []
            local_sum_parts: list[torch.Tensor] = []
            local_output_parts: list[torch.Tensor] = []

            for entry, (segment_context, segment_row_max, segment_row_sum) in zip(partial_entries, partial_outputs):
                logical_idx = int(entry["logical_idx"])
                score_scale = float(entry["payload"][4])
                segment_numerator = segment_context.to(torch.float32).contiguous()
                segment_row_max = (segment_row_max.to(torch.float32) * score_scale).contiguous()
                segment_row_sum = segment_row_sum.to(torch.float32).contiguous()
                ordered_indices.append(logical_idx)
                local_max_parts.append(segment_row_max)
                local_sum_parts.append(segment_row_sum)
                local_output_parts.append(segment_numerator)

            if ordered_indices:
                reduced = host_reduce.merge_partial_contexts(
                    ordered_indices,
                    local_max_parts,
                    local_sum_parts,
                    local_output_parts,
                )
                return {int(logical_idx): context for logical_idx, context in reduced.items()}

        merged_numerators: Dict[int, torch.Tensor] = {}
        merged_row_max: Dict[int, torch.Tensor] = {}
        merged_row_sum: Dict[int, torch.Tensor] = {}

        for entry, (segment_context, segment_row_max, segment_row_sum) in zip(partial_entries, partial_outputs):
            logical_idx = int(entry["logical_idx"])
            score_scale = float(entry["payload"][4])
            segment_numerator = segment_context.to(torch.float32)
            segment_row_max = segment_row_max.to(torch.float32) * score_scale
            segment_row_sum = segment_row_sum.to(torch.float32)
            if logical_idx not in merged_numerators:
                merged_numerators[logical_idx] = segment_numerator
                merged_row_max[logical_idx] = segment_row_max
                merged_row_sum[logical_idx] = segment_row_sum
                continue

            prev_row_max = merged_row_max[logical_idx]
            prev_row_sum = merged_row_sum[logical_idx]
            prev_numerator = merged_numerators[logical_idx]
            combined_row_max = torch.maximum(prev_row_max, segment_row_max)
            prev_scale = torch.exp(prev_row_max - combined_row_max)
            seg_scale = torch.exp(segment_row_max - combined_row_max)
            combined_row_sum = prev_row_sum * prev_scale + segment_row_sum * seg_scale
            merged_numerators[logical_idx] = (
                prev_numerator * prev_scale.unsqueeze(1)
            ) + (segment_numerator * seg_scale.unsqueeze(1))
            merged_row_max[logical_idx] = combined_row_max
            merged_row_sum[logical_idx] = combined_row_sum

        return {
            logical_idx: numerator / torch.clamp(merged_row_sum[logical_idx], min=1e-12).unsqueeze(1)
            for logical_idx, numerator in merged_numerators.items()
        }

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
        if self._helper_topology_cache:
            return
        try:
            helper_topology = self.helper.get_topology()
        except Exception:
            self._helper_topology_cache = {}
            return
        self._helper_topology_cache = {
            int(item["logical_dpu_id"]): {
                "rank_index": int(item["rank_index"]),
                "rank_id": int(item["rank_id"]),
            }
            for item in helper_topology.get("items", [])
        }

    def get_rank_groups(self) -> list[list[int]]:
        self._ensure_topology_cache()
        if not self._helper_topology_cache:
            return []
        by_rank: Dict[int, list[int]] = {}
        for physical_dpu, item in self._helper_topology_cache.items():
            rank_index = int(item.get("rank_index", 0))
            by_rank.setdefault(rank_index, []).append(int(physical_dpu))
        return [sorted(items) for _, items in sorted(by_rank.items()) if items]

    def update_group_allowed_dpus(
        self,
        k_slot: str,
        v_slot: str,
        allowed_dpus: list[int] | None,
    ) -> Dict[str, object]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        normalized = []
        if allowed_dpus and self.num_dpus > 0:
            normalized = sorted(
                {
                    int(physical_dpu) % self.num_dpus
                    for physical_dpu in allowed_dpus
                }
            )
        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            slot_info["allowed_physical_dpus"] = list(normalized)
            if normalized:
                slot_info["base_physical_dpu"] = int(normalized[0])
            return {
                "updated": True,
                "allowed_dpus": list(normalized),
            }
        return {
            "updated": False,
            "allowed_dpus": list(normalized),
        }

    def _helper_rank_sort_value(self, physical_dpu: int) -> int:
        self._ensure_topology_cache()
        rank_index = self._topology_rank_index(int(physical_dpu))
        if rank_index is None:
            return max(self.num_dpus, 1)
        return int(rank_index)

    def _helper_submit_sort_key(
        self,
        *,
        physical_dpu: int,
        shape_key: tuple[int, ...],
        slot_id: int,
        logical_idx: int,
        segment_ordinal: int = 0,
    ) -> tuple[int, ...]:
        return (
            self._helper_rank_sort_value(int(physical_dpu)),
            *[int(value) for value in shape_key],
            int(physical_dpu),
            int(slot_id),
            int(logical_idx),
            int(segment_ordinal),
        )

    def _should_rollover_tail_block(
        self,
        slot_info: Dict[str, object],
        tail_block: Dict[str, object] | None,
    ) -> bool:
        if tail_block is None:
            return False
        allowed_dpus = slot_info.get("allowed_physical_dpus")
        if not isinstance(allowed_dpus, list) or len(allowed_dpus) <= 1:
            return False
        tail_physical_dpu = int(tail_block.get("physical_dpu", 0))
        normalized_allowed = [
            int(physical_dpu) % max(self.num_dpus, 1)
            for physical_dpu in allowed_dpus
        ]
        tail_idx = -1
        try:
            tail_idx = normalized_allowed.index(tail_physical_dpu % max(self.num_dpus, 1))
        except ValueError:
            tail_idx = -1
        if tail_idx < 0:
            return False
        unused_candidates = normalized_allowed[tail_idx + 1 :]
        if not unused_candidates:
            return False
        tail_capacity = int(tail_block.get("capacity", 0))
        tail_seq_len = int(tail_block.get("seq_len", 0))
        tail_available = max(0, tail_capacity - tail_seq_len)
        block_kind = str(tail_block.get("block_kind", "growth"))
        if block_kind == "base":
            # Keep shorter prompts below the base resident length on a compact
            # single-block path. Rollover on base blocks is meant to create new
            # placement freedom once decode is actually pushing the request
            # beyond its base resident budget, not while it is still fitting
            # comfortably under that budget.
            if int(slot_info.get("seq_len", 0)) < int(self.block_tokens):
                return False
            rollover_threshold = max(8, int(self.base_block_rollover_tokens))
        else:
            rollover_threshold = max(8, self.growth_block_tokens // 4)
        return tail_available <= rollover_threshold

    def _record_timing(self, name: str, started_at: float) -> None:
        self.op_timing_totals_s[name] += float(time.perf_counter() - started_at)
        self.op_timing_counts[name] += 1

    def _complete_batch_outputs(
        self,
        op_name: str,
        outputs: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        missing = [idx for idx, output in enumerate(outputs) if output is None]
        if missing:
            raise RuntimeError(
                f"{op_name} produced incomplete batch outputs: "
                f"expected={len(outputs)} missing_indices={missing[:16]}"
            )
        return [output for output in outputs if output is not None]

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

    def _max_slots_per_dpu(self) -> int:
        return max(0, int(getattr(self.helper, "MAX_SLOTS_PER_DPU", 0)))

    def _slot_id_is_valid_for_dpu(self, slot_id: int, physical_dpu: int) -> bool:
        if self.num_dpus <= 0:
            return True
        slot_id = int(slot_id)
        if slot_id < 0:
            return False
        max_slots_per_dpu = self._max_slots_per_dpu()
        if max_slots_per_dpu <= 0:
            return False
        max_slot_count = int(self.num_dpus) * int(max_slots_per_dpu)
        return slot_id < max_slot_count and (slot_id % self.num_dpus) == (
            int(physical_dpu) % self.num_dpus
        )

    def _valid_free_slot_ids_for_dpu(self, physical_dpu: int) -> list[int]:
        if self.num_dpus <= 0:
            return []
        normalized_dpu = int(physical_dpu) % self.num_dpus
        free_ids = self._free_slot_ids_by_dpu[normalized_dpu]
        valid_ids = [
            int(slot_id)
            for slot_id in free_ids
            if self._slot_id_is_valid_for_dpu(int(slot_id), normalized_dpu)
        ]
        if len(valid_ids) != len(free_ids):
            # Older error paths could recycle helper-unsupported slot ids. Drop
            # them eagerly so capacity checks cannot be fooled by stale state.
            self._free_slot_ids_by_dpu[normalized_dpu] = valid_ids
        return valid_ids

    def _pop_free_slot_id_for_dpu(self, physical_dpu: int) -> int | None:
        if self.num_dpus <= 0:
            return None
        normalized_dpu = int(physical_dpu) % self.num_dpus
        free_ids = self._free_slot_ids_by_dpu[normalized_dpu]
        while free_ids:
            slot_id = int(free_ids.pop())
            if self._slot_id_is_valid_for_dpu(slot_id, normalized_dpu):
                return slot_id
        return None

    def _remember_free_slot_id(self, physical_dpu: int, slot_id: int) -> None:
        if self.num_dpus <= 0:
            return
        normalized_dpu = int(physical_dpu) % self.num_dpus
        slot_id = int(slot_id)
        if not self._slot_id_is_valid_for_dpu(slot_id, normalized_dpu):
            return
        free_ids = self._free_slot_ids_by_dpu[normalized_dpu]
        if slot_id not in free_ids:
            free_ids.append(slot_id)

    def _assign_slot_id(self, key: tuple[str, str], preferred_dpu: int | None = None) -> int:
        if key in self._slot_id_map:
            return self._slot_id_map[key]
        physical_dpu = self._normalize_preferred_dpu(preferred_dpu)
        free_slot_id = self._pop_free_slot_id_for_dpu(physical_dpu)
        if free_slot_id is not None:
            slot_id = int(free_slot_id)
        else:
            max_slots_per_dpu = self._max_slots_per_dpu()
            seq = self._next_slot_seq_by_dpu[physical_dpu]
            if seq >= max_slots_per_dpu:
                raise RuntimeError(
                    "No DPU KV slot capacity remains "
                    f"for physical_dpu={physical_dpu}: "
                    f"next_seq={seq} max_slots_per_dpu={max_slots_per_dpu}"
                )
            slot_id = physical_dpu + (seq * max(self.num_dpus, 1))
            self._next_slot_seq_by_dpu[physical_dpu] += 1
        self._slot_id_map[key] = slot_id
        return slot_id

    def _slot_id_to_physical_dpu(self, slot_id: int | None) -> int | None:
        if slot_id is None or self.num_dpus <= 0:
            return None
        return int(slot_id) % self.num_dpus

    def _candidate_dpus(self, preferred_dpu: int | None = None) -> list[int]:
        if self.num_dpus <= 0:
            return [0]
        normalized = self._normalize_preferred_dpu(preferred_dpu)
        ordered = list(range(self.num_dpus))
        if normalized != 0:
            ordered = ordered[normalized:] + ordered[:normalized]
        return ordered

    def _allowed_candidate_dpus(
        self,
        preferred_dpu: int | None = None,
        allowed_dpus: list[int] | None = None,
    ) -> list[int]:
        ordered = self._candidate_dpus(preferred_dpu)
        if allowed_dpus is None:
            return ordered
        allowed_set = {
            int(physical_dpu) % self.num_dpus
            for physical_dpu in allowed_dpus
            if self.num_dpus > 0
        }
        filtered = [physical_dpu for physical_dpu in ordered if physical_dpu in allowed_set]
        return filtered or ordered

    def _dpu_has_slot_capacity(self, physical_dpu: int) -> bool:
        if self.num_dpus <= 0:
            return True
        normalized_dpu = int(physical_dpu) % self.num_dpus
        max_slots = self._max_slots_per_dpu()
        next_seq = int(self._next_slot_seq_by_dpu[normalized_dpu])
        free_ids = len(self._valid_free_slot_ids_for_dpu(normalized_dpu))
        return free_ids > 0 or next_seq < max_slots

    def _dpu_has_allocation_capacity(self, physical_dpu: int, elem_count: int) -> bool:
        if self.num_dpus <= 0:
            return True
        normalized = int(physical_dpu) % self.num_dpus
        return (
            self._dpu_has_slot_capacity(normalized)
            and self.dpu_live_elems_by_dpu[normalized] + int(elem_count) <= self.POOL_CAPACITY_ELEMS
        )

    def _record_slot_capacity_reroute(self) -> None:
        self.slot_capacity_reroutes = int(getattr(self, "slot_capacity_reroutes", 0)) + 1

    def _record_dpu_allocate_failure(self, stage: str, exc: BaseException, **context: object) -> None:
        reason = str(exc).strip() or exc.__class__.__name__
        reason_key = reason[:240]
        failure_reasons = getattr(self, "dpu_allocate_failure_reasons", {})
        if not isinstance(failure_reasons, dict):
            failure_reasons = {}
        failure_reasons[reason_key] = int(failure_reasons.get(reason_key, 0)) + 1
        self.dpu_allocate_failure_reasons = failure_reasons
        self.dpu_allocate_last_failure = {
            "stage": str(stage),
            "reason": reason,
            **dict(context),
        }

    def _choose_physical_dpu_with_slot_capacity(
        self,
        *,
        preferred_dpu: int,
        elem_count: int,
        allowed_dpus: list[int] | None = None,
    ) -> int:
        candidates = self._allowed_candidate_dpus(
            preferred_dpu=preferred_dpu,
            allowed_dpus=allowed_dpus,
        )
        viable = [
            int(physical_dpu)
            for physical_dpu in candidates
            if self._dpu_has_allocation_capacity(int(physical_dpu), int(elem_count))
        ]
        if viable:
            return min(
                viable,
                key=lambda physical_dpu: self._score_physical_dpu(
                    physical_dpu,
                    int(elem_count),
                    preferred_dpu=preferred_dpu,
                ),
            )
        spill_enabled = bool(
            self.slot_spill_alloc_enabled
            or (
                getattr(self, "emergency_slot_spill_enabled", False)
                and allowed_dpus is not None
            )
        )
        if spill_enabled:
            global_candidates = self._candidate_dpus(preferred_dpu)
            global_viable = [
                int(physical_dpu)
                for physical_dpu in global_candidates
                if self._dpu_has_allocation_capacity(int(physical_dpu), int(elem_count))
            ]
            if global_viable:
                if self.slot_spill_alloc_enabled:
                    self.slot_spill_allocations += 1
                else:
                    self.emergency_slot_spill_allocations += 1
                self._record_slot_capacity_reroute()
                return min(
                    global_viable,
                    key=lambda physical_dpu: self._score_physical_dpu(
                        physical_dpu,
                        int(elem_count),
                        preferred_dpu=preferred_dpu,
                    ),
                )
        first_candidate = int(candidates[0])
        if not self._dpu_has_slot_capacity(first_candidate):
            raise RuntimeError(
                "No DPU KV slot capacity remains in allowed placement set: "
                f"preferred_dpu={preferred_dpu} "
                f"allowed_dpus={[int(item) for item in candidates]} "
                f"slot_spill_alloc_enabled={bool(self.slot_spill_alloc_enabled)} "
                f"emergency_slot_spill_enabled={bool(getattr(self, 'emergency_slot_spill_enabled', False))}"
            )
        return first_candidate

    def _dpu_capacity_headroom(self, physical_dpu: int, elem_count: int = 0) -> int:
        return max(
            0,
            int(self.POOL_CAPACITY_ELEMS)
            - int(self.dpu_live_elems_by_dpu[physical_dpu])
            - max(0, int(elem_count)),
        )

    def _circular_dpu_distance(self, lhs: int, rhs: int) -> int:
        if self.num_dpus <= 0:
            return 0
        lhs_n = int(lhs) % self.num_dpus
        rhs_n = int(rhs) % self.num_dpus
        forward = (lhs_n - rhs_n) % self.num_dpus
        backward = (rhs_n - lhs_n) % self.num_dpus
        return min(forward, backward)

    def _locality_score_components(
        self,
        physical_dpu: int,
        preferred_dpu: int | None = None,
    ) -> tuple[int, int, int]:
        if preferred_dpu is None or self.num_dpus <= 0:
            return (0, 0, int(physical_dpu))
        normalized_preferred = self._normalize_preferred_dpu(preferred_dpu)
        candidate_rank = self._topology_rank_index(int(physical_dpu))
        preferred_rank = self._topology_rank_index(normalized_preferred)
        rank_miss = 0
        if candidate_rank is not None and preferred_rank is not None:
            rank_miss = 0 if int(candidate_rank) == int(preferred_rank) else 1
        dpu_distance = self._circular_dpu_distance(int(physical_dpu), normalized_preferred)
        return (rank_miss, dpu_distance, int(physical_dpu))

    def _score_physical_dpu(self, physical_dpu: int, elem_count: int, preferred_dpu: int | None = None) -> tuple:
        live_elems = int(self.dpu_live_elems_by_dpu[physical_dpu])
        live_slot_counts = getattr(self, "dpu_live_slot_counts_by_dpu", None)
        if not isinstance(live_slot_counts, list) or physical_dpu >= len(live_slot_counts):
            live_slot_counts = self._count_slots_by_dpu()
            self.dpu_live_slot_counts_by_dpu = list(live_slot_counts)
        live_slot_count = int(live_slot_counts[physical_dpu])
        free_ids = len(self._valid_free_slot_ids_for_dpu(physical_dpu))
        next_seq = int(self._next_slot_seq_by_dpu[physical_dpu])
        rank_index = self._topology_rank_index(physical_dpu)
        rank_miss, preferred_distance, _ = self._locality_score_components(
            physical_dpu,
            preferred_dpu=preferred_dpu,
        )
        over_capacity = 0 if live_elems + int(elem_count) <= self.POOL_CAPACITY_ELEMS else 1
        no_slot_capacity = 0 if self._dpu_has_slot_capacity(physical_dpu) else 1
        slot_pressure_aware = bool(getattr(self, "slot_pressure_aware_alloc_enabled", True))
        soft_limit = int(getattr(self, "slot_pressure_soft_limit", int(self.helper.MAX_SLOTS_PER_DPU)))
        if slot_pressure_aware:
            slot_pressure_excess = max(0, int(live_slot_count) - int(soft_limit))
            projected_live_elems = int(live_elems) + max(0, int(elem_count))
        else:
            slot_pressure_excess = 0
            projected_live_elems = 0
        return (
            over_capacity,
            no_slot_capacity,
            rank_miss,
            slot_pressure_excess,
            max(
                0,
                live_slot_count - int(soft_limit),
            )
            if slot_pressure_aware
            else 0,
            preferred_distance,
            projected_live_elems,
            live_elems,
            next_seq,
            -free_ids,
            rank_index if rank_index is not None else physical_dpu,
            physical_dpu,
        )

    def choose_physical_dpu(
        self,
        *,
        elem_count: int,
        preferred_dpu: int | None = None,
        placement_policy: str | None = None,
        allowed_dpus: list[int] | None = None,
    ) -> int:
        if self.num_dpus <= 0:
            return 0
        policy = str(placement_policy or self.placement_policy)
        candidates = self._allowed_candidate_dpus(
            preferred_dpu=preferred_dpu,
            allowed_dpus=allowed_dpus,
        )
        if policy != "load_aware":
            first_candidate = int(candidates[0])
            if self._dpu_has_allocation_capacity(first_candidate, int(elem_count)):
                return first_candidate
            for physical_dpu in candidates[1:]:
                candidate = int(physical_dpu)
                if self._dpu_has_allocation_capacity(candidate, int(elem_count)):
                    self._record_slot_capacity_reroute()
                    return candidate
            if self.slot_spill_alloc_enabled:
                global_candidates = self._candidate_dpus(preferred_dpu)
                for physical_dpu in global_candidates:
                    candidate = int(physical_dpu)
                    if self._dpu_has_allocation_capacity(candidate, int(elem_count)):
                        self.slot_spill_allocations += 1
                        self._record_slot_capacity_reroute()
                        return candidate
            return first_candidate
        best_dpu = min(
            candidates,
            key=lambda physical_dpu: self._score_physical_dpu(
                physical_dpu,
                int(elem_count),
                preferred_dpu=preferred_dpu,
            ),
        )
        return int(best_dpu)

    def _count_slots_by_dpu(self) -> list[int]:
        counts = [0 for _ in range(max(self.num_dpus, 0))]
        for slot_info in self.slot_mapping.values():
            if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
                for block in slot_info.get("blocks", slot_info.get("segments", [])):
                    physical_dpu = int(block.get("physical_dpu", 0))
                    if 0 <= physical_dpu < len(counts):
                        counts[physical_dpu] += 1
                continue
            if slot_info["backend"] != "dpu":
                continue
            physical_dpu = int(slot_info.get("physical_dpu", 0))
            if 0 <= physical_dpu < len(counts):
                counts[physical_dpu] += 1
        return counts

    def _summarize_allocator_stats(self, allocator_stats: list[Dict[str, object]]) -> Dict[str, float]:
        if not allocator_stats:
            return {
                "max_usage_ratio": 0.0,
                "avg_usage_ratio": 0.0,
                "min_usage_ratio": 0.0,
                "max_free_range_count": 0.0,
                "min_largest_free_range": 0.0,
                "max_live_slot_count": 0.0,
            }
        usage_ratios = [float(item.get("usage_ratio", 0.0)) for item in allocator_stats]
        free_range_counts = [float(item.get("free_range_count", 0.0)) for item in allocator_stats]
        largest_free_ranges = [float(item.get("largest_free_range", 0.0)) for item in allocator_stats]
        live_slot_counts = [float(item.get("live_slot_count", 0.0)) for item in allocator_stats]
        return {
            "max_usage_ratio": max(usage_ratios),
            "avg_usage_ratio": sum(usage_ratios) / len(usage_ratios),
            "min_usage_ratio": min(usage_ratios),
            "max_free_range_count": max(free_range_counts),
            "min_largest_free_range": min(largest_free_ranges),
            "max_live_slot_count": max(live_slot_counts),
        }

    def _summarize_balance(self) -> Dict[str, object]:
        live_elems = [int(item) for item in self.dpu_live_elems_by_dpu]
        slot_counts = self._count_slots_by_dpu()
        active_indices = [idx for idx, value in enumerate(live_elems) if value > 0]
        active_live = [live_elems[idx] for idx in active_indices]
        active_slots = [slot_counts[idx] for idx in active_indices]
        active_count = len(active_indices)
        avg_live = (sum(active_live) / active_count) if active_count > 0 else 0.0
        max_live = max(active_live) if active_live else 0
        min_live = min(active_live) if active_live else 0
        imbalance_ratio = (max_live / avg_live) if avg_live > 0 else 0.0
        spread_ratio = ((max_live - min_live) / avg_live) if avg_live > 0 else 0.0
        return {
            "active_dpus": active_count,
            "active_ratio": (float(active_count) / float(self.num_dpus)) if self.num_dpus > 0 else 0.0,
            "avg_live_elems_active": avg_live,
            "max_live_elems": max_live,
            "min_live_elems_active": min_live,
            "imbalance_ratio": imbalance_ratio,
            "spread_ratio": spread_ratio,
            "active_slot_counts": active_slots,
        }

    def _summarize_rank_balance(self) -> Dict[str, object]:
        rank_live: Dict[int, int] = {}
        rank_slots: Dict[int, int] = {}
        for physical_dpu, live_elems in enumerate(self.dpu_live_elems_by_dpu):
            rank_id = self._topology_rank_id(physical_dpu)
            if rank_id is None:
                continue
            rank_live[rank_id] = rank_live.get(rank_id, 0) + int(live_elems)
        for physical_dpu, slot_count in enumerate(self._count_slots_by_dpu()):
            rank_id = self._topology_rank_id(physical_dpu)
            if rank_id is None:
                continue
            rank_slots[rank_id] = rank_slots.get(rank_id, 0) + int(slot_count)
        if not rank_live and not rank_slots:
            return {
                "rank_count": 0,
                "active_rank_count": 0,
                "rank_live_elems": {},
                "rank_slot_counts": {},
                "imbalance_ratio": 0.0,
            }
        live_values = list(rank_live.values()) or [0]
        avg_live = sum(live_values) / len(live_values) if live_values else 0.0
        return {
            "rank_count": len(set(rank_live) | set(rank_slots)),
            "active_rank_count": sum(1 for value in live_values if value > 0),
            "rank_live_elems": {str(key): int(value) for key, value in sorted(rank_live.items())},
            "rank_slot_counts": {str(key): int(value) for key, value in sorted(rank_slots.items())},
            "imbalance_ratio": (max(live_values) / avg_live) if avg_live > 0 else 0.0,
        }

    def _summarize_blocks(self) -> Dict[str, object]:
        blocked_slots = 0
        total_blocks = 0
        max_blocks_per_slot = 0
        total_block_capacity = 0
        total_block_live = 0
        for slot_info in self.slot_mapping.values():
            if slot_info["backend"] not in {"dpu_segmented", "dpu_blocked"}:
                continue
            blocks = slot_info.get("blocks", slot_info.get("segments", []))
            blocked_slots += 1
            total_blocks += len(blocks)
            max_blocks_per_slot = max(max_blocks_per_slot, len(blocks))
            for block in blocks:
                total_block_capacity += int(block.get("capacity", 0))
                total_block_live += int(block.get("seq_len", 0))
        return {
            "blocked_slots": blocked_slots,
            "total_blocks": total_blocks,
            "max_blocks_per_slot": max_blocks_per_slot,
            "avg_blocks_per_slot": (float(total_blocks) / float(blocked_slots)) if blocked_slots > 0 else 0.0,
            "total_block_capacity_tokens": total_block_capacity,
            "total_block_live_tokens": total_block_live,
            "block_fill_ratio": (float(total_block_live) / float(total_block_capacity)) if total_block_capacity > 0 else 0.0,
        }

    def _supports_dpu_slot(self, initial_k: torch.Tensor, capacity: int, slot_id: int | None) -> bool:
        if slot_id is None:
            return False
        if int(slot_id) < 0:
            return False
        max_slots = self.num_dpus * self.helper.MAX_SLOTS_PER_DPU
        if slot_id >= max_slots:
            return False
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        return (
            capacity <= int(self.max_dpu_capacity)
            and group_heads <= int(self.max_heads)
            and head_dim <= 128
            and seq_len <= capacity
        )

    def _encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._encode_tensor_for_dtype(tensor, resident_kv_dtype_code(self.kv_dtype, "k"))[0]

    def _encode_tensor_for_dtype(
        self,
        tensor: torch.Tensor,
        dtype_code: int,
        *,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        if int(dtype_code) == KVSLOT_DTYPE_INT8:
            return self._encode_tensor_int8(tensor, scale=scale)
        if int(dtype_code) == KVSLOT_DTYPE_FP16:
            return tensor.detach().cpu().to(torch.float16).contiguous().view(torch.int16), 1.0
        return tensor.detach().cpu().float().contiguous().view(torch.int32), 1.0

    def _encode_tensor_int8(
        self,
        tensor: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        tensor_fp32 = tensor.detach().cpu().to(torch.float32).contiguous()
        if scale is None:
            max_abs = float(torch.max(torch.abs(tensor_fp32)).item()) if tensor_fp32.numel() > 0 else 0.0
            scale = max(max_abs / 127.0, 1.0e-8)
        else:
            scale = max(float(scale), 1.0e-8)
        encoded = torch.clamp(torch.round(tensor_fp32 / float(scale)), -127, 127).to(torch.int8).contiguous()
        encoded._clover_quant_scale = float(scale)  # type: ignore[attr-defined]
        return encoded, float(scale)

    def _encode_kv_pair(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        encoded_k, resolved_k_scale = self._encode_tensor_for_dtype(
            k_tensor,
            resident_kv_dtype_code(self.kv_dtype, "k"),
            scale=k_scale,
        )
        encoded_v, resolved_v_scale = self._encode_tensor_for_dtype(
            v_tensor,
            resident_kv_dtype_code(self.kv_dtype, "v"),
            scale=v_scale,
        )
        return encoded_k, encoded_v, resolved_k_scale, resolved_v_scale

    def _decode_tensor(
        self,
        tensor: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        return self._decode_tensor_for_dtype(
            tensor,
            resident_kv_dtype_code(self.kv_dtype, "k"),
            scale=scale,
        )

    def _decode_tensor_for_dtype(
        self,
        tensor: torch.Tensor,
        dtype_code: int,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if int(dtype_code) == KVSLOT_DTYPE_INT8:
            return (tensor.to(torch.float32) * float(1.0 if scale is None else scale)).contiguous()
        if int(dtype_code) == KVSLOT_DTYPE_FP16:
            return tensor.view(torch.float16).to(torch.float32).contiguous()
        return tensor.view(torch.float32).contiguous()

    def _slot_elem_count(self, capacity: int, group_heads: int, head_dim: int) -> int:
        elem_count = int(capacity) * int(group_heads) * int(head_dim)
        k_words = kvslot_packed_word_count(elem_count, resident_kv_dtype_code(self.kv_dtype, "k"))
        v_words = kvslot_packed_word_count(elem_count, resident_kv_dtype_code(self.kv_dtype, "v"))
        return max(k_words, v_words)

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

    def _reserve_tail_block_capacities(
        self,
        live_lengths: list[int],
        total_capacity: int,
        max_reserve_tokens: int | None = None,
    ) -> list[int]:
        capacities = [max(0, int(length)) for length in live_lengths]
        reserve = max(0, int(total_capacity) - sum(capacities))
        if max_reserve_tokens is not None and int(max_reserve_tokens) > 0:
            reserve = min(reserve, int(max_reserve_tokens))
        if reserve <= 0:
            return capacities

        # Decode appends extend the sequence tail, so place spare capacity in
        # the last token blocks first. This avoids allocating a new growth slot
        # immediately after prefill when seq_len < resident capacity.
        for idx in range(len(capacities) - 1, -1, -1):
            room = max(0, int(self.block_tokens) - int(capacities[idx]))
            if room <= 0:
                continue
            take = min(room, reserve)
            capacities[idx] += int(take)
            reserve -= int(take)
            if reserve <= 0:
                break
        return capacities

    def _supports_dpu_shape(self, group_heads: int, head_dim: int) -> bool:
        return int(group_heads) <= 32 and int(head_dim) <= 128

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
        self._remember_free_slot_id(normalized_dpu, int(slot_id))

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
            self.dpu_live_slot_counts_by_dpu[normalized_dpu] = max(
                0, self.dpu_live_slot_counts_by_dpu[normalized_dpu] - 1
            )
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
            self.dpu_live_slot_counts_by_dpu[physical_dpu] = max(
                0, self.dpu_live_slot_counts_by_dpu[physical_dpu] - 1
            )
            elem_count = int(block.get("elem_count", 0))
            self.dpu_live_elems_by_dpu[physical_dpu] = max(
                0, self.dpu_live_elems_by_dpu[physical_dpu] - elem_count
            )
            block_key = tuple(block["block_key"])
            self._slot_id_map.pop(block_key, None)
            self._remember_free_slot_id(physical_dpu, slot_id)
        self.helper.persistent_state_active = self.dpu_live_slots > 0
        if self.dpu_live_slots == 0:
            self.helper.close()

    def _active_block_plan(
        self,
        slot_info: Dict[str, object],
        window: int | None = None,
    ) -> list[tuple[Dict[str, object], int]]:
        blocks = slot_info.get("blocks", slot_info.get("segments", []))
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

    def _normalize_token_segment_plan(
        self,
        *,
        seq_len: int,
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
        fallback_physical_dpu: int = 0,
    ) -> list[_TokenSegmentSpec]:
        if not segment_plan:
            return [
                _TokenSegmentSpec(
                    physical_dpu=int(fallback_physical_dpu),
                    token_start=0,
                    token_end=int(seq_len),
                )
            ]

        normalized_specs = [
            _TokenSegmentSpec(
                physical_dpu=int(item.get("physical_dpu", item.get("dpu_id", fallback_physical_dpu))),
                token_start=int(item.get("token_range_start", 0)),
                token_end=int(item.get("token_range_end", 0)),
            )
            for item in segment_plan
        ]
        normalized_specs = [spec for spec in normalized_specs if spec.token_count > 0]
        if not normalized_specs:
            raise ValueError("segment_plan produced no non-empty token segments")

        normalized_specs = sorted(normalized_specs, key=lambda spec: (int(spec.token_start), int(spec.token_end)))
        cursor = 0
        allowed = None if allowed_dpus is None else {int(physical_dpu) % max(self.num_dpus, 1) for physical_dpu in allowed_dpus}
        for spec in normalized_specs:
            if int(spec.token_start) != int(cursor):
                raise ValueError(
                    f"segment_plan must be contiguous and non-overlapping: expected_start={cursor} got={spec.token_start}"
                )
            if int(spec.token_end) <= int(spec.token_start):
                raise ValueError(f"invalid token segment range: {spec}")
            if allowed is not None and (int(spec.physical_dpu) % max(self.num_dpus, 1)) not in allowed:
                raise ValueError(
                    "segment_plan physical_dpu is outside allowed_dpus: "
                    f"physical_dpu={spec.physical_dpu} allowed_dpus={sorted(allowed)}"
                )
            cursor = int(spec.token_end)
        if int(cursor) != int(seq_len):
            raise ValueError(f"segment_plan coverage mismatch: expected_seq_len={seq_len} covered={cursor}")
        return normalized_specs

    def _allocate_block_append_only(
        self,
        *,
        key: tuple[str, str],
        slot_info: Dict[str, object],
        group_heads: int,
        head_dim: int,
        block_k: torch.Tensor,
        block_v: torch.Tensor,
        block_capacity_override: int | None = None,
        block_kind: str = "growth",
        physical_dpu_override: int | None = None,
        segment_meta: Dict[str, int] | None = None,
    ) -> Dict[str, object]:
        blocks = self._blocked_slot_blocks(slot_info)
        block_idx = len(blocks)
        base_physical_dpu = int(slot_info.get("base_physical_dpu", slot_info.get("physical_dpu", 0)))
        locality_anchor_dpu = int(blocks[-1]["physical_dpu"]) if blocks else base_physical_dpu
        allowed_dpus = slot_info.get("allowed_physical_dpus")
        if block_capacity_override is None:
            default_capacity = self.block_tokens if block_kind == "base" else self.growth_block_tokens
            block_capacity = max(int(block_k.shape[0]), int(default_capacity))
        else:
            block_capacity = max(int(block_k.shape[0]), int(block_capacity_override))
        block_key = self._block_slot_key(key, block_idx)
        block_elem_count = self._slot_elem_count(block_capacity, group_heads, head_dim)
        normalized_allowed = None
        if isinstance(allowed_dpus, list) and allowed_dpus:
            normalized_allowed = [
                int(physical_dpu) % max(self.num_dpus, 1) for physical_dpu in allowed_dpus
            ]
        block_physical_dpu = self._normalize_preferred_dpu(
            physical_dpu_override if physical_dpu_override is not None else locality_anchor_dpu
        )
        block_slot_id: int | None = None
        helper_allocated = False
        counters_applied = False
        try:
            if physical_dpu_override is not None:
                preferred_override = int(physical_dpu_override) % max(self.num_dpus, 1)
                block_physical_dpu = self._choose_physical_dpu_with_slot_capacity(
                    preferred_dpu=preferred_override,
                    elem_count=block_elem_count,
                    allowed_dpus=normalized_allowed,
                )
            elif self.placement_policy == "load_aware":
                block_preferred_dpu = locality_anchor_dpu
                block_physical_dpu = self.choose_physical_dpu(
                    elem_count=block_elem_count,
                    preferred_dpu=block_preferred_dpu,
                    allowed_dpus=normalized_allowed,
                )
            else:
                if normalized_allowed:
                    preferred_physical_dpu = normalized_allowed[block_idx % len(normalized_allowed)]
                else:
                    preferred_physical_dpu = (
                        (base_physical_dpu + block_idx) % max(self.num_dpus, 1) if self.num_dpus > 0 else 0
                    )
                block_physical_dpu = self._choose_physical_dpu_with_slot_capacity(
                    preferred_dpu=preferred_physical_dpu,
                    elem_count=block_elem_count,
                    allowed_dpus=normalized_allowed,
                )
            if not self._dpu_has_slot_capacity(block_physical_dpu):
                raise RuntimeError(
                    "No DPU KV slot capacity remains for block allocation: "
                    f"block_key={block_key} block_idx={block_idx} "
                    f"block_kind={block_kind} physical_dpu={block_physical_dpu}"
                )
            if self.dpu_live_elems_by_dpu[block_physical_dpu] + block_elem_count > self.POOL_CAPACITY_ELEMS:
                self.dpu_capacity_fallbacks += 1
                raise RuntimeError(
                    f"Blocked DPU pool capacity exceeded for {key} on physical_dpu={block_physical_dpu}"
                )
            block_slot_id = self._assign_slot_id(block_key, preferred_dpu=block_physical_dpu)
            if not self._supports_dpu_slot(block_k, block_capacity, block_slot_id):
                raise RuntimeError(
                    f"Blocked DPU slot shape unsupported for {key}: "
                    f"shape={tuple(block_k.shape)} capacity={block_capacity} slot_id={block_slot_id}"
                )

            encoded_k, encoded_v, k_scale, v_scale = self._encode_kv_pair(block_k, block_v)
            info = self.helper.allocate_group(
                block_slot_id,
                block_capacity,
                encoded_k,
                encoded_v,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            helper_allocated = True
            block = {
                "block_key": list(block_key),
                "slot_id": int(block_slot_id),
                "physical_dpu": int(block_physical_dpu),
                "block_index": int(block_idx),
                "block_kind": str(block_kind),
                "elem_count": int(block_elem_count),
                "seq_len": int(info["seq_len"]),
                "capacity": int(info["capacity"]),
                "group_heads": int(info["group_heads"]),
                "head_dim": int(info["head_dim"]),
                "k_scale": float(info.get("k_scale", k_scale)),
                "v_scale": float(info.get("v_scale", v_scale)),
            }
            if segment_meta:
                for key_name, value in dict(segment_meta).items():
                    block[str(key_name)] = int(value)
            self.dpu_allocations += 1
            self.dpu_live_slots += 1
            self.dpu_live_slot_counts_by_dpu[block_physical_dpu] += 1
            self.dpu_live_elems_by_dpu[block_physical_dpu] += block_elem_count
            self.helper.persistent_state_active = True
            counters_applied = True
            return block
        except Exception:
            self._record_dpu_allocate_failure(
                "block_allocate",
                sys.exc_info()[1] if sys.exc_info()[1] is not None else RuntimeError("block allocation failed"),
                block_key=block_key,
                block_idx=block_idx,
                block_kind=block_kind,
                slot_id=-1 if block_slot_id is None else int(block_slot_id),
                physical_dpu=block_physical_dpu,
                elem_count=block_elem_count,
                capacity=block_capacity,
            )
            if block_slot_id is not None:
                self._rollback_block_allocation(
                    block_key=block_key,
                    slot_id=block_slot_id,
                    physical_dpu=block_physical_dpu,
                    elem_count=block_elem_count,
                    helper_allocated=helper_allocated,
                    counters_applied=counters_applied,
                )
            exc = sys.exc_info()[1]
            raise RuntimeError(
                "block allocation failed "
                f"block_key={block_key} block_idx={block_idx} block_kind={block_kind} "
                f"slot_id={-1 if block_slot_id is None else int(block_slot_id)} "
                f"physical_dpu={block_physical_dpu} "
                f"elem_count={block_elem_count} capacity={block_capacity}: {exc}"
            ) from exc

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
        allowed_dpus: list[int] | None = None,
    ) -> Dict[str, object]:
        allocated_blocks: list[Dict[str, object]] = []
        seq_len = int(initial_k.shape[0])
        seq_offset = 0
        slot_info = {
            "backend": "dpu_blocked",
            "blocks": allocated_blocks,
            "segments": allocated_blocks,
            "seq_len": 0,
            "capacity": int(capacity),
            "group_heads": int(group_heads),
            "head_dim": int(head_dim),
            "block_tokens": int(self.block_tokens),
            "growth_block_tokens": int(self.growth_block_tokens),
            "base_physical_dpu": int(physical_dpu),
            "allowed_physical_dpus": [int(physical_dpu) % self.num_dpus for physical_dpu in allowed_dpus]
            if self.num_dpus > 0 and allowed_dpus
            else [],
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
                    block_capacity_override=self.block_tokens,
                    block_kind="base",
                    segment_meta={
                        "logical_segment_index": int(len(allocated_blocks)),
                        "token_range_start": int(seq_offset),
                        "token_range_end": int(seq_offset + block_seq_len),
                    },
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

    def _allocate_segmented_group(
        self,
        *,
        key: tuple[str, str],
        initial_k: torch.Tensor,
        initial_v: torch.Tensor,
        capacity: int,
        physical_dpu: int,
        group_heads: int,
        head_dim: int,
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
    ) -> Dict[str, object]:
        allocated_blocks: list[Dict[str, object]] = []
        seq_len = int(initial_k.shape[0])
        normalized_segments = self._normalize_token_segment_plan(
            seq_len=seq_len,
            allowed_dpus=allowed_dpus,
            segment_plan=segment_plan,
            fallback_physical_dpu=physical_dpu,
        )
        unique_segment_dpus = []
        seen_dpus = set()
        for segment in normalized_segments:
            normalized_dpu = int(segment.physical_dpu) % max(self.num_dpus, 1)
            if normalized_dpu in seen_dpus:
                continue
            seen_dpus.add(normalized_dpu)
            unique_segment_dpus.append(normalized_dpu)
        allowed_physical_dpus = list(unique_segment_dpus)
        if allowed_dpus and self.num_dpus > 0:
            allowed_physical_dpus = []
            seen_allowed_dpus = set()
            for physical_dpu in allowed_dpus:
                normalized_dpu = int(physical_dpu) % self.num_dpus
                if normalized_dpu in seen_allowed_dpus:
                    continue
                seen_allowed_dpus.add(normalized_dpu)
                allowed_physical_dpus.append(normalized_dpu)

        slot_info = {
            "backend": "dpu_segmented",
            "blocks": allocated_blocks,
            "segments": allocated_blocks,
            "seq_len": 0,
            "capacity": int(capacity),
            "group_heads": int(group_heads),
            "head_dim": int(head_dim),
            "block_tokens": int(self.block_tokens),
            "growth_block_tokens": int(self.growth_block_tokens),
            "base_physical_dpu": int(physical_dpu),
            "allowed_physical_dpus": list(allowed_physical_dpus),
            "planner_segment_count": int(len(normalized_segments)),
        }
        try:
            block_specs: list[tuple[int, int, int, int]] = []
            for logical_segment_index, segment in enumerate(normalized_segments):
                segment_start = int(segment.token_start)
                segment_end = int(segment.token_end)
                segment_physical_dpu = int(segment.physical_dpu) % max(self.num_dpus, 1)
                block_cursor = int(segment_start)
                while block_cursor < segment_end:
                    block_end = min(segment_end, block_cursor + int(self.block_tokens))
                    block_specs.append(
                        (
                            int(logical_segment_index),
                            int(segment_physical_dpu),
                            int(block_cursor),
                            int(block_end),
                        )
                    )
                    block_cursor = int(block_end)

            block_live_lengths = [int(block_end - block_start) for _, _, block_start, block_end in block_specs]
            if bool(getattr(self, "reserve_segment_tail_capacity_enabled", False)):
                max_reserve_tokens = int(getattr(self, "reserve_segment_tail_capacity_tokens", 0) or 0)
                block_capacities = self._reserve_tail_block_capacities(
                    block_live_lengths,
                    int(capacity),
                    max_reserve_tokens=max_reserve_tokens if max_reserve_tokens > 0 else None,
                )
            else:
                block_capacities = list(block_live_lengths)
            for (logical_segment_index, segment_physical_dpu, block_cursor, block_end), block_capacity in zip(
                block_specs,
                block_capacities,
            ):
                block_initial_k = initial_k[block_cursor:block_end].contiguous()
                block_initial_v = initial_v[block_cursor:block_end].contiguous()
                block = self._allocate_block_append_only(
                    key=key,
                    slot_info=slot_info,
                    group_heads=group_heads,
                    head_dim=head_dim,
                    block_k=block_initial_k,
                    block_v=block_initial_v,
                    block_capacity_override=int(block_capacity),
                    block_kind="base",
                    physical_dpu_override=segment_physical_dpu,
                    segment_meta={
                        "logical_segment_index": int(logical_segment_index),
                        "token_range_start": int(block_cursor),
                        "token_range_end": int(block_end),
                    },
                )
                allocated_blocks.append(block)
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
        allowed_dpus: list[int] | None = None,
        segment_plan: list[Dict[str, int]] | None = None,
    ) -> Dict[str, object]:
        started_at = time.perf_counter()
        key = self._slot_key(k_slot, v_slot)
        seq_len, group_heads, head_dim = (int(dim) for dim in initial_k.shape)
        elem_count = self._slot_elem_count(capacity, group_heads, head_dim)
        physical_dpu = (
            self.choose_physical_dpu(
                elem_count=elem_count,
                preferred_dpu=preferred_dpu,
                allowed_dpus=allowed_dpus,
            )
            if self.num_dpus > 0
            else 0
        )
        if not force_host_fallback and self._supports_dpu_shape(group_heads, head_dim):
            try:
                if segment_plan:
                    blocked_slot_info = self._allocate_segmented_group(
                        key=key,
                        initial_k=initial_k,
                        initial_v=initial_v,
                        capacity=capacity,
                        physical_dpu=physical_dpu,
                        group_heads=group_heads,
                        head_dim=head_dim,
                        allowed_dpus=allowed_dpus,
                        segment_plan=segment_plan,
                    )
                else:
                    blocked_slot_info = self._allocate_blocked_group(
                        key=key,
                        initial_k=initial_k,
                        initial_v=initial_v,
                        capacity=capacity,
                        physical_dpu=physical_dpu,
                        group_heads=group_heads,
                        head_dim=head_dim,
                        allowed_dpus=allowed_dpus,
                    )
            except Exception:
                self.dpu_allocate_failures += 1
                self._record_dpu_allocate_failure(
                    "segmented_allocate" if segment_plan else "blocked_allocate",
                    sys.exc_info()[1]
                    if sys.exc_info()[1] is not None
                    else RuntimeError("blocked allocation failed"),
                    slot_key=key,
                    physical_dpu=physical_dpu,
                    elem_count=elem_count,
                    capacity=capacity,
                    seq_len=seq_len,
                    group_heads=group_heads,
                    head_dim=head_dim,
                )
                blocked_slot_info = None
            if blocked_slot_info is not None:
                self.slot_mapping[key] = blocked_slot_info
                self._record_timing("allocate_group", started_at)
                return {
                    "backend": str(blocked_slot_info.get("backend", "dpu_blocked")),
                    "physical_dpu": int(blocked_slot_info.get("base_physical_dpu", physical_dpu)),
                    "storage": str(blocked_slot_info.get("backend", "dpu_blocked")),
                    "segments": [
                        {
                            "physical_dpu": int(block.get("physical_dpu", physical_dpu)),
                            "token_start": int(block.get("token_range_start", 0)),
                            "token_end": int(block.get("token_range_end", block.get("seq_len", 0))),
                            "seq_len": int(block.get("seq_len", 0)),
                            "capacity": int(block.get("capacity", 0)),
                            "logical_segment_index": int(block.get("logical_segment_index", 0)),
                        }
                        for block in blocked_slot_info.get("blocks", blocked_slot_info.get("segments", []))
                    ],
                }
            slot_id: int | None = None
            try:
                slot_id = self._assign_slot_id(key, preferred_dpu=physical_dpu)
            except Exception:
                self.dpu_allocate_failures += 1
                self._record_dpu_allocate_failure(
                    "slot_reservation",
                    sys.exc_info()[1]
                    if sys.exc_info()[1] is not None
                    else RuntimeError("slot reservation failed"),
                    slot_key=key,
                    physical_dpu=physical_dpu,
                    elem_count=elem_count,
                    capacity=capacity,
                    seq_len=seq_len,
                    group_heads=group_heads,
                    head_dim=head_dim,
                )
        else:
            slot_id = None
            if not force_host_fallback:
                try:
                    slot_id = self._assign_slot_id(key, preferred_dpu=physical_dpu)
                except Exception:
                    self.dpu_allocate_failures += 1
                    self._record_dpu_allocate_failure(
                        "slot_reservation",
                        sys.exc_info()[1]
                        if sys.exc_info()[1] is not None
                        else RuntimeError("slot reservation failed"),
                        slot_key=key,
                        physical_dpu=physical_dpu,
                        elem_count=elem_count,
                        capacity=capacity,
                        seq_len=seq_len,
                        group_heads=group_heads,
                        head_dim=head_dim,
                    )

        if slot_id is not None and not force_host_fallback and self._supports_dpu_slot(initial_k, capacity, slot_id):
            if self.dpu_live_elems_by_dpu[physical_dpu] + elem_count > self.POOL_CAPACITY_ELEMS:
                self.dpu_capacity_fallbacks += 1
            else:
                try:
                    encoded_k, encoded_v, k_scale, v_scale = self._encode_kv_pair(initial_k, initial_v)
                    info = self.helper.allocate_group(
                        slot_id,
                        capacity,
                        encoded_k,
                        encoded_v,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                except Exception:
                    self.dpu_allocate_failures += 1
                    self._record_dpu_allocate_failure(
                        "legacy_allocate",
                        sys.exc_info()[1]
                        if sys.exc_info()[1] is not None
                        else RuntimeError("legacy allocation failed"),
                        slot_key=key,
                        slot_id=slot_id,
                        physical_dpu=physical_dpu,
                        elem_count=elem_count,
                        capacity=capacity,
                    )
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
                        "k_scale": float(info.get("k_scale", k_scale)),
                        "v_scale": float(info.get("v_scale", v_scale)),
                    }
                    self.dpu_allocations += 1
                    self.dpu_live_slots += 1
                    self.dpu_live_slot_counts_by_dpu[physical_dpu] += 1
                    self.dpu_live_elems_by_dpu[physical_dpu] += elem_count
                    self.helper.persistent_state_active = True
                    self._record_timing("allocate_group", started_at)
                    return {
                        "backend": "dpu",
                        "physical_dpu": int(physical_dpu),
                        "storage": "dpu",
                    }

        self.host_fallback.allocate_group(
            k_slot,
            v_slot,
            initial_k,
            initial_v,
            capacity,
            preferred_dpu=preferred_dpu,
            allowed_dpus=allowed_dpus,
            segment_plan=segment_plan,
        )
        self.slot_mapping[key] = {
            "backend": "host_fallback",
            "slot_id": slot_id,
            "physical_dpu": physical_dpu,
            "elem_count": elem_count,
        }
        self.fallback_allocations += 1
        self._record_timing("allocate_group", started_at)
        return {
            "backend": "host_fallback",
            "physical_dpu": int(physical_dpu),
            "storage": "host_fallback",
        }

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
        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            append_total = int(k_new.shape[0])
            blocks = slot_info.get("blocks", slot_info.get("segments", []))
            group_heads = int(slot_info["group_heads"])
            head_dim = int(slot_info["head_dim"])
            append_base_seq_len = int(slot_info["seq_len"])
            if append_total <= 0:
                out = {
                    "seq_len": int(slot_info["seq_len"]),
                    "capacity": int(slot_info["capacity"]),
                }
                self._record_timing("append_group", started_at)
                return out

            tail_block = blocks[-1] if blocks else None
            tail_available = 0 if tail_block is None else max(0, int(tail_block["capacity"]) - int(tail_block["seq_len"]))
            if self._should_rollover_tail_block(slot_info, tail_block):
                tail_available = 0
            tail_take_len = min(tail_available, append_total)
            new_blocks: list[Dict[str, object]] = []
            append_offset = tail_take_len
            try:
                staged_blocks = list(blocks)
                while append_offset < append_total:
                    take_len = min(self.growth_block_tokens, append_total - append_offset)
                    block_k = k_new[append_offset : append_offset + take_len].contiguous()
                    block_v = v_new[append_offset : append_offset + take_len].contiguous()
                    staged_slot_info = dict(slot_info)
                    staged_slot_info["blocks"] = staged_blocks
                    staged_slot_info["segments"] = staged_blocks
                    new_block = self._allocate_block_append_only(
                        key=key,
                        slot_info=staged_slot_info,
                        group_heads=group_heads,
                        head_dim=head_dim,
                        block_k=block_k,
                        block_v=block_v,
                        block_capacity_override=self.growth_block_tokens,
                        block_kind="growth",
                        segment_meta={
                            "logical_segment_index": int(len(staged_blocks)),
                            "token_range_start": int(append_base_seq_len + append_offset),
                            "token_range_end": int(append_base_seq_len + append_offset + take_len),
                        },
                    )
                    staged_blocks.append(new_block)
                    new_blocks.append(new_block)
                    append_offset += take_len

                if tail_take_len > 0 and tail_block is not None:
                    tail_k_scale = (
                        float(tail_block.get("k_scale", 1.0))
                        if resident_kv_dtype_code(self.kv_dtype, "k") == KVSLOT_DTYPE_INT8
                        else None
                    )
                    tail_v_scale = (
                        float(tail_block.get("v_scale", 1.0))
                        if resident_kv_dtype_code(self.kv_dtype, "v") == KVSLOT_DTYPE_INT8
                        else None
                    )
                    encoded_k, encoded_v, k_scale, v_scale = self._encode_kv_pair(
                        k_new[:tail_take_len].contiguous(),
                        v_new[:tail_take_len].contiguous(),
                        k_scale=tail_k_scale,
                        v_scale=tail_v_scale,
                    )
                    result = self.helper.append_group(
                        int(tail_block["slot_id"]),
                        encoded_k,
                        encoded_v,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                    tail_block["seq_len"] = int(result["seq_len"])
                    tail_block["capacity"] = int(result["capacity"])
                    tail_block["k_scale"] = float(result.get("k_scale", tail_block.get("k_scale", 1.0)))
                    tail_block["v_scale"] = float(result.get("v_scale", tail_block.get("v_scale", 1.0)))
                    tail_start = int(tail_block.get("token_range_start", append_base_seq_len - int(tail_block["seq_len"]) + tail_take_len))
                    tail_block["token_range_start"] = int(tail_start)
                    tail_block["token_range_end"] = int(tail_start + int(tail_block["seq_len"]))
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
            slot_k_scale = (
                float(slot_info.get("k_scale", 1.0))
                if resident_kv_dtype_code(self.kv_dtype, "k") == KVSLOT_DTYPE_INT8
                else None
            )
            slot_v_scale = (
                float(slot_info.get("v_scale", 1.0))
                if resident_kv_dtype_code(self.kv_dtype, "v") == KVSLOT_DTYPE_INT8
                else None
            )
            encoded_k, encoded_v, k_scale, v_scale = self._encode_kv_pair(
                k_new,
                v_new,
                k_scale=slot_k_scale,
                v_scale=slot_v_scale,
            )
            result = self.helper.append_group(
                int(slot_info["slot_id"]),
                encoded_k,
                encoded_v,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            slot_info["seq_len"] = int(result["seq_len"])
            slot_info["capacity"] = int(result["capacity"])
            slot_info["k_scale"] = float(result.get("k_scale", slot_info.get("k_scale", 1.0)))
            slot_info["v_scale"] = float(result.get("v_scale", slot_info.get("v_scale", 1.0)))
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
        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            materialized_blocks = []
            for block in slot_info.get("blocks", slot_info.get("segments", [])):
                if int(block["seq_len"]) <= 0:
                    continue
                k, v, info = self.helper.materialize_group(int(block["slot_id"]))
                block["seq_len"] = int(info["seq_len"])
                block["capacity"] = int(info["capacity"])
                materialized_blocks.append(
                    (
                        self._decode_tensor_for_dtype(
                            k,
                            int(info.get("dtype_code", resident_kv_dtype_code(self.kv_dtype, "k"))),
                            scale=float(info.get("k_scale", 1.0)),
                        ),
                        self._decode_tensor_for_dtype(
                            v,
                            int(info.get("v_dtype_code", resident_kv_dtype_code(self.kv_dtype, "v"))),
                            scale=float(info.get("v_scale", 1.0)),
                        ),
                    )
                )
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
            out = (
                self._decode_tensor_for_dtype(
                    k,
                    int(info.get("dtype_code", resident_kv_dtype_code(self.kv_dtype, "k"))),
                    scale=float(info.get("k_scale", 1.0)),
                ),
                self._decode_tensor_for_dtype(
                    v,
                    int(info.get("v_dtype_code", resident_kv_dtype_code(self.kv_dtype, "v"))),
                    scale=float(info.get("v_scale", 1.0)),
                ),
            )
            self._record_timing("materialize_group", started_at)
            return out
        out = self.host_fallback.materialize_group(k_slot, v_slot)
        self._record_timing("materialize_group", started_at)
        return out

    def migrate_group_to_host_fallback(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        """Move one DPU-backed group into host fallback for append recovery."""
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "host_fallback":
            return {
                "backend": "host_fallback",
                "storage": "host_fallback",
                "migrated": False,
            }

        keys, values = self.materialize_group(k_slot, v_slot)
        capacity = max(int(slot_info.get("capacity", int(keys.shape[0]))), int(keys.shape[0]), 1)
        preferred_dpu = int(slot_info.get("physical_dpu", slot_info.get("base_physical_dpu", 0)))
        allowed_dpus = list(slot_info.get("allowed_physical_dpus", []) or [])

        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            self._free_block_infos(slot_info.get("blocks", slot_info.get("segments", [])))
        elif slot_info["backend"] == "dpu":
            slot_id = self._slot_id_map.pop(key, None)
            try:
                self.helper.free_group(int(slot_info["slot_id"]))
            except Exception:
                pass
            self.dpu_free_ops += 1
            self.dpu_live_slots = max(0, self.dpu_live_slots - 1)
            physical_dpu = int(slot_info.get("physical_dpu", 0)) % max(self.num_dpus, 1)
            elem_count = int(slot_info.get("elem_count", 0))
            self.dpu_live_slot_counts_by_dpu[physical_dpu] = max(
                0,
                self.dpu_live_slot_counts_by_dpu[physical_dpu] - 1,
            )
            self.dpu_live_elems_by_dpu[physical_dpu] = max(
                0,
                self.dpu_live_elems_by_dpu[physical_dpu] - elem_count,
            )
            if slot_id is not None:
                self._remember_free_slot_id(physical_dpu, int(slot_id))
            self.helper.persistent_state_active = self.dpu_live_slots > 0
            if self.dpu_live_slots == 0:
                self.helper.close()

        self.host_fallback.allocate_group(
            k_slot,
            v_slot,
            keys,
            values,
            capacity=capacity,
            preferred_dpu=preferred_dpu,
            allowed_dpus=allowed_dpus or None,
        )
        self.slot_mapping[key] = {
            "backend": "host_fallback",
            "slot_id": None,
            "physical_dpu": preferred_dpu,
            "elem_count": 0,
            "migrated_from": str(slot_info.get("backend", "")),
        }
        self.fallback_allocations += 1
        self.host_fallback_migrations += 1
        return {
            "backend": "host_fallback",
            "storage": "host_fallback",
            "migrated": True,
            "seq_len": int(keys.shape[0]),
            "capacity": int(capacity),
        }

    def slot_debug(self, k_slot: str, v_slot: str) -> Dict[str, object]:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            blocks = slot_info.get("blocks", slot_info.get("segments", []))
            return {
                "backend": self.backend_name,
                "storage": "dpu_blocked",
                "seq_len": int(slot_info["seq_len"]),
                "capacity": int(slot_info["capacity"]),
                "group_heads": int(slot_info["group_heads"]),
                "head_dim": int(slot_info["head_dim"]),
                "block_tokens": int(slot_info.get("block_tokens", self.block_tokens)),
                "growth_block_tokens": int(slot_info.get("growth_block_tokens", self.growth_block_tokens)),
                "allowed_physical_dpus": [
                    int(physical_dpu)
                    for physical_dpu in slot_info.get("allowed_physical_dpus", [])
                ],
                "base_physical_dpu": int(slot_info.get("base_physical_dpu", slot_info.get("physical_dpu", 0))),
                "block_count": len(blocks),
                "blocks": [
                    {
                        "slot_id": int(block["slot_id"]),
                        "physical_dpu": int(block["physical_dpu"]),
                        "block_index": int(block.get("block_index", 0)),
                        "logical_segment_index": int(block.get("logical_segment_index", 0)),
                        "block_kind": str(block.get("block_kind", "growth")),
                        "token_range_start": int(block.get("token_range_start", 0)),
                        "token_range_end": int(block.get("token_range_end", block.get("seq_len", 0))),
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
                        "logical_segment_index": int(block.get("logical_segment_index", 0)),
                        "block_kind": str(block.get("block_kind", "growth")),
                        "token_range_start": int(block.get("token_range_start", 0)),
                        "token_range_end": int(block.get("token_range_end", block.get("seq_len", 0))),
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
        if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
            self._free_block_infos(slot_info.get("blocks", slot_info.get("segments", [])))
            self._record_timing("free_group", started_at)
            return
        slot_id = self._slot_id_map.pop(key, None)
        if slot_info["backend"] == "dpu":
            self.helper.free_group(int(slot_info["slot_id"]))
            self.dpu_free_ops += 1
            self.dpu_live_slots = max(0, self.dpu_live_slots - 1)
            physical_dpu = int(slot_info.get("physical_dpu", 0))
            elem_count = int(slot_info.get("elem_count", 0))
            self.dpu_live_slot_counts_by_dpu[physical_dpu] = max(
                0, self.dpu_live_slot_counts_by_dpu[physical_dpu] - 1
            )
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
            self._remember_free_slot_id(physical_dpu, int(slot_id))
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
        pool_capacity_elems = int(self.POOL_CAPACITY_ELEMS)
        for stats in allocator_stats:
            tail_free = max(pool_capacity_elems - int(stats["next_free_elem"]), 0)
            total_free = int(stats["free_elems_total"]) + tail_free
            stats["tail_free_elems"] = tail_free
            stats["total_free_elems"] = total_free
            stats["pool_capacity_elems"] = pool_capacity_elems
            stats["used_elems_estimate"] = max(pool_capacity_elems - total_free, 0)
            stats["usage_ratio"] = float(stats["used_elems_estimate"]) / float(pool_capacity_elems) if pool_capacity_elems else 0.0
        allocator_summary = self._summarize_allocator_stats(allocator_stats)
        dpu_balance_summary = self._summarize_balance()
        rank_balance_summary = self._summarize_rank_balance()
        block_summary = self._summarize_blocks()
        return {
            "backend": self.backend_name,
            "num_dpus": self.num_dpus,
            "kv_dtype": self.kv_dtype,
            "placement_policy": self.placement_policy,
            "host_partial_reduce_enabled": bool(self.host_partial_reduce_enabled),
            "helper_env": dict(self.helper.helper_env),
            "helper_profile": helper_profile,
            "helper_topology": helper_topology,
            "live_slots": len(self.slot_mapping),
            "dpu_allocations": self.dpu_allocations,
            "dpu_free_ops": self.dpu_free_ops,
            "dpu_allocate_failures": self.dpu_allocate_failures,
            "dpu_allocate_failure_reasons": dict(getattr(self, "dpu_allocate_failure_reasons", {})),
            "dpu_allocate_last_failure": dict(getattr(self, "dpu_allocate_last_failure", {})),
            "dpu_live_slots": self.dpu_live_slots,
            "dpu_capacity_fallbacks": self.dpu_capacity_fallbacks,
            "slot_spill_alloc_enabled": self.slot_spill_alloc_enabled,
            "slot_spill_allocations": self.slot_spill_allocations,
            "emergency_slot_spill_enabled": bool(getattr(self, "emergency_slot_spill_enabled", False)),
            "emergency_slot_spill_allocations": int(getattr(self, "emergency_slot_spill_allocations", 0)),
            "host_fallback_migrations": int(getattr(self, "host_fallback_migrations", 0)),
            "slot_capacity_reroutes": int(getattr(self, "slot_capacity_reroutes", 0)),
            "reserve_segment_tail_capacity_enabled": bool(
                getattr(self, "reserve_segment_tail_capacity_enabled", False)
            ),
            "reserve_segment_tail_capacity_tokens": int(
                getattr(self, "reserve_segment_tail_capacity_tokens", 0)
            ),
            "slot_pressure_aware_alloc_enabled": bool(self.slot_pressure_aware_alloc_enabled),
            "slot_pressure_soft_limit": int(self.slot_pressure_soft_limit),
            "dpu_live_slot_counts_by_dpu": list(self.dpu_live_slot_counts_by_dpu),
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
            "allocator_summary": allocator_summary,
            "dpu_balance_summary": dpu_balance_summary,
            "rank_balance_summary": rank_balance_summary,
            "block_summary": block_summary,
            "host_fallback": self.host_fallback.get_debug_info(),
            "op_timing_totals_s": dict(self.op_timing_totals_s),
            "op_timing_counts": dict(self.op_timing_counts),
            "batch_item_totals": dict(self.batch_item_totals),
        }

    def qk_scores_batch(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return self.helper.qk_scores_batch(queries, keys)

    def _regular_dpu_host_weighted_value_sum(
        self,
        slot_info: Dict[str, object],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _k, values, info = self.helper.materialize_group(int(slot_info["slot_id"]))
        slot_info["seq_len"] = int(info["seq_len"])
        slot_info["capacity"] = int(info["capacity"])
        slot_info["k_scale"] = float(info.get("k_scale", slot_info.get("k_scale", 1.0)))
        slot_info["v_scale"] = float(info.get("v_scale", slot_info.get("v_scale", 1.0)))
        decoded_values = self._decode_tensor_for_dtype(
            values,
            int(info.get("v_dtype_code", resident_kv_dtype_code(self.kv_dtype, "v"))),
            scale=float(info.get("v_scale", 1.0)),
        )
        w = weights.detach().cpu().to(torch.float32).contiguous()
        if w.dim() != 2 or int(w.shape[0]) != int(decoded_values.shape[1]) or int(w.shape[1]) > int(decoded_values.shape[0]):
            raise ValueError(
                f"regular dpu host AV weight shape mismatch: weights={tuple(w.shape)} "
                f"values={tuple(decoded_values.shape)}"
            )
        weight_len = int(w.shape[1])
        tail_values = decoded_values[int(decoded_values.shape[0]) - weight_len : int(decoded_values.shape[0])].float()
        return torch.einsum("hl,lhd->hd", w, tail_values).contiguous()

    def _materialized_local_weighted_value_sum(
        self,
        k_slot: str,
        v_slot: str,
        local_head_indices: list[int],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _keys, values = self.materialize_group(k_slot, v_slot)
        w = weights.detach().cpu().to(torch.float32).contiguous()
        if w.dim() != 2 or int(w.shape[0]) != len(local_head_indices) or int(w.shape[1]) > int(values.shape[0]):
            raise ValueError(
                "local materialized AV weight shape mismatch: "
                f"weights={tuple(w.shape)} local_heads={len(local_head_indices)} values={tuple(values.shape)}"
            )
        weight_len = int(w.shape[1])
        tail_values = values[int(values.shape[0]) - weight_len : int(values.shape[0])].float()
        contexts = []
        for row_idx, local_head_idx in enumerate(local_head_indices):
            if int(local_head_idx) < 0 or int(local_head_idx) >= int(values.shape[1]):
                raise ValueError(
                    f"local_head_idx out of range for materialized AV: got={local_head_idx} "
                    f"group_heads={int(values.shape[1])}"
                )
            contexts.append(
                torch.einsum("l,ld->d", w[row_idx], tail_values[:, int(local_head_idx), :]).contiguous()
            )
        if not contexts:
            return torch.empty((0, int(values.shape[2])), dtype=torch.float32)
        return torch.stack(contexts, dim=0).contiguous()

    def qk_slot_scores_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        total_started_at = time.perf_counter()

        outputs: list[torch.Tensor | None] = [None for _ in slot_queries]
        dpu_entries: list[Dict[str, object]] = []
        host_fallback_queries: list[tuple[int, tuple[str, str, list[int], int, torch.Tensor]]] = []
        segmented_outputs: Dict[int, list[tuple[int, torch.Tensor]]] = {}

        for idx, (k_slot, v_slot, local_head_indices, window, queries) in enumerate(slot_queries):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
                self.batch_item_totals["qk_slot_scores_batch_blocked_logical_items"] += 1
                self.batch_item_totals["qk_slot_scores_batch_segmented_logical_items"] += 1
                actual_window = min(int(window), int(slot_info["seq_len"]))
                if actual_window <= 0:
                    outputs[idx] = torch.empty((len(local_head_indices), 0), dtype=torch.float32)
                    continue
                for segment_ordinal, (block, take_len) in enumerate(self._active_block_plan(slot_info, actual_window)):
                    # Keep segmented raw-score QK requests ungrouped even when
                    # multiple active blocks land on the same physical DPU.
                    #
                    # The UPMEM grouped-qk helper path is currently unstable
                    # once decode growth creates a base+growth block mix on the
                    # same DPU. Launching each segment independently preserves
                    # correctness, and we still concatenate the per-segment
                    # score tiles in host order below.
                    physical_dpu = int(block["physical_dpu"])
                    normalized_local_head_indices = [int(v) for v in local_head_indices]
                    dpu_entries.append(
                        {
                            "payload": (
                                int(block["slot_id"]),
                                normalized_local_head_indices,
                                int(take_len),
                                queries,
                            ),
                            "physical_dpu": int(physical_dpu),
                            "shape_key": (
                                len(normalized_local_head_indices),
                                int(take_len),
                                int(queries.shape[1]),
                            ),
                            "slot_id": int(block["slot_id"]),
                            "logical_idx": int(idx),
                            "segment_ordinal": int(segment_ordinal),
                            "ref_kind": "segmented",
                        }
                    )
            elif slot_info["backend"] == "dpu":
                actual_window = min(int(window), int(slot_info["seq_len"]))
                dpu_entries.append(
                    {
                        "payload": (
                            int(slot_info["slot_id"]),
                            [int(v) for v in local_head_indices],
                            actual_window,
                            queries,
                        ),
                        "physical_dpu": int(slot_info["physical_dpu"]),
                        "shape_key": (
                            len(local_head_indices),
                            int(actual_window),
                            int(queries.shape[1]),
                        ),
                        "slot_id": int(slot_info["slot_id"]),
                        "logical_idx": int(idx),
                        "segment_ordinal": 0,
                        "ref_kind": "regular",
                    }
                )
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

        if dpu_entries:
            ordered_entries = sorted(
                dpu_entries,
                key=lambda item: self._helper_submit_sort_key(
                    physical_dpu=int(item["physical_dpu"]),
                    shape_key=tuple(item["shape_key"]),
                    slot_id=int(item["slot_id"]),
                    logical_idx=int(item["logical_idx"]),
                    segment_ordinal=int(item["segment_ordinal"]),
                ),
            )
            dpu_started_at = time.perf_counter()
            dpu_outputs = self.helper.qk_slot_scores_batch([item["payload"] for item in ordered_entries])
            self._record_timing("qk_slot_scores_batch_dpu", dpu_started_at)
            self.batch_item_totals["qk_slot_scores_batch_dpu_items"] += len(ordered_entries)
            for entry, scores in zip(ordered_entries, dpu_outputs):
                ref_kind = str(entry["ref_kind"])
                logical_idx = int(entry["logical_idx"])
                if ref_kind == "regular":
                    outputs[logical_idx] = scores
                else:
                    segmented_outputs.setdefault(logical_idx, []).append((int(entry["segment_ordinal"]), scores))

        for logical_idx, score_parts in segmented_outputs.items():
            ordered_parts = [scores for _, scores in sorted(score_parts, key=lambda item: item[0])]
            outputs[logical_idx] = torch.cat(ordered_parts, dim=1).contiguous()

        self._record_timing("qk_slot_scores_batch_total", total_started_at)
        self.batch_item_totals["qk_slot_scores_batch_total"] += len(slot_queries)
        return self._complete_batch_outputs("qk_slot_scores_batch", outputs)

    def weighted_value_sum(self, k_slot: str, v_slot: str, weights: torch.Tensor) -> torch.Tensor:
        key = self._slot_key(k_slot, v_slot)
        slot_info = self.slot_mapping[key]
        if slot_info["backend"] == "dpu":
            if int(weights.shape[1]) > int(slot_info["seq_len"]):
                raise ValueError(
                    f"regular dpu AV weight window exceeds slot seq_len: weights={tuple(weights.shape)} "
                    f"slot_seq_len={slot_info['seq_len']}"
                )
            if int(weights.shape[1]) < int(slot_info["seq_len"]):
                return self._regular_dpu_host_weighted_value_sum(slot_info, weights)
            return self.helper.weighted_value_sum(int(slot_info["slot_id"]), weights)
        return self.host_fallback.weighted_value_sum(k_slot, v_slot, weights)

    def weighted_value_sum_batch(self, slot_weights: list[tuple[str, str, torch.Tensor]]) -> list[torch.Tensor]:
        if not slot_weights:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_weights]
        dpu_entries: list[Dict[str, object]] = []
        grouped_dpu_entries: list[Dict[str, object]] = []
        host_fallback_weights: list[tuple[int, tuple[str, str, torch.Tensor]]] = []
        segmented_contexts: Dict[int, torch.Tensor] = {}

        for idx, (k_slot, v_slot, weights) in enumerate(slot_weights):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
                self.batch_item_totals["weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["weighted_value_sum_batch_segmented_logical_items"] += 1
                weight_offset = 0
                per_dpu_grouped: Dict[int, list[tuple[int, int, torch.Tensor, int]]] = {}
                active_plan = self._active_block_plan(slot_info, int(weights.shape[1]))
                active_total = sum(int(block_len) for _, block_len in active_plan)
                if int(active_total) != int(weights.shape[1]):
                    raise ValueError(
                        "segmented AV weight window mismatch: "
                        f"weights={tuple(weights.shape)} active_total={active_total} slot_seq_len={slot_info['seq_len']}"
                    )
                for segment_ordinal, (block, block_len) in enumerate(active_plan):
                    block_weights = weights[:, weight_offset : weight_offset + int(block_len)].contiguous()
                    weight_offset += int(block_len)
                    physical_dpu = int(block["physical_dpu"])
                    per_dpu_grouped.setdefault(physical_dpu, []).append(
                        (
                            int(block["slot_id"]),
                            int(block_len),
                            block_weights,
                            int(segment_ordinal),
                            int(block["seq_len"]),
                        )
                    )
                for physical_dpu, group_items in per_dpu_grouped.items():
                    needs_grouped_av = len(group_items) > 1 or any(
                        int(segment_len) != int(block_seq_len)
                        for _, segment_len, _, _, block_seq_len in group_items
                    )
                    if needs_grouped_av:
                        grouped_dpu_entries.append(
                            {
                                "payload": [
                                    (slot_id, segment_len, block_weights)
                                    for slot_id, segment_len, block_weights, _, _ in group_items
                                ],
                                "physical_dpu": int(physical_dpu),
                                "shape_key": (
                                    int(group_items[0][2].shape[0]),
                                    sum(int(segment_len) for _, segment_len, _, _, _ in group_items),
                                    int(slot_info["head_dim"]),
                                ),
                                "slot_id": int(group_items[0][0]),
                                "logical_idx": int(idx),
                                "segment_ordinal": min(
                                    int(segment_ordinal) for _, _, _, segment_ordinal, _ in group_items
                                ),
                                "ref_kind": "segmented_grouped",
                            }
                        )
                    else:
                        slot_id, segment_len, block_weights, segment_ordinal, _block_seq_len = group_items[0]
                        dpu_entries.append(
                            {
                                "payload": (int(slot_id), block_weights),
                                "physical_dpu": int(physical_dpu),
                                "shape_key": (
                                    int(block_weights.shape[0]),
                                    int(block_weights.shape[1]),
                                    int(slot_info["head_dim"]),
                                ),
                                "slot_id": int(slot_id),
                                "logical_idx": int(idx),
                                "segment_ordinal": int(segment_ordinal),
                                "ref_kind": "segmented",
                            }
                        )
            elif slot_info["backend"] == "dpu":
                if int(weights.shape[1]) > int(slot_info["seq_len"]):
                    raise ValueError(
                        f"regular dpu AV weight window exceeds slot seq_len: weights={tuple(weights.shape)} "
                        f"slot_seq_len={slot_info['seq_len']}"
                    )
                if int(weights.shape[1]) < int(slot_info["seq_len"]):
                    host_started_at = time.perf_counter()
                    contexts[idx] = self._regular_dpu_host_weighted_value_sum(slot_info, weights)
                    self._record_timing("weighted_value_sum_batch_host_fallback", host_started_at)
                    self.batch_item_totals["weighted_value_sum_batch_host_fallback_items"] += 1
                else:
                    dpu_entries.append(
                        {
                            "payload": (int(slot_info["slot_id"]), weights),
                            "physical_dpu": int(slot_info["physical_dpu"]),
                            "shape_key": (
                                int(weights.shape[0]),
                                int(weights.shape[1]),
                                int(slot_info["head_dim"]),
                            ),
                            "slot_id": int(slot_info["slot_id"]),
                            "logical_idx": int(idx),
                            "segment_ordinal": 0,
                            "ref_kind": "regular",
                        }
                    )
            else:
                host_fallback_weights.append((idx, (k_slot, v_slot, weights)))

        if host_fallback_weights:
            host_started_at = time.perf_counter()
            host_contexts = self.host_fallback.weighted_value_sum_batch([item for _, item in host_fallback_weights])
            self._record_timing("weighted_value_sum_batch_host_fallback", host_started_at)
            self.batch_item_totals["weighted_value_sum_batch_host_fallback_items"] += len(host_fallback_weights)
            for (idx, _), context in zip(host_fallback_weights, host_contexts):
                contexts[idx] = context

        if grouped_dpu_entries:
            ordered_grouped_entries = sorted(
                grouped_dpu_entries,
                key=lambda item: self._helper_submit_sort_key(
                    physical_dpu=int(item["physical_dpu"]),
                    shape_key=tuple(item["shape_key"]),
                    slot_id=int(item["slot_id"]),
                    logical_idx=int(item["logical_idx"]),
                    segment_ordinal=int(item["segment_ordinal"]),
                ),
            )
            try:
                dpu_started_at = time.perf_counter()
                grouped_contexts = self.helper.weighted_value_sum_grouped_batch(
                    [item["payload"] for item in ordered_grouped_entries]
                )
                self._record_timing("weighted_value_sum_batch_dpu", dpu_started_at)
                self.batch_item_totals["weighted_value_sum_batch_dpu_items"] += len(ordered_grouped_entries)
                for entry, context in zip(ordered_grouped_entries, grouped_contexts):
                    idx = int(entry["logical_idx"])
                    if idx not in segmented_contexts:
                        segmented_contexts[idx] = context
                    else:
                        segmented_contexts[idx] = segmented_contexts[idx] + context
            except RuntimeError:
                for entry in ordered_grouped_entries:
                    base_segment_ordinal = int(entry["segment_ordinal"])
                    for segment_offset, (slot_id, segment_len, block_weights) in enumerate(entry["payload"]):
                        dpu_entries.append(
                            {
                                "payload": (int(slot_id), block_weights),
                                "physical_dpu": int(entry["physical_dpu"]),
                                "shape_key": (
                                    int(block_weights.shape[0]),
                                    int(block_weights.shape[1]),
                                    0,
                                ),
                                "slot_id": int(slot_id),
                                "logical_idx": int(entry["logical_idx"]),
                                "segment_ordinal": base_segment_ordinal + int(segment_offset),
                                "ref_kind": "segmented",
                            }
                        )

        if dpu_entries:
            ordered_entries = sorted(
                dpu_entries,
                key=lambda item: self._helper_submit_sort_key(
                    physical_dpu=int(item["physical_dpu"]),
                    shape_key=tuple(item["shape_key"]),
                    slot_id=int(item["slot_id"]),
                    logical_idx=int(item["logical_idx"]),
                    segment_ordinal=int(item["segment_ordinal"]),
                ),
            )
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.weighted_value_sum_batch([item["payload"] for item in ordered_entries])
            self._record_timing("weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["weighted_value_sum_batch_dpu_items"] += len(ordered_entries)
            for entry, context in zip(ordered_entries, dpu_contexts):
                ref_kind = str(entry["ref_kind"])
                idx = int(entry["logical_idx"])
                if ref_kind == "regular":
                    contexts[idx] = context
                else:
                    if idx not in segmented_contexts:
                        segmented_contexts[idx] = context
                    else:
                        segmented_contexts[idx] = segmented_contexts[idx] + context

        for idx, context in segmented_contexts.items():
            contexts[idx] = context

        self._record_timing("weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["weighted_value_sum_batch_total"] += len(slot_weights)
        return self._complete_batch_outputs("weighted_value_sum_batch", contexts)

    def softmax_weighted_value_sum_batch(
        self,
        slot_scores: list[tuple[str, str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not slot_scores:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_scores]
        dpu_entries: list[Dict[str, object]] = []
        segmented_scores: list[tuple[int, tuple[str, str, torch.Tensor]]] = []
        host_fallback_scores: list[tuple[int, tuple[str, str, torch.Tensor]]] = []

        for idx, (k_slot, v_slot, scores) in enumerate(slot_scores):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
                self.batch_item_totals["softmax_weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["softmax_weighted_value_sum_batch_segmented_logical_items"] += 1
                segmented_scores.append((idx, (k_slot, v_slot, scores)))
            elif slot_info["backend"] == "dpu":
                if int(scores.shape[1]) > int(slot_info["seq_len"]):
                    raise ValueError(
                        f"regular dpu softmax-AV score window exceeds slot seq_len: scores={tuple(scores.shape)} "
                        f"slot_seq_len={slot_info['seq_len']}"
                    )
                if int(scores.shape[1]) < int(slot_info["seq_len"]):
                    host_started_at = time.perf_counter()
                    weights = torch.softmax(scores.detach().cpu().to(torch.float32).contiguous(), dim=-1)
                    contexts[idx] = self._regular_dpu_host_weighted_value_sum(slot_info, weights)
                    self._record_timing("softmax_weighted_value_sum_batch_host_fallback", host_started_at)
                    self.batch_item_totals["softmax_weighted_value_sum_batch_host_fallback_items"] += 1
                else:
                    dpu_entries.append(
                        {
                            "payload": (int(slot_info["slot_id"]), scores),
                            "physical_dpu": int(slot_info["physical_dpu"]),
                            "shape_key": (
                                int(scores.shape[0]),
                                int(scores.shape[1]),
                                int(slot_info["head_dim"]),
                            ),
                            "slot_id": int(slot_info["slot_id"]),
                            "logical_idx": int(idx),
                            "segment_ordinal": 0,
                        }
                    )
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

        if dpu_entries:
            ordered_entries = sorted(
                dpu_entries,
                key=lambda item: self._helper_submit_sort_key(
                    physical_dpu=int(item["physical_dpu"]),
                    shape_key=tuple(item["shape_key"]),
                    slot_id=int(item["slot_id"]),
                    logical_idx=int(item["logical_idx"]),
                    segment_ordinal=0,
                ),
            )
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.softmax_weighted_value_sum_batch(
                [item["payload"] for item in ordered_entries]
            )
            self._record_timing("softmax_weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["softmax_weighted_value_sum_batch_dpu_items"] += len(ordered_entries)
            for entry, context in zip(ordered_entries, dpu_contexts):
                contexts[int(entry["logical_idx"])] = context

        if segmented_scores:
            segmented_weights = []
            for _, (k_slot, v_slot, scores) in segmented_scores:
                normalized_scores = scores.detach().cpu().to(torch.float32).contiguous()
                segmented_weights.append(
                    (k_slot, v_slot, torch.softmax(normalized_scores, dim=-1))
                )
            segmented_contexts = self.weighted_value_sum_batch(segmented_weights)
            for (idx, _), context in zip(segmented_scores, segmented_contexts):
                contexts[idx] = context

        self._record_timing("softmax_weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["softmax_weighted_value_sum_batch_total"] += len(slot_scores)
        return self._complete_batch_outputs("softmax_weighted_value_sum_batch", contexts)

    def qk_softmax_weighted_value_sum_batch(
        self,
        slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]],
    ) -> list[torch.Tensor]:
        if not slot_queries:
            return []
        total_started_at = time.perf_counter()

        contexts: list[torch.Tensor | None] = [None for _ in slot_queries]
        dpu_entries: list[Dict[str, object]] = []
        segmented_queries: list[
            tuple[int, tuple[str, str, list[int], int, torch.Tensor, float]]
        ] = []
        host_fallback_queries: list[
            tuple[int, tuple[str, str, list[int], int, torch.Tensor, float]]
        ] = []
        sparse_two_stage_queries: list[
            tuple[int, tuple[str, str, list[int], int, torch.Tensor, float]]
        ] = []

        for idx, (k_slot, v_slot, local_head_indices, window, queries, score_scale) in enumerate(slot_queries):
            key = self._slot_key(k_slot, v_slot)
            slot_info = self.slot_mapping[key]
            if slot_info["backend"] in {"dpu_segmented", "dpu_blocked"}:
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_segmented_logical_items"] += 1
                actual_window = min(int(window), int(slot_info["seq_len"]))
                segmented_queries.append(
                    (
                        idx,
                        (
                            k_slot,
                            v_slot,
                            [int(v) for v in local_head_indices],
                            int(actual_window),
                            queries,
                            float(score_scale),
                        ),
                    )
                )
            elif slot_info["backend"] == "dpu":
                actual_window = min(int(window), int(slot_info["seq_len"]))
                if actual_window < int(slot_info["seq_len"]):
                    sparse_two_stage_queries.append(
                        (
                            idx,
                            (
                                k_slot,
                                v_slot,
                                [int(v) for v in local_head_indices],
                                int(actual_window),
                                queries,
                                float(score_scale),
                            ),
                        )
                    )
                    continue
                dpu_entries.append(
                    {
                        "payload": (
                            int(slot_info["slot_id"]),
                            [int(v) for v in local_head_indices],
                            actual_window,
                            queries,
                            float(score_scale),
                        ),
                        "physical_dpu": int(slot_info["physical_dpu"]),
                        "shape_key": (
                            len(local_head_indices),
                            int(actual_window),
                            int(queries.shape[1]),
                        ),
                        "slot_id": int(slot_info["slot_id"]),
                        "logical_idx": int(idx),
                        "segment_ordinal": 0,
                    }
                )
            else:
                host_fallback_queries.append(
                    (
                        idx,
                        (k_slot, v_slot, [int(v) for v in local_head_indices], window, queries, float(score_scale)),
                    )
                )

        if sparse_two_stage_queries:
            score_queries = [
                (k_slot, v_slot, local_head_indices, window, queries)
                for _, (k_slot, v_slot, local_head_indices, window, queries, _score_scale) in sparse_two_stage_queries
            ]
            score_mats = self.qk_slot_scores_batch(score_queries)
            full_group_scores: list[tuple[int, tuple[str, str, torch.Tensor]]] = []
            for score_mat, (idx, (k_slot, v_slot, local_head_indices, _window, _queries, score_scale)) in zip(
                score_mats,
                sparse_two_stage_queries,
            ):
                key = self._slot_key(k_slot, v_slot)
                slot_info = self.slot_mapping[key]
                scaled_scores = score_mat.to(torch.float32) * float(score_scale)
                if [int(v) for v in local_head_indices] == list(range(int(slot_info["group_heads"]))):
                    full_group_scores.append((idx, (k_slot, v_slot, scaled_scores)))
                    continue
                weights = torch.softmax(scaled_scores, dim=-1)
                contexts[idx] = self._materialized_local_weighted_value_sum(
                    k_slot,
                    v_slot,
                    [int(v) for v in local_head_indices],
                    weights,
                )
            if full_group_scores:
                sparse_contexts = self.softmax_weighted_value_sum_batch([item for _, item in full_group_scores])
                for (idx, _), context in zip(full_group_scores, sparse_contexts):
                    contexts[idx] = context

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

        if dpu_entries:
            ordered_entries = sorted(
                dpu_entries,
                key=lambda item: self._helper_submit_sort_key(
                    physical_dpu=int(item["physical_dpu"]),
                    shape_key=tuple(item["shape_key"]),
                    slot_id=int(item["slot_id"]),
                    logical_idx=int(item["logical_idx"]),
                    segment_ordinal=0,
                ),
            )
            dpu_started_at = time.perf_counter()
            dpu_contexts = self.helper.qk_softmax_weighted_value_sum_batch(
                [item["payload"] for item in ordered_entries]
            )
            self._record_timing("qk_softmax_weighted_value_sum_batch_dpu", dpu_started_at)
            self.batch_item_totals["qk_softmax_weighted_value_sum_batch_dpu_items"] += len(ordered_entries)
            for entry, context in zip(ordered_entries, dpu_contexts):
                contexts[int(entry["logical_idx"])] = context

        if segmented_queries:
            partial_entries: list[Dict[str, object]] = []

            for logical_idx, (k_slot, v_slot, local_head_indices, window, queries, score_scale) in segmented_queries:
                key = self._slot_key(k_slot, v_slot)
                slot_info = self.slot_mapping[key]
                actual_window = min(int(window), int(slot_info["seq_len"]))
                if actual_window <= 0:
                    contexts[logical_idx] = torch.empty((len(local_head_indices), int(slot_info["head_dim"])), dtype=torch.float32)
                    continue
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_blocked_logical_items"] += 1
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_segmented_logical_items"] += 1
                for segment_ordinal, (block, take_len) in enumerate(self._active_block_plan(slot_info, actual_window)):
                    partial_entries.append(
                        {
                            "payload": (
                                int(block["slot_id"]),
                                [int(v) for v in local_head_indices],
                                int(take_len),
                                queries,
                                float(score_scale),
                            ),
                            "physical_dpu": int(block["physical_dpu"]),
                            "shape_key": (
                                len(local_head_indices),
                                int(take_len),
                                int(queries.shape[1]),
                            ),
                            "slot_id": int(block["slot_id"]),
                            "logical_idx": int(logical_idx),
                            "segment_ordinal": int(segment_ordinal),
                        }
                    )

            if partial_entries:
                ordered_entries = sorted(
                    partial_entries,
                    key=lambda item: self._helper_submit_sort_key(
                        physical_dpu=int(item["physical_dpu"]),
                        shape_key=tuple(item["shape_key"]),
                        slot_id=int(item["slot_id"]),
                        logical_idx=int(item["logical_idx"]),
                        segment_ordinal=int(item["segment_ordinal"]),
                    ),
                )
                dpu_started_at = time.perf_counter()
                partial_outputs = self.helper.qk_softmax_weighted_value_sum_partial_batch(
                    [item["payload"] for item in ordered_entries]
                )
                self._record_timing("qk_softmax_weighted_value_sum_batch_dpu", dpu_started_at)
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_dpu_items"] += len(ordered_entries)
                reduce_started_at = time.perf_counter()
                if self.host_partial_reduce_enabled:
                    merged_contexts = self._merge_partial_contexts(ordered_entries, partial_outputs)
                else:
                    merged_numerators: Dict[int, torch.Tensor] = {}
                    merged_row_max: Dict[int, torch.Tensor] = {}
                    merged_row_sum: Dict[int, torch.Tensor] = {}
                    for entry, (segment_context, segment_row_max, segment_row_sum) in zip(ordered_entries, partial_outputs):
                        logical_idx = int(entry["logical_idx"])
                        score_scale = float(entry["payload"][4])
                        segment_numerator = segment_context.to(torch.float32)
                        segment_row_max = segment_row_max.to(torch.float32) * score_scale
                        segment_row_sum = segment_row_sum.to(torch.float32)
                        if logical_idx not in merged_numerators:
                            merged_numerators[logical_idx] = segment_numerator
                            merged_row_max[logical_idx] = segment_row_max
                            merged_row_sum[logical_idx] = segment_row_sum
                            continue

                        prev_row_max = merged_row_max[logical_idx]
                        prev_row_sum = merged_row_sum[logical_idx]
                        prev_numerator = merged_numerators[logical_idx]
                        combined_row_max = torch.maximum(prev_row_max, segment_row_max)
                        prev_scale = torch.exp(prev_row_max - combined_row_max)
                        seg_scale = torch.exp(segment_row_max - combined_row_max)
                        combined_row_sum = prev_row_sum * prev_scale + segment_row_sum * seg_scale
                        merged_numerators[logical_idx] = (
                            prev_numerator * prev_scale.unsqueeze(1)
                        ) + (segment_numerator * seg_scale.unsqueeze(1))
                        merged_row_max[logical_idx] = combined_row_max
                        merged_row_sum[logical_idx] = combined_row_sum
                    merged_contexts = {
                        logical_idx: numerator
                        / torch.clamp(merged_row_sum[logical_idx], min=1e-12).unsqueeze(1)
                        for logical_idx, numerator in merged_numerators.items()
                    }
                self._record_timing("qk_softmax_weighted_value_sum_batch_host_reduce", reduce_started_at)
                self.batch_item_totals["qk_softmax_weighted_value_sum_batch_host_reduce_items"] += len(ordered_entries)

                for logical_idx, context in merged_contexts.items():
                    contexts[logical_idx] = context

        self._record_timing("qk_softmax_weighted_value_sum_batch_total", total_started_at)
        self.batch_item_totals["qk_softmax_weighted_value_sum_batch_total"] += len(slot_queries)
        return self._complete_batch_outputs("qk_softmax_weighted_value_sum_batch", contexts)
