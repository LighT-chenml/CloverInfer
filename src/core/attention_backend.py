from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
from typing import Dict, List

import torch


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
        qk_check_interval: int = 1,
        qk_check_limit: int = 1,
        qk_mixed_enabled: bool = True,
        qk_mixed_heads: int = 2,
        qk_mixed_window: int = 128,
    ):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.num_dpus = num_dpus
        self.length = length
        self.qk_check_interval = qk_check_interval
        self.qk_check_limit = qk_check_limit
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
        self._run_dot_smoke_test()

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

        header = struct.pack("<IIII", 0x514B494F, head_dim, num_keys, num_queries)
        qk_dir = os.path.join(self.repo_root, "src", "pim", "upmem_qk")
        binary_path = self._ensure_qk_binary()

        with tempfile.TemporaryDirectory(prefix="clover_qk_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.bin")
            output_path = os.path.join(tmpdir, "output.bin")
            with open(input_path, "wb") as f:
                f.write(header)
                f.write(q_i32.numpy().tobytes(order="C"))
                f.write(k_i32.numpy().tobytes(order="C"))

            completed = subprocess.run(
                [binary_path, "--input", input_path, "--output", output_path, "--num-dpus", str(min(self.num_dpus, num_keys))],
                cwd=qk_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.qk_last_output = (completed.stdout + completed.stderr).strip()
            if completed.returncode != 0:
                raise RuntimeError(f"UPMEM qk runner failed:\n{self.qk_last_output}")

            with open(output_path, "rb") as f:
                out_header = f.read(16)
                magic, out_head_dim, out_num_keys, out_num_queries = struct.unpack("<IIII", out_header)
                if magic != 0x514B494F or out_head_dim != head_dim or out_num_keys != num_keys or out_num_queries != num_queries:
                    raise RuntimeError("UPMEM qk runner returned an invalid header")
                raw_scores = f.read(num_queries * num_keys * 8)

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

    def init_request(self, request_id: str, initial_kv: List[Dict[str, torch.Tensor]]) -> int:
        return self.cpu_backend.init_request(request_id, initial_kv)

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

        q = query.detach().cpu().contiguous()
        k_new = key.detach().cpu().contiguous()
        v_new = value.detach().cpu().contiguous()

        if q.dim() == 3:
            q = q.squeeze(0)
        if k_new.dim() == 3:
            k_new = k_new.squeeze(0)
        if v_new.dim() == 3:
            v_new = v_new.squeeze(0)

        self.cpu_backend.k_cache[request_id][layer_idx] = torch.cat(
            [self.cpu_backend.k_cache[request_id][layer_idx], k_new.unsqueeze(0)], dim=0
        )
        self.cpu_backend.v_cache[request_id][layer_idx] = torch.cat(
            [self.cpu_backend.v_cache[request_id][layer_idx], v_new.unsqueeze(0)], dim=0
        )

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
        context = torch.einsum("hl,lhd->hd", weights, values.float()).to(query.dtype)

        if layer_idx == len(self.cpu_backend.k_cache[request_id]) - 1:
            self.cpu_backend.context_lens[request_id] += 1

        return context.unsqueeze(0)

    def get_context_len(self, request_id: str) -> int:
        return self.cpu_backend.get_context_len(request_id)

    def free_request(self, request_id: str) -> None:
        self.cpu_backend.free_request(request_id)

    def get_debug_info(self) -> Dict[str, object]:
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
        }
