from __future__ import annotations

import math
import os
import subprocess
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
        scores = torch.einsum("hd,lhd->hl", q.float(), keys.float())
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
    ) -> torch.Tensor:
        should_run_qk_check = (
            self.qk_check_interval > 0
            and self.qk_check_count < self.qk_check_limit
            and (self.qk_check_count % self.qk_check_interval) == 0
        )
        if should_run_qk_check:
            try:
                q = query.detach().cpu()
                if q.dim() == 3:
                    q = q.squeeze(0)
                keys_per_dpu = min(8, self.cpu_backend.k_cache[request_id][layer_idx].shape[0] + 1)
                self._run_qk_smoke_test(head_dim=q.shape[-1], keys_per_dpu=keys_per_dpu)
            except Exception:
                self.qk_check_failures += 1
                raise
        return self.cpu_backend.decode_layer(request_id, layer_idx, query, key, value)

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
        }
