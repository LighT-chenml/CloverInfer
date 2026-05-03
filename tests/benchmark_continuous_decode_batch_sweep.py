import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import ray
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def parse_int_list(value: str) -> List[int]:
    items: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return items


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def build_ray_runtime_env(*, force_rebuild_kvslot_helper: bool = False) -> Dict[str, object]:
    excludes = [
        ".git/",
        "artifacts/",
        "model/",
        "**/__pycache__/",
        "*.pyc",
    ]
    env_vars: Dict[str, str] = {
        "PYTHONPATH": REPO_ROOT,
        "CLOVER_SHARED_REPO_ROOT": REPO_ROOT,
    }
    # Note: we intentionally do not set CLOVER_KVSLOT_MAX_GROUP_CAPACITY_TOKENS
    # globally here. Capacity-capping is model/workload dependent and should be
    # controlled explicitly by the caller to keep baselines comparable.
    if force_rebuild_kvslot_helper:
        env_vars["CLOVER_FORCE_REBUILD_KVSLOT"] = "1"
    # Allow callers to override kvslot-related policies via the driver env.
    # This keeps the sweep script generic while still letting experiments opt-in
    # to escape hatches (e.g., host fallback when blocked groups overflow).
    passthrough_keys = [
        "CLOVER_KVSLOT_MAX_GROUP_CAPACITY_TOKENS",
        "CLOVER_KVSLOT_BLOCKED_OVERFLOW_POLICY",
    ]
    for key in passthrough_keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            env_vars[key] = str(value)
    return {
        "working_dir": REPO_ROOT,
        "excludes": excludes,
        "env_vars": env_vars,
    }


def build_prompt_for_target_tokens(tokenizer, target_tokens: int, prefix: str = "Context:") -> Tuple[str, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    filler_ids = tokenizer.encode(" clover", add_special_tokens=False)
    token_ids = list(prefix_ids)
    while len(token_ids) < target_tokens:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:target_tokens]
    prompt = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(actual_ids)


def topk_timings(totals: object, k: int = 5) -> List[Dict[str, float]]:
    if not isinstance(totals, dict):
        return []
    items = []
    for key, value in totals.items():
        try:
            sec = float(value)
        except Exception:
            continue
        items.append((str(key), sec))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [{"op": op, "s": float(sec)} for op, sec in items[: max(0, int(k))]]


def make_cluster_config(args, decode_max_batch_size: int) -> ClusterConfig:
    return ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource="prefill_gpu",
        decode_dense_resource="decode_dense_gpu",
        attention_resource="attention_pim",
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend="cloverinfer",
        pim_num_dpus=int(args.pim_num_dpus),
        pim_length=int(args.pim_length),
        pim_block_tokens=int(args.pim_block_tokens),
        pim_head_grouping_policy=str(args.pim_head_grouping_policy),
        pim_dpu_placement_policy=str(args.pim_dpu_placement_policy),
        pim_resident_store_backend=str(args.pim_resident_store_backend),
        pim_resident_kv_dtype=str(args.pim_resident_kv_dtype),
        pim_qk_full_enabled=True,
        pim_softmax_av_fused_enabled=True,
        clover_pim_attention_enabled=True,
        # Disable cpu shadow during sweeps to reduce pressure and avoid double bookkeeping.
        clover_cpu_shadow_enabled=bool(args.clover_cpu_shadow_enabled),
        clover_shadow_checks_enabled=bool(args.clover_shadow_checks_enabled),
        clover_op_profiling_enabled=bool(args.clover_op_profiling_enabled),
        decode_continuous_batching_enabled=True,
        decode_continuous_max_batch_size=int(decode_max_batch_size),
        decode_continuous_batch_window_s=float(args.decode_batch_window_s),
        # Keep attention RPC-side batching off; continuous engine should provide batching.
        attention_rpc_cross_key_batch_enabled=True,
        attention_rpc_batch_window_s=float(args.attention_rpc_batch_window_s),
        attention_rpc_batch_max_size=int(args.attention_rpc_batch_max_size),
        attention_wavefront_cohort_policy=str(getattr(args, "attention_wavefront_cohort_policy", "batch")),
        attention_actor_side_batching_enabled=False,
    )


def summarize_case(records: List[Dict[str, object]]) -> Dict[str, object]:
    latencies = [float(r["metrics"]["latency"]) for r in records]
    ttfts = [float(r["metrics"]["ttft"]) for r in records]
    tpots = [float(r["metrics"]["tpot"]) for r in records]
    throughputs = [float(r["metrics"]["throughput"]) for r in records]
    total_tokens = [int(r["metrics"]["total_tokens"]) for r in records]
    attention_rpc_s = [
        float(r["metrics"]["stage_timing"]["scheduler"].get("attention_decode_rpc_s", 0.0))
        for r in records
        if isinstance(r["metrics"].get("stage_timing"), dict)
    ]
    per_token_attention_rpc = []
    for r in records:
        st = r["metrics"].get("stage_timing", {})
        attn = float(st.get("scheduler", {}).get("attention_decode_rpc_s", 0.0))
        denom = max(int(r["metrics"]["total_tokens"]) - 1, 1)
        per_token_attention_rpc.append(attn / float(denom))

    last_backend_debug = {}
    if records:
        last_backend_debug = (
            records[-1]["metrics"]
            .get("attention_backend_before_free", {})
            .get("backend_debug", {})
        )
    clover_totals = last_backend_debug.get("clover_op_timing_totals_s", {})
    last_store_debug = last_backend_debug.get("resident_store_debug", {})
    last_engine_stats = (
        records[-1]["metrics"].get("continuous_engine", {}) if records else {}
    )

    dpu_allocate_failures = int(last_store_debug.get("dpu_allocate_failures", 0) or 0)
    fallback_allocations = int(last_store_debug.get("fallback_allocations", 0) or 0)
    dpu_capacity_fallbacks = int(last_store_debug.get("dpu_capacity_fallbacks", 0) or 0)
    engine_last_attention_batch_size = int(last_engine_stats.get("engine_last_attention_batch_size", 0) or 0)
    engine_avg_attention_batch_size = float(last_engine_stats.get("engine_attention_batch_avg_size", 0.0) or 0.0)

    return {
        "num_requests": int(len(records)),
        "avg_latency": float(statistics.mean(latencies)) if latencies else 0.0,
        "p95_latency": percentile(latencies, 0.95),
        "avg_ttft": float(statistics.mean(ttfts)) if ttfts else 0.0,
        "avg_tpot": float(statistics.mean(tpots)) if tpots else 0.0,
        "avg_throughput": float(statistics.mean(throughputs)) if throughputs else 0.0,
        "avg_total_tokens": float(statistics.mean(total_tokens)) if total_tokens else 0.0,
        "avg_attention_decode_rpc_s": float(statistics.mean(attention_rpc_s)) if attention_rpc_s else 0.0,
        "avg_attention_decode_rpc_s_per_token": float(statistics.mean(per_token_attention_rpc))
        if per_token_attention_rpc
        else 0.0,
        "last_engine_last_attention_batch_size": int(engine_last_attention_batch_size),
        "last_engine_avg_attention_batch_size": float(engine_avg_attention_batch_size),
        "last_store_dpu_allocate_failures": int(dpu_allocate_failures),
        "last_store_dpu_capacity_fallbacks": int(dpu_capacity_fallbacks),
        "last_store_fallback_allocations": int(fallback_allocations),
        "last_clover_op_top5": topk_timings(clover_totals, k=5),
    }


def run_case(args, decode_max_batch_size: int, prompt: str) -> Dict[str, object]:
    cluster = make_cluster_config(args, decode_max_batch_size)
    model = ModelConfig(
        model_name=str(args.model_name),
        model_path=str(args.model),
        max_seq_len=int(args.max_seq_len),
        max_new_tokens=int(args.max_new_tokens),
        dtype=str(args.dtype),
    )
    scheduler = GlobalScheduler.remote(cluster, model)
    placement = ray.get(scheduler.initialize_cluster.remote())

    # Warmup one request to stabilize caches.
    _ = ray.get(scheduler.submit_request.remote(prompt, return_metrics=False, max_new_tokens=2))

    total_requests = int(args.num_requests)
    concurrency = max(1, int(args.concurrency))
    records: List[Dict[str, object]] = []
    pending: Dict[ray.ObjectRef, float] = {}
    sent = 0
    completed = 0

    try:
        while completed < total_requests:
            while sent < total_requests and len(pending) < concurrency:
                started_at = time.time()
                fut = scheduler.submit_request.remote(
                    prompt,
                    return_metrics=True,
                    max_new_tokens=int(args.max_new_tokens),
                )
                pending[fut] = started_at
                sent += 1

            ready, _ = ray.wait(list(pending.keys()), num_returns=1)
            fut = ready[0]
            started_at = pending.pop(fut)
            completion, metrics = ray.get(fut)
            completed += 1
            records.append(
                {
                    "started_at": float(started_at),
                    "finished_at": float(time.time()),
                    "completion": completion,
                    "metrics": metrics,
                }
            )
    finally:
        # Best-effort cleanup: release kvslot helper / actors between cases.
        try:
            ray.get(scheduler.shutdown_cluster.remote())
        except Exception:
            pass
        try:
            ray.kill(scheduler)
        except Exception:
            pass

    return {
        "case": {
            "decode_continuous_max_batch_size": int(decode_max_batch_size),
            "decode_continuous_batch_window_s": float(args.decode_batch_window_s),
            "concurrency": int(concurrency),
            "num_requests": int(total_requests),
            "prompt_token_length": int(args.prompt_token_length),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "placement": placement,
        "records": records,
        "summary": summarize_case(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--model-name", default="opt-125m")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompt-token-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--decode-batch-sizes", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--decode-batch-window-s", type=float, default=0.002)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--pim-num-dpus", type=int, default=16)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument("--pim-head-grouping-policy", default="balanced")
    parser.add_argument("--pim-dpu-placement-policy", default="rotated")
    parser.add_argument("--pim-resident-store-backend", default="upmem_kvslot")
    parser.add_argument("--pim-resident-kv-dtype", default="fp16")
    parser.add_argument("--attention-rpc-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-rpc-batch-max-size", type=int, default=8)
    parser.add_argument(
        "--attention-wavefront-cohort-policy",
        default="batch",
        choices=["batch", "step"],
        help="Forwarded to ClusterConfig.attention_wavefront_cohort_policy. "
        "Use 'step' to batch attention by (decode_step, layer) across cohorts.",
    )
    parser.add_argument("--clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--no-clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--no-clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--no-clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--pim-force-rebuild-kvslot-helper", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "continuous_decode_batch_sweep.jsonl"),
    )
    args = parser.parse_args()

    clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = False
    elif args.clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = True
    # Sweep stability: default to no cpu shadow unless explicitly enabled.
    if not args.clover_cpu_shadow_enabled and not args.no_clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = False
    clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = False
    elif args.clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = True
    if not args.clover_shadow_checks_enabled and not args.no_clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = False
    clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        clover_op_profiling_enabled = False
    elif args.clover_op_profiling_enabled:
        clover_op_profiling_enabled = True
    args.clover_cpu_shadow_enabled = clover_cpu_shadow_enabled
    args.clover_shadow_checks_enabled = clover_shadow_checks_enabled
    args.clover_op_profiling_enabled = clover_op_profiling_enabled

    decode_batch_sizes = sorted({max(1, int(x)) for x in args.decode_batch_sizes})
    if not decode_batch_sizes:
        decode_batch_sizes = [1]
    max_bs = max(decode_batch_sizes)
    if int(args.concurrency) <= 0:
        args.concurrency = int(max_bs)
    if int(args.num_requests) <= 0:
        # Default: 2 waves per case.
        args.num_requests = int(args.concurrency) * 2

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Qwen models often rely on custom tokenizer code. Since we're running from a
    # local, pinned model snapshot for experiments, allow local custom code.
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    prompt, actual_tokens = build_prompt_for_target_tokens(tokenizer, int(args.prompt_token_length))
    args.prompt_token_length = int(actual_tokens)

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env=build_ray_runtime_env(force_rebuild_kvslot_helper=bool(args.pim_force_rebuild_kvslot_helper)),
    )

    outputs: List[Dict[str, object]] = []
    with open(args.output, "w", encoding="utf-8") as f:
        for bs in decode_batch_sizes:
            case_out = run_case(args, int(bs), prompt)
            outputs.append(case_out)
            f.write(json.dumps(case_out, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "model": str(args.model),
        "prompt_token_length": int(args.prompt_token_length),
        "max_new_tokens": int(args.max_new_tokens),
        "decode_batch_sizes": decode_batch_sizes,
        "cases": [
            {
                **case["case"],
                **case["summary"],
            }
            for case in outputs
        ],
        "output": args.output,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
