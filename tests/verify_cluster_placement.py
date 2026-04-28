import argparse
import os
import sys
import ray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler

FLOAT_TOL = 1e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--prompt", default="Hello CloverInfer")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--attention-backend", default="cpu", choices=["cpu", "pim_naive", "cloverinfer"])
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-resident-store-backend", default="host", choices=["host", "upmem_kvslot"])
    parser.add_argument("--pim-qk-full-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-full-enabled", action="store_true")
    parser.add_argument("--pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--no-pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--no-pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--no-pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--no-clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--no-clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--no-clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--clover-shadow-check-token-interval", type=int, default=4)
    parser.add_argument("--clover-shadow-check-layer-interval", type=int, default=4)
    parser.add_argument("--clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--decode-step-sync-window-s", type=float, default=0.0)
    parser.add_argument("--decode-step-sync-max-size", type=int, default=8)
    parser.add_argument("--attention-decode-wave-persist-enabled", action="store_true")
    parser.add_argument("--attention-layer-barrier-window-s", type=float, default=0.0)
    parser.add_argument("--attention-layer-barrier-max-size", type=int, default=8)
    parser.add_argument("--attention-rpc-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-rpc-batch-max-size", type=int, default=8)
    parser.add_argument("--attention-actor-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-actor-batch-max-size", type=int, default=8)
    parser.add_argument("--expected-prefill-ip", default="192.168.123.3")
    parser.add_argument("--expected-dense-ip", default="192.168.123.4")
    parser.add_argument("--expected-attention-ip", default="192.168.123.7")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pim_qk_full_enabled and args.no_pim_qk_full_enabled:
        raise ValueError("cannot set both --pim-qk-full-enabled and --no-pim-qk-full-enabled")
    if args.pim_qk_full_shadow_check and args.no_pim_qk_full_shadow_check:
        raise ValueError("cannot set both --pim-qk-full-shadow-check and --no-pim-qk-full-shadow-check")
    if args.pim_softmax_av_fused_enabled and args.no_pim_softmax_av_fused_enabled:
        raise ValueError("cannot set both --pim-softmax-av-fused-enabled and --no-pim-softmax-av-fused-enabled")
    if args.pim_softmax_av_shadow_check and args.no_pim_softmax_av_shadow_check:
        raise ValueError("cannot set both --pim-softmax-av-shadow-check and --no-pim-softmax-av-shadow-check")
    if args.clover_cpu_shadow_enabled and args.no_clover_cpu_shadow_enabled:
        raise ValueError("cannot set both --clover-cpu-shadow-enabled and --no-clover-cpu-shadow-enabled")
    if args.clover_shadow_checks_enabled and args.no_clover_shadow_checks_enabled:
        raise ValueError("cannot set both --clover-shadow-checks-enabled and --no-clover-shadow-checks-enabled")
    if args.clover_op_profiling_enabled and args.no_clover_op_profiling_enabled:
        raise ValueError("cannot set both --clover-op-profiling-enabled and --no-clover-op-profiling-enabled")
    if args.clover_host_qk_mixed_enabled and args.no_clover_host_qk_mixed_enabled:
        raise ValueError("cannot set both --clover-host-qk-mixed-enabled and --no-clover-host-qk-mixed-enabled")
    pim_qk_mixed_enabled = True
    if args.pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = False
    pim_qk_full_enabled = False
    if args.pim_qk_full_enabled:
        pim_qk_full_enabled = True
    if args.no_pim_qk_full_enabled:
        pim_qk_full_enabled = False
    pim_qk_full_shadow_check = True
    if args.no_pim_qk_full_shadow_check:
        pim_qk_full_shadow_check = False
    pim_softmax_av_fused_enabled = False
    if args.pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = True
    if args.no_pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = False
    pim_softmax_av_shadow_check = True
    if args.no_pim_softmax_av_shadow_check:
        pim_softmax_av_shadow_check = False
    clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = False
    clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = False
    clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        clover_op_profiling_enabled = False
    clover_host_qk_mixed_enabled = False
    if args.clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = True
    if args.no_clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = False

    ray.init(
        address=args.address,
        runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}},
    )

    cluster = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource="prefill_gpu",
        decode_dense_resource="decode_dense_gpu",
        attention_resource="attention_pim",
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend=args.attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_resident_store_backend=args.pim_resident_store_backend,
        pim_qk_full_enabled=pim_qk_full_enabled,
        pim_qk_full_shadow_check=pim_qk_full_shadow_check,
        pim_softmax_av_fused_enabled=pim_softmax_av_fused_enabled,
        pim_softmax_av_shadow_check=pim_softmax_av_shadow_check,
        pim_qk_mixed_enabled=pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
        decode_step_sync_window_s=args.decode_step_sync_window_s,
        decode_step_sync_max_size=args.decode_step_sync_max_size,
        attention_decode_wave_persist_enabled=args.attention_decode_wave_persist_enabled,
        attention_layer_barrier_window_s=args.attention_layer_barrier_window_s,
        attention_layer_barrier_max_size=args.attention_layer_barrier_max_size,
        attention_rpc_batch_window_s=args.attention_rpc_batch_window_s,
        attention_rpc_batch_max_size=args.attention_rpc_batch_max_size,
        attention_actor_batch_window_s=args.attention_actor_batch_window_s,
        attention_actor_batch_max_size=args.attention_actor_batch_max_size,
        clover_cpu_shadow_enabled=clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=clover_shadow_checks_enabled,
        clover_op_profiling_enabled=clover_op_profiling_enabled,
        clover_shadow_check_token_interval=args.clover_shadow_check_token_interval,
        clover_shadow_check_layer_interval=args.clover_shadow_check_layer_interval,
        clover_host_qk_mixed_enabled=clover_host_qk_mixed_enabled,
    )
    model = ModelConfig(model_path=args.model, max_new_tokens=args.max_new_tokens)

    scheduler = GlobalScheduler.remote(cluster, model)
    info = ray.get(scheduler.initialize_cluster.remote())
    print("Cluster placement:", info)

    assert info["prefill"]["ip"] == args.expected_prefill_ip, info
    assert info["decode_dense"]["ip"] == args.expected_dense_ip, info
    assert info["attention"]["ip"] == args.expected_attention_ip, info
    assert info["prefill"]["device"] == "cuda", info
    assert info["decode_dense"]["device"] == "cuda", info
    assert info["attention"]["device"] == "cpu", info
    assert info["attention"]["backend"] == args.attention_backend, info
    if args.attention_backend in {"pim_naive", "cloverinfer"}:
        debug = info["attention"]["backend_debug"]
        assert debug["num_dpus"] == args.pim_num_dpus, debug
        assert debug["length"] == args.pim_length, debug
        assert debug["resident_store_backend"] == args.pim_resident_store_backend, debug
        assert debug["qk_full_enabled"] == pim_qk_full_enabled, debug
        assert debug["qk_full_shadow_check"] == pim_qk_full_shadow_check, debug
        assert debug["softmax_av_fused_enabled"] == pim_softmax_av_fused_enabled, debug
        assert debug["softmax_av_shadow_check"] == pim_softmax_av_shadow_check, debug
        assert debug["qk_mixed_enabled"] == pim_qk_mixed_enabled, debug
        assert debug["qk_mixed_heads"] == args.pim_qk_mixed_heads, debug
        assert debug["qk_mixed_window"] == args.pim_qk_mixed_window, debug
        assert debug["resident_metadata_enabled"] is True, debug
        assert debug["resident_compute_enabled"] is True, debug
        if args.attention_backend == "cloverinfer":
            assert debug["backend_variant"] == "cloverinfer", debug
            assert debug["clover_cpu_shadow_enabled"] == clover_cpu_shadow_enabled, debug
            assert debug["clover_shadow_checks_enabled"] == clover_shadow_checks_enabled, debug
            assert debug["clover_op_profiling_enabled"] == clover_op_profiling_enabled, debug
            assert debug["clover_shadow_check_token_interval"] == args.clover_shadow_check_token_interval, debug
            assert debug["clover_shadow_check_layer_interval"] == args.clover_shadow_check_layer_interval, debug
            assert debug["clover_host_qk_mixed_enabled"] == clover_host_qk_mixed_enabled, debug

    if not args.skip_generation:
        output, metrics = ray.get(
            scheduler.submit_request.remote(
                args.prompt,
                return_metrics=True,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print("Output:", repr(output))
        print("Metrics:", metrics)
        assert metrics["total_tokens"] >= 1
        stage_timing = metrics["stage_timing"]
        assert stage_timing["counts"]["decode_steps"] >= 1
        assert stage_timing["counts"]["decode_layers"] >= stage_timing["counts"]["decode_steps"]
        assert stage_timing["scheduler"]["total_rpc_s"] >= 0
        assert stage_timing["actors"]["total_compute_s"] >= 0
        if args.attention_backend in {"pim_naive", "cloverinfer"}:
            debug = metrics["attention_backend"]["backend_debug"]
            assert debug["resident_append_ops"] > 0, debug
            if args.attention_backend == "cloverinfer":
                assert "clover_op_timing_totals_s" in debug, debug
                assert "prepare_decode_record_s" in debug["clover_op_timing_totals_s"], debug
            if pim_softmax_av_fused_enabled:
                assert debug["softmax_av_fused_ops"] > 0, debug
                if args.attention_backend == "pim_naive" or clover_shadow_checks_enabled:
                    assert debug["softmax_av_fused_shadow_max_abs_diff"] <= FLOAT_TOL, debug
            elif debug.get("resident_av_enabled", False):
                assert debug["resident_av_ops"] > 0, debug
                if args.attention_backend == "pim_naive" or clover_shadow_checks_enabled:
                    assert debug["resident_av_shadow_max_abs_diff"] <= FLOAT_TOL, debug
            else:
                assert debug["resident_materialize_ops"] > 0, debug
                if args.attention_backend == "pim_naive" or clover_shadow_checks_enabled:
                    assert debug["resident_shadow_max_abs_diff"] <= FLOAT_TOL, debug
            assert debug["resident_last_freed_request_id"], debug
            assert debug["resident_request_count"] == 0, debug
            if args.pim_resident_store_backend == "upmem_kvslot":
                store_debug = debug["resident_store_debug"]
                assert store_debug["backend"] == "upmem_kvslot_store", store_debug
                assert store_debug["dpu_allocations"] > 0, store_debug
                assert store_debug["helper_restarts"] >= 1, store_debug
                allocator_stats = store_debug["allocator_stats"]
                if allocator_stats:
                    assert len(allocator_stats) == args.pim_num_dpus, allocator_stats
                    for stats in allocator_stats:
                        assert stats["pool_capacity_elems"] > 0, stats
                        assert 0 <= stats["usage_ratio"] <= 1.0, stats
                        assert stats["total_free_elems"] + stats["used_elems_estimate"] == stats["pool_capacity_elems"], stats
                else:
                    assert store_debug["dpu_live_slots"] == 0, store_debug

    print("Placement verification passed.")


if __name__ == "__main__":
    main()
