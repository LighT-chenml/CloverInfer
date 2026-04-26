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
    parser.add_argument("--attention-backend", default="cpu", choices=["cpu", "pim_naive"])
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-resident-store-backend", default="host", choices=["host", "upmem_kvslot"])
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--decode-step-sync-window-s", type=float, default=0.0)
    parser.add_argument("--decode-step-sync-max-size", type=int, default=8)
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
    pim_qk_mixed_enabled = True
    if args.pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = False

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
        pim_qk_mixed_enabled=pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
        decode_step_sync_window_s=args.decode_step_sync_window_s,
        decode_step_sync_max_size=args.decode_step_sync_max_size,
        attention_layer_barrier_window_s=args.attention_layer_barrier_window_s,
        attention_layer_barrier_max_size=args.attention_layer_barrier_max_size,
        attention_rpc_batch_window_s=args.attention_rpc_batch_window_s,
        attention_rpc_batch_max_size=args.attention_rpc_batch_max_size,
        attention_actor_batch_window_s=args.attention_actor_batch_window_s,
        attention_actor_batch_max_size=args.attention_actor_batch_max_size,
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
    if args.attention_backend == "pim_naive":
        debug = info["attention"]["backend_debug"]
        assert debug["num_dpus"] == args.pim_num_dpus, debug
        assert debug["length"] == args.pim_length, debug
        assert debug["resident_store_backend"] == args.pim_resident_store_backend, debug
        assert debug["qk_mixed_enabled"] == pim_qk_mixed_enabled, debug
        assert debug["qk_mixed_heads"] == args.pim_qk_mixed_heads, debug
        assert debug["qk_mixed_window"] == args.pim_qk_mixed_window, debug
        assert debug["resident_metadata_enabled"] is True, debug
        assert debug["resident_compute_enabled"] is True, debug

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
        if args.attention_backend == "pim_naive":
            debug = metrics["attention_backend"]["backend_debug"]
            assert debug["resident_append_ops"] > 0, debug
            if debug.get("resident_av_enabled", False):
                assert debug["resident_av_ops"] > 0, debug
                assert debug["resident_av_shadow_max_abs_diff"] <= FLOAT_TOL, debug
            else:
                assert debug["resident_materialize_ops"] > 0, debug
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
