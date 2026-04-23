import argparse
import os
import sys
import ray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--prompt", default="Hello CloverInfer")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--expected-prefill-ip", default="192.168.123.3")
    parser.add_argument("--expected-dense-ip", default="192.168.123.4")
    parser.add_argument("--expected-attention-ip", default="192.168.123.7")
    return parser.parse_args()


def main():
    args = parse_args()
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
        attention_backend="cpu",
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

    print("Placement verification passed.")


if __name__ == "__main__":
    main()
