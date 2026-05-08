import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.clover_planner import plan_sharding
from src.core.clover_pipeline_sim import SimulationConfig, SimplePipelineSimulator
from src.core.clover_scheduler_components import (
    CapacityAwareMicroBatchScheduler,
    default_capacity_checker,
    default_predict_host_time,
    default_predict_pim_time,
)


def main():
    scheduler = CapacityAwareMicroBatchScheduler(
        planner=plan_sharding,
        predict_pim_time=lambda reqs, plan: default_predict_pim_time(reqs, plan, a=0.35, b=0.01),
        predict_host_time=lambda total_tokens: default_predict_host_time(total_tokens, c=0.08),
        capacity_checker=default_capacity_checker,
        num_dpus=16,
        num_heads=4,
        time_gap_threshold=0.02,
        lookahead_window=4,
    )
    simulator = SimplePipelineSimulator(
        SimulationConfig(
            num_dpus=16,
            num_heads=4,
            max_capacity_per_dpu=256,
            transfer_latency_s=0.01,
            transfer_bandwidth_tokens_per_s=300.0,
            dpu_compute_scale_s=0.35,
            dpu_compute_bias_s=0.01,
            fc_compute_per_token_s=0.02,
            fixed_batch_size=3,
        ),
        scheduler,
    )

    workload = [
        {"request_id": "req0", "seq_len": 96, "num_new_tokens": 1},
        {"request_id": "req1", "seq_len": 8, "num_new_tokens": 1},
        {"request_id": "req2", "seq_len": 80, "num_new_tokens": 1},
        {"request_id": "req3", "seq_len": 12, "num_new_tokens": 1},
        {"request_id": "req4", "seq_len": 64, "num_new_tokens": 1},
        {"request_id": "req5", "seq_len": 16, "num_new_tokens": 1},
    ]

    optimized = simulator.run(workload, optimized=True)
    fixed = simulator.run(workload, optimized=False)

    print("optimized_tokens_per_s", round(optimized.tokens_per_s, 4))
    print("fixed_tokens_per_s", round(fixed.tokens_per_s, 4))
    print("optimized_utilization", {k: round(v, 3) for k, v in optimized.utilization.items()})
    print("fixed_utilization", {k: round(v, 3) for k, v in fixed.utilization.items()})
    print("optimized_batches", len(optimized.batches))
    print("fixed_batches", len(fixed.batches))
    print("optimized_first_batch", optimized.batches[0]["log"] if optimized.batches else "none")
    print("fixed_first_batch", fixed.batches[0]["log"] if fixed.batches else "none")


if __name__ == "__main__":
    main()
