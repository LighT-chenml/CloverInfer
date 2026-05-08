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


def build_scheduler():
    return CapacityAwareMicroBatchScheduler(
        planner=plan_sharding,
        predict_pim_time=lambda reqs, plan: default_predict_pim_time(reqs, plan, a=0.25, b=0.01),
        predict_host_time=lambda total_tokens: default_predict_host_time(total_tokens, c=0.05),
        capacity_checker=default_capacity_checker,
        num_dpus=8,
        num_heads=2,
        time_gap_threshold=0.01,
        lookahead_window=3,
    )


def test_pipeline_sim_runs_and_reports_utilization():
    scheduler = build_scheduler()
    simulator = SimplePipelineSimulator(
        SimulationConfig(
            num_dpus=8,
            num_heads=2,
            max_capacity_per_dpu=128,
            transfer_latency_s=0.01,
            transfer_bandwidth_tokens_per_s=200.0,
            dpu_compute_scale_s=0.01,
            dpu_compute_bias_s=0.0,
            fc_compute_per_token_s=0.005,
        ),
        scheduler,
    )
    requests = [
        {"request_id": "r0", "seq_len": 48, "num_new_tokens": 1},
        {"request_id": "r1", "seq_len": 8, "num_new_tokens": 1},
        {"request_id": "r2", "seq_len": 40, "num_new_tokens": 1},
        {"request_id": "r3", "seq_len": 4, "num_new_tokens": 1},
    ]
    result = simulator.run(requests, optimized=True)
    assert result.tokens_per_s >= 0.0
    assert result.batches
    assert "dpu" in result.utilization

