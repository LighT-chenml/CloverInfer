import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.clover_planner import DynamicTokenMetadata, plan_sharding, update_sharding


def test_plan_sharding_balances_loads():
    requests = [
        {"request_id": "r0", "seq_len": 16},
        {"request_id": "r1", "seq_len": 8},
        {"request_id": "r2", "seq_len": 4},
    ]
    plan = plan_sharding(requests, D=8, H=2)
    assert plan["num_dpus"] == 8
    assert plan["num_heads"] == 2
    assert len(plan["dpu_groups"]) == 2
    assert len(plan["dpu_loads"]) == 8
    assert plan["load_stddev"] >= 0.0

    for head_id, request_map in plan["per_head_shards"].items():
        for request in requests:
            shards = request_map[request["request_id"]]
            covered = sum(int(item["token_count"]) for item in shards)
            assert covered == int(request["seq_len"])


def test_update_sharding_replans_with_new_requests():
    base = plan_sharding([{"request_id": "base", "seq_len": 12}], D=4, H=2)
    updated = update_sharding(base, [{"request_id": "new", "seq_len": 6}])
    assert updated["metadata"]["updated_from_existing"] is True
    for head_id, request_map in updated["per_head_shards"].items():
        assert "base" in request_map
        assert "new" in request_map


def test_plan_sharding_supports_fewer_dpus_than_heads():
    plan = plan_sharding(
        [
            {"request_id": "r0", "seq_len": 10},
            {"request_id": "r1", "seq_len": 6},
        ],
        D=4,
        H=12,
    )
    assert plan["metadata"]["effective_group_count"] == 4
    assert plan["metadata"]["planner_mode"] == "single_dpu_multi_head_group"
    assert len(plan["head_group_ranges"]) == 4
    assert len(plan["dpu_groups"]) == 4
    total_group_heads = 0
    for head_id, head_range in plan["head_group_ranges"].items():
        total_group_heads += int(head_range["group_heads"])
        shards = plan["per_head_shards"][head_id]["r0"]
        assert len(shards) == 1
        assert shards[0]["group_heads"] == head_range["group_heads"]
    assert total_group_heads == 12


def test_plan_sharding_prefill_uses_capacity_hints():
    plan = plan_sharding(
        [{"request_id": "r0", "seq_len": 12}],
        D=4,
        H=1,
        dpu_free_capacity={0: 1, 1: 1, 2: 20, 3: 2},
    )
    shards = plan["per_head_shards"][0]["r0"]
    widest = max(shards, key=lambda item: int(item["token_count"]))
    assert plan["metadata"]["capacity_aware_prefill"] is True
    assert widest["dpu_id"] == 2


def test_dynamic_metadata_tracks_decode_release_and_rebalance():
    plan = plan_sharding([{"request_id": "r0", "seq_len": 8}], D=4, H=1)
    metadata = DynamicTokenMetadata.from_plan(plan)
    assert metadata.group_counts(0) == {0: 2, 1: 2, 2: 2, 3: 2}

    appended = metadata.append_decode_token("r0", 0, token_idx=8)
    assert appended.token_range.end == 9
    assert sum(metadata.group_counts(0).values()) == 9

    metadata.range_table[0]["r0"] = [
        shard
        for shard in metadata.range_table[0]["r0"]
        if int(shard.dpu_id) == 0
    ]
    metadata.recompute_counts()
    before = metadata.imbalance_cv(0)
    migration = metadata.rebalance_once(0, threshold=0.1)
    assert migration["migrated"] is True
    assert metadata.imbalance_cv(0) < before

    metadata.release_request("r0")
    assert sum(metadata.group_counts(0).values()) == 0
