import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.attention_backend import CpuAttentionBackend, PimNaiveAttentionBackend
from src.core.clover_attention_backend import CloverInferAttentionBackend
from src.core.resident_kv_store import HostResidentKVStore


class _NoSmokePimBackend(PimNaiveAttentionBackend):
    def _run_dot_smoke_test(self) -> None:
        self.smoke_test_ok = True
        self.smoke_test_output = "skipped"


class _NoSmokeCloverBackend(CloverInferAttentionBackend):
    def _run_dot_smoke_test(self) -> None:
        self.smoke_test_ok = True
        self.smoke_test_output = "skipped"


def _one_layer_kv(seq_len=5, num_heads=2, head_dim=3):
    values = torch.arange(seq_len * num_heads * head_dim, dtype=torch.float32).view(seq_len, num_heads, head_dim)
    keys = (values + 1.0) / 10.0
    return [{"key": keys.contiguous(), "value": values.contiguous()}]


def _manual_decode(initial_kv, query, key, value, score_scale, window):
    keys = torch.cat([initial_kv[0]["key"], key.squeeze(0).unsqueeze(0)], dim=0)
    values = torch.cat([initial_kv[0]["value"], value.squeeze(0).unsqueeze(0)], dim=0)
    active_window = min(int(window), int(keys.shape[0])) if int(window) > 0 else int(keys.shape[0])
    keys = keys[-active_window:]
    values = values[-active_window:]
    q = query.squeeze(0).float()
    scores = torch.einsum("hd,lhd->hl", q, keys.float()) * float(score_scale)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hl,lhd->hd", weights, values.float()).unsqueeze(0)


def _manual_gqa_decode(initial_kv, query, key, value, score_scale, window):
    keys = torch.cat([initial_kv[0]["key"], key.squeeze(0).unsqueeze(0)], dim=0)
    values = torch.cat([initial_kv[0]["value"], value.squeeze(0).unsqueeze(0)], dim=0)
    active_window = min(int(window), int(keys.shape[0])) if int(window) > 0 else int(keys.shape[0])
    keys = keys[-active_window:]
    values = values[-active_window:]
    q = query.squeeze(0).float()
    repeat_factor = int(q.shape[0]) // int(keys.shape[1])
    head_map = torch.arange(int(q.shape[0]), dtype=torch.long) // repeat_factor
    keys_for_q = keys[:, head_map, :]
    values_for_q = values[:, head_map, :]
    scores = torch.einsum("hd,lhd->hl", q, keys_for_q.float()) * float(score_scale)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hl,lhd->hd", weights, values_for_q.float()).unsqueeze(0)


def test_cpu_sparse_window_uses_tail_tokens_only():
    initial_kv = _one_layer_kv()
    query = torch.tensor([[[0.2, -0.1, 0.3], [0.4, 0.1, -0.2]]], dtype=torch.float32)
    key = torch.tensor([[[0.5, 0.6, -0.7], [0.2, -0.3, 0.1]]], dtype=torch.float32)
    value = torch.tensor([[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]]], dtype=torch.float32)

    backend = CpuAttentionBackend(attention_sparse_window=2)
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.75)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.75, window=2)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cpu_dense_window_zero_keeps_full_attention():
    initial_kv = _one_layer_kv()
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = CpuAttentionBackend(attention_sparse_window=0)
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=1.0, window=0)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_host_resident_av_accepts_short_tail_weights():
    store = HostResidentKVStore()
    initial_kv = _one_layer_kv(seq_len=4, num_heads=2, head_dim=2)[0]
    store.allocate_group("k", "v", initial_kv["key"], initial_kv["value"], capacity=8)

    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    actual = store.weighted_value_sum("k", "v", weights)
    expected = torch.einsum("hl,lhd->hd", weights, initial_kv["value"][-2:].float())

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_host_resident_softmax_av_accepts_short_tail_scores():
    store = HostResidentKVStore()
    initial_kv = _one_layer_kv(seq_len=4, num_heads=2, head_dim=2)[0]
    store.allocate_group("k", "v", initial_kv["key"], initial_kv["value"], capacity=8)

    scores = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=torch.float32)
    actual = store.softmax_weighted_value_sum_batch([("k", "v", scores)])[0]
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("hl,lhd->hd", weights, initial_kv["value"][-2:].float())

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pim_host_backend_sparse_matches_cpu_sparse_reference():
    initial_kv = _one_layer_kv(seq_len=5, num_heads=2, head_dim=3)
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = _NoSmokePimBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=3,
    )
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.5)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.5, window=3)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert backend.get_debug_info()["attention_sparse_window"] == 3


def test_pim_sparse_resident_init_keeps_only_tail_window():
    initial_kv = _one_layer_kv(seq_len=6, num_heads=2, head_dim=3)
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = _NoSmokePimBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=3,
    )
    returned_context_len = backend.init_request("req", initial_kv)
    state = backend.request_states["req"]

    assert returned_context_len == 6
    assert state.context_len == 3
    assert state.logical_context_len == 6
    assert {group.seq_len for group in state.layer_states[0].head_groups} == {3}
    debug = backend.get_debug_info()
    footprint = debug["resident_request_footprints"][0]
    assert footprint["context_len"] == 3
    assert footprint["logical_context_len"] == 6

    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.5)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.5, window=3)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert state.context_len == 4
    assert state.logical_context_len == 7


def test_clover_host_backend_sparse_matches_cpu_sparse_reference():
    initial_kv = _one_layer_kv(seq_len=6, num_heads=2, head_dim=3)
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = _NoSmokeCloverBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=4,
        shadow_checks_enabled=True,
        shadow_check_token_interval=1,
        shadow_check_layer_interval=1,
    )
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.25)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.25, window=4)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert backend.get_debug_info()["attention_sparse_window"] == 4


def test_clover_qk_only_host_av_path_matches_cpu_sparse_reference():
    initial_kv = _one_layer_kv(seq_len=6, num_heads=2, head_dim=3)
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = _NoSmokeCloverBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=4,
        pim_attention_enabled=True,
        pim_qk_only_host_av_experimental_enabled=True,
        shadow_checks_enabled=True,
        shadow_check_token_interval=1,
        shadow_check_layer_interval=1,
    )
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.25)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.25, window=4)
    debug = backend.get_debug_info()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert debug["clover_pim_qk_only_host_av_experimental_enabled"] is True
    assert debug["clover_pim_qk_only_host_av_decode_items"] == 1


def test_clover_sparse_resident_init_keeps_only_tail_window_with_full_shadow():
    initial_kv = _one_layer_kv(seq_len=7, num_heads=2, head_dim=3)
    query = torch.randn(1, 2, 3)
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)

    backend = _NoSmokeCloverBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=4,
        shadow_checks_enabled=True,
        shadow_check_token_interval=1,
        shadow_check_layer_interval=1,
    )
    returned_context_len = backend.init_request("req", initial_kv)
    state = backend.request_states["req"]

    assert returned_context_len == 7
    assert backend.get_context_len("req") == 7
    assert state.context_len == 4
    assert state.logical_context_len == 7
    assert backend.shadow_layer_lens["req"] == [7]
    assert {group.seq_len for group in state.layer_states[0].head_groups} == {4}

    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.25)
    expected = _manual_decode(initial_kv, query, key, value, score_scale=0.25, window=4)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert backend.get_context_len("req") == 8
    assert state.context_len == 5
    assert state.logical_context_len == 8


def test_pim_host_backend_preserves_compressed_gqa_kv_for_resident_store():
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 3
    seq_len = 5
    initial_kv = [
        {
            "key": torch.randn(seq_len, num_kv_heads, head_dim),
            "value": torch.randn(seq_len, num_kv_heads, head_dim),
            "num_query_heads": num_query_heads,
            "num_key_value_heads": num_kv_heads,
        }
    ]
    query = torch.randn(1, num_query_heads, head_dim)
    key = torch.randn(1, num_kv_heads, head_dim)
    value = torch.randn(1, num_kv_heads, head_dim)

    backend = _NoSmokePimBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=3,
    )
    backend.init_request("req", initial_kv)
    state = backend.request_states["req"]

    assert state.layer_states[0].num_heads == num_query_heads
    assert state.layer_states[0].kv_heads == num_kv_heads
    assert sum(group.slot_group_heads for group in state.layer_states[0].head_groups) == num_kv_heads

    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.5)
    expected = _manual_gqa_decode(initial_kv, query, key, value, score_scale=0.5, window=3)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clover_host_backend_gqa_fused_path_matches_reference():
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 3
    seq_len = 6
    initial_kv = [
        {
            "key": torch.randn(seq_len, num_kv_heads, head_dim),
            "value": torch.randn(seq_len, num_kv_heads, head_dim),
            "num_query_heads": num_query_heads,
            "num_key_value_heads": num_kv_heads,
        }
    ]
    query = torch.randn(1, num_query_heads, head_dim)
    key = torch.randn(1, num_kv_heads, head_dim)
    value = torch.randn(1, num_kv_heads, head_dim)

    backend = _NoSmokeCloverBackend(
        resident_store_backend="host",
        qk_full_enabled=True,
        softmax_av_fused_enabled=True,
        attention_sparse_window=4,
        shadow_checks_enabled=True,
        shadow_check_token_interval=1,
        shadow_check_layer_interval=1,
    )
    backend.init_request("req", initial_kv)
    actual = backend.decode_layer("req", 0, query, key, value, score_scale=0.25)
    expected = _manual_gqa_decode(initial_kv, query, key, value, score_scale=0.25, window=4)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
