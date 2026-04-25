import torch

from src.core.resident_kv_store import UpmemKVSlotStore


def main() -> None:
    repo_root = "/home/cml/CloverInfer"
    torch.manual_seed(0)
    for kv_dtype in ["fp32", "fp16"]:
        store = UpmemKVSlotStore(repo_root, num_dpus=1, kv_dtype=kv_dtype)
        seq_len = 7
        heads = 3
        head_dim = 10
        k = torch.randn(seq_len, heads, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, heads, head_dim, dtype=torch.float32)
        w = torch.softmax(torch.randn(heads, seq_len, dtype=torch.float32), dim=-1)
        store.allocate_group("k", "v", k, v, capacity=16, preferred_dpu=0)
        out = store.weighted_value_sum("k", "v", w)
        ref = torch.einsum("hl,lhd->hd", w, v.float()).contiguous()
        diff = float(torch.max(torch.abs(out - ref)).item())
        print(kv_dtype, diff)
        store.free_group("k", "v")


if __name__ == "__main__":
    main()
