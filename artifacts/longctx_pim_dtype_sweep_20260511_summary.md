# Long Context PIM Dtype Sweep, 2026-05-11

Purpose: compare CPU-Attention against true CloverInfer PIM with resident KV stored as fp32 vs fp16, and test whether fp16 helps or hurts under longer context.

## Results

| Prompt tokens | Baseline | KV dtype | Limit/concurrency | Output throughput | Avg latency | Avg TPOT | Avg TTFT | Avg attention decode compute |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | CPU-Attention | - | 4/2 | 3.369264 tok/s | 2.274921 s | 0.597023 s | 0.483852 s | 0.112308 s |
| 512 | CloverInfer PIM | fp32 | 4/2 | 0.705600 tok/s | 10.712299 s | 3.411971 s | 0.476387 s | 3.751405 s |
| 512 | CloverInfer PIM | fp16 | 4/2 | 0.662280 tok/s | 11.312821 s | 3.609099 s | 0.485524 s | 4.303735 s |
| 1024 | CPU-Attention | - | 2/1 | 1.818834 tok/s | 2.185261 s | 0.489330 s | 0.717272 s | 0.205974 s |
| 1024 | CloverInfer PIM | fp32 | 2/1 | 0.399787 tok/s | 9.795473 s | 3.033848 s | 0.693929 s | 6.132860 s |
| 1024 | CloverInfer PIM | fp16 | 2/1 | 0.416790 tok/s | 9.405777 s | 2.901931 s | 0.699984 s | 6.218936 s |

## Artifacts

- `artifacts/longctx512_cpu_attention_limit4_c2_tok4_20260511.jsonl`
- `artifacts/longctx512_clover_pim_fp32_limit4_c2_tok4_20260511.jsonl`
- `artifacts/longctx512_clover_pim_fp16_limit4_c2_tok4_20260511.jsonl`
- `artifacts/longctx1024_cpu_attention_limit2_c1_tok4_20260511.jsonl`
- `artifacts/longctx1024_clover_pim_fp32_limit2_c1_tok4_20260511.jsonl`
- `artifacts/longctx1024_clover_pim_fp16_limit2_c1_tok4_20260511.jsonl`

## Interpretation

- At 512 prompt tokens, CPU-Attention is 3.369264 tok/s, CloverInfer PIM fp32 is 0.705600 tok/s, and CloverInfer PIM fp16 is 0.662280 tok/s. fp16 is slower than fp32 in this setting.
- At 1024 prompt tokens, CPU-Attention is 1.818834 tok/s, CloverInfer PIM fp32 is 0.399787 tok/s, and CloverInfer PIM fp16 is 0.416790 tok/s. fp16 is slightly faster than fp32, but still far below CPU-Attention.
- The fp16 path stores resident KV more compactly, but DPU compute converts fp16 values to float in software before accumulation, so fp16 is not a true half-precision acceleration path on UPMEM.
- These data support moving the next PIM optimization toward integer or fixed-point KV/QK kernels rather than expecting fp16 alone to win.
