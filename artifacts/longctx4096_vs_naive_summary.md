| baseline | tok/s | speedup vs Naive | latency s | TPOT s | DPUs | KV | grouping | placement | fused | QK launches s | DPU items | host fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive PIM | 0.0830 | 1.00x | 96.30 | 12.12 | 32 | int8 | coarse | load_aware |  | 55.35 | 3465 | 63 |
| CloverInfer | 0.1176 | 1.42x | 67.65 | 8.08 | 32 | int8 | balanced | rotated | True | 34.14 | 13328 | 1904 |

Target status: PASS for `CloverInfer > Naive PIM`.
