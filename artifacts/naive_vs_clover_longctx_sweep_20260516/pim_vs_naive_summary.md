| prompt | out | baseline | tok/s | speedup vs Naive | latency s | TPOT s | DPUs | KV | grouping | placement | fused | QK launches s | DPU items | host fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 8 | Naive PIM | 0.0994 | 1.00x | 80.36 | 10.61 | 32 | int8 | coarse | load_aware |  | 52.71 | 4704 | 0 |
| 2048 | 8 | CloverInfer | 0.0694 | 0.70x | 114.79 | 15.53 | 32 | int8 | balanced | rotated | True | 88.22 | 11718 | 1386 |
| 4096 | 8 | Naive PIM | 0.0833 | 1.00x | 95.94 | 12.04 | 32 | int8 | coarse | load_aware |  | 55.35 | 3465 | 63 |
| 4096 | 8 | CloverInfer | 0.1153 | 1.38x | 68.96 | 8.17 | 32 | int8 | balanced | rotated | True | 34.14 | 13328 | 1904 |

Target status: PASS for at least one matched `CloverInfer > Naive PIM` case.
