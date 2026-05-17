# Motivation 1 Imbalance Range Sweep

Main metric: `straggler_ratio = max_active_dpu_tokens / avg_active_dpu_tokens`.
Tail proxy: `max_active_dpu_tokens`; smaller is better.

| D | H | D/H | long/short | naive straggler | naive tail | tail reduction vs Clover | straggler reduction vs Clover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 4 | 4 | 8 | 2.909 | 8192 | 2.909 | 2.909 |
| 16 | 4 | 4 | 32 | 3.657 | 32768 | 3.657 | 3.657 |
| 16 | 4 | 4 | 128 | 3.908 | 131072 | 3.908 | 3.908 |
| 32 | 4 | 8 | 8 | 4.267 | 8192 | 4.267 | 4.267 |
| 32 | 4 | 8 | 32 | 6.564 | 32768 | 6.564 | 6.564 |
| 32 | 4 | 8 | 128 | 7.585 | 131072 | 7.585 | 7.585 |
| 64 | 4 | 16 | 8 | 5.565 | 8192 | 5.565 | 5.565 |
| 64 | 4 | 16 | 32 | 10.894 | 32768 | 10.894 | 10.894 |
| 64 | 4 | 16 | 128 | 14.322 | 131072 | 14.322 | 14.322 |
