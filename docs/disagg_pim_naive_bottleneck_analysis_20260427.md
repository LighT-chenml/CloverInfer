# `disagg_pim_naive` 性能瓶颈分析与 CloverInfer 优化路线

Date: 2026-04-27

## 目的

本文面向下一步的 CloverInfer 设计，先回答两个问题：

1. 当前仓库里的 `disagg_pim_naive` 为什么慢
2. 后续优化应该优先改哪里

这里不再停留在“PIM kernel 还很 naive”这个抽象判断，而是结合当前代码路径，定位端到端延迟真正花在什么地方。

## 现有基线结论

从已有文档看，`disagg_pim_naive` 已经从最初约 `5.764s` 降到约 `2.425s`，明显改善，但仍慢于 `disagg_cpu`，更远慢于 `split_gpu_full_decode` 和 `monolithic_gpu`。

参考：

- [docs/qwen_baseline_refresh_after_stdio_20260424.md](/home/cml/CloverInfer/docs/qwen_baseline_refresh_after_stdio_20260424.md)
- [docs/pim_challenge_analysis_20260424.md](/home/cml/CloverInfer/docs/pim_challenge_analysis_20260424.md)
- [docs/pim_baseline_findings_20260425.md](/home/cml/CloverInfer/docs/pim_baseline_findings_20260425.md)

这说明当前版本的主要问题已经不只是“子进程太慢”一项，而是整个远端 attention 路径仍然有大量固定开销、重复数据路径和容量压力。

## 当前真实执行路径

### 1. Scheduler 侧 decode 编排仍然非常细粒度

`GlobalScheduler.submit_request()` 在每个 decode step、每一层里，都会顺序执行：

1. `dense.start_token`
2. `dense.prepare_attention`
3. `attention.decode_layer`
4. `dense.finish_layer`
5. 最后 `dense.sample_next_token`

对应代码见：

- [src/core/scheduler.py#L376](/home/cml/CloverInfer/src/core/scheduler.py#L376)
- [src/core/scheduler.py#L425](/home/cml/CloverInfer/src/core/scheduler.py#L425)

这意味着即使 attention backend 本身更快，系统仍然要承受：

- 每 token 每 layer 的 `.4 -> .7 -> .4` 往返
- Ray actor RPC
- 张量序列化 / 反序列化
- 调度等待和事件循环开销

因此 `disagg_cpu` 本身就已经比 `split_gpu_full_decode` 慢很多，这部分是 `disagg_pim_naive` 的地基成本。

### 2. AttentionNode 虽然支持 batching，但默认 batching 很轻

当前配置默认值：

- `attention_rpc_batch_window_s = 0.001`
- `attention_actor_batch_window_s = 0.001`
- `*_max_size = 8`

见：

- [src/core/config.py#L38](/home/cml/CloverInfer/src/core/config.py#L38)

这说明系统具备“攒批”机制，但默认仍是毫秒级小窗口，且只有单 attention node。对于低并发或短 decode，批次往往不够大，无法真正摊薄固定成本。

### 3. `pim_naive` fast path 里仍然保留了大量 host 路径

`PimNaiveAttentionBackend.decode_layer_batch()` 的主路径见：

- [src/core/attention_backend.py#L1137](/home/cml/CloverInfer/src/core/attention_backend.py#L1137)

每个 item 在 `_prepare_decode_record()` 里都会先：

- 把 `query/key/value` 拉到 CPU 并 `contiguous`
- 追加 resident KV
- 同时也追加一份 CPU shadow KV

见：

- [src/core/attention_backend.py#L745](/home/cml/CloverInfer/src/core/attention_backend.py#L745)
- [src/core/attention_backend.py#L763](/home/cml/CloverInfer/src/core/attention_backend.py#L763)
- [src/core/attention_backend.py#L783](/home/cml/CloverInfer/src/core/attention_backend.py#L783)
- [src/core/attention_backend.py#L785](/home/cml/CloverInfer/src/core/attention_backend.py#L785)

这意味着当前 backend 并不是“PIM 算 attention，host 只管调度”，而是：

- resident store 保持一份 KV
- CPU backend 再保持一份 KV
- 某些路径还要把 resident KV materialize 回 host

这会造成明显的重复内存流量与重复状态维护。

## 核心瓶颈

### 瓶颈 1: 远端 attention 的 RPC 频率过高

这是当前最大的系统级瓶颈。

证据：

- scheduler 在每层都同步调用 `prepare_attention`、`attention.decode_layer`、`finish_layer`
- 每个 decode token 都重复整套流程
- 这部分成本在 `disagg_cpu` 中就已经存在

代码位置：

- [src/core/scheduler.py#L409](/home/cml/CloverInfer/src/core/scheduler.py#L409)
- [src/core/scheduler.py#L428](/home/cml/CloverInfer/src/core/scheduler.py#L428)
- [src/core/scheduler.py#L435](/home/cml/CloverInfer/src/core/scheduler.py#L435)
- [src/core/scheduler.py#L448](/home/cml/CloverInfer/src/core/scheduler.py#L448)

为什么重要：

- 这部分与 PIM 无关，但它决定了 `disagg_pim_naive` 的性能天花板
- 如果不先降低 call granularity，任何 PIM kernel 提升都会被 RPC 税吞掉一大块

结论：

- CloverInfer 不能只优化 attention backend
- 必须同时优化 decode choreography

### 瓶颈 2: `pim_naive` 仍有大量 CPU shadow 和重复数据路径

在 `_prepare_decode_record()` 中，当前实现把新 token 的 KV 同时写到：

- resident store
- `cpu_backend.k_cache / v_cache`

并且在非 resident AV 路径下，还会把 resident KV materialize 回 host：

- [src/core/attention_backend.py#L792](/home/cml/CloverInfer/src/core/attention_backend.py#L792)
- [src/core/attention_backend.py#L793](/home/cml/CloverInfer/src/core/attention_backend.py#L793)
- [src/core/attention_backend.py#L794](/home/cml/CloverInfer/src/core/attention_backend.py#L794)

这使得当前 fast path 里混入了三类额外开销：

- GPU/actor 输出转 CPU
- resident append 之外再做 CPU append
- resident readback / host-side score 或 context 校验

为什么重要：

- 这类开销与 context 变长后会持续累积
- 即使 DPU 端计算完全免费，host 端仍会很重

结论：

- CloverInfer 的第一条原则应是“fast path 去 shadow 化”
- correctness oracle 需要保留，但不能与生产路径共用同一条热路径

### 瓶颈 3: PIM 调用粒度仍然偏向 group，而不是 layer/request

当前实现已经不再是最早的纯逐 head 调用，但主粒度仍是 `head_group`。

代码位置：

- head group 构造：[src/core/attention_backend.py#L320](/home/cml/CloverInfer/src/core/attention_backend.py#L320)
- full-QK 批处理：[src/core/attention_backend.py#L832](/home/cml/CloverInfer/src/core/attention_backend.py#L832)
- fused QK+softmax+AV：[src/core/attention_backend.py#L882](/home/cml/CloverInfer/src/core/attention_backend.py#L882)

虽然 `_effective_head_group_count()` 已经尝试避免把 layer 切得太碎，但真实执行仍然是：

- 每条请求
- 每一层
- 每个 resident head group

都要组装 slot query、拼接结果、再在 host 端重组。

这会留下两类固定成本：

- control path 成本仍按 group 付费
- host 端仍要做 group-level metadata 管理和结果拼接

结论：

- CloverInfer 应该把目标从“group-aware implementation”
- 提升到“layer-resident execution”

### 瓶颈 4: helper/stdin-stdout 协议仍然是主热路径的一部分

当前 `UpmemKVSlotStore` 的 DPU 操作通过 `_KVSlotHelperClient` 走持久化 helper 进程，所有 batch 请求都经由 `stdin/stdout` 二进制协议发送：

- QK batch
- AV batch
- softmax+AV batch
- QK+softmax+AV batch

代码位置：

- helper client：[src/core/resident_kv_store.py#L82](/home/cml/CloverInfer/src/core/resident_kv_store.py#L82)
- slot QK batch：[src/core/resident_kv_store.py#L1055](/home/cml/CloverInfer/src/core/resident_kv_store.py#L1055)
- AV batch：[src/core/resident_kv_store.py#L1092](/home/cml/CloverInfer/src/core/resident_kv_store.py#L1092)
- fused batch：[src/core/resident_kv_store.py#L1143](/home/cml/CloverInfer/src/core/resident_kv_store.py#L1143)

相比最初的 subprocess/file-heavy 版本，这已经前进了一大步；但它仍然有明显限制：

- query/weight/score/context 仍然要在 host 侧 pack/unpack
- tensor 仍要转 CPU contiguous
- helper 协议仍然是串行边界

结论：

- 当前 helper 方案适合 bring-up 和功能迭代
- 但不应作为 CloverInfer 的最终高性能控制面

### 瓶颈 5: 真实模型下 resident KV 容量仍会触发 fallback

已有结果表明，轻量 workload 上 `4 DPU + fp16` 可以恢复 full-resident path，但在更重的 `Qwen-1_8B` decode 条件下仍然会 overflow，直到 `8 DPU + fp16` 才恢复稳定。

参考：

- [docs/pim_baseline_findings_20260425.md](/home/cml/CloverInfer/docs/pim_baseline_findings_20260425.md)

代码上也能看到当前 resident store 明确支持：

- DPU allocation
- host fallback

见：

- [src/core/resident_kv_store.py#L921](/home/cml/CloverInfer/src/core/resident_kv_store.py#L921)
- [src/core/resident_kv_store.py#L1029](/home/cml/CloverInfer/src/core/resident_kv_store.py#L1029)

为什么重要：

- 一旦 fallback 发生，PIM 路径会退化成混合路径
- 这不仅影响平均延迟，也让结果更难解释

结论：

- CloverInfer 需要把“容量稳定性”当作一等目标
- 不是只有算得快，还要保证 resident path 不频繁失效

## 对 CloverInfer 的总体判断

当前 `disagg_pim_naive` 的主要矛盾不是“QK kernel 再快一点就够了”，而是三层问题叠加：

1. 系统层：远端 attention RPC 太细
2. 后端层：fast path 仍夹带大量 host shadow / materialize
3. 存储层：resident KV 容量和布局还不够稳定

因此 CloverInfer 不应被定义为“更快的 naive PIM backend”，而应被定义为：

- 一个面向 decode 热路径的 coarse-grained remote attention system
- 其中 PIM resident KV 和 PIM-side attention compute 是核心执行资源
- host 只保留必要控制面，而不再承担热路径中的重复数据处理

## 建议的优化顺序

### Stage 0: 先把 profiling 证据补齐

虽然仓库已记录 stage timing，但对 CloverInfer 还不够细。

建议新增的计时项：

- `prepare_decode_record_s`
- `resident_append_s`
- `cpu_shadow_append_s`
- `resident_materialize_s`
- `qk_helper_io_s`
- `av_helper_io_s`
- `group_reassemble_s`

目的不是做最终 profiling 系统，而是先回答：

- `attention_decode_rpc_s` 里到底多少是 helper I/O
- 多少是 host tensor copy
- 多少是真正 DPU compute

这是 CloverInfer 第一周最值得做的基础工作。

### Stage 1: 从 fast path 中拆掉 correctness shadow

优先级很高。

目标：

- 将 correctness check 变成可采样、可旁路的 debug path
- 默认生产路径不再同时维护完整 CPU shadow KV

建议做法：

- 保留 `CpuAttentionBackend` 作为 oracle
- 但只在抽样请求 / 抽样层 / 抽样 step 上运行
- 不再在每次 decode 的热路径里同步执行

如果这一点不做，后续很多优化都会被 shadow 成本污染。

### Stage 2: 提升 attention 执行粒度

这是 CloverInfer 的核心系统改造。

目标不是继续把单层 `decode_layer_batch()` 打磨得更漂亮，而是进一步减少：

- `.4 -> .7`
- `.7 -> .4`

往返次数。

优先方向：

1. 把更多 layer 内逻辑合并到 attention wave 中
2. 将 scheduler 级 batch 与 attention actor 级 batch 统一
3. 让一个 attention RPC 承载更多 request/layer work

如果只能选一个方向，优先减少远端同步次数，而不是先追求更复杂的 kernel。

### Stage 3: 从 group-oriented 走向 layer-resident execution

当前 resident slot 已经是好的基础，但还需要再往前走一步。

目标：

- 对 host 来说，一个 layer 更像一个整体对象
- host 不再频繁关心 group 拼接和 slot row 映射

这会带来两个收益：

- 降低 control path 复杂度
- 为后续更深的 PIM-side fusion 留出接口空间

### Stage 4: 真正把 QK/softmax/AV 融成热路径主实现

当前已经有：

- `qk_full_enabled`
- `softmax_av_fused_enabled`
- `qk_softmax_weighted_value_sum_batch`

这是很好的起点，但现在仍带 shadow 和 host-side fallback 逻辑。

下一阶段的目标应是：

- fused path 成为默认主路径
- shadow path 完全旁路化
- host 不再依赖 materialized scores/weights/context 做主流程

### Stage 5: 处理容量与布局

这部分同样重要，但不建议先于 Stage 1/2。

优先项：

- resident KV `fp16` 作为默认主线配置继续推进
- 扩大 DPU 数量与 KV 压缩共同评估
- 研究更稳定的 reserve policy 和 layout

不建议太早做的事情：

- 一上来改复杂 allocator
- 一上来做不规则 sparse attention
- 一上来围绕 GQA 重写全局设计

这些都值得做，但不是当前最大 bottleneck。

## 推荐的 CloverInfer 第一阶段实现目标

我建议把 CloverInfer 第一阶段定义为下面三个明确目标：

### 目标 1: 建立“无 shadow 污染”的真实快路径

验收标准：

- 默认运行时 CPU shadow 不在每层热路径执行
- correctness 可以单独抽样验证

### 目标 2: 把 attention 远端交互从 layer-by-layer 推向 wave/cohort 粒度

验收标准：

- `attention_decode_rpc_s` 占比明显下降
- 同等 workload 下 `disagg_cpu` 和 `disagg_pim` 都能受益

### 目标 3: 让 fused resident attention 成为主路径

验收标准：

- 默认使用 resident QK + resident softmax/AV
- host 不再频繁 materialize layer KV

## 不建议当前优先投入的方向

短期内不建议把主要时间放在以下方向：

1. 单独继续微调 `qk_mixed_heads` 之类小参数
2. 只做更多 DPU kernel micro-optimization
3. 只做 allocator policy 打磨

原因很简单：

- 这些方向能带来改进
- 但还没有打中当前端到端主瓶颈

## Bottom Line

`disagg_pim_naive` 现在慢，不是因为“PIM 还不够强”，而是因为：

- 系统交互粒度太细
- host 热路径仍承担了太多 shadow / materialize / pack-unpack 工作
- resident KV 容量稳定性还不够强

所以 CloverInfer 的第一目标，不应是“把 naive kernel 再优化一点”，而应是：

- 建立真正 coarse-grained 的 remote attention fast path
- 让 resident attention 成为默认主执行路径
- 让 correctness 和 fallback 退出热路径

如果后续按这个顺序推进，`disagg_pim_naive -> CloverInfer` 的演进逻辑会非常清晰：

1. 先去掉假性开销
2. 再放大真实 PIM 优势
3. 最后再做容量、布局和更高级 attention 设计
