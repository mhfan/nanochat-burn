# 推理：采样、分页 KV cache 与推测解码

推理路径把 prompt prefill 与逐 token decode 分开。`ReferenceSampler` 保留易读的 CPU 语义，
`DeviceSampler` 避免每步把完整 vocabulary logits 回传 CPU；两者由确定性测试约束。

## 自回归与采样契约

给定历史 token，模型产生下一 token logits `z`。repetition penalty 对历史中每个不同 token 应用：

```math
z'_i=\begin{cases}
z_i/r,&z_i>0\\
z_i\,r,&z_i\le 0
\end{cases}
```

其中 `r > 0`。`temperature = 0` 使用 argmax；否则：

```math
p_i=\operatorname{softmax}(z'_i/T),
```

可先只保留 top-k logits，再用有种子的 RNG 从 CDF 采样。配置拒绝负温度、`top_k = 0` 和非正
repetition penalty。生成长度还受模型 `sequence_len` 限制。

DeviceSampler 只把选中的 token 小向量传回 host；ReferenceSampler 会读取 logits，主要用于检查
语义，不应作为 GPU 性能基线。

## KV cache 与分页 attention

prefill 一次写入 prompt 的 K/V，decode 每步只计算新 token。分页 cache 包含：

- 全局 K/V page pools 与 free list；
- 每个 request 的 block table；
- request slot 与逻辑位置映射；
- attention 逐页执行 online softmax，不把历史页重新拼成连续张量。

请求完成或取消时必须释放页面。continuous scheduler 可在迭代边界接纳新请求、完成或取消旧请求，
并复用 slot/page。attention sink 实验保留开头页面并驱逐中间历史。

## 推测解码

greedy 模式中 draft 连续提出 K 个 token，target 一次验证该块；接受共同前缀，在首个分歧处采用
target token，并 truncate/rollback cache。接受率为：

```math
\text{acceptance rate}=\frac{\text{accepted draft tokens}}
{\text{proposed draft tokens}}.
```

当前实现只在 `temperature = 0` 时提供数学无损的 speculative 路径。随机采样会回退 target 模型，
没有实现概率修正版本；这是明确的正确性边界。

## 源码入口

- `src/engine/inference.rs`：采样配置、Reference/Device sampler、prefill 与 decode。
- `src/gpt/cache.rs`：page allocator、block table、分页 K/V 与 online-softmax attention。
- `src/engine/scheduler.rs`：continuous batching admission/completion/cancel 状态机。
- `src/engine/serving.rs`：动态 batch、request slot、位置映射与页面回收。
- `src/engine/speculative.rs`：draft/target 验证、cache rollback 与统计。
- `src/gpt/quant.rs`：W8/W4 weight-only 推理。
- `src/benchmark.rs`、`src/bin/bench/`：可复现推理、推测解码与算子基准。

## 最小实验

验证同一 seed 的随机 decode 可重复，以及设备/参考 greedy sampler 一致：

```bash
cargo test engine::inference::tests::test_seeded_decode_is_deterministic
cargo test engine::inference::tests::test_device_and_reference_greedy_samplers_match
```

验证 full/cached forward、分页 attention 和页面复用：

```bash
cargo test gpt::tests::test_cached_forward_matches_full_and_incremental_forward
cargo test gpt::tests::test_paged_attention_roundtrip
cargo test gpt::tests::test_page_allocator_reuses_released_pages
```

验证 greedy speculative 输出与 target-only 相同：

```bash
cargo test engine::speculative::tests::test_speculative_decoding_lossless
```

对真实 artifact 测吞吐而不是从单元测试计时：

```bash
cargo run --release --no-default-features --features wgpu --bin bench -- infer --artifact runs/sft --batches 1,2,4
cargo run --release --no-default-features --features wgpu --bin bench -- speculative --target runs/sft --draft runs/pretrain
```

结果写入 `runs/benchmarks/`，包含 prefill、TTFT、同步逐步测得的 median TPOT、异步流水的 decode
tokens/s、batch scaling、cache 字节数和设备 allocator 峰值。同一 prompt 的多样本请求只运行一次
prefill，随后复制到物理独立的 KV 页，避免重复 Transformer 计算且允许各样本安全分叉。Flex
没有设备 allocator，对应字段为 `null`。

## 常见错误

- 将 `temperature = 0` 代入除法。实现把它定义为独立的 argmax 分支。
- 认为 repetition penalty 会按重复次数多次应用。历史 token 先去重。
- 把 SWA、KV cache 和 PagedAttention 当作同一件事：它们分别控制可见性、复用计算和存储布局。
- 随机温度下仍宣称 speculative 加速。当前会正确回退 target-only。
- 取消请求却绕过 serving 的 release 路径，会泄漏 page/slot。
- 用 Flex 进程 RSS 冒充显存。只有 CubeCL allocator 字段代表设备内存。
- 只报告 acceptance rate，不报告 target 调用、TTFT 和端到端 speedup。

## 能力边界

DeviceSampler 减少 host 传输，但实际收益依赖设备、词表、batch 和后端融合。分页 cache 的测试验证
分配、数值和回收；是否优于连续 cache 必须由目标负载基准决定。Python 工具调用输出也会进入 token
队列，但其执行限制见对齐章节，不能视为安全隔离。
