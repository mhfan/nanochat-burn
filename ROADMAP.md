# nanochat-burn Roadmap

`nanochat-burn` 的目标是成为一个可信、可复现、容易读懂的 Rust mini-LLM 学习项目。
项目优先保证数学正确性、实验闭环和教学价值，再逐步推进 GPU 性能与服务化能力。

## 开发原则

- 每项能力明确标记为参考实现、实验实现或性能实现。
- 所有性能结论必须附带可复现基准，不以实现名称代替性能证据。
- 所有数学 parity 结论必须由 Python/Rust fixture 测试支持。
- 每个训练阶段都必须能保存、加载并恢复完整实验状态。
- 保留清晰的参考路径，再为热点增加后端特化路径和 parity 测试。
- WGPU 是主要加速后端，NdArray 是可移植性和快速验证后端。

## M0：项目可信度与文档基线

- [x] 在 README 中增加能力状态表：`Stable`、`Reference`、`Experimental`、`Planned`。
- [x] 校准测试数量、feature 参数、后端说明和已知限制。
- [x] 区分 blocked KV cache 与完整 PagedAttention。
- [x] 区分 group-normalized REINFORCE 与完整 GRPO。
- [x] 将 Python 子进程执行器标记为受限执行器，而不是安全隔离边界。
- [x] 为所有公开运行命令提供最小输入、输出产物和预期结果。

验收条件：README 中的每项能力都能指向实现、测试或明确的计划项。

## M1：可复现实验与三阶段训练闭环

- [x] 引入统一实验配置，替代训练入口中的硬编码超参数。
- [x] 定义标准 artifact 目录和 manifest。
- [x] 保存模型配置、tokenizer、模型权重和训练阶段。
- [x] 保存并追加结构化训练指标。
- [x] 保存 optimizer、trainer 与 dataloader 状态，支持预训练精确断点续训。
- [x] 在引入 dropout、随机采样训练或数据增强时保存并恢复对应随机状态。
- [x] Pretrain 输出可被 SFT 加载。
- [x] SFT 输出可被 RL 加载。
- [x] Eval、CLI Chat 和 Web Chat 加载同一 artifact。
- [x] 提供一个 TinyStories 或小型文本的端到端 recipe。

标准产物布局：

```text
runs/<run-name>/
├── manifest.json
├── experiment.toml
├── config.json
├── tokenizer.json
├── model.safetensors
├── optimizer.safetensors
├── trainer-state.json
└── metrics.jsonl
```

验收条件：一个命令可完成 tokenizer → pretrain → SFT → eval，进程中断后可恢复且结果一致。

## M2：数值 Parity 与正确性证据

- [x] 从 Python nanochat 导出固定输入和参数 fixtures。
- [x] 验证 tokenizer token IDs 与 conversation rendering。
- [x] 验证 attention、MLP、RoPE、RMSNorm 单元输出。
- [x] 验证完整 logits、loss 和参数梯度。
- [x] 验证 optimizer Muon/AdamW 单步参数更新。
- [x] 验证 full forward、chunked cache 和逐 token cache 一致性。
- [x] 建立 f32、f16、W8、W4 的明确误差预算。
- [x] 自动生成 parity 报告。

验收条件：README 中的 parity 声明均由可重复运行的测试和误差表支持。

## M3：教学体验与消融实验

- [x] 编写 tokenizer 章节。
- [ ] 编写数据章节。
- [ ] 编写模型章节。
- [ ] 编写优化器章节。
- [ ] 编写训练章节。
- [ ] 编写推理章节。
- [ ] 编写对齐章节。
- [ ] 每章包含公式、源码入口、最小实验和常见错误。
- [x] 支持 ReLU2、QK Norm、GQA、SWA、Smear、Backout 开关。
- [x] 支持 Muon 与 AdamW 对照实验。
- [x] 生成 loss、BPB、tokens/s、内存占用和模型质量报告。
- [x] 增加 tiny overfit 集成测试。
- [x] 增加 resume equivalence 集成测试。
- [x] 增加 deterministic decode 集成测试。

验收条件：读者可通过一组小实验观察每项架构设计对数值和训练的影响。

## M4：推理性能基线

- [x] 保留 CPU `ReferenceSampler`，增加设备端 `DeviceSampler`。
- [x] 避免每个 token 回传完整 vocabulary logits。
- [x] 建立 prefill/decode 分离基准。
- [x] 记录首 token 延迟、tokens/s、显存和 batch scaling。
- [x] 为量化记录模型大小、误差和吞吐收益。
- [x] 为 speculative decoding 记录 acceptance rate 和真实加速比。
- [x] 实现 draft KV cache truncate/rollback，移除全量重建。

验收条件：WGPU 性能路径和参考路径数值一致，且性能提升由基准证明。

## M5：真正的分页缓存与并发调度

- [x] 抽象连续 KV cache 与分页 KV cache 的公共接口。
- [x] 实现 page allocator、free list 和请求级 block table。
- [x] attention 直接消费分页 KV，不在每步重构连续张量。
- [x] 支持请求动态加入、完成、取消和页回收。
- [x] 实现 iteration-level continuous batching。
- [x] 增加 StreamingLLM attention sinks 与页面驱逐实验。

验收条件：多请求负载下页面能复用和回收，并优于静态 batch 基线。

## M6：对齐算法

- [x] 将现有算法明确为 group-normalized REINFORCE。
- [x] 保存 rollout token log-probs 和生成策略版本。
- [x] 实现 reference model KL penalty。
- [x] 实现 old-policy ratio 和 clipped GRPO objective。
- [x] 增加 reward、KL、clip fraction 和 response length 指标。
- [x] 对比 SFT、REINFORCE 和 GRPO 的小规模实验。

验收条件：目标函数、日志指标和测试能够区分 REINFORCE 与 GRPO。

## M7：GPU 算子优化

- [x] 基于可复现的 `bench_ops` profiling 选择 CubeCL 融合热点。
- [x] 实验 RMSNorm、RoPE 和 Softmax；无稳定收益的算子保留 Burn Fusion 泛型实现。
- [x] 接入 Burn/CubeCL attention，并评估原生 causal、显式 mask 与 Flash/autotune 路径。

验收条件：每个实际采用的特化算子均有参考实现、数值 parity 测试和端到端性能数据。结果与
复现命令见 [`docs/gpu-operators.md`](docs/gpu-operators.md)。

## 发布标准

### v0.2

- M0 完成。
- M1 支持统一 artifact 和阶段加载。
- 提供一个可复现的端到端小模型实验。

### v0.5

- M2、M3 完成。
- CPU 与 WGPU 均有持续集成验证。
- 文档包含完整学习路径和消融报告。

### v1.0

- M4、M5 的核心性能路径完成。
- M6 至少提供一个经过验证的完整 GRPO 实验。
- 所有 Stable 能力都有测试、文档和可复现基准。
