# 训练：预训练、SFT 与精确恢复

训练层把模型、数据、optimizer、调度器和 artifact 组合成可复现实验。预训练与 SFT 都使用
next-token loss，但 batch 组织、target mask 和恢复边界不同。

## 损失与梯度累积

对有效 target 的平均交叉熵为：

```math
L=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_{\le i}).
```

预训练的每个 optimizer step 包含：

```math
K=\frac{\text{total\_batch\_size}}{\text{device\_batch\_size}}
```

个 micro-batch。每次 backward 前先用 `L/K` 缩放，Burn 的 `GradientsAccumulator` 累加梯度，
第 K 次后才更新参数。配置要求两种 batch size 可整除。当前 SFT 循环每步使用一个
`device_batch_size`，不经过这条梯度累积路径。

SFT 将 user、system、tool 和 padding 位置 target 设为忽略值，只对 assistant token 计算同一损失。

## 训练调度

学习率先线性 warmup，然后保持常数，最后线性 warmdown 到 `final_lr_frac * learning_rate`。
Muon momentum 在前 400 step 从 0.85 升到 0.97，末段降到 0.90。weight decay 使用余弦调度：

```math
\lambda_t=\frac{\lambda_0}{2}\left(1+\cos\frac{\pi t}{N}\right).
```

BPB（bits per byte）按文本字节而非 token 数归一化：

```math
\operatorname{BPB}=\frac{\text{total negative log-likelihood}}
{\ln 2\cdot\text{total text bytes}}.
```

特殊 token 的字节长度视为 0，忽略 target 不进入分子或分母。

## 三阶段 artifact 链路

```text
text -> tokenizer + pretrain -> SFT -> RL -> eval/chat/report
```

统一 artifact 至少包含 manifest、实际 experiment TOML、模型配置、tokenizer、模型权重和 metrics。
Pretrain 额外保存 optimizer、trainer 与 dataloader 位置，能够精确恢复下一步。RL 也保存 optimizer、
trainer、采样 RNG 与 rollout 日志。SFT 当前输出可供 RL 加载，但不承诺 optimizer 级精确续训。

预训练恢复由 `NANOCHAT_RESUME_ARTIFACT` 指定。`NANOCHAT_CHECKPOINT_INTERVAL` 控制 checkpoint
间隔，`NANOCHAT_OUTPUT_ARTIFACT` 可把恢复结果写到新目录并继承历史 metrics。

## 源码入口

- `src/engine.rs`：`TrainingConfig`、梯度累积、调度器、BPB 和训练集成测试。
- `src/engine/pretrain.rs`：tokenizer、二进制数据、checkpoint 与恢复工作流。
- `src/engine/sft.rs`：conversation packing 与 assistant-only targets。
- `src/engine/recipe.rs`：tokenizer → pretrain → SFT → eval 的离线 tiny recipe。
- `src/artifact.rs`：统一产物、manifest 和 resume state。
- `src/experiment.rs`：强类型 TOML、默认值和跨字段校验。
- `src/bin/train.rs`、`src/bin/report.rs`：命令入口和实验汇总。

## 最小实验

确认一个极小语料能被模型记住，loss 确实下降：

```bash
cargo test engine::tests::test_tiny_corpus_overfit
```

确认中断后恢复与不中断训练的 step、数据位置和 logits 一致：

```bash
cargo test engine::tests::test_training_resume_equivalence
```

一条命令运行完整离线链路：

```bash
cargo run --no-default-features --features ndarray --bin train -- --recipe --config configs/tiny.toml
```

输入位于 `data/fixtures/tiny/`，产物写入 `runs/tiny/`。一层模型和少量 step 只验证链路，评测分数
不代表能力。查看实验指标：

```bash
cargo run --bin report -- runs/tiny/pretrain runs/tiny/sft
```

## 正确性证据与观察点

- overfit 测试证明参数、loss 和 optimizer 路径形成有效闭环。
- resume equivalence 同时覆盖 model、optimizer、trainer 和 dataloader，而不只比较 step 数。
- experiment 配置和 manifest 随 artifact 保存，可追溯模型形状、阶段和输入关系。
- metrics JSONL 记录 loss、BPB、吞吐、内存和阶段指标，`report` 统一汇总。
- Python/Rust fixture 另行约束模型与 optimizer 数学；端到端 loss 下降不能代替 parity。

## 常见错误

- 把预训练的 `total_batch_size` 当作单设备张量第一维。它决定累积后的有效 batch；当前 SFT
  每步实际 batch 是 `device_batch_size`。
- 忘记把 loss 除以累积步数，导致有效学习率随 K 成比例放大。
- 用新的 tokenizer 加载旧模型或旧 `.bin`。artifact 的 tokenizer 是模型语义的一部分。
- 只复制 `model.safetensors` 继续训练，却称为相同轨迹。
- 让 SFT 对 prompt/padding 求 loss，会把模型训练成复述用户和 padding。
- 从 tiny recipe 的最终分数推断真实模型质量；它是集成测试，不是 benchmark。
- 在同一输出目录做对照实验，覆盖配置和 metrics。

## 能力边界

精确恢复目前由 Pretrain 和 RL 明确实现；SFT 保证阶段产物可加载，不保证任意中断点的 optimizer
轨迹恢复。多设备分片规则已经定义，但改变 world size 后不能沿用原 rank 的 dataloader 位置。
