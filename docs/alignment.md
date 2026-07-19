# 对齐：组内归一化 REINFORCE 与 GRPO

RL 阶段从 SFT artifact 加载 policy，冻结一份 reference model，并对每个问题采样一组回答。实现同时
支持无 ratio clip 的 group-normalized REINFORCE 和带 old-policy ratio 的 clipped GRPO；两者共享
rollout、优势和 KL 统计。

## Rollout 契约

每条 `RolloutRecord` 保存训练 step、policy version、完整 tokens、sampled mask、old-policy token
log-probs、reward 和 advantage。只有模型实际采样的 assistant token mask 为 1；prompt、强制 token
和 tool 输出不参与策略 loss。

当前数学任务 reward 从 assistant answer 提取整数并做 exact match：正确为 1，错误为 0。对同一
问题的 G 个回答，使用总体方差计算组内优势：

```math
\mu=\frac1G\sum_i r_i,\qquad
\sigma^2=\frac1G\sum_i(r_i-\mu)^2,
```

```math
A_i=\frac{r_i-\mu}{\sqrt{\sigma^2}+10^{-4}}.
```

组内 reward 全相同时优势接近 0，因此该问题不提供方向性策略信号。

## 两种目标函数

old-policy ratio 为：

```math
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t|s_t)
-\log\pi_{old}(a_t|s_t)\right).
```

group-normalized REINFORCE 使用当前 log-prob 与组内优势的乘积。GRPO 使用 clipped surrogate：

```math
J_t=\min\left(\rho_tA,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A\right).
```

reference KL 使用非负估计器。令 `x = log p_ref - log p_policy`：

```math
D_{KL}^{est}=e^x-x-1.
```

最终最小化“负策略目标 + `kl_coeff` × 有效采样 token 的平均 KL”。日志同时记录 reward、KL、clip
fraction 和 response length，使两种算法不会只靠配置名称区分。

## 恢复与执行器边界

RL checkpoint 保存 model、optimizer、trainer step、采样 RNG、耗时和 rollout JSONL，因此同一配置
可以恢复采样序列。prompt 过长时从左侧截断，为 generation budget 保留空间。

代码题评测可启动受限 Python 子进程，设置超时和输出上限。这只能限制资源与挂起风险，不提供
容器、权限或系统调用隔离，不能运行不可信代码。

## 源码入口

- `src/engine/rl.rs`：rollout、group advantage、REINFORCE/GRPO、KL、指标与恢复。
- `src/engine/inference.rs`：生成与 old-policy token log-prob 来源。
- `src/engine/sandbox.rs`：带超时/输出限制的 Python 子进程。
- `src/engine/eval.rs`：categorical、generative 与 HumanEval 风格评测。
- `src/artifact.rs`：RL model/optimizer/trainer 状态和 rollout 文件声明。
- `src/bin/train.rs`、`src/bin/eval.rs`、`src/bin/report.rs`：训练、评测与对照报告入口。

## 最小实验

验证组内优势均值、ratio clip 和两种 objective：

```bash
cargo test engine::rl::tests::test_group_advantages_and_clipped_objective
```

验证执行器成功、异常、大输出和超时路径：

```bash
cargo test engine::sandbox::tests
```

在同一 SFT artifact 上做控制变量对照：

```bash
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/reinforce cargo run --bin train -- --rl \
  --rl-algorithm group_normalized_reinforce
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/grpo cargo run --bin train -- --rl \
  --rl-algorithm grpo
```

分别评测后汇总：

```bash
NANOCHAT_ARTIFACT=runs/compare/reinforce cargo run --bin eval
NANOCHAT_ARTIFACT=runs/compare/grpo cargo run --bin eval
cargo run --bin report -- runs/sft runs/compare/reinforce runs/compare/grpo
```

比较时至少查看 reward、KL、clip fraction、response length 和任务质量；单看训练 loss 无法说明策略
是否更好。

## 正确性证据与观察点

- objective 单元测试覆盖正/负优势时 clip 的方向，避免 `min` 在负优势下写反。
- rollout 显式保存 old log-probs 与 policy version，ratio 不会误用更新后的 policy 作为 old policy。
- sampled mask 确保 tool/forced token 不产生策略梯度。
- reference model 在 RL 阶段冻结，只用于 KL，不被 optimizer 收集。
- resume state 包含采样 RNG；否则即使参数和 step 相同，下一组 rollout 也会变化。

## 常见错误

- 把“组内归一化”直接等同于 GRPO。没有 old-policy ratio/clip 的路径是 REINFORCE 变体。
- 用 sample variance（分母 G-1）复算优势。实现使用 population variance（分母 G）。
- 把 NLL 当作 log-prob。训练内部先得到 NLL，再取负值形成 `log p`。
- 让 prompt、calculator 输出或 padding 参与 objective。
- reference model 与 old policy 混为一谈：前者用于 KL，后者用于 ratio。
- 将受限 subprocess 称为安全 sandbox，或用它执行来自互联网的任意代码。
- reward 全相同仍期待组内优势产生学习信号。

## 能力边界

当前 0/1 exact-match reward 适合演示可验证数学回答，不是通用偏好模型。GRPO 实现具备 ratio、clip、
reference KL 和可观察指标，但小规模结果不代表生产对齐效果；任何算法结论都应附带 SFT 基线、固定
配置、rollout 规模、随机种子和评测结果。
