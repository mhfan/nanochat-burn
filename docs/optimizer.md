# 优化器：Muon、AdamW 与参数分组

训练可选择纯 AdamW，或按参数角色混合 Muon 与 AdamW。混合方案不是让用户手写参数列表；
`HybridOptimizer` 根据模型结构稳定地收集参数，并把二维 Transformer 权重交给 Muon，其余参数交给
带角色缩放的 AdamW。

## AdamW 更新

第 `t` 步梯度为 `g_t`：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
```

偏差修正后，解耦 weight decay 的更新为：

```math
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

AdamW 的参数、梯度和一二阶矩在更新期间显式转换为 fp32，`epsilon=1e-10`，更新完成后才把
parameter 转回模型 dtype。这样 f16 embedding 不会把 `1 - beta` 或小二阶矩提前舍入掉；optimizer
state 也始终以 fp32 保存。Muon 的低精度回写路径仍使用适配后的 clamp/epsilon。

## Muon 路径

Muon 先更新 momentum，并使用 Nesterov 风格组合得到方向。MuonEq 把每行缩放到平均行范数，
随后在 fp32 中执行五次 Polar Express 多项式迭代，逼近矩阵极分解中的正交方向。Muon+ 再把
Frobenius norm 校正为 `sqrt(min(rows, cols))`。NorMuon 最后沿矩阵较长维维护二阶统计、保留更新
范数，并按形状缩放学习率：

```math
\eta_{matrix}=\eta\sqrt{\max(\text{rows}/\text{cols},1)}.
```

weight decay 只作用于正交更新与当前参数同号的位置，即 `(g_ortho * parameter) >= 0`。具体多项式
系数保存在源码中并由 Python fixture 固定；文档不复制一份容易漂移的“第二实现”。

## 参数分组

选择 `muon_adam_w` 时：

- Transformer 内二维矩阵使用 Muon。
- token embedding、LM head、value embedding、标量系数、Smear 与 Backout 参数使用 AdamW。
- AdamW 子组根据角色使用学习率倍率；常量定义在 `src/optim.rs`。

选择 `adam_w` 时，所有参数都由 AdamW 更新。两种模式共享相同的外部基础学习率、weight decay 和
训练调度器，因此适合控制变量对照。

optimizer state 包含每个参数的 momentum/二阶统计和 optimizer kind，并保存为 safetensors。
恢复时 optimizer kind 必须与当前训练配置一致。

## 源码入口

- `src/optim.rs`：参数收集、fp32 AdamW、MuonEq/Polar Express/Muon+/NorMuon 与状态序列化。
- `src/optim/parity.rs`：固定 Python fixture 的 AdamW/Muon 单步更新。
- `src/engine.rs`：梯度累积后调用 optimizer，并提供学习率、momentum 和 weight decay 调度。
- `src/artifact.rs`：optimizer 与 trainer resume state 的保存/加载。
- `configs/mini.toml`：`optimizer = "muon_adam_w"` 的完整实验配置。

## 最小实验

验证矩阵正交化的形状和有限值：

```bash
cargo test optim::tests::test_muon_orthogonalization
```

验证 Rust 与 Python 的 AdamW、宽矩阵 Muon 和高矩阵 Muon 单步结果：

```bash
cargo test optim::parity -- --show-output
```

验证 safetensors 状态往返：

```bash
cargo test optim::tests::test_optimizer_state_roundtrip
```

对照实验应复制同一 TOML，只改变：

```toml
optimizer = "adam_w" # 另一组使用 "muon_adam_w"
```

并设置不同的 `artifacts.pretrain`。训练后用 `report` 同时读取两个目录，比较 loss/BPB、tokens/s 与
内存；不要让两次运行覆盖同一个 artifact。

## 正确性证据与观察点

- parity fixture 同时比较更新后的参数和 optimizer state，能发现“参数暂时接近但状态已漂移”。
- 宽/高矩阵分别覆盖 NorMuon 归约维度选择。
- resume equivalence 测试把 optimizer、trainer 和 dataloader 一起恢复并比较下一步 logits。
- WGPU 测试检查 f16 参数回写和 fp32 AdamW state；NdArray f32 测试不能代替低精度验证。

## 常见错误

- 把所有二维参数都机械交给 Muon。embedding 与 LM head 按模型角色走专门 AdamW 组。
- 用 AdamW 的 weight decay 公式解释 Muon。Muon 的 decay 还受同号 mask 控制。
- 只保存模型权重后声称“精确续训”。缺少 momentum、二阶统计、step 和 dataloader 位置会改变轨迹。
- 比较优化器时同时改学习率、seed 或 batch size，导致无法归因。
- 把模型 dtype 误当 optimizer state dtype。f16 forward/backward 不意味着 AdamW moment 也应是 f16。
- 假定正交化输出满足数学上的精确 `Q^TQ=I`。有限次多项式是高效近似，按 fixture 容差验证。

## 能力边界

Muon 的收益依赖模型形状、batch 和训练预算；项目提供实现、parity 与对照工具，不保证它在每个
tiny run 都优于 AdamW。性能或质量结论应附配置、随机种子、artifact 指标和运行后端。
