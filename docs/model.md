# 模型：GPT、RoPE、GQA 与残差路径

`Gpt` 是一个 pre-norm decoder-only Transformer。实现保留清晰的参考数学，同时让 Burn 后端为
attention、广播和量化选择合适路径。所有结构开关都来自 artifact 中的 `GptConfig`。

## 一层 Transformer 的契约

忽略可选门控时，第 `l` 层为：

```math
h'_l = h_l + \operatorname{Attention}(\operatorname{RMSNorm}(h_l)),
```

```math
h_{l+1} = h'_l + \operatorname{MLP}(\operatorname{RMSNorm}(h'_l)).
```

RMSNorm 没有可学习 scale：

```math
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
```

`epsilon` 会按后端精度调整，避免 f16 下溢。MLP 可在普通 ReLU 与 nanochat 的 ReLU² 间切换：

```math
\operatorname{ReLU^2}(x)=\max(x,0)^2.
```

## RoPE、QK Norm 与 GQA

每个 attention head 的偶数维度拆成两半。对位置 `p` 的旋转为：

```math
y_1=x_1\cos\theta_p+x_2\sin\theta_p,\qquad
y_2=x_2\cos\theta_p-x_1\sin\theta_p,
```

频率基数为 100000。启用 QK Norm 时，Q/K 在 RoPE 与点积前分别 RMS 归一化并乘 1.2。

参考 attention 是：

```math
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V.
```

因果 mask 禁止看到未来 token；SWA 还禁止看到窗口外历史。GQA 令多个 query head 共用一组 K/V：

```math
g=\frac{n_{head}}{n_{kv\_head}}.
```

`repeat_kv` 用 reshape/expand 表达这个共享关系。配置要求 `n_head % n_kv_head == 0`，关闭 GQA 时
则要求两者相等。

## nanochat 结构开关

- `relu_squared`：ReLU² 或普通 ReLU。
- `qk_norm`：对 Q/K 分别做无 scale RMSNorm。
- `gqa`：共享 KV heads。
- `swa`：按 `window_pattern` 选择短窗/长窗，最后一层强制长窗。
- `smear`：用 sigmoid gate 和学习系数混入前一个 token embedding。
- `backout`：保存中层特征，在最终输出前减去学习系数乘该特征。

模型还支持交替层 value embedding gate、每层 residual/x0 可学习系数，以及
`15 * tanh(logits / 15)` logit softcap。token embedding 与 LM head 不共享权重；padding vocab 只为
后端形状服务，返回 logits 会切回实际词表大小。

## 配置不变量

- `n_embd` 必须被 `n_head` 整除，head dimension 必须为偶数。
- `n_head` 必须被 `n_kv_head` 整除。
- `window_pattern` 只能由 `S`/`L` 构成，且模型维度、层数和序列长度都非零。
- weight-only 量化只接受 W8 或 W4；不支持的门控形状保留浮点路径。

## 源码入口

- `src/gpt.rs`：配置校验、RMSNorm、RoPE、attention、MLP、残差和 logits。
- `src/gpt/cache.rs`：连续/分页 KV cache 与逐页 attention。
- `src/gpt/quant.rs`：W8/W4 Linear 和可移植回退。
- `src/gpt/parity.rs`：Python/Rust 模块、完整模型、梯度与 cache parity。
- `src/gpt/tests.rs`：消融、attention 后端与 cached forward 测试。
- `docs/gpu-operators.md`：GPU attention 与算子 profiling 的复现证据。

## 最小实验

观察只切换 ReLU² 是否改变 MLP 输出：

```bash
cargo test gpt::tests::test_relu_squared_ablation_changes_mlp_output
```

验证固定 Python fixture 的模块、logits、loss 和梯度：

```bash
cargo test gpt::parity -- --show-output
```

验证 full forward、分块 cache 与逐 token decode：

```bash
cargo test gpt::tests::test_cached_forward_matches_full_and_incremental_forward
```

真实训练消融可复制 `configs/mini.toml`，一次只修改
`pretrain.model.features` 的一个布尔值，并为每组设置不同 artifact 目录；最后用：

```bash
cargo run --bin report -- runs/baseline runs/ablation
```

比较 loss、BPB、吞吐和质量。小实验只能说明该配置与随机种子下的差异，不能单独证明普适收益。

## 常见错误

- 关闭 GQA 却保留较少的 `n_kv_head`，这不是 MHA，会被配置校验拒绝。
- 把 SWA 当成 KV cache。SWA 是可见性规则；cache 是避免重复计算历史 K/V 的存储机制。
- 忽略 RoPE 符号约定。这里第二半是 `x2*cos - x1*sin`，fixture 固定了该约定。
- 认为 padded vocab 的额外 logits 会暴露给采样器。forward 在返回前切回真实 vocab。
- 用量化输出与 Python f32 直接归因于量化误差。误差预算把后端精度误差与量化误差分开比较。
- 看到“GPU attention”就假定必然是 Flash。Burn/CubeCL 会按设备和形状选择融合、Flash 或 fallback。

## 能力边界

参考和后端路径由 parity 测试约束“数值足够一致”，不是逐 bit 相同。f16、W8 和 W4 使用 README
中公开的误差预算。吞吐结论也必须由目标设备上的 `bench_ops`/`bench_infer` 支持，不能从算子名称
推断。
