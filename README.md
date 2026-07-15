# 🚀 nanochat-burn

一个基于 **Rust 与 Burn 深度学习框架** 高性能、地道实现的极简且完整的 GPT/LLM 训练与自回归推理引擎
。它完整复刻了 [`nanochat`](https://github.com/karpathy/nanochat) 的独创模型架构与优化器设计，完美支持 **Foundational Pretraining (预训练)*
*、**Supervised Fine-Tuning (SFT 监督微调)** 以及 **Reinforcement Learning (RL 强化学习对齐)** 的完
整三阶段生命周期。项目以可读、可运行、可验证为目标，覆盖 BPE、数据管线、GPT、Muon/AdamW、预训练、SFT、组内归一化 REINFORCE、量化和自回归推理，并逐步建立与 Python `nanochat` 的数值 parity 证据。

---

## 🌟 能力状态

| 能力 | 状态 | 说明 |
|---|---|---|
| GPT、RoPE、GQA、SWA、QK Norm、ReLU² | Stable | 有单元测试和 cached/full forward 一致性测试 |
| Muon + AdamW | Stable | 支持梯度累积和 f16 数值保护 |
| Pretrain → SFT → RL artifact 衔接 | Experimental | 共享模型、配置和 tokenizer；Pretrain 与 RL 保存 optimizer/trainer，RL 额外恢复采样 RNG |
| TOML 实验配置 | Experimental | 一个强类型配置统一模型、数据、三阶段超参数和 artifact 链路，并随产物保存 |
| W8/W4 weight-only quantization | Experimental | WGPU 使用 Burn QFloat 快路径，NdArray 和特殊形状使用可移植回退 |
| Paged KV cache | Experimental | free-list、请求 block table 与逐页 online-softmax attention；服务级动态 batching 尚未接入 |
| Speculative decoding | Reference | greedy 模式数学无损，支持增量 draft rollback、acceptance 与真实加速比基准 |
| REINFORCE / GRPO | Experimental | 支持组内优势、old-policy ratio clip、reference KL、rollout 记录和恢复 |
| Python code executor | Experimental | 有超时和输出限制，但不是 OS 级安全沙箱 |
| Python/Rust 数值 parity | Reference | Fixtures 覆盖 tokenizer、模型、optimizer、cache 与 f32/f16/W8/W4 误差预算，并可自动生成报告 |

完整实施计划见 [ROADMAP.md](ROADMAP.md)。

## 学习路径

面向源码的 mini-LLM 教程从 [docs/README.md](docs/README.md) 开始，按 tokenizer、数据、模型、
优化器、训练、推理和对齐组织。每章包含数学或数据契约、源码入口、可运行的 tiny 实验、正确性
证据和常见错误。第一章 [Tokenizer：从字节到训练目标](docs/tokenizer.md) 已完成，可直接运行：

```bash
cargo run --features ndarray --example tokenizer
```

## 核心模型与系统特性

*   **独创 GPT Transformer 架构**：
    *   **RoPE 位置编码**：采用旋转位置编码（Rotary Position Embeddings），且 LM Head 与 Embedding 权重采用 **untied weights** 机制。
    *   **激活函数**：使用独特的 `ReLU²`（$x = \text{ReLU}(x)^2$）而非标准的 GeLU/SwiGLU。
    *   **QK Norm**：自注意力计算前对 Query 和 Key 投影张量分别应用无 Scale 参数的 RMSNorm 归一化。
    *   **分组查询注意力（GQA）**：支持 KV 头数 `n_kv_head` 小于 `n_head` 的多路查询。
    *   **滑动窗口自注意力（SWA）**：支持各层独立的滑动窗口注意力机制。
    *   **Smear Bigram 混入**：前向传播中以 bigram mixing 形式动态融合前一个 token 的 embedding。
    *   **Backout 残差扣除**：在最后 Logits 预测前，减去中层特征以有效扣除低级冗余信息。
    *   **Logit Softcap**：对输出 logits 应用 `15.0 * tanh(logits / 15.0)` 限制数值波动。
*   **Muon + AdamW 混合优化器**：
    *   **Muon 优化器**：对于 2D 线性层矩阵参数，使用高精度 **Polar Express Sign Method** 进行正交化，并融合 **NorMuon** 方差归约算法加速收敛。
    *   **AdamW 优化器**：一维标量参数、Smear/Backout 门控及 Embedding 使用 AdamW 更新。
    *   **半精度稳定性 (f16)**：对 epsilon、clamp 和 `.sqrt()` 下界进行 half-precision 适配，并通过 WGPU 测试持续验证。
*   **系统级优化**：
    *   **静态 attention 掩码**：在模型初始化时预计算 Attention 掩码并在前向中 slice 访问。
    *   **广播式 GQA**：`repeat_kv` 使用 `.reshape()` 与 `.expand()` 表达分组广播，减少显式复制。
    *   **设备切换**：默认使用 WGPU，也可通过 `--features ndarray` 进行纯 CPU 快速验证。
*   **低比特量化与生态对接 (Quantization & Serialization)**：
    *   **低比特权重量化**：Linear 投影支持 W8A16/W4A16 和通道/块对称缩放。
    *   **Safetensors 序列化**：支持模型参数导入导出和 Burn/PyTorch Linear layout 转置。
    *   **统一 Artifact**：模型配置、tokenizer、权重、optimizer、trainer 状态和阶段 manifest 保存在同一目录中。
*   **推理与对齐参考实现**：
    *   **KV Cache 与推测解码**：包含 blocked cache、完整前向 parity 测试和 greedy speculative reference path。
    *   **组内归一化 REINFORCE**：无需 Critic，按问题内 rollout reward 标准化优势。

---

## 📂 目录结构与架构设计

项目采用极致扁平化与高度模块化的 Rust Idiomatic 工程结构设计：

```text
    burn/
    ├── Cargo.toml
    ├── README.md               # 项目主说明文档
    ├── ROADMAP.md              # 分阶段实施计划与验收标准
    ├── docs/                   # mini-LLM 分章节学习路径
    ├── examples/               # 与章节配套的可运行 tiny 实验
    ├── configs/mini.toml       # 默认 mini-LLM 三阶段实验配置
    ├── src/
    │   ├── lib.rs              # 导出所有子模块
    │   ├── artifact.rs         # 统一实验产物、manifest 与阶段加载
    │   ├── common.rs           # 阶段 0: 设备检测（BURN_DEVICE=cpu支持）、类型与数值校验器
    │   ├── tokenizer.rs        # 阶段 1: Byte-level BPE 与并行编码
    │   ├── dataset.rs          # 阶段 2: 数据集载入与 mmap 闪存加载支持
    │   ├── dataloader.rs       # 阶段 2: 分片、预取与断点位置
    │   ├── gpt.rs              # 阶段 3: GPT 架构实现 (Rotary Embeddings, Softcap, GQA 等)
    │   ├── gpt/
    │   │   ├── parity.rs       # Python/Rust 模型 parity 与误差预算
    │   │   └── quant.rs        # W8/W4 权重量化与后端快路径
    │   ├── checkpoint.rs       # 阶段 3: Checkpoint 序列化与 Safetensors 对接
    │   ├── benchmark.rs        # Prefill/decode、量化与 speculative 基准
    │   ├── optim.rs            # 阶段 4: Polar Express Muon + AdamW 混合正交优化器
    │   ├── engine.rs           # 阶段 4/5: 训练与推理引擎底座、BPB 评估器
    │   ├── experiment.rs       # 强类型 TOML 实验配置、校验与持久化
    │   ├── engine/
    │   │   ├── calculator.rs   # 内置 Tool-Use 计算器状态机算子
    │   │   ├── pretrain.rs     # 阶段 4/5: 异步预训练工作流
    │   │   ├── inference.rs    # 阶段 5: 支持 KV-Cache 的批量自回归采样
    │   │   ├── rl.rs           # 组内归一化 REINFORCE 工作流
    │   │   ├── speculative.rs  # 阶段 5: 无损推测解码双模型推理引擎 (Draft + Target Model)
    │   │   ├── sandbox.rs      # 阶段 6: 带超时和输出限制的 Python 子进程
    │   │   ├── scheduler.rs    # Continuous batching admission/cancel 调度器
    │   │   ├── eval.rs         # 阶段 6: 评测子系统及 benchmark 评估 (gsm8k, spellingbee 等)
    │   │   └── sft.rs          # 阶段 6: 监督微调 (packed SFT) 工作流
    │   └── bin/
    │       ├── train.rs        # 训练入口 (支持 --pretrain, --sft, --rl 参数动态切换)
    │       ├── eval.rs         # 多任务评测入口
    │       ├── report.rs       # 训练、BPB、吞吐与质量汇总
    │       ├── bench_infer.rs       # Prefill/decode 与量化基准
    │       ├── bench_spec.rs        # 推测解码 acceptance/speedup 基准
    │       ├── chat.rs         # CLI 命令行多轮流式对话客户端
    │       └── chat_web.rs     # Axum SSE 多轮对话服务器
```

---

## ⚡ 快速开始与命令指南

### 1. 运行单元测试
使用 NdArray 进行快速、可移植验证：
```bash
cargo test --features ndarray
```

使用默认 WGPU 后端验证：

```bash
cargo test
```

Python nanochat parity fixtures 位于 `data/fixtures/parity/`。Tokenizer fixture 覆盖 BPE
训练结果、普通 token IDs、conversation masks、tool parts、截断和 completion rendering；module
fixture 使用固定输入和参数验证 RMSNorm、RoPE、MLP 与含 value embedding gate 的 GQA attention。
Full-model fixture 进一步验证两层 GPT 的 logits、mean loss 和代表性参数梯度。Rust 测试不依赖
Python，并以同一模型验证 full、chunked、非均匀 chunk 和逐 token cache logits。Optimizer
fixture 验证 AdamW 与宽/长矩阵 Muon 的单步参数和状态更新：

```bash
cargo test --features ndarray tokenizer::tests::test_python_tokenizer
cargo test --features ndarray gpt::parity
cargo test --features ndarray optim::parity
cargo test gpt::parity::test_f16_w8_w4_logit_error_budgets -- --nocapture
```

固定 `model.json` fixture 的 logits 最大绝对误差预算如下。f32/f16 以 Python f32
fixture 为参照；W8/W4 以同后端未量化 logits 为参照，从而单独衡量 weight-only
量化误差。表中实测值来自当前 NdArray 与 Metal/WGPU 测试环境，预算由测试常量持续执行：

| 路径 | 参照 | 实测误差 | 预算 |
|---|---|---:|---:|
| NdArray f32 | Python f32 | 6.6e-7 | 5e-5 |
| WGPU/Metal f16 | Python f32 | 2.63619e-3 | 5e-3 |
| NdArray portable W8 | NdArray f32 | 9.6667e-4 | 5e-3 |
| WGPU native W8 | WGPU f16 | 2.31934e-3 | 5e-3 |
| NdArray portable W4, block 8 | NdArray f32 | 2.99752e-3 | 2e-2 |
| WGPU native W4, block 8 | WGPU f16 | 4.15039e-3 | 2e-2 |

一条命令运行 NdArray 与 WGPU parity suites，并将测试清单、实测误差、预算、revision
和运行环境写入 `target/parity-report.md`：

```bash
python3 tools/parity_report.py
```

无 GPU 环境可使用 `--backend ndarray`；`--backend wgpu` 可单独验证加速后端，
`--output -` 将 Markdown 输出到标准输出。任一测试失败、指标缺失或误差超出预算时，
命令仍会生成带诊断信息的报告并返回非零状态。

使用脚本声明的固定 Python 依赖重新导出 fixtures：

```bash
uv run tools/export_tokenizer_parity.py --nanochat-root /path/to/python-nanochat
uv run tools/export_torch_parity.py all --nanochat-root /path/to/python-nanochat
```

也可以设置 `NANOCHAT_ROOT=/path/to/python-nanochat`。该路径仅在重新生成跨语言 fixtures
时需要，日常构建、测试、训练和 Web UI 不依赖父目录；`all` 可替换为 `modules`、`model` 或
`optimizer`，只重新生成对应 fixture。

### 消融、报告与推理基准

`pretrain.model.features` 可独立关闭 `relu_squared`、`qk_norm`、`gqa`、`swa`、`smear`
和 `backout`；关闭 GQA 时需令 `n_kv_head = n_head`。训练配置中的
`optimizer = "muon_adam_w"` 可替换为 `"adam_w"` 进行同配置对照。
RL 可用 `--rl-algorithm group_normalized_reinforce` 或 `--rl-algorithm grpo` 覆盖配置，
便于保持其余参数完全一致地生成两组 artifact。

```bash
cargo run --bin report -- runs/pretrain runs/sft runs/rl
cargo run --release --bin bench_infer -- --artifact runs/sft --batches 1,2,4
cargo run --release --bin bench_infer -- --artifact runs/sft --quantization 4
cargo run --release --bin bench_spec -- runs/sft runs/pretrain
```

报告写入 `runs/report.json`，推理基准写入 `runs/benchmarks/`。基准记录 prefill、首 token、
decode tokens/s、batch scaling、理论 KV cache 字节数，以及量化误差/估算线性权重大小；目前不声称
这些估算值等同于驱动报告的峰值显存。

同一 SFT artifact 的小规模对齐对照可用两个输出目录运行；随后分别执行 `eval`，再交给
`report` 汇总，报告会从 `experiment.toml` 区分两种算法：

```bash
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/reinforce cargo run --bin train -- --rl \
  --rl-algorithm group_normalized_reinforce
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/grpo cargo run --bin train -- --rl \
  --rl-algorithm grpo
NANOCHAT_ARTIFACT=runs/compare/reinforce cargo run --bin eval
NANOCHAT_ARTIFACT=runs/compare/grpo cargo run --bin eval
cargo run --bin report -- runs/sft runs/compare/reinforce runs/compare/grpo
```

### 端到端 Tiny Recipe

仓库内置一套离线小文本 recipe，一条命令完成 tokenizer 训练、pretrain、SFT 和 eval：

```bash
cargo run --features ndarray --bin train -- --recipe --config configs/tiny.toml
```

完整配置位于 `configs/tiny.toml`，可复现输入位于 `data/fixtures/tiny/`，产物写入
`runs/tiny/`。它用于快速验证完整实验链路；模型只有一层且仅训练少量步骤，因此评测分数不代表
实际模型能力。移除 `--features ndarray` 即可使用默认 WGPU 后端运行同一 recipe。

### 2. 基础预训练 (Pretraining)
训练命令默认读取 `configs/mini.toml`。该文件统一描述模型、随机种子、预训练语料、数据路径、
三阶段训练参数、评测任务和 artifact 链路；未知字段或非法组合会在设备初始化前报错。启动小型
合成数据预训练：
```bash
cargo run --bin train --release -- --pretrain
```

使用自定义配置：

```bash
cargo run --bin train --release -- --pretrain --config configs/mini.toml
NANOCHAT_CONFIG=configs/mini.toml cargo run --bin train --release -- --pretrain
```

每个训练产物都会保存一份实际配置为 `experiment.toml`。

`pretrain.model.sequence_len` 是模型的最大上下文容量，随模型 artifact 固定。Pretrain 和 SFT
默认继承该值；若某阶段需要更短的训练序列，可在对应的 `training` 表中增加
`sequence_length = 128`，但不能超过模型容量。RL 的 `max_generation_tokens` 是生成预算，不会改变模型容量。

预训练默认每 5 步更新一次可恢复 checkpoint；中断后从同一 artifact 继续：

```bash
NANOCHAT_RESUME_ARTIFACT=runs/pretrain cargo run --bin train --release -- --pretrain
```

可通过 `NANOCHAT_CHECKPOINT_INTERVAL` 调整保存间隔，设为 `0` 时仅保存最终状态。设置
`NANOCHAT_OUTPUT_ARTIFACT` 可将恢复后的训练写入新目录，并保留 checkpoint 之前的 metrics 历史。

### 3. 监督微调 (SFT)
加载 `runs/pretrain/`，执行 Packed SFT，并输出 `runs/sft/`：
```bash
cargo run --bin train --release -- --sft
```

### 4. 在线强化学习对齐 (RL)
加载 `runs/sft/`，执行基于 GSM8K 的组内归一化 REINFORCE，并输出 `runs/rl/`：
*   **在 GPU (Metal) 上运行**（注：在 macOS GPU 运行时自回归动态切片会导致 Metal 产生 JIT 编译热身耗时）：
    ```bash
    cargo run --bin train --release -- --rl
    ```
*   **选择 WGPU CPU device 运行**：
    ```bash
    BURN_DEVICE=cpu cargo run --bin train --release -- --rl
    ```

### 5. 评测与交互式对话

Eval、CLI Chat 和 Web Chat 默认按 `runs/rl`、`runs/sft`、`runs/pretrain` 的顺序加载最新可用 artifact。可通过 `NANOCHAT_ARTIFACT` 显式指定目录。

```bash
cargo run --bin eval --release
NANOCHAT_CONFIG=configs/tiny.toml cargo run --features ndarray --bin eval
NANOCHAT_ARTIFACT=runs/sft cargo run --bin chat --release
```

*   **CLI 命令行对话客户端**：
    启动基于终端的交互式多轮对话客户端，体验自回归流式生成与内置 Calculator Tool-Use 状态机：
    ```bash
    cargo run --bin chat --release
    ```
    *（在 CPU 下极速体验自回归生成：`BURN_DEVICE=cpu cargo run --bin chat --release`）*

*   **Web 对话服务端**：
    启动基于 Axum 的流式 SSE Web 服务：
    ```bash
    cargo run --features web --bin chat_web --release
    ```
    启动后可在浏览器中访问 [http://127.0.0.1:8080](http://127.0.0.1:8080)。

---

## 🛠️ 技术规范与开发准则

1.  **泛型后端抽象**：模型、注意力和优化器基于 `<B: Backend>` 或 `<B: AutodiffBackend>`；当前持续验证 WGPU 与 NdArray。
2.  **参考实现优先**：先保留容易阅读和验证的实现，再为热点增加后端特化快路径。
3.  **显式同步边界**：训练日志、评测和当前 CPU sampler 会发生设备回读；设备端 sampler 列入性能路线图。
4.  **可验证主张**：Stable 能力需要测试，性能结论需要基准，parity 结论需要跨语言 fixtures。

---

## 🗺️ 未来展望与路线图 (Future Roadmap / TODO)

项目按“可信文档 → 可复现实验 → 数值 parity → 教学消融 → 推理性能 → 并发调度 → 对齐算法 → GPU 特化”的顺序推进。任务状态和发布标准统一维护在 [ROADMAP.md](ROADMAP.md)，不在 README 重复维护容易失真的功能清单。
