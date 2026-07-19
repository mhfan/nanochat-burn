# 🚀 nanochat-burn

一个基于 **Rust 与 Burn 深度学习框架**、高性能且地道实现的极简 GPT/LLM 训练与自回归推理
引擎。它复现 [`nanochat`](https://github.com/karpathy/nanochat) 的模型架构与优化器设计，支持
**Foundational Pretraining（预训练）**、**Supervised Fine-Tuning（SFT 监督微调）** 和
**Reinforcement Learning（RL 强化学习对齐）** 的完整三阶段生命周期。项目以可读、可运行、
可验证为目标，覆盖 BPE、数据管线、GPT、Muon/AdamW、预训练、SFT、REINFORCE/GRPO、量化、
分页 KV cache 和自回归推理，并建立与 Python `nanochat` 的数值 parity 证据。

---

## 🌟 能力状态

| 能力 | 状态 | 说明 |
|---|---|---|
| GPT、RoPE、GQA、SWA、QK Norm、ReLU² | Stable | 有单元测试和 cached/full forward 一致性测试 |
| Muon + AdamW | Stable | AdamW 使用 fp32 状态/计算；Muon 包含 Polar Express、MuonEq、Muon+ 与 NorMuon |
| Pretrain → SFT → RL artifact 衔接 | Experimental | 共享模型、配置和 tokenizer；Pretrain 与 RL 保存 optimizer/trainer，RL 额外恢复采样 RNG |
| TOML 实验配置 | Experimental | 一个强类型配置统一模型、数据、三阶段超参数和 artifact 链路，并随产物保存 |
| W8/W4 weight-only quantization | Experimental | WGPU 使用 Burn QFloat 快路径，NdArray 和特殊形状使用可移植回退 |
| GPU attention | Experimental | 完整因果层和 SWA 层统一调用 Burn/CubeCL attention，由后端选择融合、Flash 或 fallback 路径 |
| Paged KV cache | Experimental | free-list、请求 block table、逐页 online-softmax attention，以及按 request slot/position 合并的 iteration-level continuous batching |
| Speculative decoding | Reference | greedy 模式数学无损，支持增量 draft rollback、acceptance 与真实加速比基准 |
| REINFORCE / GRPO | Experimental | 支持组内优势、old-policy ratio clip、reference KL、rollout 记录和恢复 |
| Python code executor | Experimental | 独立临时目录、环境清理、stdin/内存/输出限制和防误删 guard；仍不是安全边界 |
| Python/Rust 数值 parity | Reference | Fixtures 覆盖 tokenizer、模型、optimizer、cache 与 f32/f16/W8/W4 误差预算，并可自动生成报告 |

## 学习路径

面向源码的 mini-LLM 教程从 [docs/README.md](docs/README.md) 开始，七章均已可用：
[tokenizer](docs/tokenizer.md)、[数据](docs/data.md)、[模型](docs/model.md)、
[优化器](docs/optimizer.md)、[训练](docs/training.md)、[推理](docs/inference.md) 和
[对齐](docs/alignment.md)。每章包含数学或数据契约、源码入口、可运行的 tiny 实验、正确性证据
和常见错误。可以从 tokenizer 示例开始：

```bash
cargo run --example tokenizer
```

## 与原生 nanochat 的差异和取舍

本项目以当前 Python [`nanochat`](https://github.com/karpathy/nanochat) 的模型数学和训练方法为
参考，但不是逐行翻译。原生实现优先服务 CUDA/H100 上的大吞吐研究训练；nanochat-burn 优先保证
Rust 类型边界、跨平台后端、端到端 artifact、自包含测试和可扩展推理系统。两者适合解决的问题
并不完全相同：

| 方面 | Python nanochat | nanochat-burn | 当前优势方 |
|---|---|---|---|
| 后端与主要目标 | PyTorch/CUDA、FlashAttention、`torch.compile`，面向 H100 speedrun | Burn + WGPU/Metal/Vulkan 或 NdArray，面向可移植实验和 Rust 集成 | Python：训练性能；Burn：可移植性 |
| 模型 | 上游基准定义 | 对齐核心 GPT 数学，并提供 ReLU²、QK Norm、GQA、SWA、Smear、Backout 消融开关和跨语言 fixtures | nanochat-burn：消融与验证 |
| 预训练数据 | Parquet 流式文档、BOS-aligned best-fit packing、真实多 rank 分片 | 本地文本预分词为 mmap u32 shards；现在按空行识别文档、长文档分片，并以 BOS-aligned best-fit 固定行写盘 | Python nanochat |
| 训练规模 | 多 GPU optimizer sharding，并有 CUDA FP8 路径 | 当前训练执行仍是单进程；支持可恢复梯度累积，但没有真实 collective 多 GPU 和 FP8 训练 | Python nanochat |
| 优化器 | 融合/编译的 AdamW + 分布式 Muon | 泛型 Burn 实现；AdamW 在 f16 模型下仍以 fp32 保存和计算状态，Muon 对齐 MuonEq/Muon+/NorMuon | Python nanochat：性能与扩展规模 |
| Checkpoint | PyTorch checkpoint，数据位置近似恢复 | 模型、tokenizer、配置、optimizer、trainer、dataloader/RNG 组成统一 safetensors artifact，强调精确恢复 | nanochat-burn |
| 推理 | 连续 KV cache；同 prompt 先 prefill 一次再扩展样本 | Paged KV、请求页表/回收、attention sinks、continuous batching；同 prompt 只计算一次并复制独立 KV 页 | nanochat-burn |
| 推理扩展 | 基础生成和当前上游 benchmark | W8/W4 weight-only、greedy speculative decoding、设备采样器、SSE Web 服务和内存/TPOT 基准 | nanochat-burn |
| 对齐 | SFT 与较精简的 group-relative RL | Packed SFT（含梯度累积）、REINFORCE/GRPO、old-policy clip、reference KL、rollout/RNG 恢复 | nanochat-burn：功能覆盖 |
| 产品边界 | 上游主动保持最小，已移除 Web UI、报告和部分教学任务 | 保留报告、Web UI、离线 tiny recipe 和学习文档，作为 Rust 系统实验面的一部分 | Python：极简；Burn：完整工具链 |

模型 dtype 已集中到 `src/common.rs` 的 `ModelFloat`。但 Burn 0.21 尚不支持可训练的 FP8，当前
不能只把 `f16` 换成八位类型；W8/W4 也只是推理期权重量化。待 Burn 提供 FP8 kernel、缩放和
autograd 支持后，可从该入口接入；现阶段继续使用 f16 训练和 fp32 AdamW 状态。

## 除学习之外的潜在用途

nanochat-burn 目前最现实的定位是“单机、小模型、可嵌入的 LLM 系统实验栈”，而不是取代大型
CUDA 集群训练框架。除了理解 LLM 全生命周期，还可以用于：

- **单工作站领域模型原型**：用自有文本完成 tokenizer、预训练、SFT、GRPO、评测和可恢复
  artifact，适合验证小模型的数据配方、损失与对齐策略。
- **Rust 应用内嵌推理**：在已有 Rust 服务或桌面程序中直接加载 safetensors，使用 WGPU 在
  Metal/Vulkan 设备运行，减少跨语言进程和 Python 运行时依赖。
- **服务系统研究原型**：评估 paged KV、continuous batching、请求取消、attention sinks、
  speculative decoding、SSE backpressure 和不同 batch 下的 TTFT/TPOT/吞吐/显存取舍。
- **低比特部署实验**：比较 f16、W8A16、W4A16 的 logits 误差、模型字节数和真实延迟，筛选
  适合本地或资源受限设备的模型配置。
- **优化器与后端回归平台**：利用同一 Python fixture、NdArray reference 和 WGPU 路径验证新
  optimizer、算子或 Burn 版本升级，捕获 dtype、layout、cache 和恢复语义漂移。
- **可复现实验与 CI 样板**：统一 artifact、强类型 TOML、离线 tiny recipe、数值预算和报告
  适合做小型模型研究的持续集成基线。

这些用途仍受三个边界约束：训练尚未实现真实多 GPU collective；WGPU 性能不能等同于原生
CUDA/FlashAttention；模型生成的 Python 即使经过防误伤 guard，也不能执行不受信任的恶意代码。

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
    *   **Muon 优化器**：对于 2D 线性层矩阵参数，使用 **MuonEq** 行均衡、**Polar Express Sign Method** 正交化、**Muon+** 范数校正和 **NorMuon** 方差归约。
    *   **AdamW 优化器**：一维标量参数、Smear/Backout 门控及 Embedding 使用 AdamW 更新；即使模型参数为 f16，moment 和更新数学也保持 fp32，最后才转换回参数 dtype。
    *   **半精度稳定性 (f16)**：Muon 的正交化阶段升至 fp32，低精度回写路径保留 epsilon、clamp 和 `.sqrt()` 下界保护，并通过 WGPU 测试持续验证。
*   **系统级优化**：
    *   **attention 掩码复用**：完整因果层使用后端原生 causal 语义；SWA 与缓存路径共享初始化时预计算的 bool 掩码。
    *   **广播式 GQA**：`repeat_kv` 使用 `.reshape()` 与 `.expand()` 表达分组广播，减少显式复制。
    *   **设备切换**：默认使用可移植的 NdArray；训练、推理和 GPU 基准通过 `--no-default-features --features wgpu` 显式启用 WGPU。
*   **低比特量化与生态对接 (Quantization & Serialization)**：
    *   **低比特权重量化**：Linear 投影支持 W8A16/W4A16 和通道/块对称缩放。
    *   **Safetensors 序列化**：支持模型参数导入导出和 Burn/PyTorch Linear layout 转置。
    *   **统一 Artifact**：模型配置、tokenizer、权重、optimizer、trainer 状态和阶段 manifest 保存在同一目录中。
*   **推理与对齐参考实现**：
    *   **分页 KV Cache 与推测解码**：包含 page allocator、逐页 attention、continuous batching、完整前向 parity 测试和 greedy speculative reference path。
    *   **REINFORCE 与 GRPO**：无需 Critic，支持问题内 rollout reward 标准化优势、old-policy ratio clip 和 reference KL。

---

## 📂 目录结构与架构设计

项目采用极致扁平化与高度模块化的 Rust Idiomatic 工程结构设计：

```text
    burn/
    ├── Cargo.toml
    ├── README.md               # 项目主说明文档
    ├── docs/                   # 七章 mini-LLM 学习路径与 GPU 算子报告
    ├── examples/               # 与章节配套的可运行 tiny 实验
    ├── configs/
    │   ├── mini.toml           # 默认 mini-LLM 三阶段实验配置
    │   └── tiny.toml           # 离线 tiny 全流程冒烟测试配置
    └── src/
        ├── lib.rs              # 导出所有子模块
        ├── artifact.rs         # 统一实验产物、manifest 与阶段加载
        ├── common.rs           # 阶段 0: 设备检测（BURN_DEVICE=cpu支持）、类型与数值校验器
        ├── tokenizer.rs        # 阶段 1: Byte-level BPE 与并行编码
        ├── dataset.rs          # 阶段 2: 数据集载入与 mmap 闪存加载支持
        ├── dataloader.rs       # 阶段 2: 分片、预取与断点位置
        ├── gpt.rs              # 阶段 3: GPT 架构实现 (Rotary Embeddings, Softcap, GQA 等)
        ├── gpt/
        │   ├── cache.rs        # 分页 KV cache、page allocator 与 block table
        │   ├── parity.rs       # Python/Rust 模型 parity 与误差预算
        │   ├── quant.rs        # W8/W4 权重量化与后端快路径
        │   └── tests.rs        # GPT、cache 与量化集成测试
        ├── checkpoint.rs       # 阶段 3: Checkpoint 序列化与 Safetensors 对接
        ├── benchmark.rs        # Prefill/decode、量化与 speculative 基准
        ├── optim.rs            # 阶段 4: Polar Express Muon + AdamW 混合正交优化器
        ├── engine.rs           # 阶段 4/5: 训练与推理引擎底座、BPB 评估器
        ├── experiment.rs       # 强类型 TOML 实验配置、校验与持久化
        ├── engine/
        │   ├── calculator.rs   # 内置 Tool-Use 计算器状态机算子
        │   ├── pretrain.rs     # 阶段 4/5: 异步预训练工作流
        │   ├── inference.rs    # 阶段 5: 支持 KV-Cache 的批量自回归采样
        │   ├── rl.rs           # 组内归一化 REINFORCE 工作流
        │   ├── speculative.rs  # 阶段 5: 无损推测解码双模型推理引擎 (Draft + Target Model)
        │   ├── sandbox.rs      # 阶段 6: 带超时和输出限制的 Python 子进程
        │   ├── scheduler.rs    # Continuous batching admission/cancel 调度器
        │   ├── serving.rs      # 动态请求迭代、取消和 KV cache slot 回收
        │   ├── eval.rs         # 阶段 6: 评测子系统及 benchmark 评估 (gsm8k, spellingbee 等)
        │   └── sft.rs          # 阶段 6: 监督微调 (packed SFT) 工作流
        └── bin/
            ├── train.rs        # 训练入口 (支持 --pretrain, --sft, --rl 参数动态切换)
            ├── eval.rs         # 多任务评测入口
            ├── report.rs       # 训练、BPB、吞吐与质量汇总
            ├── bench_infer.rs  # Prefill/decode 与量化基准
            ├── bench_spec.rs   # 推测解码 acceptance/speedup 基准
            ├── bench_ops.rs    # RMSNorm、RoPE、Softmax 与 attention 算子基准
            ├── chat.rs         # CLI 命令行多轮流式对话客户端
            └── chat_web.rs     # Axum SSE 多轮对话服务器
```

---

## ⚡ 快速开始与命令指南

### 1. 运行单元测试
单元测试默认使用 F32 NdArray 后端：
```bash
cargo test
```

需要验证 WGPU/f16、原生量化或显存统计时额外启用 WGPU feature；普通测试仍保持 NdArray：

```bash
cargo test --features wgpu gpt::parity::test_f16_w8_w4_logit_error_budgets -- --nocapture
```

Python nanochat parity fixtures 位于 `data/fixtures/parity/`。Tokenizer fixture 覆盖 BPE
训练结果、普通 token IDs、conversation masks、tool parts、截断和 completion rendering；module
fixture 使用固定输入和参数验证 RMSNorm、RoPE、MLP 与含 value embedding gate 的 GQA attention。
Full-model fixture 进一步验证两层 GPT 的 logits、mean loss 和代表性参数梯度。Rust 测试不依赖
Python，并以同一模型验证 full、chunked、非均匀 chunk 和逐 token cache logits。Optimizer
fixture 验证 AdamW 与宽/长矩阵 Muon 的单步参数和状态更新：

```bash
cargo test tokenizer::tests::test_python_tokenizer
cargo test gpt::parity
cargo test optim::parity
cargo test --features wgpu gpt::parity::test_f16_w8_w4_logit_error_budgets -- --nocapture
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
uv run --no-project tools/parity_report.py
```

无 GPU 环境可使用 `--backend ndarray`；`--backend wgpu` 可单独验证加速后端，
`--output -` 将 Markdown 输出到标准输出。任一测试失败、指标缺失或误差超出预算时，
命令仍会生成带诊断信息的报告并返回非零状态。

使用脚本声明的固定 Python 依赖重新导出 fixtures：

```bash
uv run --no-project tools/export_tokenizer_parity.py --nanochat-root /path/to/python-nanochat
uv run --no-project tools/export_torch_parity.py all --nanochat-root /path/to/python-nanochat
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
cargo run --release --no-default-features --features wgpu --bin bench_infer -- --artifact runs/sft --batches 1,2,4
cargo run --release --no-default-features --features wgpu --bin bench_infer -- --artifact runs/sft --quantization 4
cargo run --release --no-default-features --features wgpu --bin bench_spec -- runs/sft runs/pretrain
cargo run --release --no-default-features --features wgpu --bin bench_ops
```

报告写入 `runs/report.json`，推理及算子基准写入 `runs/benchmarks/`。训练报告记录各指标时刻观测到的
进程 RSS 峰值；推理基准记录 prefill、首 token、median TPOT、decode tokens/s、batch scaling、理论 KV cache
字节数，以及 CubeCL allocator 实测的设备 `bytes_in_use`/`bytes_reserved` 峰值。NdArray 没有设备
allocator，相关字段为 `null`，不会用进程内存冒充显存。

同一 SFT artifact 的小规模对齐对照可用两个输出目录运行；随后分别执行 `eval`，再交给
`report` 汇总，报告会从 `experiment.toml` 区分两种算法：

```bash
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/reinforce cargo run --no-default-features --features wgpu --bin train -- --rl \
  --rl-algorithm group_normalized_reinforce
NANOCHAT_OUTPUT_ARTIFACT=runs/compare/grpo cargo run --no-default-features --features wgpu --bin train -- --rl \
  --rl-algorithm grpo
NANOCHAT_ARTIFACT=runs/compare/reinforce cargo run --no-default-features --features wgpu --bin eval
NANOCHAT_ARTIFACT=runs/compare/grpo cargo run --no-default-features --features wgpu --bin eval
cargo run --bin report -- runs/sft runs/compare/reinforce runs/compare/grpo
```

### 端到端 Tiny Recipe

仓库内置一套离线小文本 recipe，一条命令完成 tokenizer 训练、pretrain、SFT 和 eval：

```bash
cargo run --bin train -- --recipe --config configs/tiny.toml
```

完整配置位于 `configs/tiny.toml`，可复现输入位于 `data/fixtures/tiny/`，产物写入
`runs/tiny/`。它用于快速验证完整实验链路；模型只有一层且仅训练少量步骤，因此评测分数不代表
实际模型能力。加入 `--no-default-features --features wgpu` 可使用 WGPU 后端运行同一 recipe。

### 2. 基础预训练 (Pretraining)
训练命令默认读取 `configs/mini.toml`。该文件统一描述模型、随机种子、预训练语料、数据路径、
三阶段训练参数、评测任务和 artifact 链路；未知字段或非法组合会在设备初始化前报错。启动小型
合成数据预训练：
```bash
cargo run --release --no-default-features --features wgpu --bin train -- --pretrain
```

使用自定义配置：

```bash
cargo run --release --no-default-features --features wgpu --bin train -- --pretrain --config configs/mini.toml
NANOCHAT_CONFIG=configs/mini.toml cargo run --release --no-default-features --features wgpu --bin train -- --pretrain
```

每个训练产物都会保存一份实际配置为 `experiment.toml`。

`pretrain.model.sequence_len` 是模型的最大上下文容量，随模型 artifact 固定。Pretrain 和 SFT
默认继承该值；若某阶段需要更短的训练序列，可在对应的 `training` 表中增加
`sequence_length = 128`，但不能超过模型容量。RL 的 `max_generation_tokens` 是生成预算，不会改变模型容量。

预训练默认每 5 步更新一次可恢复 checkpoint；中断后从同一 artifact 继续：

```bash
NANOCHAT_RESUME_ARTIFACT=runs/pretrain cargo run --release --no-default-features --features wgpu --bin train -- --pretrain
```

可通过 `NANOCHAT_CHECKPOINT_INTERVAL` 调整保存间隔，设为 `0` 时仅保存最终状态。设置
`NANOCHAT_OUTPUT_ARTIFACT` 可将恢复后的训练写入新目录，并保留 checkpoint 之前的 metrics 历史。

### 3. 监督微调 (SFT)
加载 `runs/pretrain/`，执行 Packed SFT，并输出 `runs/sft/`：
```bash
cargo run --release --no-default-features --features wgpu --bin train -- --sft
```

### 4. 在线强化学习对齐 (RL)
加载 `runs/sft/`，执行基于 GSM8K 的组内归一化 REINFORCE，并输出 `runs/rl/`：
*   **在 GPU (Metal) 上运行**（注：在 macOS GPU 运行时自回归动态切片会导致 Metal 产生 JIT 编译热身耗时）：
    ```bash
    cargo run --release --no-default-features --features wgpu --bin train -- --rl
    ```
*   **选择 WGPU CPU device 运行**：
    ```bash
    BURN_DEVICE=cpu cargo run --release --no-default-features --features wgpu --bin train -- --rl
    ```

### 5. 评测与交互式对话

Eval、CLI Chat 和 Web Chat 默认按 `runs/rl`、`runs/sft`、`runs/pretrain` 的顺序加载最新可用 artifact。可通过 `NANOCHAT_ARTIFACT` 显式指定目录。

```bash
cargo run --release --no-default-features --features wgpu --bin eval
NANOCHAT_CONFIG=configs/tiny.toml cargo run --bin eval
NANOCHAT_ARTIFACT=runs/sft cargo run --release --no-default-features --features wgpu --bin chat
```

*   **CLI 命令行对话客户端**：
    启动基于终端的交互式多轮对话客户端，体验自回归流式生成与内置 Calculator Tool-Use 状态机：
    ```bash
    cargo run --release --no-default-features --features wgpu --bin chat
    ```
    *（使用默认 NdArray 在 CPU 验证：`cargo run --release --bin chat`）*

*   **Web 对话服务端**：
    启动基于 Axum 的流式 SSE Web 服务。服务通过单一迭代 worker 动态接纳请求，并将不同上下文
    位置的 active 请求合入一次 batched decode forward；完成、断连或输出积压时会回收对应 KV
    cache slot。`NANOCHAT_MAX_BATCH` 控制同时 active 的请求上限（默认 8）：
    ```bash
    NANOCHAT_MAX_BATCH=8 cargo run --release --no-default-features --features wgpu,web --bin chat_web
    ```
    启动后可在浏览器中访问 [http://127.0.0.1:8080](http://127.0.0.1:8080)。

---

## 🛠️ 技术规范与开发准则

1.  **泛型后端抽象**：模型、注意力和优化器基于 `<B: Backend>` 或 `<B: AutodiffBackend>`；当前持续验证 WGPU 与 NdArray。
2.  **参考实现优先**：先保留容易阅读和验证的实现，再为热点增加后端特化快路径。
3.  **显式同步边界**：训练日志与评测会发生设备回读；推理保留 CPU `ReferenceSampler` 做语义校验，性能路径使用只回传选中 token 的 `DeviceSampler`。
4.  **可验证主张**：Stable 能力需要测试，性能结论需要基准，parity 结论需要跨语言 fixtures。

---

## 🧭 后续研究方向

当前学习路径、实验闭环、数值 parity、推理基线、并发调度、对齐算法和 GPU 算子里程碑均已
落地。以下方向具有研究和教学价值，但会显著扩大模型、训练或采样系统的复杂度，不作为当前版本
的交付承诺：

- **长上下文 RoPE scaling**：评估 NTK-aware 或 YaRN，同时验证短上下文质量、长上下文任务
  表现和 artifact 配置兼容性。
- **随机采样下严格无损的 speculative decoding**：实现基于 draft/target 概率比的接受拒绝
  采样与残差分布，证明输出分布与独立 target sampling 一致。
- **Medusa heads**：增加多 token 预测 heads 和候选树验证；需要额外训练流程、模型结构、
  checkpoint 格式及与双模型 speculative decoding 的收益对比。
- **多机/多 GPU collective**：保留现有 rank/shard 数据契约，未来评估 Burn collective、参数/optimizer
  state 分片、跨 rank checkpoint 与失败恢复；当前版本不实施，也不把 mock DDP 描述为分布式训练。
- **FP8 训练**：`ModelFloat` 是项目内唯一模型元素类型入口，fp32 AdamW state 也已与模型 dtype
  解耦。待 Burn/WGPU 或 Burn/CUDA 提供可训练的 E4M3/E5M2 element、带 scale/amax 的矩阵与
  attention kernel 及 autograd 支持后，从该入口接入，并以 f16 parity、收敛和吞吐测试作为启用门槛。
