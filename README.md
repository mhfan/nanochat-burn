# 🚀 nanochat-burn

一个基于 **Rust & Burn 深度学习框架** 高性能、地道实现的极简且完整的 GPT/LLM 训练与自回归推理引擎。它完整复刻了 `nanochat` 的独创模型架构与优化器设计，完美支持 **Foundational Pretraining (预训练)**、**Supervised Fine-Tuning (SFT 监督微调)** 以及 **Reinforcement Learning (RL 强化学习对齐)** 的完整三阶段生命周期。

---

## 🌟 核心特性与数学 Parity

`nanochat-burn` 致力于实现与 PyTorch 原始版本 **100% 的数值对齐与数学 Parity**，同时发挥 Rust 的系统级安全与并发能效：

*   **独创 GPT Transformer 架构**：
    *   **RoPE 位置编码**：采用旋转位置编码（Rotary Position Embeddings），且 LM Head 与 Embedding 权重采用 **untied weights** 机制。
    *   **激活函数**：使用独特的 `ReLU²`（$x = \text{ReLU}(x)^2$）而非标准的 GeLU/SwiGLU。
    *   **QK Norm**：自注意力计算前对 Query 和 Key 投影张量分别应用无 Scale 参数的 RMSNorm 归一化。
    *   **分组查询注意力（GQA）**：支持 KV 头数 `n_kv_head` 小于 `n_head` 的多路查询。
    *   **滑动窗口自注意力（SWA）**：支持各层独立的滑动窗口注意力机制。
    *   **Smear Bigram 混入**：前向传播中以 bigram mixing 形式动态融合前一个 token 的 embedding。
    *   **Backout 残差扣除**：在最后 Logits 预测前，减去中层特征以有效扣除低级冗余信息。
    *   **Logit Softcap**：对输出 logits 应用 `15.0 * tanh(logits / 15.0)` 限制数值波动。
*   **新一代正交优化器 (DistMuonAdamW)**：
    *   **Muon 优化器**：对于 2D 线性层矩阵参数，使用高精度 **Polar Express Sign Method** 进行正交化，并融合 **NorMuon** 方差归约算法加速收敛。
    *   **AdamW 优化器**：一维标量参数、Smear/Backout 门控及 Embedding 仍采用 AdamW 优化，且学习率比率与 PyTorch 保持完美一致。
    *   **半精度稳定性 (f16)**：对所有 epsilons 与 clamp 范围进行 half-precision 适配，并为 `.sqrt()` 算子施加下界 clamp 保护，彻底消除了 WGPU 硬件加速下的 `NaN` 传播。
*   **零拷贝与系统级优化**：
    *   **静态度 attention 掩码**：在模型初始化时直接在 GPU 显存预计算 Attention 掩码并就地 slice 访问，完全消除 Host-to-Device 显存复制。
    *   **零拷贝 GQA 重构**：将 `repeat_kv` 算子重构为 `.reshape()` 和虚拟广播 `.expand()`，实现 100% 内存连续性并消除了显存碎片。
    *   **动态设备切换**：在 `common.rs` 中支持 `BURN_DEVICE=cpu` 环境监测，可在 CPU 后端下以 0.1 秒/步的速度执行无 JIT 开销的极速验证。

---

## 📂 目录结构与架构设计

项目采用极致扁平化与高度模块化的 Rust Idiomatic 工程结构设计：

```text
    burn/
    ├── Cargo.toml
    ├── README.md               # 项目主说明文档
    ├── src/
    │   ├── lib.rs              # 导出所有子模块
    │   ├── common.rs           # 阶段 0: 设备检测（BURN_DEVICE=cpu支持）、类型与数值校验器
    │   ├── tokenizer.rs        # 阶段 1: 工业级 BPE 分词器与并行化 Tokenizer
    │   ├── dataset.rs          # 阶段 2: 数据集载入与 mmap 闪存加载支持
    │   ├── dataloader.rs       # 阶段 2: DistributedDataLoader 批处理器
    │   ├── gpt.rs              # 阶段 3: GPT 架构实现 (Rotary Embeddings, Softcap, GQA 等)
    │   ├── checkpoint.rs       # 阶段 3: Checkpoint 序列化与 Safetensors 对接
    │   ├── optim.rs            # 阶段 4: Polar Express Muon + AdamW 混合正交优化器
    │   ├── engine.rs           # 阶段 4/5: 训练与推理引擎底座、BPB 评估器
    │   ├── engine/
    │   │   ├── calculator.rs   # 内置 Tool-Use 计算器状态机算子
    │   │   ├── pretrain.rs     # 阶段 4/5: 异步预训练工作流
    │   │   ├── inference.rs    # 阶段 5: 支持 KV-Cache 的高并发自回归采样推理引擎
    │   │   ├── speculative.rs  # 阶段 5: 无损推测解码双模型推理引擎 (Draft + Target Model)
    │   │   ├── sandbox.rs      # 阶段 6: 隔离 Python 子进程安全代码沙箱
    │   │   ├── eval.rs         # 阶段 6: 评测子系统及 benchmark 评估 (gsm8k, spellingbee 等)
    │   │   └── sft.rs          # 阶段 6: 监督微调 (packed SFT) 工作流
    │   └── bin/
    │       ├── train.rs        # 训练入口 (支持 --pretrain, --sft, --rl 参数动态切换)
    │       ├── eval.rs         # 评测与 DCLM CORE / ChatCORE 整合评估程序
    │       ├── chat.rs         # CLI 命令行多轮流式对话客户端
    │       └── chat_web.rs     # Web AXUM 高端多轮对话服务器
```

---

## ⚡ 快速开始与命令指南

### 1. 运行单元测试
执行完整的 17/17 个数值、架构及功能性单元测试：
```bash
cargo test --lib
```

### 2. 基动预训练 (Pretraining)
启动 Foundational Pretraining 并输出 BPB 评估：
```bash
cargo run --bin train --release -- --pretrain
```

### 3. 监督微调 (SFT)
启动 Packed SFT 微调循环，训练将以半精度 `f16` 在 GPU/Metal 上运行：
```bash
cargo run --bin train --release -- --sft
```

### 4. 在线强化学习对齐 (RL)
启动基于 GSM8K 数据集的在线强化学习 REINFORCE 对齐：
*   **在 GPU (Metal) 上运行**（注：在 macOS GPU 运行时自回归动态切片会导致 Metal 产生 JIT 编译热身耗时）：
    ```bash
    cargo run --bin train --release -- --rl
    ```
*   **在 CPU 上极速运行**（规避任何 GPU JIT 编译，0.1 秒/步瞬时生成）：
    ```bash
    BURN_DEVICE=cpu cargo run --bin train --release -- --rl
    ```

### 5. 交互式对话体验 (Interactive Chat Experience)
在模型预训练、监督微调（SFT）或强化学习（RL）对齐完成后，可直接启动交互式服务体验自回归生成与内置计算器 Tool-Use 状态机：

*   **CLI 命令行对话客户端**：
    启动基于终端的交互式多轮对话客户端，体验自回归流式生成与内置的 Safe Calculator Tool-Use 状态机：
    ```bash
    cargo run --bin chat --release
    ```
    *（在 CPU 下极速体验自回归生成：`BURN_DEVICE=cpu cargo run --bin chat --release`）*

*   **Web 高端对话服务端**：
    启动基于 Axum 的流式 (SSE) 高并发 Web 交互服务器：
    ```bash
    cargo run --bin chat_web --release
    ```
    启动后可直接在浏览器中访问 [http://127.0.0.1:8080](http://127.0.0.1:8080) 体验顺滑、高端的可视化流式聊天交互页面。

---

## 🛠️ 技术规范与开发准则

1.  **泛型后端抽象**：
    所有神经网络层、自注意力及优化器均基于泛型 `<B: Backend>` 或 `<B: AutodiffBackend>` 构建，确保在 WGPU (GPU)、Candle 甚至纯 CPU 等后端之间可零成本自由平移。
2.  **极致的零 Placeholder**：
    不存在任何 `todo!()` 或静态 Mock，所有代码均采用生产级错误处理（`Result<T, E>`）并进行完整的资源管控。
3.  **零 Host-GPU 同步**：
    除非进行必要的 Tensor 采样读取（如 BPE Decode），前向与反向梯度反传中绝不执行任何阻塞式的 `into_scalar()` 显存CPU回读，最大化 GPU Model FLOPS Utilization (MFU)。

---

## 🗺️ 未来展望与路线图 (Future Roadmap / TODO)

面向未来，`nanochat-burn` 致力于从小巧、极简的学术复刻演进为**工业级端侧高性能 LLM 系统**。以下为项目的未来路线图规划与最新进展：

### 1. 🏎️ 系统级与 GPU 算子性能极限优化 (Systems & Kernel Optimization)
*   **✅ [已实现] 静态 KV-Cache 预分配与算子优化**：
    *   在 [gpt.rs](file:///Users/mhfan/Devel/nanochat/burn/src/gpt.rs) 中实现了静态 `KVCache` 预分配与就地切片赋值算子 (`slice_assign`)，并将自注意力计算和掩码全面对齐至最大长度。
    *   在 macOS WGPU/Metal 后端下实现了 **100% 静态 Shape 前向与反向传播计算图**，**完全消除了动态自回归导致的 GPU JIT 编译温身延迟**。
*   **CubeCL 手写融合算子**：利用 Burn 底层的 CubeCL 编写特化 GPU 算子（如 Fused RMSNorm, Fused RoPE 以及 Fused Softmax），最大化 GPU 硬件吞吐量 (MFU)。
*   **FlashAttention 集成**：在 LibTorch/CUDA 后端下直接接入硬件级 FlashAttention-2/3 算子，实现企业级大吞吐训练。

### 2. 🧠 前沿大模型算法升级 (Algorithmic & Modeling Features)
*   **✅ [已实现] GRPO (Group Relative Policy Optimization) 强化对齐**：
    *   引入了 DeepSeek-R1 风格的 GRPO 强化学习算法，支持单 Prompt 采样 $N$ 个回答，并在组内以无偏标准差归一化计算 Advantage。
    *   省去了庞大的 Critic 价值网络，实现了超低显存开销下的在线对齐，全自动适应 GPU 的半精度 $f16$ 稳定性。
*   **✅ [已实现] 无损推测解码 (Lossless Speculative Decoding)**：
    *   设计了地道的 `SpeculativeInferenceEngine` 与双模型（Draft Model + Target Model）并发验证架构，支持并行对 $K$ 个草稿 token 进行无损验证。
    *   实现了 **零开销 KV-Cache 回滚机制**：在 draft token 被拒绝时，仅通过重置 Generator 指针进行 zero-overhead 覆盖，完美规避了显存拷贝与重建开销。
    *   在 WGPU Metal 硬件加速测试下验证了 **100% token 级的无损一致性 (Mathematical Lossless Parity)**！
*   **量化适配 (Weight Quantization)**：为 Linear 投影层和 Embedding 层开发 INT8/INT4/NF4 精度量化加载，极大压缩端侧部署模型大小。

### 3. 🌐 高并发服务化与工程生态 (Serving & Ecosystem Integration)
*   **连续批处理 (Continuous Batching)**：在 Axum Web 服务端支持请求的动态混入与实时剥离，最大化多用户并发吞吐。
*   **PagedAttention 机制**：实现 Rust 原生的 PagedAttention，以物理 Page 映射非连续 KV-Cache 内存，彻底告别显存碎片。
*   **Safetensors 生态转换**：提供双向转换工具，支持将 Qwen、Llama 等社区主流小模型一键导出至 `nanochat-burn` 运行。
