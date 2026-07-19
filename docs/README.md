# nanochat-burn 学习路径

这组文档从数据进入系统开始，沿着一次完整的 mini-LLM 生命周期阅读 Rust 实现。每章固定包含
数学或数据契约、源码入口、可运行的最小实验、正确性证据和常见错误；建议按顺序阅读，也可以按
源码模块独立查阅。

| 章节 | 主题 | 状态 | 主要源码 |
|---|---|---|---|
| 1 | [Tokenizer：从字节到训练目标](tokenizer.md) | Available | `src/tokenizer.rs` |
| 2 | [数据集、mmap 与 batch](data.md) | Available | `src/dataset.rs`、`src/dataloader.rs` |
| 3 | [GPT、RoPE、GQA 与残差路径](model.md) | Available | `src/gpt.rs` |
| 4 | [Muon、AdamW 与参数分组](optimizer.md) | Available | `src/optim.rs` |
| 5 | [预训练、SFT 与实验恢复](training.md) | Available | `src/engine.rs`、`src/engine/pretrain.rs` |
| 6 | [KV cache、采样、量化与推测解码](inference.md) | Available | `src/engine/inference.rs` |
| 7 | [Group-normalized REINFORCE 与 GRPO](alignment.md) | Available | `src/engine/rl.rs` |

## 阅读约定

- `Reference` 路径优先表达数学和数据契约，性能特化必须有 parity 测试保护。
- 示例使用 tiny 输入，只用于观察机制和验证不变量，不代表模型质量或性能结论。
- NdArray 是默认的可移植验证后端；涉及 f16、原生量化或 GPU 行为时显式启用 WGPU feature。
- 文档中的命令均从仓库根目录执行。
- 每个性能结论都需要在目标设备运行相应 benchmark；tiny 测试只验证机制和不变量。
