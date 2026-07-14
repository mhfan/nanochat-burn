# nanochat-burn 学习路径

这组文档从数据进入系统开始，沿着一次完整的 mini-LLM 生命周期阅读 Rust 实现。每章固定包含
数学或数据契约、源码入口、可运行的最小实验、正确性证据和常见错误；建议按顺序阅读，也可以按
源码模块独立查阅。

| 章节 | 主题 | 状态 | 主要源码 |
|---|---|---|---|
| 1 | [Tokenizer：从字节到训练目标](tokenizer.md) | Available | `src/tokenizer.rs` |
| 2 | 数据集、mmap 与 batch | Planned | `src/dataset.rs`、`src/dataloader.rs` |
| 3 | GPT、RoPE、GQA 与残差路径 | Planned | `src/gpt.rs` |
| 4 | Muon、AdamW 与参数分组 | Planned | `src/optim.rs` |
| 5 | 预训练、SFT 与实验恢复 | Planned | `src/engine.rs`、`src/engine/pretrain.rs` |
| 6 | KV cache、采样、量化与推测解码 | Planned | `src/engine/inference.rs` |
| 7 | Group-normalized REINFORCE | Planned | `src/engine/rl.rs` |

## 阅读约定

- `Reference` 路径优先表达数学和数据契约，性能特化必须有 parity 测试保护。
- 示例使用 tiny 输入，只用于观察机制和验证不变量，不代表模型质量或性能结论。
- NdArray 是可移植的快速验证后端；涉及 f16、原生量化或 GPU 行为时再使用默认 WGPU 后端。
- 文档中的命令均从 `burn/` 目录执行。

完整功能状态与后续实验计划见 [ROADMAP.md](../ROADMAP.md)。
