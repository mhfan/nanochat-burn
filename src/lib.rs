
pub mod common;
pub mod tokenizer;
pub mod dataset;
pub mod dataloader;
pub mod checkpoint;
pub mod gpt;
pub mod optim;
pub mod engine;

/*
    nanochat-burn/
    ├── Cargo.toml
    └── src/
        ├── lib.rs              # 导出所有子模块
        ├── common.rs           # 阶段 0: 设备检测、日志、配置参数
        ├── tokenizer.rs        # 阶段 1: BPE 分词器
        ├── dataset.rs          # 阶段 2: 数据集载入
        ├── dataloader.rs       # 阶段 2: 批处理器
        ├── gpt.rs              # 阶段 3: GPT 架构实现
        ├── checkpoint.rs       # 阶段 3: Checkpoint 序列化与 Safetensors 转换
        ├── optim/              # 阶段 4: 优化器
        │   ├── mod.rs
        │   └── muon.rs         # 极客挑战: Muon 优化器实现
        ├── engine.rs           # 阶段 4/5: 训练与推理引擎
        └── bin/
            ├── train.rs        # 训练入口
            └── chat.rs         # 推理与交互入口
 */
