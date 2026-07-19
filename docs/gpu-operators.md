# GPU 算子基线

M7 优先优化已被基准证明的热点。WGPU 类型本身是 Burn Fusion 包装的 CubeCL 后端，
因此普通 element-wise 表达式已经进入图融合；直接维护私有 CubeCL kernel 只适合能稳定跨过
reduction 或 materialization 边界、并有显著收益的算子。

## 可复现基准

```bash
cargo run --release --no-default-features --features wgpu --bin bench -- ops
cargo run --release --no-default-features --features wgpu --bin bench -- ops \
  --batch 4 --sequence 256 --heads 8 --head-dim 64
cargo run --release --no-default-features --features wgpu --bin bench -- ops \
  --batch 1 --sequence 1024 --heads 8 --head-dim 64
```

`bench ops` 每轮都同步设备，分别记录 RMSNorm、RoPE、standalone Softmax、参考 attention、
显式 mask 的 Burn attention 和使用 `is_causal` 的 Burn attention。JSON 默认写入
`runs/benchmarks/operators.json`。

当前 Metal/WGPU f16 环境的测量如下；这些数据用于决定默认策略，不代表其他 GPU 的固定结论：

| shape `[B, T, H, D]` | reference | Burn masked | Burn causal |
|---|---:|---:|---:|
| `[2, 256, 2, 8]` | 3.6065 ms | 3.3878 ms (1.06x) | 4.8168 ms (0.75x) |
| `[4, 256, 8, 64]` | 4.5062 ms | 5.6118 ms (0.80x) | 4.6493 ms (0.97x) |
| `[1, 1024, 8, 64]` | 7.1775 ms | 6.9113 ms (1.04x) | 7.3430 ms (0.98x) |

项目生产路径统一调用 Burn 标准 attention API：完整因果层使用 `is_causal`，SWA 层传入
bool window mask。WGPU/CUDA 的具体 Flash、融合或 fallback 选择属于 CubeCL 后端 autotune
职责；项目不根据单机基准维护容易失效的 shape 阈值。当前 Metal 数据也没有显示某条路径
稳定获胜。reference 仅保留给基准和 Flex/WGPU 数值 parity 测试；本次两条 Burn 路径的
最大误差均未超过测试预算。

## 融合实验结论

- **RMSNorm**：最后一维 reduction 是主要边界；归一化链已由 Burn Fusion 处理。mini 默认形状
  的实测延迟约 2.00 ms，暂不足以支撑维护私有 kernel。
- **RoPE**：乘加链可融合，但切片与拼接仍会 materialize。mini 默认形状约 1.74 ms，暂保留
  泛型实现以维持 Flex/WGPU 一致性。
- **Softmax**：standalone 实现约 2.05 ms；完整 attention 已改用标准 Burn API，使后端可将
  scale、mask、softmax 和 value matmul 纳入融合或 Flash 候选，无需另写单独 Softmax kernel。

后续只有在真实模型 profiling 显示 RMSNorm 或 RoPE 成为稳定热点，并且端到端收益明显高于维护
成本时，才引入 CubeCL 特化实现。
