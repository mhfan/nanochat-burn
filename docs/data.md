# 数据：从文本分片到 next-token batch

本章追踪文本如何变成训练张量，并解释分片、预取和断点位置之间的契约。这里的重点不是数据集
规模，而是保证每个 token 只以预期顺序出现，恢复训练后拿到的下一批数据与不中断时相同。

## 数据契约

预训练分片是无头的 little-endian `u32` token ID 流：

```text
token_0 | token_1 | ... | token_n
 4 bytes   4 bytes          4 bytes
```

文件长度必须是 4 的倍数。加载器优先把 mmap 字节零拷贝解释为 `u32`；地址未对齐时逐块解码，
两条路径产生相同 token 序列。写入前会检查 tokenizer ID 能否装入 `u32`。

语言模型长度为 `T` 的一行需要读取 `T + 1` 个 token：

```math
x = (t_0, t_1, \ldots, t_{T-1}),\qquad
y = (t_1, t_2, \ldots, t_T).
```

因此 batch 所需 token 数是 `batch_size * (sequence_len + 1)`。末尾不足一批的部分不会被拼成
形状不完整的张量，而是切换到该 rank 的下一个分片。

SFT 输入是 conversation JSONL。conversation 先渲染为 token IDs 与 assistant mask，随后同样做一位
右移；padding 使用 BOS token，padding 与非 assistant token 的训练 mask 都为 0。

## 分片、预取与恢复

第 `i` 个分片分配给满足下式的 rank：

```math
i \bmod \text{world\_size} = \text{rank}.
```

`DistributedDataLoader` 在 Tokio 任务中用容量为 4 的 channel 预取 batch。它保存
`shard_idx`、`token_offset` 和 `epoch`；恢复时从这个逻辑位置继续，而不是根据已训练 step 猜测偏移。
一个 rank 走完自己负责的全部分片后 epoch 才加一。

这些约束使数据顺序独立于消费者速度，但不允许恢复到另一个 rank 未被分配的分片，也不允许偏移
超过分片长度。

## 源码入口

- `src/dataset.rs`：文本预分词、little-endian 二进制格式、mmap 数据集和 SFT JSONL。
- `src/dataloader.rs`：next-token shift、rank 分片、异步预取与 `DataLoaderPosition`。
- `src/tokenizer.rs`：conversation rendering、assistant mask 和特殊 token。
- `src/engine/sft.rs`：SFT conversation 的排序、greedy packing、截断和 padding。
- `data/fixtures/tiny/`：端到端 recipe 使用的最小离线输入。

## 最小实验

先验证文本到 mmap token 的往返和格式校验：

```bash
cargo test --features ndarray dataset::tests::test_bin_pretokenization_and_mmap_dataset -- --show-output
```

再验证两个 rank 只读取自己的分片、预取顺序稳定，并且保存位置后的下一批与恢复后完全一致：

```bash
cargo test --features ndarray dataloader::tests::test_distributed_dataloader_prefetch_and_sharding -- --show-output
```

SFT packing 的最小边界测试：

```bash
cargo test --features ndarray engine::sft::tests::test_sft_packer
```

## 正确性证据与观察点

- 二进制往返测试比较原 token IDs 与 mmap 读取结果，而不只检查文件存在。
- dataloader 测试检查 rank 对应的分片选择，并比较 uninterrupted/resumed 的下一批张量。
- SFT packer 测试检查 `T + 1` 行形状、超长样本截断和 padding；tokenizer fixture 另行检查
  conversation assistant mask。
- tiny recipe 将同一数据契约贯穿 tokenizer、pretrain、SFT 和 eval，见训练章节。

## 常见错误

- 把 `.bin` 当成本机端序整数数组。格式固定为 little-endian，手工生成时也必须遵守。
- 只为一行准备 `T` 个 token。目标右移需要 `T + 1`，否则最后一个目标不存在。
- 修改 world size 后直接沿用旧 dataloader 位置。旧 `shard_idx` 可能不再属于该 rank。
- 把 channel 容量理解为训练 batch 数。它只是生产者与消费者之间的预取深度。
- 让 padding 或 user token 参与 SFT loss。只有右移后的 assistant mask 应产生有效 target。
- 认为 mmap 等于所有场景零拷贝。未对齐输入会走安全的 little-endian 解码回退。

## 能力边界

当前格式追求简单和确定性，不携带 shard 元数据、校验和或压缩索引。更换 tokenizer 后必须重新
预分词；artifact 中保存 tokenizer 是为了让模型与 token 语义一起迁移，而不是让旧分片自动兼容
新词表。
