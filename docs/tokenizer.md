# Tokenizer：从字节到训练目标

Tokenizer 是文本、conversation 数据和模型词表之间的边界。本项目使用 byte-level BPE：任何
UTF-8 输入都先由单字节 token 表示，再按训练得到的 merge rank 合并，因此不需要 unknown token。
同一个 tokenizer artifact 必须贯穿 pretrain、SFT、RL、eval 和 inference。

## 数据契约

设预切分后的 piece 集合为 \(W\)，piece \(w\) 在语料中的次数为 \(c(w)\)。对相邻 token pair
\((a,b)\)，训练时重新计算加权频次：

\[
F(a,b)=\sum_{w\in W}c(w)N_w(a,b)
\]

其中 \(N_w(a,b)\) 是 pair 在当前 tokenization 中的相邻出现次数。每轮选择
\(\arg\max F(a,b)\)，将 pair 对应的字节串加入词表，并替换所有不重叠出现。Rust 实现在频次
相同时按左右 token ID 排序，因此 fixture 可重复生成。

编码时从单字节分片开始，反复合并当前可用且 rank 最小的 pair。训练顺序就是 merge rank，较早
学到的 pair 优先：

\[
E(p)=\operatorname{merge}_{\text{lowest rank}}([p_0,p_1,\ldots,p_{n-1}])
\]

请求的 `vocab_size` 包含 256 个基础字节和 9 个 conversation special tokens。若语料已没有可合并
pair，训练会提前结束，因此实际词表大小应通过 `get_vocab_size()` 获取，而不是假定总能达到请求值。

## 源码地图

| 数据流 | 入口 | 关键行为 |
|---|---|---|
| 训练 | [`BpeTokenizer::train_from_iterator`](../src/tokenizer.rs) | regex 预切分、pair 计数、确定性 merge、special token 分配 |
| 普通编码 | `encode_ordinary`、`encode_ordinary_batch` | special token 字符串按普通文本处理；batch 使用 Rayon |
| 解码 | `decode` | 拼接 token bytes，再以 loss-tolerant UTF-8 构造字符串 |
| Conversation | `render_conversation` | 插入角色边界 token，并生成 SFT target mask |
| Completion | `render_for_completion` | 移除最后一条 assistant 答案，只保留待生成前缀 |
| Artifact | `save`、`load` | 使用带版本号的 JSON，并在加载后重建反向映射 |
| SFT 消费方 | [`flatten_sft_batch`](../src/engine/sft.rs) | 对 mask 做 next-token shift，非目标位置写为 `-1` |

当前训练实现每轮 merge 都重新扫描所有 word token lists。设目标 merge 数为 \(M\)，当前语料 token
总数为 \(T\)，核心计数成本约为 \(O(MT)\)；单个 piece 的编码还会扫描候选 pair 并在 `Vec` 中删除，
最坏约为 \(O(n^2)\)。这是便于审阅和 parity 的 reference 实现，大语料训练应换成增量 pair 统计，
但必须保持 merge 顺序一致。

## Conversation 与 Loss Mask

`render_conversation` 返回等长的 token IDs \(z\) 和 mask \(m\)。用户文本、角色边界和
`python_output` 的 mask 为 0；assistant 文本、Python 调用及 assistant end token 的 mask 为 1。
SFT 以当前位置预测下一个 token，因此真正送入 loss 的目标为：

\[
y_t=\begin{cases}
z_{t+1}, & m_{t+1}=1\\
-1, & m_{t+1}=0
\end{cases}
\]

`-1` 是模型 loss 的 ignore index。注意 mask 判断的是目标 token `t + 1`，不是输入 token `t`；
否则每个 assistant span 的第一个 token 会被漏训，并错误训练 span 后的 prompt token。

System message 不使用独立边界，而是与第一条 user message 以两个换行连接。Conversation 必须按
user/assistant 交替；multipart content 只允许出现在 assistant，其中 `python_output` 是环境观察，
不会成为 SFT 目标。

## 最小实验

运行可移植示例：

```bash
cargo run --features ndarray --example tokenizer
```

示例训练一个 tiny tokenizer，然后验证三个不变量：普通文本 encode/decode roundtrip、conversation
IDs 与 mask 等长、assistant span 至少包含一个监督目标。输出会展示普通 token IDs、conversation
token 数以及被监督的 target 数。

运行 Rust/Python fixture parity：

```bash
cargo test --features ndarray tokenizer::tests::test_python_tokenizer -- --show-output
```

它验证固定语料的 merge table、普通编码、conversation rendering、截断和 completion prompt 与
Python nanochat 一致。

## 常见错误

| 症状 | 原因 | 检查方式 |
|---|---|---|
| 模型 embedding 越界 | 模型 `vocab_size` 与 tokenizer artifact 不一致 | 使用 `get_vocab_size()` 建模；artifact loader 会拒绝不一致配置 |
| special token 被拆成字节 | 对 special token 字符串调用了 `encode_ordinary` | 精确边界使用 `encode_special`；conversation 使用 renderer |
| 加载后 decode 为空 | 绕过 `BpeTokenizer::load` 直接反序列化，未重建 inverse mappings | 通过 `load` 读取 artifact，或显式调用 `build_inverse_mappings` |
| SFT 监督位置偏移一位 | 用 `m_t` 过滤 target，而不是 `m_{t+1}` | 对照 `flatten_sft_batch` 的 `1..=sequence_len` |
| 对话在角色边界中间结束 | `max_tokens` 对渲染结果执行硬截断 | 训练前统计长度并预留边界 token；不要把截断结果当完整 conversation |
| system 或 multipart panic | system 后不是 user，或 user 使用 multipart | 在数据导入阶段验证角色交替和 content 类型 |

## 正确性边界

- Python fixture 固定了训练、编码和 conversation 语义，入口在 `src/tokenizer.rs` 的 parity tests。
- `save` 会按 token ID 排序 merge table，使 tokenizer JSON 稳定；`load` 检查格式版本、重复 ID、
  重复 rank 和缺失 special tokens。
- `decode(encode_ordinary(text)) == text` 对合法 UTF-8 文本成立；任意不完整字节序列使用 replacement
  character 解码，这是 `String::from_utf8_lossy` 的明确行为。
