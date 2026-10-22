# Beauty TIGER 原始轨迹 Schema v2

本采集保存现有三位 SID、Beam=15、词表=769 的实际搜索。没有训练、正负标签、候选过滤、错误定位或重新排序。所有新增说明文档统一位于 `docs/generated/tiger_raw_beauty_20261013/`；数据目录中的本文件为随数据携带的副本。

## 商品与 Token

内部商品索引从 0 开始；Embedding 行号等于内部索引，原 Embedding `item_id` 保留原值。商品 ASIN 不参与 Token 计算。原始 SID 每位为 0..255，第 j 位（j=0,1,2）的 Offset Token 为 `code + j*256 + 1`。因此合法位置区间为 1..256、257..512、513..768。Start、PAD、EOS 的 ID 都为 0，但原搜索固定生成三步，不因 EOS 提前终止。没有位置词表约束或非法 ID 过滤。

## 文件和定位

每个 Split 有独立的 `shard_START_END.npz` 与对应 `shard_START_END.jsonl.gz`。`END` 是不包含的原始样本索引。`samples.jsonl.gz` 为这些 gzip 成员顺序连接，可直接用 Python gzip 读取。它与分片 JSONL 内容相同；读取时选择其中一种，避免重复。JSON 的 `tensor_shard` 相对数据根目录，`tensor_row` 为分片内部从 0 开始的行号。`collection_manifest.json` 记录完成分片、精确行范围、形状、文件 SHA256 和计数；`download_inventory.json` 给出除自身以外所有最终文件的哈希与大小。

## 样本 JSONL

`sample_id=split_原始索引7位补零`，严格区分 train/valid/test，不能用哈希 user_id 替代。`dataset_index` 是原划分的 `sample_index`。`history_item_ids` 是实际截断到最后 50 个商品的历史；`history_sid` 为对应 N×3 原始 SID；`input_ids` 和 `attention_mask` 是原 `t5_collate` 的实际左 Padding 输入。`history_length` 为有效 Token 数，`history_length_items` 为有效商品数，`full_history_length_items` 为截断前商品数。保留 `target_item_id/target_asin/target_sid/target_tokens`。

`steps` 有三项，`depth` 为 1..3。每项 `parents` 包含原排名、完整父前缀（含 Start 和不含 Start 两种）、原累计分数、有效性、占位标记、是否独立前向计算、概率来自哪个父行。`surviving_beams` 是实际 torch.topk 的结果，包含原排名、父索引、新 Token、flattened index、完整新前缀、单 Token Log Probability、父累计分数和新累计分数。父索引从 0 开始；排名从 1 开始。

`final_beams` 保留所有 15 条原始结果：完整 Token、去除 Start 的三位 Token、累计分数、原排名、每层祖先排名。`raw_sid` 仅在位置解码有效时提供；`raw_sid_arithmetic` 始终保留原 Token 减去 Offset 的算术值；非法 Token 及重复 SID 不作过滤。

`gold_teacher_forced` 是自由生成结束后另一冻结前向得到的独立诊断。第 d 步输入只包含 Start 与真实目标前 d-1 位，因果 Mask 防止未来泄漏。记录正确父前缀、真实下一 Token、单步和累计 logprob、排名。排名为 `1 + 严格更高分的词表 Token 数`；并列 Token 数另存。该信息从不参与自由 Beam 选择。

## NPZ 张量

记 N=分片实际样本数，D=3，K=15，V=769，H=128。索引张量均为整数；概率、分数、可选 Logits/Hidden 都无损存为 float32，ZIP 压缩只压缩字节、不量化。模型前向仍使用原运行的 AMP 配置；FP32 存储不表示额外重跑 FP32 模型。

| Key | Shape | 含义 |
| --- | --- | --- |
| next_token_logprobs | N,D,K,V | 原搜索实际 F.log_softmax(logits.float()) 完整结果 |
| next_token_logits（可选） | N,D,K,V | 模型实际 Logits，转换为 FP32 不丢数值 |
| beam_scores_before | N,D,K | 扩展前原父分数 |
| selected_flat_indices | N,D,K | 原 torch.topk 索引，parent*769+token |
| selected_parent_indices | N,D,K | 父 Beam 索引 |
| selected_token_ids | N,D,K | 实际新增 Token |
| selected_cumulative_scores | N,D,K | 原 torch.topk 返回的 FP32 分数 |
| active_beam_mask | N,D,K | 真实有效父 Beam |
| initial_placeholder_mask | N,D,K | 初始化占位父 Beam |
| forward_evaluated_mask | N,D,K | 本轮是否独立执行了该行 T5 前向 |
| logprob_source_parent_index | N,D,K | 原搜索广播复用的源行 |
| parent_prefix_tokens_with_start | N,D,K,3 | 扩展前父路径，实际长度等于 depth，未使用尾部填 -1 |
| parent_last_hidden_state（可选） | N,D,K,H | 最后一层 final norm 输出，当前父前缀末位置，扩展候选前，尚未处理新 Token |
| final_tokens_with_start | N,K,4 | 原始最终全部 Beam |
| final_cumulative_scores | N,K | 最终累计分数 |
| gold_next_token_logprobs | N,D,V | 独立正确前缀条件下完整词表分布 |
| gold_next_token_ids | N,D | 正确下一 Offset Token |
| gold_token_logprobs | N,D | 正确 Token 单步概率 |
| gold_cumulative_logprobs | N,D | FP32 顺序累加诊断分数 |
| gold_token_ranks | N,D | 1+严格更高概率 Token 数 |
| gold_equal_score_token_counts | N,D | 与正确 Token 严格同分的 Token 数，含自身 |
| dataset_indices | N | 原始 Split 索引 |

第一步只有父 0 有效，其他父累计分数为 -1e9。`first_beam_only=True` 时原实现仅执行第一行前向，再广播到 15 行。张量原样保留广播后的分布：只有父 0 的 `forward_evaluated_mask=True`，所有复制行 `logprob_source_parent_index=0`。这些复制行不能被当成 14 条独立计算的真实路径。后两步全部 15 个父 Beam 有效并分别执行前向。不存在未生成路径的伪造概率。

## 离线重建与精确并列

候选分数采用 FP32 `next_token_logprobs + beam_scores_before[...,None]`，按父优先、Token 次优先 flatten。可重建任意 Top-M 的分数、父索引和前缀。离线验证独立重建全部 Top-15 的分数多重集、严格更好候选的覆盖，并验证原选择索引对应的分数、父子路径、排名和最终 Token。

FP32 会出现严格同分候选。CUDA torch.topk 未约定并列排序；CPU 稳定排序不保证产生相同并列索引。实际保存的 `selected_flat_indices` 是原搜索并列顺序的权威记录。重建保留该原顺序，不人为引入新的 Beam 规则。报告明确计数并列步骤和 CPU 稳定排序顺序差异；分数重建要求绝对误差为 0。完整概率归一化容差为 3e-6。

验证程序的 `--cuda-topk` 可在不加载或调用 T5 的条件下，仅对磁盘读回的完整矩阵重新执行原 CUDA Top-K，独立核对全部父索引、Token、排名及分数。Debug 和 1,000 条阶段均执行该检查。正式分片采用 DEFLATE level 3 的标准 NPZ 无损压缩；FP32 数值不变。

## SID 映射与恢复

`metadata/item_sid_mapping.parquet` 包含内部索引、ASIN、原 Embedding 行和 ID、原 SID、Offset Token、碰撞组大小、同 SID 全部商品索引和 ASIN。`sid_mapping_check.json` 只检查与报告，不修复 SID。`run_metadata.json` 包含模型/RQ-VAE checkpoint 哈希、Git commit、完整原配置、GPU、AMP、源码哈希、划分和种子。模型权重不重复放进每个采集目录；使用其标识的既有本地 checkpoint。

采集原子写入两个分片文件后才登记完成。失败保留完整分片，未登记分片在 `--resume` 时重做；完整分片先验证哈希再跳过。原输入或采集设置变化会拒绝混用。GPU OOM 自动缩小批次并记录重试，重新验证基线一致性；单条无法处理则报告该 ID 并停止，不虚报全覆盖。没有以“失败跳过”掩盖缺失行。
