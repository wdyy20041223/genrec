# CoTIGER 复现说明

本仓库是独立的 TIGER 项目。`cotiger/core/` 提取了 GenRec 中 TIGER、T5、RQ-VAE、Amazon 数据和必要训练工具；不需要安装或克隆完整 GenRec。运行以下命令前，先进入仓库根目录并激活 Python 3.11 环境。安装方法见[README](../../../README.md)。

## 1. 两种复现范围

- **复核主推理结果**：使用已有 [Beauty v1 Release](https://github.com/wdyy20041223/CoTIGER/releases/tag/beauty-v1)。它包含原版 100-epoch TIGER 权重、三位 SID、Train/Validation 加工数据、训练统计和参考预测。不需要 Sentence-T5、RQ-VAE 权重或原始评价轨迹。
- **重新训练或重建研究数据**：自行准备 Amazon Reviews 2014 Beauty 5-core、商品 metadata、Sentence-T5-XL、RQ-VAE 和 TIGER 训练产物。历史微调权重、完整 Beam NPZ/JSONL、全部 RQ-VAE 快照没有随 Git 上传；不能通过克隆恢复。重新训练生成的 SID 或模型可能不同，不能据此声称逐位复现原实验。

代码安装、单元测试和 `--help` 不会启动训练或下载模型。原始数据集类在缺少文件时支持下载；只有明确准备从头训练时才执行训练命令。

## 2. 无资产检查

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s tiger_prefix_ft -t . -p "test_*.py" -v
python -m cotiger.evaluate --help
python -m cotiger.core.trainers.rqvae_trainer --help
python -m cotiger.core.trainers.tiger_trainer --help
```

测试使用很小的合成张量和随机模型验证实现。`test_finetune` 中需要真实 Beauty 文件的检查在缺少资产时明确跳过；资产无关的梯度/分数检查在 `tests/test_prefix_loss.py` 中独立执行。不要把这些检查当作推荐效果实验。

## 3. 使用已有推理资产

将 Release ZIP 解压到根目录，得到 `artifacts/beauty/`。先核对 ZIP 的 SHA256：

```text
cotiger-beauty-inference-v1.zip
07364a212a40014a8c0bb537c36cdd24632760b4346e375e06a4f8232ea4d048
```

Windows 可用 `Get-FileHash ... -Algorithm SHA256`，Linux 可用 `sha256sum ...`。包的旧文件名、数据格式与当前独立包兼容。

```bash
# 每个输出目录必须尚不存在。
python -m cotiger.evaluate --max-samples 64 --check-reference --output out/cotiger/smoke15
python -m cotiger.evaluate --beam 30 --max-samples 64 --check-reference --output out/cotiger/smoke30
python -m cotiger.evaluate --engine reference --beam 30 --max-samples 64 --check-reference --output out/cotiger/reference_smoke30

# 完整 22,363 条 Validation；不训练。
python -m cotiger.evaluate --warmup-full --repeats 2 --check-reference --output out/cotiger/valid15
python -m cotiger.evaluate --beam 30 --warmup-full --repeats 2 --check-reference --output out/cotiger/valid30
python -m cotiger.evaluate --method tiger --check-reference --output out/cotiger/original15
python -m cotiger.transition_baseline --output out/cotiger/transition_baseline
python -m cotiger.statistics --output out/cotiger/rebuilt_train_statistics.npz
```

默认校准参数为 `alpha=[0,0.4,0.4]`、最近三项、衰减 0.5、平滑 20，三层 Beam=15；无 Trie、无位置过滤。`--method tiger` 使用相同权重的原始搜索。严格对照固定 Batch=64、FP16 前向、FP32 LogSoftmax，比较最终全部有序 Beam；小样本须取完整 Batch。不同设备或 PyTorch 内核对接近并列的 FP16 排序可能有差异，应保留实际检查结果。

`predictions.npz` 包含有序 `tokens` 和 `sample_ids`，`result.json` 包含 Recall/NDCG、计时和显存。每次独立执行均有首批预热；`--warmup-full` 再完整预热一次。正式比较 reference/fast 应交错执行，使用相同的预热、Batch、精度与收集方式。历史 18.91→10.48 s、30.86→16.61 s 是同机匹配计时，不能与其他时段日志混算。

## 4. 从原始数据开始

使用 Amazon **2014** Beauty，不替换成 2018/2023 版本。数据代码保留 GenRec 的清洗、商品顺序和 leave-one-out 划分。准备：

```text
dataset/amazon/raw/beauty/reviews_Beauty_5.json.gz
dataset/amazon/raw/beauty/meta_Beauty.json.gz
models_hub/sentence-t5-xl/          # 与历史实验相同的文本编码器
```

先自行检查文件与模型来源；本次整理未下载这些资产。Sentence-T5 仅用于商品向量，TIGER 本身是小型 T5，不能用通用预训练 T5 权重替代。

以下是正式训练入口示例，会进行长时间训练，仅在需要重新训练时自行执行：

```bash
python -m cotiger.core.trainers.rqvae_trainer config/tiger/amazon/rqvae.gin --split beauty --gin "train.wandb_logging=False"
python -m cotiger.core.trainers.tiger_trainer config/tiger/amazon/tiger.gin --split beauty --gin "train.wandb_logging=False" --gin "train.save_dir_root='out/tiger/amazon/{split}/tiger'"
```

当前配置 RQ-VAE 2000 epoch、三层码本各 256，TIGER 100 epoch、4 Encoder/4 Decoder、d_model=128。TIGER 配置指向 `out/tiger/amazon/beauty/rqvae/checkpoint_epoch_1999.pt`。上面的 TIGER 命令显式将模型产物写入 `beauty/tiger/`，与后面的导出器目录结构对齐；不传覆盖参数时原配置写入 `beauty/`。`checkpoint_epoch_1999` 是零起始名称，对应 RQ-VAE 第 2000 epoch，不是 TIGER 的 100 epoch。

配置中的文本模型路径可用重复 `--gin` 覆盖，例如 `--gin "MODEL_HUB_SENTENCE_T5_XL='/your/local/sentence-t5-xl'"`。Windows 如需避免多进程开销，可加 `--gin "train.num_workers=0"`。保持 `add_disambiguation=False`，完整 SID 为三位，Offset `[1,257,513]`，0 是 Padding/Decoder Start。

若已有**与历史 Beauty 实验匹配**的 100-epoch 训练目录，可导出分析输入：

```bash
python scripts/tiger_export_analysis.py --run-dir out/tiger/amazon/beauty --dataset-root dataset/amazon --output-dir out/beauty_export/analysis
```

该导出器按原实验进行严格规模核对，读取 `tiger/results.json`、`tiger/resolved.gin`、`tiger/best_model.pt`、`rqvae/checkpoint_epoch_*.pt` 和已处理商品向量，并导出 SID、模型配置、用户序列、Train/Valid/Test 样本和 RQ-VAE 快照。需要 CUDA，会执行原版评价；不是小规模检查命令。旧 resolved.gin 中的 `genrec.` 命名空间在读取时转换为 `cotiger.core.`，不要求安装旧包。

从新导出构造校准资产时，放入同一目录的四个核心输入为 `best_model.pt`、`model_config.json`、`item_semantic_ids.npy`、`valid_samples.parquet`。`cotiger.statistics` 的 Train 输入需具备前缀加工后的 Token 列，下一节说明生成方式。切勿把导出的“商品索引历史 Parquet”与“Token 训练 Parquet”混用。

## 5. 轨迹采集与 H2/H4/H6 数据

这些是可选研究步骤，默认校准推理不需要它们。原始轨迹按 JSONL.GZ+NPZ 成对保存，通过 `tensor_shard`、`tensor_row`、`dataset_index` 核对；不是只截取最终 Beam。

先执行最多 100 条/划分的检查采集：

```bash
python -m tiger_prefix_ft.raw_collection.collect_full_beam --checkpoint out/tiger/amazon/beauty/tiger/best_model.pt --rqvae-checkpoint out/tiger/amazon/beauty/rqvae/checkpoint_epoch_1999.pt --analysis-dir out/beauty_export/analysis --resolved-gin out/tiger/amazon/beauty/tiger/resolved.gin --embedding-parquet dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet --reviews-gzip dataset/amazon/raw/beauty/reviews_Beauty_5.json.gz --output-dir out/beauty_raw_small --splits train valid --max-samples 100 --batch-size 8
python -m tiger_prefix_ft.raw_collection.verify_full_beam out/beauty_raw_small --report out/beauty_raw_small_verification.json
python -m tiger_prefix_ft.prepare_data --raw-root out/beauty_raw_small --output-dir out/prefix_small --max-samples 100 --report out/prefix_small/prepare_report.json
python -m tiger_prefix_ft.check_data --raw-root out/beauty_raw_small --output-dir out/prefix_small --prepare-report out/prefix_small/prepare_report.json --report out/prefix_small/check_report.json
```

通过后，使用新的采集目录，省略 `--max-samples` 执行全量；原始 NPZ 体积很大，应提前预留磁盘。加工阶段始终只使用 Train 目标：

| 文件 | 一行表示什么 | 主要字段 |
| --- | --- | --- |
| `data/train_samples.parquet` | 一个用户历史和完整目标 | `sample_id`、`dataset_index`、`input_ids`、`attention_mask`、`target_tokens`、`target_sid`、原始 NPZ 定位 |
| `data/sid_catalog.parquet` | 一个去重的合法完整 SID | `raw_sid`、`offset_tokens`、`collision_group_size` |
| `data/hard_negatives.parquet` | 某用户、某深度的一个错误前缀 | `sample_id`、`dataset_index`、`depth`、`negative_tokens`、`score`、`gold_score`、`source`、`survived_beam` 等 |

历史全量规模分别为 131,413、11,401、11,824,179 行。历史 Token 输入长度 24～152，保留其原有左 Padding 和 Mask；采样不会给每个前缀再复制用户历史。注意 Release 的 Train 是这一 Token 格式，Valid 是原导出的商品索引历史格式，两个用途不同。

Hard 必须用相同 `sample_id+depth` 与 Positive 关联，分数属于对应深度；仅合法、不同于目标的前缀可作为负例。同 SID 碰撞商品不能互相构成 SID 负例。完整定义见采集器的 [schema](../../../tiger_prefix_ft/raw_collection/schema.md)。

```bash
python -m tiger_prefix_ft.prepare_data --raw-root out/beauty_raw_full --output-dir out/prefix_full --report out/prefix_full/prepare_report.json
python -m tiger_prefix_ft.check_data --raw-root out/beauty_raw_full --output-dir out/prefix_full --prepare-report out/prefix_full/prepare_report.json --report out/prefix_full/check_report.json
python -m tiger_prefix_ft.build_fixed_k8 --data-dir out/prefix_full/data --output-dir out/prefix_full/data --report out/prefix_full/fixed_k8_report.json
```

得到 `train_k8_h2.parquet`、`train_k8_h4.parquet`、`train_k8_h6.parquet`；相同 sample_id 使用相同 rank_depth。三者计划比例为 E3/M3/H2、E2/M2/H4、E1/M1/H6，Depth 1 不采 Medium；不足用真实 Easy 补足并记录实际计数，去重后必须 K=8。每行保存完整 CE labels、当前深度 Positive、8 个 Negative 及类型。

自建主推理资产时，将原版权重、模型配置、SID 和导出的 Valid 放入 `out/beauty_assets/`，然后运行 `python -m cotiger.statistics --train out/prefix_full/data/train_samples.parquet --output out/beauty_assets/train_statistics.npz`。评估时传 `--artifacts out/beauty_assets`。自训权重没有历史参考预测时不要加 `--check-reference`；比较其实际指标，不能期待与历史 checkpoint 逐位相同。

可选微调入口为 `python -m tiger_prefix_ft.finetune --help`，支持 Mixed InfoNCE / ce_only、λ、训练 epoch 和显式检查点路径。当前模型重新计算各自前缀的 Log Probability，负前缀共享 Encoder，缓存 NPZ 分数不用于反向传播。已有实验未证明 CE+InfoNCE 能提升 Recall，不能将其写成主方法的有效改进。本次仓库整理没有重新训练。

## 6. 诊断与历史实验

- `python -m tiger_prefix_ft.beam_diagnosis --help`：读取原版、CE-only 和 H6 的实际检查点，在完整 Validation 上记录每次 Top-15 后的真实存活、首次剪枝、排名/分数、Recall/NDCG。需原分析目录、微调模型和原训练报告；Release 未提供这些微调权重。
- `python -m tiger_prefix_ft.cascade_analysis --help`：检查原始 NPZ 的错误前缀和级联；需真实原始轨迹。
- `python -m tiger_prefix_ft.analyze_sid_collisions --input artifacts/beauty/item_semantic_ids.npy --output-dir out/cotiger/collisions`：仅用 Release 中 SID 即可分析碰撞，不训练。
- `python -m tiger_prefix_ft.analyze_codebook_usage --help`：完整码本审计还需要原导出的码本向量和各 epoch 分配快照。不能仅凭 Release SID 重现全部几何与历史快照结论。
- `python -m tiger_prefix_ft.analyze_sid_similarity --help`：需要原商品文本向量、SID、用户序列和原 Train 导出，用 Train 交互做离线相似度分析。
- `profile_prior_speed`、`benchmark_prior_speed`、`verify_prior_speed`：保留历史精细 Profiling/逐层校验程序，需要原实验输出。只复核主推理输出与速度时，优先使用不依赖旧实验目录的 `cotiger.evaluate`。

历史研究脚本默认目录仍沿用原实验名称，可通过 CLI 的显式路径覆盖。`COTIGER_ROOT`、`COTIGER_BEAUTY_BASE`、`COTIGER_STUDY_ROOT`、`COTIGER_RAW_ROOT`、`COTIGER_PREFIX_DATA` 环境变量在启动 Python 前设置；代码中的默认路径不表示这些数据已随 Git 提供。

## 7. 删除本地目录前

代码、配置、测试和本文可从 Git 恢复；主推理所需资产当前可从已有 Release 恢复。未上传的微调权重、全量轨迹、全部快照、原始模型缓存及历史逐样本中间结果应按需要另行备份。README 中的历史汇总报告保留结论，但不代替全部原始结果。本次不额外上传权重/数据，不执行本地删除或云端操作。
