# CoTIGER — Collaborative-Guided TIGER for Generative Retrieval

围绕 TIGER 的独立生成式推荐项目，聚焦 **自回归前缀错误传播、协同信号引导解码和推理加速**。在冻结的 TIGER 上引入 Train 商品转移统计，引导 Semantic ID 前缀搜索，并通过稀疏查表和 KV Cache 优化降低推理开销。

项目围绕一个具体问题展开：**正确 SID 的早期前缀被 Beam 剪掉后，后续解码无法恢复目标商品。如何利用用户历史中的协同信号改善搜索，同时保留可核对的评分规则？**

## 项目实现

- **历史前缀校准**：从 Train 构建商品 SID 转移计数，按最近三件历史的时间衰减计算 Log-Lift，在 D2 引导搜索、D3 校准最终排序。
- **公平对照**：比较原版 TIGER、纯 SID 转移推荐器、CoTIGER；同时保留 Beam=15/30 的结果，区分统计信号贡献与模型互补收益。
- **无损推理加速**：预计算 Log-Lift、CSR 转移存储、用户级奖励表、GPU 批量索引、已见 SID 过滤，以及仅重排 Self-Attention KV Cache。
- **输出验收**：完整 Validation 逐样本核对 Top-10 SID 与顺序，并核对最终全部 Beam；小样本进一步检查三层路径及 FP32 原始累计分数。

仓库只包含 TIGER、生成 SID 必需的 RQ-VAE，以及本项目的改进和诊断代码。TIGER 基础实现从 [GenRec](https://github.com/phonism/genrec) 提取至 `cotiger/core/`；不依赖完整 GenRec 仓库，也不包含其他推荐模型。本项目沿用原始 T5 结构与既有 100-epoch TIGER checkpoint，历史校准无需梯度训练。

```text
cotiger/
├── core/                     # TIGER、T5、RQ-VAE、Amazon 数据和原训练/评估流程
├── prior.py                  # 历史转移 Log-Lift 参考公式
├── reference.py              # 校准 Beam Search 参考实现
├── decoding.py               # CSR/GPU 查表、用户奖励复用、KV Cache 加速
├── evaluate.py               # 原版/校准、参考/加速实现的统一评估入口
└── statistics.py             # 仅从 Train 重建转移统计
tiger_prefix_ft/              # Beam 诊断、原始轨迹采集、负例加工和微调研究
config/tiger/amazon/          # RQ-VAE 与 TIGER 训练配置
tests/                       # 不依赖外部权重的数据、搜索及梯度检查
scripts/                     # TIGER 小规模流程、分析导出和备份核对
docs/generated/tiger_prefix_ft_beauty_20261013/
                             # 集中保存方法、结果和复现说明
```

另保留 CE + Prefix InfoNCE、Trie 和 Beam 宽度的研究代码，便于复查已有实验。这些可选分支不参与默认 CoTIGER 推理；主要结果使用原版权重、无 Trie、完整词表搜索。

## 实验结果

Amazon Beauty 5-core：Train **131,413** 条，Validation **22,363** 条；12,101 个商品对应 **11,401** 个唯一三位 SID。下表为完整 Validation 的 **SID 匹配指标**，数值均为百分比。

| 方法 | Beam | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原版 TIGER | 15 | 5.9742 | 8.9120 | 3.9788 | 4.9222 |
| CoTIGER（默认） | 15 | 8.2100 | **11.4341** | 5.6649 | **6.7024** |
| 原版 TIGER | 30 | 6.0010 | 9.0328 | 4.0081 | 4.9823 |
| CoTIGER | 30 | 8.2458 | **11.5369** | 5.6926 | **6.7532** |
| 纯 SID 转移统计（最近三项，平滑 20） | 全目录 Top-10 | — | 10.2178 | — | 6.4953 |

默认 Beam=15 的 Recall@10 相对原版提高 **28.30%**，绝对提高 **2.5220 个百分点**。Beam=30 相对原版提高 **27.72%**，相对较强的纯统计基线提高 **12.91%**。

统计推荐器本身已有较强效果，收益包含额外的协同信号。历史 Test 结果中，CoTIGER 全 30 相对纯统计基线的 Recall@10 为 8.5722% vs 7.9730%，NDCG@10 为 5.0632% vs 5.0925%；没有在所有指标上超过统计基线。Test 已用于历史分析，不能视为尚未接触的全盲测试。完整对照与限制见[实验记录](docs/generated/tiger_prefix_ft_beauty_20261013/cotiger_results.md)。

### 无损加速

RTX 4060 Laptop GPU（8187.5 MiB），PyTorch 2.6.0+cu124；相同 Batch=64、FP16 前向、FP32 LogSoftmax 和累计分数。下表比较**同一校准算法**优化前后，采用同进程交错执行、每次完整预热、两次热运行的中位数，计时包含评估及相同的 CPU 输出收集，不含加载与落盘。

| Beam | 校准参考实现 | 优化实现 | 加速比 | 峰值显存（前 → 后） | 优化后吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 | 18.91 s | **10.48 s** | **1.80×** | 1918 → 1082 MiB | 2134 条/s |
| 30 | 30.86 s | **16.61 s** | **1.86×** | 3807 → 2046 MiB | 1346 条/s |

完整 Validation 的 Top-10 SID、顺序及最终全部 Beam **完全一致**。不同时间段的 Windows/GPU 耗时有明显波动；不能把旧日志中的 76.35 s 与另一时间段的 16.61 s 相除作为加速比。

## 快速复现

### 1. 安装

在有 CUDA 的机器上使用 Python 3.11。验证环境的依赖版本记录在 `requirements.txt`；不要在已有环境中盲目覆盖 PyTorch。

```bash
git clone https://github.com/wdyy20041223/CoTIGER.git
cd CoTIGER
conda env create -f environment.yml
conda activate cotiger
```

`environment.yml` 对应 Windows + CUDA 12.4。Linux 或已有 CUDA 环境可先安装匹配驱动的 PyTorch，再运行 `pip install -r requirements.txt` 和 `pip install -e .`。研究脚本可使用 `pip install -e ".[research]"`。无需安装 PyTorch Geometric、Triton 或 LoRA 组件。复现实验请尽量保持 PyTorch、精度、Batch 及 Padding 相同；不同 GPU 内核可能影响接近并列的 FP16 排序。

### 2. 获取必要资产

下载 [Beauty v1 发布包](https://github.com/wdyy20041223/CoTIGER/releases/tag/beauty-v1) 中的 `cotiger-beauty-inference-v1.zip`，在仓库根目录解压。包内提供冻结权重、配置、SID、Train/Validation 加工数据、Train 统计及参考预测，不需要下载 Sentence-T5 或重新训练 RQ-VAE。

```text
artifacts/beauty/
├── best_model.pt
├── model_config.json
├── item_semantic_ids.npy
├── train_samples.parquet
├── valid_samples.parquet
├── train_statistics.npz
├── manifest.json
├── reference/                 # 原版与校准方法的完整 Validation 预测
└── evidence/                  # 原始实验摘要与逐样本一致性证明
```

发布包附有 SHA256 校验文件；逐文件来源、大小和摘要见[资产清单](docs/generated/tiger_prefix_ft_beauty_20261013/cotiger_artifacts.json)。数据及权重作为实验资产单独发布，Git 历史仅保存代码和必要说明。

### 3. 小样本核对与完整评估

以下命令均从仓库根目录执行；`--output` 必须是尚不存在的目录，避免覆盖结果。

```bash
# CPU 检查，无需下载权重；使用微型合成模型验证搜索、Padding 和梯度
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s tiger_prefix_ft -t . -p "test_*.py" -v

# 首批 64 条真实 Validation：与发布包中的同批次参考预测严格比较
python -m cotiger.evaluate --max-samples 64 --check-reference --output out/cotiger/smoke15
python -m cotiger.evaluate --beam 30 --max-samples 64 --check-reference --output out/cotiger/smoke30

# 完整 Validation；默认全三层 Beam=15、GPU CSR 先验
python -m cotiger.evaluate --warmup-full --check-reference --output out/cotiger/valid15
python -m cotiger.evaluate --beam 30 --warmup-full --check-reference --output out/cotiger/valid30

# 同一 checkpoint 的原版 TIGER
python -m cotiger.evaluate --method tiger --check-reference --output out/cotiger/original15

# 保留的校准参考实现；用于和 fast 实现比较
python -m cotiger.evaluate --engine reference --beam 30 --warmup-full --check-reference --output out/cotiger/reference30

# 纯统计基线，无需运行 TIGER 或使用 GPU
python -m cotiger.transition_baseline --output out/cotiger/sid_markov
```

每次评估输出 `result.json`（四项指标、耗时、吞吐量、显存）及 `predictions.npz`（逐样本有序 SID 与 sample_id）。`--check-reference` 比较最终全部 Beam，而不只是聚合指标。严格参考核对固定 Batch=64；小样本必须取完整 Batch。

`--repeats 2` 可重复热运行；`--prior-backend cpu` 使用等价的 CPU 用户级奖励表。性能测试请将 reference 与 fast 的执行顺序交错，并报告本机实测；不要直接套用上表秒数。

### 4. 核对 Train 统计来源

```bash
python -m cotiger.statistics --output out/cotiger/rebuilt_train_statistics.npz
```

此步骤只读取发布包内的 131,413 条 Train，不进行模型训练。统计来自最后一个历史 SID 到目标 SID 的观察转移，Validation 目标不进入计数。重建文件可按 NPZ 中的各个数组逐一核对。

## 评分规则与代码

对深度 `d` 的候选前缀 `p`：

```text
lift(h,p) = log(1 + C_train(h,p) / (20 * P_global(p)))
bonus(H,p) = Σ_age 0.5^age * lift(history[-1-age],p)
selection_score = raw_cumulative_logP + alpha[d] * bonus(H,p)
alpha = [0, 0.4, 0.4]
```

最近三项中的重复 SID 只使用最近一次，保留原始时间距离。每一层重新计算该层前缀奖励，奖励不累计到下一层原始 LogP。D3 对历史中已见且仅对应单个商品的 SID 减 1000 分；SID 碰撞组保留。三层始终搜索完整 769 词表，未匹配合法前缀的候选奖励为 0。默认全三层 Beam=15，未启用 Trie 或位置码本过滤。

| 文件 | 职责 |
| --- | --- |
| `cotiger/core/models/` | 原始 TIGER 和 SID 量化用 RQ-VAE；检查点参数名保持兼容 |
| `cotiger/core/modules/t5.py` | 原 T5、逐层 Beam Search、可选轨迹记录与 Trie 接口 |
| `cotiger/core/trainers/` | RQ-VAE/TIGER 训练与原 Recall/NDCG 评估器 |
| `cotiger/evaluate.py` | 冻结模型加载、原评估器复用、输出核对及计时 |
| `cotiger/prior.py` | 原校准公式与字典参考实现 |
| `cotiger/reference.py` | 保留实际逐层 Top-K 的搜索参考实现 |
| `cotiger/decoding.py` | 紧凑索引、CSR GPU 查表、奖励复用及 KV Cache 优化 |
| `cotiger/statistics.py` | 从固定 Train 重建转移统计 |
| `cotiger/transition_baseline.py` | 最近三项平滑 SID Markov 基线 |
| `cotiger/data.py`、`cotiger/ids.py` | 相同历史截断、左 Padding、Offset 与 SID 索引 |
| `tests/test_cotiger.py` | 路径、分数与先验边界检查 |

更详细的公式、搜索行为和优化边界见[方法说明](docs/generated/tiger_prefix_ft_beauty_20261013/cotiger_design.md)。

## 研究复现与资产边界

[完整复现说明](docs/generated/tiger_prefix_ft_beauty_20261013/reproduction.md) 列出了从原始 Beauty、文本向量、RQ-VAE、TIGER 到校准评估的步骤，以及轨迹采集、H2/H4/H6 数据准备和逐层诊断的入口。训练和原始轨迹采集需自行准备输入；命令不会随安装自动运行。

- [推荐效果改进报告](docs/generated/tiger_prefix_ft_beauty_20261013/inference_optimization_20261016.md)
- [无损工程加速报告](docs/generated/tiger_prefix_ft_beauty_20261013/prior_speed_20261016.md)
- [当前码本使用情况](docs/generated/tiger_prefix_ft_beauty_20261013/codebook_usage_20261020.md)
- [独立仓库复现核查](docs/generated/tiger_prefix_ft_beauty_20261013/reproducibility_audit_20261022.md)

Git 只保存源码、配置、测试和必要说明，不保存模型权重、数据集、Parquet、原始 Beam 轨迹或云端登录信息。已有 Beauty v1 Release 可选用于精确重放主推理结果；历史微调检查点、完整原始轨迹和全部 RQ-VAE 快照不在该包中。删除本地实验目录前，应自行保留仍需分析的这些文件。仅克隆代码不能恢复未上传的历史数据。

## 来源与许可

模型、RQ-VAE、T5 及原始训练/评估基础设施来自 [phonism/GenRec](https://github.com/phonism/genrec)，在 `cotiger/core/` 中按 TIGER 所需依赖提取，保留 MIT 许可与上游作者信息。本项目的工作集中于历史转移校准、对照分析、推理优化和输出验证；不将上游 TIGER/RQ-VAE 描述为个人原创。

代码沿用 [MIT License](LICENSE)。Beauty 来自 Amazon Reviews；加工数据与实验权重是复现资产，使用时仍需遵守源数据的使用条件。当前结果针对这一数据集、checkpoint 和 SID 级评价协议，尚未证明跨数据集泛化或独立商品级检索效果。
