# TIGER 历史校准推理：无损工程加速

> 历史实验记录：下文默认值和目录名描述实验当时的研究入口。当前独立包默认通过 `cotiger.evaluate` 启用校准、全15；当前复现入口与资产范围见[复现说明](reproduction.md)。文中 `out/` 原始中间结果未随 Git 上传。

本次只使用本机 RTX 4060 Laptop GPU 和 Beauty Validation 的 22,363 条样本。没有训练、修改权重、SID、训练统计或历史实验数据。普通推理的默认配置仍为 15–15–15，历史校准仍需显式启用。所有新实验保存在 `out/tiger_prefix_ft/beauty_prior_speed_20261016/`。

## 验收边界

- 原始 100 epoch TIGER checkpoint：`out/cloud_reproduction/tiger_beauty_fast_20261012_194804/run/tiger/best_model.pt`，SHA256 为 `f1628a38656353e115b2739fc9c9c0ffdec74fcaf808cdbb6ef5ea615b0e4760`。
- 保留原始 T5、FP32 参数和 FP16 autocast；LogSoftmax、累计模型分数使用 FP32。校准只参与当前层筛选，奖励不累加到后续原始模型分数。
- 保留 `log(1+C(h,p)/(20*P_global(p)))`、最近三件历史、按原始时间距离的 `1, 0.5, 0.25` 衰减、重复 SID 去重、`alpha=[0,0.4,0.4]`。
- 保留 D2 搜索引导和 D3 最终校准、已见单商品 SID 减 1000、SID 碰撞保护。保留完整 769 词表搜索；未找到真实商品前缀时仅奖励为零，没有 Trie、位置过滤或额外剪枝。
- Batch 固定 64，样本顺序、左 Padding、预处理和原有 `tiger_trainer.evaluate` 一致。没有使用 Test。
- 新路径只支持本任务的三层、固定 Beam 宽度和冻结 eval 推理；原来的可复现入口未修改。

## 实际瓶颈

先对真实前 256 条样本、4 个 Batch、Beam=30 使用 `torch.profiler` 和 CUDA Events，再实现优化。Windows 上本环境的 profiler 只提供 CPU activity；报告 GPU stream 时间使用 CUDA Events。Event 跨度可能包括 stream 等待，CPU 拷贝跨度可能包括等待之前的 GPU 工作，不能相加，也不能把所有等待归因于 NumPy。

下表来自 `profile_reference30_final/profile.json` 和 `profile_fast30_final/profile.json`，单位 ms，均为 256 条的合计。它们用于定位，完整端到端 Benchmark 才是速度验收依据。

| 范围 | 原版校准 | 优化 GPU 路径 | 测量说明 |
|---|---:|---:|---|
| Encoder 前向 | 42.799 | 42.213 | CUDA Event |
| Decoder 与 logits | 145.497 | 148.640 | CUDA Event，计算未更改 |
| FP32 LogSoftmax | 0.140 | 0.142 | CUDA Event |
| KV Cache 重排 | 131.144 | 1.417 | 原版重排 self/cross，新版仅必要的 self |
| 逐候选 CPU 转移匹配 | 44.700 | 0 | CPU scope，新版预先合并用户奖励 |
| CPU 历史/已见 SID 整理 | 9.448 | 0 | CPU scope，GPU 路径使用 Tensor |
| CPU 最终已见过滤 | 4.530 | 0 | CPU scope，GPU 路径向量化 |
| 父路径 GPU→CPU | 234.058 | 0 | CPU scope，包含 GPU 等待 |
| 奖励 CPU→GPU | 15.587 | 0 | CPU scope，GPU 路径没有此拷贝 |
| GPU 用户奖励准备 | — | 15.459 | CUDA Event，含小 kernel 启动间隔 |
| GPU 候选奖励索引 | — | 1.304 | CUDA Event |
| Beam 分数/索引加法 | 4.319 | 0.865 | CUDA Event，减少冗余分数分支 |
| Top-K | 5.174 | 1.711 | CUDA Event；候选数、Top-K 算法和规则未改 |
| 结果回传 | 0.710 | 2.156 | CUDA Event；两者均需要最终 CPU 输出 |
| 压缩保存小样本结果 | 6.328 | 6.825 | CPU wall，单独计时 |

旧版本初始化已经预计算 Log-Lift：初始化调用 3 次 `np.log1p`，解码过程中为 0 次。本次没有把这项既有优化计作新增收益。旧版最后一轮 `allclose` 校验导致额外同步，新版将校验移到独立验证程序；profile 的 `aten::allclose` 从 4 次降到 0 次。`aten::item` 两边均记录到 8 次，不能声称所有隐式标量操作都已消除。

## 工程改动

### 1. 每个用户只计算一次历史奖励

`CPUCompactPrior` 使用原版相同的 NumPy 浮点运算顺序，将三个最近历史 SID 的稀疏奖励合并成 D2、D3 两张紧凑表，每个 Batch 传到 GPU 一次。后续所有父 Beam 共享它们，候选路径仍各自使用对应深度和真实前缀取值。

`CSRGPUPrior` 将同一批预计算 Log-Lift 存为 `indptr/indices/values`，按 Batch 的历史 SID 批量定位 CSR 行，在 GPU 合并奖励。三个历史按时间先后顺序依次累加，保持 FP32 加法顺序；重复历史跳过，但不改变其他历史的实际距离。每条 CSR 行中非零目标列唯一，避免非零奖励的并发原子求和改变顺序。

两条路径都使用紧凑合法前缀 ID。D2、D3 共享历史上下文查询，却保留各自的前缀表。非法位置或目录中不存在的候选映射到零奖励哨兵，仍参与原始全词表竞争。

全局索引表约 20.57 MiB；Batch=64 的 D2+D3 用户奖励表约 4.31 MiB。没有建立用户×256³ 的完整 SID 稠密矩阵。D2/D3 CSR 非零数分别为 110,175、113,704。旧版 Beam=30 全 Validation 的奖励 H2D 数据合计约 3.844 GiB；GPU CSR 路径该项为零，常规输入传入和最终结果回传仍然存在。

### 2. 已见 SID 批量过滤

预编码目录中 SID 是否唯一，对用户完整可见历史批量查找，并用布尔表去重。仅对唯一商品 SID 扣一次 1000；同 SID 多商品继续受碰撞保护。没有改成负无穷，也没有将重复出现的已见 SID 重复扣分。

### 3. 保留分数，减少 Beam 内部搬运

复用一次 Encoder 输出和一次 Beam 扩展的 Encoder Hidden State。固定 Beam 宽度下，同一用户的 cross-attention KV 对所有 Beam 相同，父索引又不会跨用户，因此保留 cross KV，只重排 self KV。第一层刚复制的 KV 也全部相同，无需立即再重排。最后一层不为不存在的下一步重排缓存。

删除只用于加权实验和轨迹收集的冗余分数分支，使用原始累计分数和当前层奖励执行同样的完整 Top-K。没有改变模型 logits、FP32 LogSoftmax 或最终排名公式。

### 4. 可选的 Linear 转换缓存

`FrozenLinearCastCache` 只缓存 CUDA autocast 原本会生成的 Linear FP16 权重副本，原来的 FP32 Parameter 对象、存储、数值及 dtype 保留；Embedding 和 LayerNorm 不改。仅支持冻结 eval 推理，切换 checkpoint 或训练前须退出上下文。它是一个独立候选，是否采用以完整 Validation 的严格输出检查和速度为准。

## 正确性证据

独立小样本验证比较每个深度实际选择后的父子路径、原始 FP32 累计分数和 D2/D3 的完整词表奖励 Tensor，不使用最终路径截断反推，也不只看四项指标。

- Beam=15：256 条，核对 5,905,920 个奖励数值、11,520 个原始累计分数。
- Beam=30：256 条，核对 11,811,840 个奖励数值、23,040 个原始累计分数。
- 两种 Beam 均逐位一致；可选 Linear 缓存也单独通过同样的逐位检查，且退出后原始 FP32 参数逐位不变。
- 合成边界测试覆盖无效 Offset 前缀零奖励、非三倍数左 Padding、Mask 下的垃圾值、重复历史保持原距离、单商品已见扣分和 SID 碰撞保护。
- 完整 Validation 的每一轮都与历史原版校准输出逐样本比较整个 Top-15/Top-30，包括顺序；任何一个不一致都会立即报错，不接受近似版本混入结果。

## 性能对照

最终主计时采用 `balanced_b15/`、`balanced_b30/`。固定 Batch=64，每个变体先完整预热，再按 **旧校准→GPU CSR→GPU CSR+Linear 缓存→缓存→GPU CSR→旧校准** 顺序执行，各取得两次正式计时。全部使用同一份 22,363 条 Validation，关闭 RSS 采样线程。主推荐为 **`CSRGPUPrior + fast_search`，不安装 Linear forward 挂钩**。

| Beam | 路径 | 中位数秒（范围） | 分摊 ms/样本 | 吞吐 样本/秒 | 峰值张量 MiB | 峰值保留 MiB | CPU 进程秒中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 15 | 旧校准 | 18.910（18.587–19.234） | 0.8456 | 1182.6 | 1918.3 | 2136.0 | 18.359 |
| 15 | 推荐优化 | 10.477（10.425–10.529） | 0.4685 | 2134.5 | 1081.6 | 1206.0 | 9.984 |
| 15 | 可选 Linear 缓存 | 10.259（10.214–10.304） | 0.4587 | 2179.9 | 1090.1 | 1216.0 | 9.953 |
| 30 | 旧校准 | 30.856（30.451–31.260） | 1.3798 | 724.8 | 3807.1 | 4006.0 | 30.016 |
| 30 | 推荐优化 | 16.614（16.590–16.637） | 0.7429 | 1346.0 | 2045.8 | 2122.0 | 16.211 |
| 30 | 可选 Linear 缓存 | 16.359（16.300–16.417） | 0.7315 | 1367.0 | 2054.3 | 2132.0 | 15.789 |

| Beam | 推荐版加速比 | 耗时下降 | 张量显存下降 | 可选 Linear 缓存的额外中位数收益 |
|---|---:|---:|---:|---:|
| 15 | 1.805× | 44.60% | 43.62% | 2.08% |
| 30 | 1.857× | 46.16% | 46.26% | 1.54% |

可选缓存增加约 8.69 MiB 的 FP16 权重副本，并需管理 Linear forward 的安装/恢复。最终双向复测只多获得约 1%–2% 的中位数收益，前一阶段也未稳定获益，因此主推荐不启用。它的结果完全一致，不属于近似版本；希望复现最小耗时版本可以显式选 `fast_gpu_linear`。

### 四项推荐指标与逐样本输出

| Beam | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 逐样本整个 Beam 差异 |
|---|---:|---:|---:|---:|---:|
| 15 | 8.2100% | 11.4341% | 5.6649% | 6.7024% | 0 / 22,363 |
| 30 | 8.2458% | 11.5369% | 5.6926% | 6.7532% | 0 / 22,363 |

上表对旧校准、推荐版和可选缓存版完全相同。`top10_equivalence.parquet` 共 **89,452 行**：两种 Beam×两个优化版本×22,363 个样本，保存 `sample_id`、`beam_size`、`variant`、Top-10/整个 Beam 的一致性标记，以及参考和优化后的完整 Top-10 Offset Token 整数数组。所有标记为真；没有浮点并列例外。它是新的校验产物，未改变原有三份训练 Parquet。

`final_summary.json` 是机器可读主汇总；`completion_audit.json` 和 `artifact_manifest.json` 记录源码与产物校验。CPU/GPU 两种先验、可选 Linear 缓存各在两种 Beam 的真实前 256 条上通过逐位奖励/路径/原始分数检查。9 项回归测试通过。

### 初始化、保存与 CPU 内存

| Beam | 模型/数据加载秒 | 推荐先验准备秒（两次） | 当前变体首次调用秒（两次） | 首次 NPZ 保存秒：旧/推荐 |
|---|---:|---|---|---|
| 15 | 0.568 | 0.402, 0.404 | 0.060, 0.049 | 0.133 / 0.148 |
| 30 | 0.493 | 0.400, 0.417 | 0.081, 0.084 | 0.246 / 0.256 |

加载/初始化/压缩保存不在主热运行计时内；首次调用并非独立进程冷启动，不能用该表计算冷启动加速比。CPU RSS 主计时仅记录起点和终点，周期峰值字段为空。另有 25 ms RSS 采样的历史诊断轮如下，受进程 allocator 与执行顺序影响，仅展示观测值，不宣称 CPU 内存固定减少多少。

| RSS 诊断轮 | 旧校准观测峰值 MiB | GPU 优化观测峰值 MiB |
|---|---:|---:|
| final_b15 | 1590.6 | 1585.0 |
| final_b30 | 1602.3 | 1594.8 |

### 性能波动与未混用的结果

`clean_b15/clean_b30` 是关闭 RSS 线程后先完成的三轮正式测试。其数值均保留，不能只展示后来的快速时段：

| Beam | 旧校准三轮秒 | CPU 表三轮秒 | GPU CSR 三轮秒 | GPU CSR+Linear 三轮秒 |
|---|---|---|---|---|
| 15 | 68.61, 68.86, 63.28 | 36.47, 26.73, 54.40 | 50.47, 49.13, 48.21 | 19.41, 10.26, 10.05 |
| 30 | 67.21, 73.80, 60.39 | 60.02, 57.33, 53.30 | 40.01, 40.71, 40.99 | 45.92, 36.69, 37.94 |

这一阶段 Beam=15 的 CPU 表中位数比 GPU 快，说明不能假设 GPU 一定优于 CPU；CPU 后端保留供显式复测。之后原版和优化版共同回到更快时段，最终正反序对照才用于主加速比。**不报告“历史 76.35 秒→当前 16.61 秒”的跨时段加速比，也不报告“68.61 秒→10.48 秒”为同状态收益。** 未定位系统时段变化的唯一原因，不能保证每次都有相同绝对秒数。

本机 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，总显存 8187.5 MiB，PyTorch 2.6.0+cu124 / CUDA 12.4。所有阶段的单轮结果、输入 SHA256、压缩预测文件和源码快照均保留。

### 分阶段对照：早期双向复测

`final_b30/summary.json` 保存了同一阶段先顺序、再反序运行的两次全量测量。下表同时展示各工程环节及范围；最终主计时使用另一个明确标识的 `balanced_b30/balanced_b15` 阶段，不将二者拼接计算加速比。

| Beam=30 路径 | 主要变化 | 两轮中位数 / 范围，秒 | 峰值张量显存 MiB |
|---|---|---:|---:|
| reference | 旧字典先验与旧 Beam 引擎 | 66.52 / 53.29–79.76 | 3807.1 |
| cpu_table | 每用户合并奖励、CPU 紧凑表、GPU 索引，旧引擎 | 31.16 / 26.20–36.12 | 3834.4 |
| gpu_csr | CSR GPU 用户奖励与向量化已见过滤，旧引擎 | 24.01 / 23.90–24.11 | 3836.1 |
| gpu_csr_kv_only | 上一行仅替换为 self KV 重排 | 15.82 / 15.79–15.86 | 2138.8 |
| fast_cpu_table | CPU 紧凑表加精简引擎 | 17.24 / 17.13–17.35 | 2043.8 |
| fast_gpu_csr | CSR GPU 加精简引擎 | 15.43 / 15.36–15.50 | 2045.8 |

这一阶段，CPU 表换为 GPU CSR 的中位数比值约 1.30；GPU CSR 路径仅调整 KV 重排后约 1.52；再去除多余 Encoder 扩展和分数跟踪后约 1.03。它们说明各环节的量级，受顺序执行、系统状态和两次样本数量限制，不将逐行差值解释成严格隔离的因果收益。所有行的整个 Top-30 均与历史参考逐样本一致。

早期 Beam=15 的两轮全量中位数：旧校准 18.71 秒，CPU 紧凑表 12.34 秒，GPU CSR 10.48 秒；对应峰值张量显存 1918.3、1079.7、1081.6 MiB。后续完整预热阶段，GPU CSR 又测得 37.44–41.26 秒。所有数据保留在 `final_b15/`、`warmed_b15/`；这些更快或更慢的值都不被静默丢弃。

计时范围包括原始 evaluator 的 Batch 读取、输入传输、历史奖励、生成、四项指标计算和相同的最终 CPU 输出收集；模型/数据加载、先验初始化、首次 CUDA 调用、完整预热和压缩保存分别记录，不混入热运行耗时。每轮开始和结束同步 CUDA。每轮释放空闲 allocator 缓存，模型和 kernel 保持预热状态，因此该数字包含每轮重新申请临时工作区的成本。延迟是整批评估分摊值，并非交互式单用户 p95。

JSON 的 `first_cuda_call_seconds` 是当前变体首次调用并同步的时间。变体在同一进程顺序执行，后续变体的 CUDA 环境已经预热，不能将这些数值当作各方案独立进程的冷启动比较。`model_data_loading_seconds` 不包含 Python 导入与预先 SHA256 审计时间；本报告不宣称命令启动到结果生成的冷启动加速比。GPU 显存为 PyTorch `max_memory_allocated`，另记 `max_memory_reserved`，不包含其他应用和完整驱动开销。

本机出现明显的跨时段波动：早期旧校准 Beam=30 在 28–80 秒间变化；优化 GPU Beam=15 既出现约 10.5 秒，也出现 37–41 秒。同一份输出始终一致，不能据此选择最好的一次作为稳定承诺。先关闭周期性 RSS 采样线程并完整预热、重复三轮；随后执行最终正反序复测。先前有 RSS 线程的结果仍单独保留，用于 CPU 内存诊断；线程影响是否为唯一原因尚未证明。

## 复现与调用

在仓库根目录的 PowerShell 中运行。所有 `--output` 必须是尚不存在的路径，防止覆盖历史结果；下面的目录名可自行更换。代码直接使用已存在的原始 checkpoint、Validation 和训练统计，不会下载或训练模型。

```powershell
$py = 'python'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
New-Item -ItemType Directory -Force out/tiger_prefix_ft/prior_speed_recheck | Out-Null

# 真实 256 条：三层路径、全词表奖励和原始 FP32 分数严格相等
& $py -m tiger_prefix_ft.verify_prior_speed --beam 15 --output out/tiger_prefix_ft/prior_speed_recheck/scores15.json
& $py -m tiger_prefix_ft.verify_prior_speed --beam 30 --output out/tiger_prefix_ft/prior_speed_recheck/scores30.json

# 完整 Validation：固定 batch=64，完整预热，正序/反序各一次
& $py -m tiger_prefix_ft.benchmark_prior_speed --beam 15 --warmup-full --variants reference fast_gpu_csr fast_gpu_linear fast_gpu_linear fast_gpu_csr reference --output out/tiger_prefix_ft/prior_speed_recheck/full15
& $py -m tiger_prefix_ft.benchmark_prior_speed --beam 30 --warmup-full --variants reference fast_gpu_csr fast_gpu_linear fast_gpu_linear fast_gpu_csr reference --output out/tiger_prefix_ft/prior_speed_recheck/full30

# 独立性能定位；不要把 profiler 下耗时当成端到端 Benchmark
& $py -m tiger_prefix_ft.profile_prior_speed --beam 30 --variant reference --output out/tiger_prefix_ft/prior_speed_recheck/profile_old
& $py -m tiger_prefix_ft.profile_prior_speed --beam 30 --variant fast_gpu --output out/tiger_prefix_ft/prior_speed_recheck/profile_fast

# 边界回归；CPU 上可运行
& $py -m unittest tiger_prefix_ft.test_inference_fast -v
```

给核对程序加 `--backend cpu` 可检查 CPU 表，增加 `--cache-linear` 可检查可选权重转换缓存；请为每次核对设置不同的输出 JSON 路径。

如需 CPU 内存周期采样，可给 Benchmark 加 `--sample-rss`，将其作为独立诊断。主计时默认关闭该线程；此时 `peak_rss_mib=null`，仅给出 RSS 起点和终点，避免把端点误称为内存峰值。

在已有模型加载代码中接入新生成器的核心调用如下。`tiger` 应已加载上述原始 FP32 checkpoint、置于 CUDA、冻结并 `eval()`；`stats` 和 `offset_sids` 沿用现有加载方式。

```python
import torch
from tiger_prefix_ft.inference_priors import PrefixPrior
from tiger_prefix_ft.inference_fast import CSRGPUPrior, CPUCompactPrior, generate_fast

reference = PrefixPrior(stats, offset_sids, [0., .4, .4], True, 3)
prior = CSRGPUPrior(reference)  # 或 CPUCompactPrior(reference)

with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    for batch in loader:
        ids = batch['input_ids'].cuda(non_blocking=True)
        mask = batch['attention_mask'].cuda(non_blocking=True)
        tokens = generate_fast(tiger, ids, mask, prior, widths=(30, 30, 30))
        # tokens: (batch * 30, 4)，首位为 Decoder Start 0。
```

新生成器默认宽度仍是 `(15,15,15)`。上面的主调用不安装任何模型 forward 挂钩。若专门复现实验性的 Linear 缓存，请把 `FrozenLinearCastCache(tiger)` 上下文放在 Batch 循环外；离开后原始 forward 恢复。切换 checkpoint、设备或训练模式前，应先退出缓存上下文。

## 新增文件

| 文件 | 用途 |
|---|---|
| `tiger_prefix_ft/inference_fast.py` | 紧凑奖励表、CSR GPU 先验、精简的等价 Beam Search、推理入口 |
| `tiger_prefix_ft/inference_weight_cache.py` | 可选的冻结 Linear FP16 转换缓存 |
| `tiger_prefix_ft/profile_prior_speed.py` | 原版/优化版 CPU profiler 与 CUDA Event 定位 |
| `tiger_prefix_ft/benchmark_prior_speed.py` | 原 evaluator 全量计时、显存统计、逐样本全 Beam 核对 |
| `tiger_prefix_ft/verify_prior_speed.py` | 真实样本逐层路径、FP32 分数、全词表奖励逐位核对 |
| `tiger_prefix_ft/test_inference_fast.py` | 独立参考与边界条件测试 |
| 本文档 | 原理、结果、限制与复现命令 |

本次没有修改旧的模型、T5、旧校准器或评估实现。新脚本每次运行在自己的输出目录归档源码和输入 SHA256；旧实验目录没有覆盖。

## 未采用与后续可能

- 逐条 Python `np.searchsorted` 并不比字典快：768 次真实上下文查询，单次微基准中位数为字典 0.184 ms、标量 searchsorted 1.791 ms、批量 CSR 行查询 0.067 ms。它只衡量行定位，不能代替端到端收益。
- 仅换成 GPU 查询而保留所有旧缓存搬运，显存约 3.84 GiB，几乎没有减少；KV 重排和 Encoder 扩展的调整才带来主要显存收益。
- 没有通过改 Batch、降低精度、改变 Beam、位置过滤、Trie、剪枝或参数调整取得加速。
- 后续可考虑固定形状 CUDA Graph、减少小 kernel 启动、异步结果收集和更紧凑的 cross KV 表示。本次不引入编译或 attention 架构改写；这些方案都需要重新做逐样本严格校验。
