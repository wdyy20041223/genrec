# 独立 TIGER 仓库复现核查（2026-10-22）

## 核查范围与结论

将 CoTIGER 整理为只包含 TIGER 及其改进的独立项目。当前代码不要求安装完整 GenRec；权重和数据不进入本次 Git 提交。本轮未重新训练、未进行完整 Validation 性能实验、未执行云端操作或删除本地实验文件。

检查开始时，本地 main 与公开 GitHub main 均为 `c16cbc9ef0cb40b782710554d7e52887b29db340`，已跟踪 92 个文件。发现旧发布范围同时包含完整 GenRec 模型目录，却忽略了实际使用的 `tiger_prefix_ft` 研究源码；部分采集/分析入口还保存在未发布的过程目录中。仅克隆旧提交不足以恢复这些研究入口。

## 本次调整

| 项目 | 处理 |
| --- | --- |
| TIGER/T5/RQ-VAE | 依赖范围提取到 `cotiger/core/`；不再导入 `genrec` |
| 其他推荐模型 | 从当前 Git 文件树移除；旧本地 `genrec/` 原文件保留并忽略 |
| 模型与评分兼容 | TIGER、RQ-VAE、T5、TIGER trainer 与迁移前逐文件比较，只更换导入命名空间；参数名、结构和计算保持一致 |
| RQ-VAE 数据默认值 | 去掉额外 P5 数据实现，默认使用本项目 Gin 已指定的 `AmazonItemDataset` |
| 无关依赖 | 移除 PyTorch Geometric、Triton、LoRA 等非 TIGER 依赖及其他模型配置 |
| 原始 Beam 采集器 | 移入 `tiger_prefix_ft/raw_collection/`，保留原张量格式、验证器及 schema |
| 研究入口 | 保留 CE/Prefix InfoNCE、负例加工、逐层诊断、SID/码本分析及历史加速实验源码 |
| 本机路径 | 基础模型路径改为可配置本地路径，分析默认路径可通过 CLI/环境变量指定 |
| 旧实验导出 | 读取旧 resolved.gin 时转换命名空间；导出记录实际当前提交，不再硬编码历史 commit |
| 文档 | README 说明独立项目范围、来源、两项改进和结果；必要报告统一放在本目录 |

完整 GenRec 的历史文件仍存在于旧 Git 提交中；本次没有重写仓库历史。HSTU/SASRec/RPG 的本地 Windows worker 修改不纳入当前项目。

## 干净副本与安装验证

在独立目录复制待发布代码，**不放入 `genrec/`、原始数据、权重或旧实验输出**，再启动隔离 Python 进程检查：

| 检查 | 实际结果 |
| --- | --- |
| `tests/` CPU 测试 | 15 项通过 |
| `tiger_prefix_ft/` CPU 测试 | 55 项通过；另 1 组真实 H4/权重集成检查因缺少资产明确跳过 |
| 无资产联合 Loss 检查 | 验证 CE、各自前缀打分、共享 Encoder、Encoder/Decoder 有限梯度 |
| 研究/推理 CLI | 22 个 `--help` 入口通过 |
| TIGER/RQ-VAE trainer | 两个 CLI 通过；两份 Beauty Gin 独立解析成功，不启动训练 |
| 源码编译 | 通过 |
| Wheel | `cotiger-0.2.0-py3-none-any.whl` 构建及无依赖本地安装通过 |
| Wheel 内容 | 74 个 Python 文件与待发布源码一致，包含采集 schema，不包含 `genrec/` |
| 导入来源 | 确认全部项目模块来自干净副本/新安装包，未回退原工作目录或旧 GenRec 安装 |

干净测试曾发现一项合成测试隐式读取私有 SID 导出，已改用明确的合成 Catalog；同时取消了通过测试类导入造成的重复执行。正常研究函数的评分逻辑没有改变。

## 真实权重推理核对

对本地已有 Beauty Release ZIP 核对 SHA256 后，解压到上述独立副本。使用 RTX 4060 Laptop GPU（8187.5 MiB）、PyTorch 2.6.0+cu124、Batch=64，重新加载原始 100-epoch TIGER 权重：

| 方法 | Beam | 真实 Validation 样本 | 全部有序 Beam 与历史参考 |
| --- | ---: | ---: | --- |
| 原版 TIGER | 15 | 64 | 完全一致 |
| 原版 TIGER | 30 | 64 | 完全一致 |
| 校准参考实现 | 15 | 64 | 完全一致 |
| 校准参考实现 | 30 | 64 | 完全一致 |
| 校准加速实现 | 15 | 64 | 完全一致 |
| 校准加速实现 | 30 | 64 | 完全一致 |

这是同一批 64 条样本在六种配置下的兼容性检查，不是 384 条独立样本，也不是重新进行完整性能实验。核对包括最终所有 Beam 的 SID 和顺序，不只比较 Recall/NDCG。原权重 SHA256 仍为 `f1628a38656353e115b2739fc9c9c0ffdec74fcaf808cdbb6ef5ea615b0e4760`。

本轮检查和原有完整 22,363 条加速实验是不同范围；完整实验的逐样本一致性证据与计时见[无损加速报告](prior_speed_20261016.md)。不能用本轮 64 条运行时间代替完整吞吐量。

## 删除本地之前需要保留什么

公开仓库：[wdyy20041223/CoTIGER](https://github.com/wdyy20041223/CoTIGER)。原有公开 [Beauty v1 Release](https://github.com/wdyy20041223/CoTIGER/releases/tag/beauty-v1) 提供主推理资产，本轮没有新增上传权重或数据。

| 内容 | Git 可恢复 | 已有 Release 可恢复 | 本地删除的影响 |
| --- | --- | --- | --- |
| 独立 TIGER/改进源码、配置、测试、必要说明 | 是 | 不需要 | 可重新克隆 |
| 原始 TIGER、SID、Train/Validation、统计及主推理参考预测 | 否 | 是 | 可从现有 Release 恢复 |
| 全量原始 Beam NPZ/JSONL | 否 | 否 | 如仍需重新挖掘/分析，必须另行备份或重新采集 |
| H2/H4/H6 微调权重和全部逐 epoch 结果 | 否 | 否 | 仅保留代码不能恢复当时权重 |
| 固定 K8 数据与完整 Hard Pool | 否 | 否 | 需保留本地，或从真实 Train 轨迹重建 |
| 全部 RQ-VAE 快照、原商品向量及原始模型缓存 | 否 | 否 | 完整码本/相似性审计需另行保留或重新生成 |

只读盘点：原 Beauty 导出目录约 0.595 GiB，原始 Beam 目录约 15.769 GiB，`out/tiger_prefix_ft` 约 2.167 GiB，`models_hub` 约 2.316 GiB。这些目录没有放入 Git；不要把约 1 MB 的源码仓库理解为全部历史实验的完整备份。

本次保留了本地原文件。后续复现命令、输入格式与各阶段前置条件见[复现说明](reproduction.md)。
