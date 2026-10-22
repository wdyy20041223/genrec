# CoTIGER 方法与实现

## 数据与模型边界

- 上游：GenRec 的 TIGER；冻结 100-epoch checkpoint，4 层 Encoder / 4 层 Decoder，d_model=128。
- Beauty Train=131413，Validation=22363；目录中有12101个商品、11401个唯一三位SID。
- 每位原始码本大小256，Offset为[1,257,513]，`token_i=code_i+256*i+1`；0为Padding/Decoder Start。
- 输入保留最近50件商品，将历史Token左Padding到当前Batch最长长度的8倍数；attention_mask区分有效历史。
- 模型前向使用FP16 autocast，参数、LogSoftmax及原始累计LogP保持FP32。
- 评价按三位SID完全匹配；第四位消歧Token关闭。同SID商品不能通过当前指标区分。

## Train 统计

只从Train样本取得最后一个完整历史SID `h` 和完整目标SID `s`，累计 `C(h,s)`。将目标SID截到深度d，即可聚合得到 `C(h,p_d)`。

全局边际概率使用同一Train的目标前缀频次：

```text
P_global(p_d) = (C_train_target(p_d)+1) / (N_train+|CatalogPrefixes_d|)
lift(h,p_d) = log1p(C(h,p_d)/(20*P_global(p_d)))
```

不存在的转移奖励为0。Log-Lift在加载统计时计算一次，以float32保存；推理不再重复进行归一化和对数运算。

## 搜索中的校准

1. 从当前用户实际历史取最近三件SID，按距今0、1、2位分别赋予1、0.5、0.25权重。
2. 相同SID只保留最近一次，较远SID仍使用其原始距离对应的权重。
3. 每个深度对当前Beam的全部769种下一Token扩展计算原始累计LogP。
4. 用相同深度的真实候选前缀查询奖励，按`raw_logP+alpha_d*bonus`执行全局Top-K；alpha=[0,0.4,0.4]。
5. 只把原始LogP带入下一轮，上一层奖励不重复累计。D2改变存活父路径，D3决定最终顺序。
6. D3将已见单商品SID减1000分；对应多个商品的SID保留，避免把SID碰撞中的其他商品一并过滤。

没有把未来目标送入先验，没有Trie或位置限制。合法目录仅用于奖励索引和碰撞判断，不作为搜索掩码。位置不合法或目录外候选仍参与原始搜索，其先验奖励为0。

## 无损优化

参考实现逐用户、逐历史上下文、逐Beam父路径匹配前缀，且在每层重排Self及Cross Attention KV。

优化实现保留相同评分操作顺序：

- 将转移关系存为CSR，values是已有float32 Log-Lift。
- 用紧凑前缀ID索引；0是未映射哨兵，奖励恒为0。
- 每个Batch按用户预先合并两个深度的稀疏奖励，三个历史位置依次相加，避免改变浮点累加顺序。
- D2、D3在GPU上批量查表；临时表只覆盖当前Batch和真实目录前缀，不构建每用户完整三位SID空间。
- 将已见判断及碰撞保护向量化，仍执行减1000分。
- 固定Beam宽度时，同用户的Cross Attention KV相同；只让Self Attention KV随父Beam重排，跳过不必要的最后一层重排。

`reference.py`保留参考路径，`decoding.py`提供CPU用户表和GPU CSR两个等价后端。推荐GPU CSR；没有采用改变候选剪枝范围的近似加速。

## 统计基线

独立SID Markov推荐器在全部11401个唯一SID中取Top10：

```text
q(s) = (C_train_target(s)+1)/(N_train+11401)
P(s|h) = (C(h,s)+20*q(s))/(C(h)+20)
score(H,s) = Σ_age 0.5^age * P(s|history[-1-age])
```

同样使用最近三项去重、单商品已见过滤。并列按Train频率降序、再按SID字典序；未知上下文回退全局平滑概率。这与校准所用的Log-Lift是两种排序，不能混为同一基线。

## 正确性验收

原工程加速实验已对两个Beam配置的完整Validation逐样本核对Top10及最终全部Beam；256条真实样本另行核对每层存活路径、先验值和FP32原始累计分数的逐位一致性。发布包保存原始预测与一致性Parquet，CLI可以重新加载相同checkpoint复核。

独立项目整理保留原模型计算与参数名，不重新训练或生成SID。2026-10-22 在不含旧 GenRec 的干净副本中，对原版、校准参考、校准加速三个实现的两个Beam配置各64条真实Validation做严格输出对照，全部有序Beam一致；见[本次核查](reproducibility_audit_20261022.md)。发布包`evidence/packaging_checks.json`保留早期发布核对记录。
