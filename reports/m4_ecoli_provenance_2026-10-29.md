# M4：大肠杆菌数据来源、强度标签与位置核验

日期：2026-10-29。目标是核实课程数据与公开数据的对应关系，为强度预测和位置分析提供可追溯依据。

## 1. 文件身份：两份课程数组与公开 PromoA 数据完全一致

课程 `strenth/data/promoter.npy` 和 `gene_expression.npy` 的 Git blob SHA-1，与 [Wang 等人公开仓库的 PromoA 训练目录](https://github.com/wangxinglong1990/Promoter_design/tree/56c1b331bfbca0665b2c3f6b02345de794fe25df/PromoS_and_PromoA_strength_prediction/PromoA_train)逐字节一致：

| 数组 | 本地和上游 Git blob SHA-1 | 本地 SHA-256 |
|---|---|---|
| `promoter.npy` | `14cf8fb3979f8824cf612870bae7cd46d875a347` | `e71911452431acb620478f0de5ae896481be5618e7a7f9712125cd98cb8e37ab` |
| `gene_expression.npy` | `264a60dd2cf439b711a24413fc10383b9f8cb598` | `a1426fa5e5806fceff2c041218b781b6dc1f0cc84b39cf4ef626d02965fd9ce5` |

此外，[CAPE 研究仓库的 Ecoli_seq.npy](https://github.com/BobYHY/CAPE/blob/7a9a0e903b46813a62b234e0a062e22def6dff44/CAPE_Final/data/Ecoli_seq.npy)也具有相同的序列文件 Git blob。**这里仅核对 CAPE 的序列文件，未核对 CAPE 的标签文件是否数值相同。**Git blob SHA-1 的比较包含文件字节、顺序和数组格式；它证明文件身份，不单独证明标签的生物学含义。

课程包中 `supplementary/E_coli.txt` 有 11,884 行数据，逐行比较后，序列和强度数值与两份数组均无差异。数组中所有序列是合法 50 bp，强度均为有限正数；原始数值范围 8.67–2,755,441.06。机器可读结果见 [审计结果](m4_ecoli_provenance_audit_2026-10-29.json)。

## 2. 公开论文提供的来源与 TSS 线索

Wang 等人的 [PromoA 论文](https://doi.org/10.1002/ggn2.202300184) 将该 11,884 条数据称为 NDB，称其来自 Thomason 等人的全基因组转录起始位点（TSS）图谱，并用于预测转录水平。原始研究 [Thomason 等人](https://doi.org/10.1128/JB.02096-14)使用大肠杆菌 MG1655 和差异 RNA 测序（dRNA-seq）；其研究涵盖多种生长条件，但**不能据此推断本地每一条数值对应哪一种条件**。

使用与本地完全相同序列数组的 [CAPE 论文](https://doi.org/10.1093/bib/bbae398)将其 11,884 条 E. coli 数据描述为 TSS 之前的 50 bp，并将该数据集的表达水平描述为 dRNA-seq 相关测量。因此，原先“完全没有 TSS 截取依据”的表述需要修正：**现在有公开文献和相同序列数组提供的数据集级依据**。

仍需注意：本地两个数组及 `E_coli.txt` 本身只保存序列与数字，没有每条记录的基因组坐标、链方向、TSS 坐标、σ 因子或测量条件。我们没有逐条重新映射原始 TSS，因此尚不能核实每条记录的方向、精确坐标编号或对所有样本指定 σ 因子。

Wang 等人说明 NDB 含不同类型启动子，并对 σ⁷⁰ 类型使用过计算分类；这不是本地每条样本的实验确认注释，不能将所有样本直接称作 σ⁷⁰ 启动子，也不能把位置扫描得到的候选 motif 当作已经确认的 −10/−35 元件。

## 3. 标签含义与模型评价

现有证据支持将该标签称为**自然启动子相关的转录水平或表达强度代理指标**，而不是直接认定为同一实验平台下的绝对启动子固有强度。dRNA-seq 信号与具体培养条件、转录环境等可能有关。课程文件和本次已核对的资料没有给出能明确对应本地每个数字的单位、测序归一化公式及逐样本条件；这些仍待从原始数据处理记录核实。不同物种或荧光实验的数值不能直接拼接当作同一尺度。

课程附带毕业论文第 3.2 节描述的是**该论文模型**对标签先取 log10 再做 min-max 归一化；这不代表本地数组一定是该步骤之后的值。我们当前的主要预测器使用 `log10(原数组数值)`，CNN 在拟合时使用训练样本均值和标准差做目标标准化。不同论文的 MAE 数值如果不在同一目标尺度上，不能直接比较。

文件身份也带来一个重要评价边界：公开 PromoA 模型使用过这两份完全相同的数组。**即使它由其他研究者训练，也不能未经训练数据重叠核验，就当作我们这 11,884 条数据的独立测试器**；它可能已经接触过我们划分中的验证或测试样本。对于新生成序列，它仍可作为一个计算参考，但不能代替实验验证或证明目标强度。

## 4. 复现

```bash
python scripts/audit_ecoli_provenance.py \
  --data-dir /path/to/promoter/strenth/data \
  --output reports/m4_ecoli_provenance_audit_2026-10-29.json
python -m unittest discover -s tests -v
```

脚本离线计算本地 Git blob 和 SHA-256，比较事先记录的固定上游版本标识，并检查文本与数组逐行一致性。上游公开仓库当时的版本标识分别为 PromoA `56c1b331bfbca0665b2c3f6b02345de794fe25df` 和 CAPE `7a9a0e903b46813a62b234e0a062e22def6dff44`。原始课程数组不提交 Git。
