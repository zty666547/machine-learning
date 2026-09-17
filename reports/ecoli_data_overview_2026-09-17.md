# 课程数据包中的大肠杆菌数据

阶段复核日期：2026年9月8日

> 结论：老师说的“数据不全是 50 bp”是正确的。课程包中有两套用途不同的大肠杆菌数据。11,884 条连续强度主数据全部为 50 bp；另一套启动子二分类数据共有 3,350 条，其中 3,349 条为 81 bp，1 条为 80 bp。两套数据不能混为一个数据集。

## 1. 两套数据的区别

| 数据集 | 样本数 | 长度分布 | 标签 | 适合的任务 | 在本项目中的用途 |
|---|---:|---|---|---|---|
| 连续强度数据 | 11,884 | 50 bp：11,884 条 | 实数强度 8.67–2755441.06 | 根据目标强度生成启动子 | **主数据集** |
| 启动子二分类数据 | 3,350 | 80 bp：1 条, 81 bp：3,349 条 | 0/1，表示是否为启动子 | 判断一条序列是否像启动子 | 辅助识别评价 |

因此，我们在 M1 报告中说“全部为 50 bp”时，必须明确限定为**连续强度主数据**，不能泛指课程包里的全部大肠杆菌数据。

## 2. 连续强度主数据

- 文件：`strenth/data/supplementary/E_coli.txt`，并与 `promoter.npy`、`gene_expression.npy` 一致；
- 样本数：11,884；
- 序列长度：全部 50 bp；
- 唯一序列：11,884，没有完全重复；
- 强度范围：8.67 至 2755441.06，中位数 137.085；
- 标签含义：连续表达强度，可以作为条件生成模型的目标条件。

前 5 条原始记录：

| 序号 | 50 bp 启动子序列 | 强度 |
|---:|---|---:|
| 1 | `ATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATT` | 1951.41 |
| 2 | `TAATTTTTATCTGTCTGTGCGCTATGCCTATATTGGTTAAAGTATTTAGT` | 36.84 |
| 3 | `AATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGC` | 2355.75 |
| 4 | `CATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAA` | 1262.81 |
| 5 | `GCACCAATGAGCGTACCTGGTGCTTGAGGATTTCCGGTATTTTTAATCAG` | 203.37 |

## 3. 大肠杆菌启动子二分类数据

- 文件：`reg_and_gen/Datasets/Escherichia coli/Dataset.csv`；
- 样本数：3,350；
- 长度分布：80 bp：1 条, 81 bp：3,349 条；
- 标签分布：正类 `1` 共 1,645 条，负类 `0` 共 1,705 条；
- 唯一序列：3,342；完全重复多出的记录 8 条，涉及 8 个序列组；
- 重复序列的标签冲突：0 组。

其中唯一的 80 bp 记录为：

| seq_id | 类型 | 长度 | 标签 | 序列 |
|---:|---|---:|---:|---|
| 2444 | gene | 80 | 0 | `TCCTGGGTTGTATCGTTTATTCCGCGCCGCCCGGAGCGGAGCGAAACGGCACAACCATCGGATCAGGCTTTGGCTCAACA` |

课程包给出的拆分情况：

| 子集 | 样本数 | 长度分布 | 正类 | 负类 | 唯一序列 |
|---|---:|---|---:|---:|---:|
| train | 2,345 | 81 bp：2,345 条 | 1,156 | 1,189 | 2,341 |
| dev | 335 | 80 bp：1 条, 81 bp：334 条 | 170 | 165 | 334 |
| test | 670 | 81 bp：670 条 | 319 | 351 | 670 |

原始 `train/dev/test` 之间各有 1 条完全相同的序列交叉出现：

- `train` 与 `dev`：`ACCCACTAATCGTCCGATTAAAAACCCTGCAGAAACGGATAATCATGCCGATAACTCATATAACGCAGGGCTGTTTATCGT`
- `train` 与 `test`：`TCGCAAAACAGACCGTGTTGCGCAATTTGTCAACGAAAACAATAATGCGTAAGGTAGAAACCCGAACTACATTGAGGAATC`
- `dev` 与 `test`：`GCGAAGGTAAGTTGATGACTCATGATGAACCCTGTTCTATGGCTCCAGATGACAAACATGATCTCATATCAGGGACTTGTT`

这套数据只能回答“是否为启动子”，不能回答“启动子强度是多少”。若以后用它训练辅助识别器，应先按序列去重并重新划分，避免相同序列跨训练集、验证集和测试集。

## 4. 对第 25 组项目的直接影响

1. 条件生成模型继续使用 11,884 条 50 bp 连续强度数据，因为我们的目标是按指定表达强度生成序列。
2. 3,350 条 80/81 bp 二分类数据不直接并入生成模型训练；长度不同，标签含义也不同。
3. 二分类数据可在后期训练一个辅助启动子识别器，但使用前需要处理重复和跨集合重叠。
4. 汇报时建议表述为：**课程包中的序列并非全部为 50 bp；本项目选用的连续强度主数据全部为 50 bp，另一套大肠杆菌二分类数据主要为 81 bp。**
5. `kucao_test.csv` 在课程代码中被明确用于 *Bacillus subtilis*（枯草芽孢杆菌）强度预测，因此不计入大肠杆菌数据。
