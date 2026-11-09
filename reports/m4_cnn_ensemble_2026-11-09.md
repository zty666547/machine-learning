# M4 三种子 CNN 等权集成

## 方法

在不重新选择结构或权重的条件下，将此前三个种子训练的同结构 motif CNN 预测在 log10 强度尺度上等权取平均。目标是检验训练随机性导致的不同预测是否可以相互抵消。三个种子、相似性簇三折、等权方式和早停规则在查看集成结果前固定，见[实验配置](../configs/m4_cnn_ensemble_2026-11-09.toml)。

训练集每条样本的三个预测均来自未使用该样本训练或早停的同一外层折；单模型每个种子的整体折外及开发验证 MAE 均重现此前记录（容差 1e-8），并检查数据摘要和 NumPy 版本。最终模型使用 8,318 条训练数据内部的拟合/早停划分训练；开发验证集只做探索评价。

## 结果

| 训练折外指标 | 三个单模型指标的平均值 | 三模型等权集成 |
|---|---:|---:|
| MAE | 0.46050 | 0.45710 |
| Pearson | 0.19842 | 0.22067 |
| Spearman | 0.17743 | 0.19626 |
| 弱组 MAE | 0.46674 | 0.46640 |
| 中组 MAE | 0.16668 | 0.15800 |
| 强组 MAE | 0.74797 | 0.74680 |
| 预测标准差 / 标签标准差 | 0.24708 | 0.22229 |

单模型折外 MAE 分别为 0.45961、0.46040、0.46149；集成 0.45710 在该折外数据上低于每个成员。中组误差下降较明显，弱组和强组变化很小，预测范围仍偏窄；集成弱组平均有符号误差 +0.46631，强组 -0.74465，仍存在两端偏差。

由于绝对误差的凸性，对同一批样本，等权集成 MAE 不超过各成员 MAE 的平均值是数学性质，不能单凭这一点证明泛化改善。集成在本次折外数据中也优于最好的单成员，但这仍是同一数据上的探索结果，不是统计显著性或独立泛化的证明。

开发验证集单模型平均 MAE 0.43718，集成为 0.43254；开发验证集之前已多轮使用，不能作为独立验证。各成员的折外预测来自同一组折和样本，三个种子也不是独立数据集。

## 边界与复现

本轮没有用开发验证集选择种子或权重；测试数据只由加载器检查格式，未提取特征或计算指标。无新的外部数据和湿实验，集成不替换现有 M3 融合筛选器。与该筛选器训练方式不同，指标不构成严格配对对照。

```sh
for seed in 20260912 20260913 20260914; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python scripts/evaluate_cnn_ensemble_member.py --data-dir "$DATA_DIR" --seed "$seed" --output-dir outputs/m4_cnn_ensemble
done
python scripts/summarize_cnn_ensemble.py --data-dir "$DATA_DIR" --input-dir outputs/m4_cnn_ensemble --output reports/m4_cnn_ensemble_2026-11-09.json
python scripts/score_cnn_ensemble.py --models outputs/m4_cnn_ensemble/cnn_20260912.npz outputs/m4_cnn_ensemble/cnn_20260913.npz outputs/m4_cnn_ensemble/cnn_20260914.npz --candidate-file "$CANDIDATES" --output outputs/m4_cnn_ensemble/candidate_scores.csv
```

候选文件每行一条 50 bp A/C/G/T 序列；分数是课程标签尺度上的预测 log10 强度，并非实测。模型检查点和成员预测保存在本地被 Git 忽略的 `outputs/m4_cnn_ensemble/`，不随代码提交，需按命令重新训练获得。完整分组指标见[汇总数据](m4_cnn_ensemble_2026-11-09.json)。
