# M4：三个生成/采样种子的目标接近度排序

## 目的与方法

上一轮使用一个种子组合，观察到中目标组中 M3 与 SVR 的目标接近度前 50 条重合较少。本轮在看过该结果之后，固定相同方法并扩展到此前条件响应实验中另外两个已有种子组合。新种子评分结果出来前固定[实验规则](../configs/m4_target_repeat_2026-11-13.toml)。

每个组合包含连续自回归和连续 VAE，各在弱、中、强目标下生成 500 条；三个组合共 9,000 条记录。沿用原 M3 融合筛选器和固定 SVR 检查点，在每个生成器、种子组合和目标组内部，分别对两个评分器按 `|预测 log10 分数 − 输入目标 log10 强度|` 从小到大排列，并统计两者前 50 条的重合。分数距离不是实测误差；并列按原样本编号处理。

前一个组合复用已有结果；另外两个组合重新训练对应的 VAE 和固定 M3，重新采样并计算 SVR 分数。两组新增运行的 12 组序列 SHA-256 均与早期条件响应记录一致，M3 的开发验证 Pearson 和各组分数均值与原记录核对。自回归模型训练为确定性，重复区别主要来自采样；VAE 训练与采样种子均改变。

## 结果：各组前 50 条重合数

| 生成器 | 输入目标 | 原种子组合 | 新组合一 | 新组合二 | 三次均值 |
|---|---|---:|---:|---:|---:|
| 自回归 | 弱 | 21 | 19 | 23 | 21.00 |
| 自回归 | 中 | 9 | 8 | 8 | 8.33 |
| 自回归 | 强 | 20 | 19 | 27 | 22.00 |
| VAE | 弱 | 19 | 20 | 24 | 21.00 |
| VAE | 中 | 6 | 6 | 8 | 6.67 |
| VAE | 强 | 26 | 20 | 25 | 23.67 |

在本次三个组合中，中目标组的重合持续较少；目标距离排序的 Spearman 均值分别为自回归 0.4176、VAE 0.4509。弱、强组的重合数较高，但仍不意味着真实强度一致。每组内 500 条序列均不重复；不同组或不同运行之间可能重复，9,000 是记录数。

两个模型的评分尺度、整体偏移和范围收缩都可能改变“到目标的距离”，因此重合较少既可能涉及序列相对判断，也可能涉及校准差异；本实验不能区分原因，更不能判定哪个评分器正确。

## 边界与复现

三个组合及其生成样本此前用于开发和展示，两个评分器和生成器共享课程训练数据；它们不是独立数据集或独立生物验证。M3 和 VAE 均使用反复使用的开发验证集；本轮没有新增外部数据、测试指标或湿实验。分数接近目标不等于真实强度接近目标，本轮不确定最终实验候选。

先按[SVR 推理记录](m4_portable_svr_2026-11-12.md)生成匹配 SHA-256 的检查点，并确保前次 `outputs/m4_generated_svr_compare/candidates.csv` 可用；再运行：

```sh
for i in 0 1; do
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 python scripts/evaluate_target_repeat.py --data-dir "$DATA_DIR" \
  --svr-model outputs/m4_portable_svr/position_rbf_svr.npz --run-index "$i" \
  --output "reports/m4_target_repeat_run_${i}.json" \
  --candidate-csv "outputs/m4_target_repeats/candidates_${i}.csv"
done
python scripts/summarize_target_repeat.py --output reports/m4_target_repeat_2026-11-13.json
python scripts/export_target_repeat_rankings.py \
 --reference-csv outputs/m4_generated_svr_compare/candidates.csv \
 --repeat-dir outputs/m4_target_repeats --output outputs/m4_target_repeats/rankings_all.csv
```

两次新增运行各六组 top-50 重合数均经独立重算；汇总 JSON 和导出排名分别重复生成一致，53 项现有测试通过。逐条结果在 Git 忽略的本地 `outputs/m4_target_repeats/`，完整汇总见[JSON](m4_target_repeat_2026-11-13.json)；输出为探索记录，不是实测结果或最终候选。
