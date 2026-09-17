# M2：位置感知强度预测器探索

复核日期：2026 年 9 月 17 日。数据、目标变换和划分均与 [M2 固定划分与基线报告](m2_split_and_strength_baseline_2026-09-17.md) 相同：输入为 50 bp 大肠杆菌启动子序列，预测目标为 `log10(强度)`。

## 问题与设计

全局 k-mer 统计不能区分同一 motif 出现在不同位置的情形。本次探索因此加入位置相关的表示：

1. **位置岭回归**：每个碱基位置的单碱基与相邻二碱基 one-hot 特征，并保留全局 1--3-mer/GC 特征；
2. **位置 MLP**：只使用每个位置的单碱基 one-hot 特征，以两层 ReLU 网络学习非线性组合；
3. **motif CNN**：7 bp 滑动卷积提取局部 motif，将卷积响应按五个连续位置区间分别平均，再用 ReLU 预测头输出强度。

第三种设计同时保留“局部模式”和“模式大致出现在启动子哪个区域”两类信息，因而是下一阶段进一步扩展的主模型方向。

## 当前结果

| 模型 | 验证 MAE | 验证 Pearson | 测试 MAE | 测试 Pearson | 结论 |
|---|---:|---:|---:|---:|---|
| 全局 1--3-mer 岭回归 | 0.4458 | 0.2326 | 0.4599 | 0.2349 | 开发基线 |
| 位置岭回归 | 0.4633 | 0.2172 | 0.4745 | 0.2237 | 未改善 |
| 位置 MLP | 0.4734 | 0.1003 | 0.4887 | 0.1118 | 未改善 |
| 位置 motif CNN | **0.4317** | **0.2607** | 0.4549 | 0.2139 | 验证集改善，需继续验证 |

CNN 在验证集上的 MAE 和相关性优于两个岭回归版本，说明局部 motif 和位置分区具有可学习信号；但其开发期测试集相关性未超过全局 k-mer 基线，尚不能冻结为生成器的独立评价器。

## 对后续工作的影响

- 保留位置岭回归和位置 MLP 作为可复现的反例基线；
- 以 CNN 的卷积 motif 表示为下一轮预测器的基础，优先通过验证集调整卷积宽度、过滤器数量、位置汇聚方式和正则化；
- 后续模型选择只看训练集和验证集；本报告中的测试指标仅作为本次开发探索的监测记录，最终模型应在冻结配置后重新进行一次测试集评估；
- 在独立强度预测器达到稳定、可解释的效果前，生成结果不能只以预测强度作为有效性证明，仍须同时报告 motif、GC、k-mer、新颖性和多样性指标。

## 复现命令

```bash
python scripts/train_position_strength_predictor.py \
  --data-dir /path/to/promoter/strenth/data \
  --output-dir outputs/position_ridge

python scripts/train_position_mlp_predictor.py \
  --data-dir /path/to/promoter/strenth/data \
  --output-dir outputs/position_mlp

python scripts/train_motif_cnn_predictor.py \
  --data-dir /path/to/promoter/strenth/data \
  --output-dir outputs/motif_cnn
```
