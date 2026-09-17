# M2：motif CNN 验证集选择记录

日期：2026 年 9 月 17 日。目标为在固定 90% 相似性聚类划分上，以 `log10(强度)` 训练位置感知 CNN。除表中列出的卷积宽度和过滤器数量外，其余设置一致：五个位置区间、64 单元 ReLU 预测头、学习率 0.001、权重衰减 0.0001、随机种子 20260912 和早停。

## 选择规则

模型选择只查看验证集，优先选择验证集 MAE 最低的候选；若 MAE 相同，再比较 Pearson 相关系数。测试集不参与结构选择。

| 候选 | 卷积窗口 | 过滤器 | 验证 MAE | 验证 RMSE | 验证 Pearson | 验证 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_k5_f32` | 5 bp | 32 | 0.4329 | 0.5824 | 0.2607 | **0.2406** |
| `cnn_k7_f32` | 7 bp | 32 | **0.4317** | 0.5826 | 0.2607 | 0.2338 |
| `cnn_k9_f32` | 9 bp | 32 | 0.4377 | 0.6001 | 0.2148 | 0.1936 |
| `cnn_k7_f64` | 7 bp | 64 | 0.4324 | 0.5827 | **0.2634** | 0.2340 |

按事先设定的 MAE 规则，选定 `cnn_k7_f32`。对应的冻结配置见 `configs/m2_motif_cnn_selected.toml`。

## 固定结构后的测试

在选定结构后进行一次测试集评估：MAE 为 0.4549、RMSE 为 0.6227、Pearson 为 0.2139、Spearman 为 0.1933。

该结果没有超过全局 1--3-mer 岭回归基线的测试 Pearson 0.2349。因此，冻结的是下一轮改进所使用的 **CNN 结构和实验流程**，不是可用于最终生成评价的强度验证器。后续应在训练/验证集上继续提高稳定性；在独立评价器达标前，生成结果须与 motif、GC、k-mer、新颖性和多样性指标一起解读。

## 复现

仅进行验证集选择：

```bash
python scripts/tune_motif_cnn.py \
  --data-dir /path/to/promoter/strenth/data
```

以冻结配置进行一次测试评估：

```bash
python scripts/train_motif_cnn_predictor.py \
  --config configs/m2_motif_cnn_selected.toml \
  --data-dir /path/to/promoter/strenth/data \
  --output-dir outputs/m2_motif_cnn_selected \
  --evaluate-test
```
