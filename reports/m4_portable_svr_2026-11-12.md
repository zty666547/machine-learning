# M4：RBF SVR 候选预测器的保存与打分

## 本轮工作

将已固定配置的 RBF SVR 用 8,318 条训练数据重新拟合，导出为可用 NumPy 读取的推理文件，并实现 50 bp 序列批量打分。该模型使用位置 1-mer/2-mer、全局 1–3-mer 和 GC 特征。导出内容包括训练集拟合的特征标准化参数、支持向量、系数、截距和核带宽；支持向量是部分训练序列的标准化特征，可能保留训练数据的信息。导出过程固定于[配置](../configs/m4_portable_svr_2026-11-12.toml)。

## 一致性检查

使用此前同一数据、模型配置、NumPy 2.5.4 与 scikit-learn 1.9.1 训练；1,783 条开发验证样本的 scikit-learn 预测 MAE 为 0.4387331964242697，与先前记录一致。将模型导出后重新加载，NumPy 实现与 scikit-learn 在这些样本上的预测最大绝对差约 **9.24×10⁻¹⁴**，小于预设容差 1×10⁻⁸。该检查用于确认实现一致，不是新的独立评价，也没有用开发验证集选参数。

额外以三条已有序列核对命令行输出与直接调用的一致性；针对 RBF 计算、保存加载和无效输入新增单元测试，项目共 51 项测试通过。跨机器或数值环境可能出现浮点差异。

## 使用

候选文件每行一条长度为 50 的 A/C/G/T 序列：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 python scripts/score_portable_svr.py \
   --model outputs/m4_portable_svr/position_rbf_svr.npz \
   --candidate-file "$CANDIDATES" \
   --output outputs/m4_portable_svr/candidate_scores.csv
```

CSV 包含输入序列与预测 log10 强度。推理只需 NumPy，不需要 scikit-learn。模型文件约 7.6 MiB，位于本地 `outputs/`，按仓库规则不会随 Git 提交。要重建文件，在安装 NumPy 2.5.4 和 scikit-learn 1.9.1 后运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 python scripts/export_portable_svr.py --data-dir "$DATA_DIR" \
   --model-output outputs/m4_portable_svr/position_rbf_svr.npz \
   --report reports/m4_portable_svr_2026-11-12.json
```

测试数组仅在加载时检查格式，本轮未提取测试特征或计算指标。当前模型是探索性候选，强组低估问题仍存在；输出不是实测强度，也没有替换 M3 融合筛选器。数值一致性不证明生物学有效性。文件摘要和验证记录见 [JSON](m4_portable_svr_2026-11-12.json)。
