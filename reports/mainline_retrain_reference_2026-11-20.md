# 主线固定参考运行：从原始数据重训与参数重载复现

## 范围

本轮补充 11 月 19 日交接核验尚未执行的训练环节：沿用历史固定数据、模型设置和种子，从课程原始数组重新拟合一组参考模型，再生成、评分并核对 3,000 条记录。没有搜索参数或根据本轮结果改选模型。

参考运行使用模型种子 20260912、采样基础种子 20261030，连续自回归与连续条件 VAE 各在弱、中、强目标生成 500 条。它只是已有三个运行中的一个；本轮没有把其他种子和全部历史消融重新训练一遍。规则见[复现配置](../configs/mainline_retrain_reference_2026-11-20.toml)，历史参考见[原双评分记录](m4_generated_svr_compare_2026-11-12.md)。

## 实际重训内容

| 部分 | 本轮执行与记录 |
|---|---|
| RBF SVR | 在 NumPy 2.5.4 / scikit-learn 1.9.1 环境，用 8,318 条训练样本重新拟合并导出 NumPy 推理参数 |
| motif CNN | 在 NumPy 2.3.5 环境重新训练，开发验证集沿用原来的早停规则；运行 53 轮，记录中第 33 轮的验证 MAE 最低 |
| 全局及位置岭回归 | 分别用原训练样本及 alpha=10 重新拟合；融合权重保持 0.5/0.4/0.1 |
| 连续自回归 | 重新统计三阶、三个条件区间的位置上下文分布，保存可重建采样的状态 |
| 连续条件 VAE | 隐藏层 128、潜变量 16、beta=0.2，重新训练至 120 轮；保存推理参数和逐轮记录 |

全部模型拟合只使用原训练部分。开发验证部分用于原 CNN/VAE 早停及数值复核；它此前已多次参与开发，不能把本轮重训解释成新的独立评价。测试数组只由加载器检查格式，本轮未计算测试集预测指标。

## 复现结果

1. 新训练的 SVR 推理文件 SHA-256 与历史检查点完全一致；导出后 NumPy 与 scikit-learn 预测最大绝对差约为 9.24×10⁻¹⁴，小于固定容差 10⁻⁸。
2. 重新训练、生成和评分后，3,000 条记录的 CSV 与历史参考字节一致；六组指标 JSON 与原结果完全相同。
3. 重新加载本轮保存的 CNN、两项岭回归、SVR、自回归状态和 VAE 参数，再次采样与评分。六组各 500 条序列全部一致，M3 与 SVR 相对于本轮训练后输出的最大分数差均为 0。
4. 项目现有 65 项单元测试通过。本轮未因复现成功重新确定参数、目标或最终候选名单。

这说明在本轮固定环境和固定参考运行下，主线可以从原始数据训练并恢复已有结果，不只是重新读取旧表格。这不是三套种子的全面重训、跨硬件逐字节复现保证，也不是生成序列的功能或表达强度验证。

## 保存的本地产物

`outputs/mainline_retrain_reference/` 包含：

- `svr.npz`、`svr_retrained.json`：重训后的 SVR 参数与导出检查结果；
- `candidates_retrained.csv`、`generated_retrained.json`：重新生成的 3,000 条记录和指标；
- `checkpoints/`：CNN、两项岭回归、VAE、自回归状态、逐轮训练记录及文件摘要。

保存的是推理参数与状态，未保存优化器动量等续训状态。不能把参数保存称为中断续训功能。模型与逐条样本按仓库规则留在本地；代码、配置、本说明与[核验 JSON](mainline_retrain_reference_2026-11-20.json)随 Git 同步。11 月 19 日交接 ZIP 保持为当时的快照，不被本轮覆盖；以下新增命令使用本轮仓库代码。

## 重训与核验命令

需要 Python 3.11 以上；本轮实际使用 Python 3.12.14。两个环境分别安装仓库记录的依赖。`DATA_DIR` 指向课程大肠杆菌 `promoter.npy` 和 `gene_expression.npy` 所在目录。

```sh
python3.12 -m venv .venv-classical
.venv-classical/bin/python -m pip install -r reproduction/requirements-classical-history.txt
python3.12 -m venv .venv-mainline
.venv-mainline/bin/python -m pip install -r reproduction/requirements-audit.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv-classical/bin/python scripts/export_portable_svr.py --data-dir "$DATA_DIR" \
  --model-output outputs/mainline_retrain_reference/svr.npz \
  --report outputs/mainline_retrain_reference/svr_retrained.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv-mainline/bin/python scripts/evaluate_generated_svr_compare.py --data-dir "$DATA_DIR" \
  --svr-model outputs/mainline_retrain_reference/svr.npz \
  --output outputs/mainline_retrain_reference/generated_retrained.json \
  --candidate-csv outputs/mainline_retrain_reference/candidates_retrained.csv \
  --checkpoint-dir outputs/mainline_retrain_reference/checkpoints

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv-mainline/bin/python scripts/verify_reference_retraining.py \
  --run-dir outputs/mainline_retrain_reference \
  --output reports/mainline_retrain_reference_2026-11-20.json
```

`--checkpoint-dir` 是本轮增加的可选参数；省略时保留原训练与输出流程。核验脚本重新载入保存参数，不再拟合模型，并将结果与固定历史序列及指标比较；任何摘要、采样或分数差异都会使核验中止。
