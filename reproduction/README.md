# 主线复现交接包 v1

本包用于交接已有主线代码、结果和计算候选。包含版本化源码、配置、报告、划分，以及既有 9,000 条生成记录和 90 条候选。原始课程数据和训练模型检查点需由使用者另行准备。本包不是课程正式提交格式声明，也没有把计算评分当作实测表达强度。

## 1. 解压并检查文件

进入解压后的项目目录，先检查 ZIP 内所有文件是否与清单一致。这一步只使用 Python 标准库，不需要课程数据：

```sh
python3 scripts/verify_mainline_handoff.py --root .
```

`PACKAGE_MANIFEST.json` 保存所有其他包内文件的路径、字节数和 SHA-256。程序检查缺失、额外、符号链接或内容变化；它不把校验通过解释为生物学验证。

## 2. 创建核验环境

需要 Python 3.11 或以上，本轮实际验证 Python 3.12.14。以下是 macOS/Linux 示例：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/requirements-audit.txt
.venv/bin/python -m unittest discover -s tests
```

先执行第 1 步，再创建 `.venv`。检查脚本忽略本地虚拟环境与 Python 缓存，但仍会检查包内已有文件。核验用的 NumPy 2.3.5 与已存生成记录一致；完整测试还需要 openpyxl 及其 XML 依赖，已列在同一依赖文件中。`requirements-classical-history.txt` 记录另一个历史经典模型环境，供需要重训 SVR 的人员单独建立；它不是本轮完整重训的保证，两个环境不要混为同一版本。

## 3. 准备课程原始数据

设置 `DATA_DIR` 为大肠杆菌 `promoter.npy` 与 `gene_expression.npy` 所在目录：

```sh
export DATA_DIR="/你的课程数据/promoter/strenth/data"
```

两份文件的预期 SHA-256 分别为：

- `promoter.npy`：`e71911452431acb620478f0de5ae896481be5618e7a7f9712125cd98cb8e37ab`
- `gene_expression.npy`：`a1426fa5e5806fceff2c041218b781b6dc1f0cc84b39cf4ef626d02965fd9ce5`

代码会检查数据摘要和相似性簇划分；不要求重新划分数据，也不要自行把不同物种或不同长度数据拼接。

## 4. 从包内生成记录核验主线结论

这一步重算已有生成序列统计，核对候选来源和文件，并运行项目测试；模型训练成绩仍引用归档结果，不会重新训练 VAE、CNN 或 SVR。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python scripts/audit_mainline_evidence_v1.py --data-dir "$DATA_DIR" \
  --ranking-csv outputs/m4_target_repeats/rankings_all.csv \
  --candidate-dir outputs/final_candidates_v1 \
  --json-output outputs/handoff_verification/evidence.json \
  --markdown-output outputs/handoff_verification/results.md
```

成功后，将新生成的 JSON 和 Markdown 与归档版本比较。程序的结果包含 NumPy 版本，因此完整字节比较应使用上面固定的版本：

```sh
cmp outputs/handoff_verification/evidence.json reports/mainline_evidence_v1_2026-11-19.json
cmp outputs/handoff_verification/results.md reports/mainline_results_v1_2026-11-19.md
```

## 5. 重建 90 条计算候选

使用包内的 9,000 条原始生成记录，重新导出到新目录，避免覆盖交接文件：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python scripts/export_final_candidates_v1.py --data-dir "$DATA_DIR" \
  --ranking-csv outputs/m4_target_repeats/rankings_all.csv \
  --output-dir outputs/candidates_rebuilt \
  --report outputs/handoff_verification/candidates_rebuilt.json

diff -r outputs/final_candidates_v1 outputs/candidates_rebuilt
```

成功比较说明现有记录可以按固定规则重建同一清单。它不证明所有模型训练在新硬件上逐字节相同，也不证明候选实际强度达到目标。

## 6. 阅读入口

先读 `reports/mainline_results_v1_2026-11-19.md` 了解数据、两种生成器、评分器限制和项目结论，再读 `reports/final_candidates_v1_2026-11-17.md` 了解清单。历史重训入口在相应报告内；所需原始数据、检查点和额外依赖应按各实验说明准备。

## 7. 从交接目录重新构建 ZIP

包内保留构建脚本及固定内容规则，重建时不需要 Git 仓库或额外读取本机工作目录：

```sh
.venv/bin/python scripts/build_mainline_handoff.py \
  --artifacts-root . --output-dir outputs/handoff_rebuilt
```

使用相同源码、生成记录和 Python 压缩运行环境时，新 ZIP 的 SHA-256 应与交接包核验记录相同。跨 Python/zlib 版本的压缩字节可能不同，应同时检查解压文件内容清单。

## 8. 最新仓库的固定参考训练复现

在 11 月 20 日仓库代码中已补充一组从原始数组训练、生成和评分的复现，并核验保存参数能重建输出。命令、两套环境和范围见[训练复现记录](../reports/mainline_retrain_reference_2026-11-20.md)。11 月 19 日 ZIP 是冻结快照，不包含这一后续新增入口。
