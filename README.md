# PR03-01 条件启动子生成

第25组“机器学习综合实践”课程项目代码仓库。

项目阶段性进展与关键实验结论见 [更新日志](CHANGELOG.md)。

本项目根据目标表达强度生成长度为 50 bp 的大肠杆菌启动子候选序列。计划比较条件 VAE 与条件自回归模型，并使用独立强度预测器以及 motif、GC、k-mer、新颖性和多样性指标统一评价生成结果。

## 当前版本

当前版本建立了所有成员共用的工程基础：

- TOML 配置加载与校验；
- 控制台和文件统一日志；
- 随机种子管理；
- 课程数据读取与合法性检查；
- 1-mer 至 3-mer 及 GC 特征；
- 可保存和加载的岭回归强度预测基线；
- CVAE 与自回归模型共用的生成器接口；
- 不依赖课程数据的端到端检查脚本。

岭回归模型用于验证数据、训练、评价、日志和保存流程是否连通。它是开发基线，不是最终的独立评估器。正式实验使用版本化的相似性聚类划分，并将比较更适合序列建模的预测器。

## 环境

需要 Python 3.10 或以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 快速检查

```bash
python scripts/smoke_test.py
```

成功时输出：

```text
Smoke test passed
```

## 复核课程数据

安装分析依赖后，可以重新生成数据核验记录：

```bash
python -m pip install -e '.[analysis]'
python scripts/audit_course_data.py \
  --data-root /path/to/promoter \
  --output reports/data_audit.md
```

当前复核结果见 `reports/data_audit_2026-09-17.md`。

课程包中两套大肠杆菌数据的区别、长度分布和样例见
`reports/ecoli_data_overview_2026-09-17.md`。其中连续强度主数据全部为
50 bp；另一套二分类数据主要为 81 bp，不能将两者混为同一数据集。

该专项记录可以用不依赖第三方库的脚本重新生成：

```bash
python scripts/summarize_ecoli_data.py \
  --data-root /path/to/promoter \
  --output reports/ecoli_data_overview.md
```

## 运行正式 EDA

固定相似性划分生成后，运行：

```bash
python scripts/run_eda.py \
  --data-dir /path/to/promoter/strenth/data
```

报告和 SVG 图表将写入 `reports/eda/`。

## 在课程数据上训练开发基线

数据目录中需要包含：

- `promoter.npy`：一维 DNA 序列数组；
- `gene_expression.npy`：一维正数强度数组。

运行：

```bash
python scripts/train_strength_baseline.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/ridge_strength_dev
```

当 `configs/base.toml` 指向 `data/splits/ecoli_similarity_split.csv` 时，程序自动使用固定的相似性聚类划分；也可以使用 `--split-file` 指定另一份经过复核的划分文件。

程序会生成：

- `logs/pipeline.log`：带时间、级别和模块名称的运行日志；
- `outputs/ridge_strength_dev/model.npz`：模型参数；
- `outputs/ridge_strength_dev/metrics.json`：验证集和测试集指标。

## 训练位置感知强度预测器

该模型使用每个位置的单碱基、相邻二碱基以及全局 k-mer/GC 特征。它是位置感知的线性候选基线。

```bash
python scripts/train_position_strength_predictor.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/position_strength_predictor
```

脚本只通过验证集选择正则强度，随后一次性报告测试集结果。

也可运行两种非线性候选：位置 MLP 和带位置分区的 motif CNN。三种方案的当前结论见 [M2 位置预测器探索报告](reports/m2_position_predictor_2026-09-17.md)。

```bash
python scripts/train_position_mlp_predictor.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/position_mlp

python scripts/train_motif_cnn_predictor.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/motif_cnn
```

完成验证集选择后，使用冻结配置并显式加入 `--evaluate-test` 才会输出测试集指标：

```bash
python scripts/train_motif_cnn_predictor.py \
  --config configs/m2_motif_cnn_selected.toml \
  --data-dir /path/to/course/data \
  --output-dir outputs/m2_motif_cnn_selected \
  --evaluate-test
```

## 配置

共享配置位于 `configs/base.toml`。随机种子、数据文件名、划分比例、特征范围、模型正则强度和日志位置均从配置读取。修改实验设置时，应新增配置文件或提交明确的配置修改，避免只在个人命令中保留参数。

## M3 无条件生成基线

位置频率生成器按训练集中每个位置的 A/C/G/T 频率独立采样。它用于验证生成和统一评价流程，不具备表达强度控制能力。

```bash
python scripts/generate_position_frequency_baseline.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/position_frequency_baseline
```

结果会包含生成序列，以及合法性、唯一性、新颖性、GC、位置碱基频率和 3-mer 分布指标。首轮结果见 [M3 位置频率生成基线报告](reports/m3_position_frequency_baseline_2026-09-17.md)。

## M3 条件生成基线

条件位置频率生成器将训练集按弱、中、强表达等级分别建模。它能验证条件输入是否改变序列分布，但不代表生成序列已被可靠地预测为相应强度。

```bash
python scripts/generate_conditional_position_baseline.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/conditional_position_frequency_baseline
```

首轮结果见 [M3 条件位置频率生成基线报告](reports/m3_conditional_position_baseline_2026-09-18.md)。

## M3 条件自回归生成基线

条件自回归生成器根据目标等级、当前位置和前 3 个已生成碱基生成下一个碱基，因此可以保留局部 k-mer 关系。

```bash
python scripts/generate_conditional_autoregressive_baseline.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/conditional_autoregressive_baseline
```

首轮结果见 [M3 条件自回归基线报告](reports/m3_conditional_autoregressive_baseline_2026-09-18.md)。

## M3 连续条件 VAE

条件 VAE 使用连续 `log10(表达强度)` 和随机潜变量生成新序列。它目前是可复现的候选模型，须与条件自回归基线共同评价。

```bash
python scripts/train_conditional_vae.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/conditional_vae_selected
```

首轮结果和限制见 [M3 连续条件 VAE 报告](reports/m3_conditional_vae_first_run_2026-09-19.md)。

连续目标响应分析表明，生成 GC 会随目标对数强度整体下降，但局部仍有波动；详见 [M3 连续条件 VAE 响应报告](reports/m3_vae_continuous_response_2026-09-20.md)。

KL 权重消融后，后续 VAE 实验优先使用 `beta=0.20`；详见 [M3 VAE KL 权重稳定性报告](reports/m3_vae_beta_stability_2026-09-20.md)。

潜变量维度消融后，后续 VAE 实验固定 `latent_size=16`；网络宽度消融后固定 `hidden_size=128`；详见 [M3 VAE 潜变量维度稳定性报告](reports/m3_vae_latent_stability_2026-09-21.md) 与 [M3 VAE 网络宽度稳定性报告](reports/m3_vae_width_stability_2026-09-21.md)。

## M3 生成器统一比较

两种条件生成器使用相同的合法性、新颖性、GC、位置碱基频率、3-mer、候选 motif 和多样性指标比较。当前条件自回归在局部模式保真度上更强；当前配置的复查结论见 [M3 阶段复查记录](reports/m3_review_2026-09-21.md)。

```bash
python scripts/compare_conditional_generators.py \
  --data-dir /path/to/course/data \
  --output reports/m3_conditional_generator_comparison.json
```

## M3 汇报前一致性检查

在汇报前可运行以下命令，确认固定划分、选定 VAE 配置、生成结果和报告结论仍然一致：

```bash
python scripts/verify_m3_release.py \
  --data-dir /path/to/promoter/strenth/data
```

小组内部汇报表述见 [M3 期中汇报工作简报](reports/m3_midterm_brief_2026-09-22.md)。

## 冻结强度预测器诊断

已冻结 CNN 的分组误差诊断只用于解释其局限，不用于继续选择模型：

```bash
python scripts/analyze_frozen_motif_cnn.py \
  --data-dir /path/to/promoter/strenth/data \
  --evaluate-test
```

结果说明见 [冻结 CNN 分组误差诊断](reports/m2_frozen_motif_cnn_diagnostic_2026-09-24.md)。

## M3 汇报图表

```bash
python scripts/render_m3_midterm_figures.py
```

图表和讲解说明见 [M3 期中汇报图表说明](reports/m3_midterm_figures_guide_2026-09-23.md)。

## 目录

```text
configs/             共享实验配置
data/                数据说明与后续固定划分
scripts/             可直接运行的训练和检查入口
src/promoter_ml/     数据、模型、评价、日志等公共代码
outputs/             本地生成结果，不进入普通 Git 历史
logs/                本地运行日志，不进入普通 Git 历史
```

## 协作约定

- `main` 保存经过检查、能够运行的阶段版本；
- 每项工作在独立任务分支完成，通过合并请求进入 `main`；
- 开始工作前同步远程更新，当天完成的有效工作当天推送；
- 一个提交只完成一个可以说明清楚的改动；
- 数据接口、固定划分和评价规则的修改必须由另一名成员复核；
- 原始数据、模型权重、缓存和个人环境不直接提交到普通 Git 历史。

推荐提交说明：

```text
feat(data): add similarity-aware dataset split
feat(eval): add independent strength predictor
feat(cvae): implement conditional decoder
feat(ar): add temperature sampling
fix(split): prevent similar sequences crossing datasets
docs(m2): record EDA findings and reproduction commands
```
