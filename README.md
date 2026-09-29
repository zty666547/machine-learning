# PR03-01 条件启动子生成

第25组“机器学习综合实践”课程项目代码仓库。

本项目根据目标表达强度生成长度为 50 bp 的大肠杆菌启动子候选序列。计划比较条件 VAE 与条件自回归模型，并使用独立强度预测器以及 motif、GC、k-mer、新颖性和多样性指标统一评价生成结果。

## 当前版本

当前版本（W0 工程基础）建立了所有成员共用的工程环境、目录结构和数据契约：

- TOML 配置加载与校验；
- 控制台和文件统一日志；
- 随机种子管理；
- 课程数据读取与合法性检查；
- 1-mer 至 3-mer 及 GC 特征；
- 可保存和加载的岭回归强度预测基线；
- CVAE 与自回归模型共用的生成器接口；
- 不依赖课程数据的端到端检查脚本；
- 环境自检脚本 `scripts/check_environment.py`；
- 数据校验、校验和与清单生成脚本 `scripts/prepare_data.py`；
- `experiments/`、`results/`、`data/splits/` 等后续工作目录。

岭回归模型用于验证数据、训练、评价、日志和保存流程是否连通。它是开发基线，不是最终的独立评估器。正式实验将使用相似性聚类划分替代当前的确定性随机划分，并比较更适合序列建模的预测器，详见 `data/splits/README.md`。

## 环境

需要 Python 3.10 或以上版本，numpy 为必需依赖；W7 的条件生成模型需要 PyTorch（Ubuntu 训练机使用 CUDA 版）。

Ubuntu（CUDA 训练机）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,dev]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Windows（编辑、数据准备与轻量脚本）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis,dev]"
```

安装后先用一条命令确认工具链与 GPU：

```bash
python scripts/check_environment.py
```

它会逐项报告 Python、numpy、torch、CUDA 设备、显存以及可选的分析依赖；缺少必需组件时返回非零退出码。镜像源、驱动与显存注意事项见 `docs/ENVIRONMENT.md`。

安装后建议依次运行 W0 的四项验收检查：

```bash
python scripts/check_layout.py        # 目录骨架与 extras 声明
python scripts/check_environment.py   # Python/numpy/torch/CUDA 与可选依赖
python -m pytest                      # 数据契约与布局的快速测试
python scripts/smoke_test.py          # 合成数据上的端到端检查
```

## 数据准备

将课程数据放入 `data/raw/`（数据契约见该目录的 README），然后校验并生成规范化副本：

```bash
python scripts/prepare_data.py --dry-run   # 只校验并打印 sha256
python scripts/prepare_data.py             # 同时写入 data/processed/
```

`--dry-run` 不写任何文件；正式运行时会在 `data/raw/` 生成 `dataset_manifest.json`，记录样本数、强度范围、GC 均值与各文件 sha256，该清单需要随数据集说明一起提交。


## 快速检查

```bash
python scripts/smoke_test.py
```

成功时输出：

```text
Smoke test passed
```

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

程序会生成：

- `logs/pipeline.log`：带时间、级别和模块名称的运行日志；
- `outputs/ridge_strength_dev/model.npz`：模型参数；
- `outputs/ridge_strength_dev/metrics.json`：验证集和测试集指标。

## 配置

共享配置位于 `configs/base.toml`。随机种子、数据文件名、划分比例、特征范围、模型正则强度和日志位置均从配置读取。修改实验设置时，应新增配置文件或提交明确的配置修改，避免只在个人命令中保留参数。

## 目录

```text
configs/             共享实验配置
data/raw/            原始课程数据与 dataset_manifest.json
data/processed/      prepare_data.py 生成的规范化数组
data/splits/         小体积、可复核的固定划分定义
docs/                环境与数据契约说明
experiments/         每个实验一个可复现脚本（exp01_… 等）
results/             实验产生的图表、表格与清单（进入 Git 历史）
scripts/             可直接运行的准备、自检、训练与评价入口
src/promoter_ml/     数据、模型、评价、日志等公共代码
tests/               数据契约与目录布局的快速测试
outputs/             本地检查点与大型产物，不进入普通 Git 历史
logs/                本地运行日志，不进入普通 Git 历史
```

## 平台与运行约定

- Windows 用于编辑、数据准备与轻量脚本，必须只依赖 numpy 即可运行；
- Ubuntu 训练机带 CUDA 版 PyTorch，用于 W7 的条件生成模型训练；
- 所有命令保持 `python scripts/xxxx.py` 的跨平台形式，不在脚本中硬编码平台路径；
- GPU 显存为 8 GiB（RTX 4060 Laptop），训练脚本需在结果中记录 batch size 与显存占用。

## 协作约定

- `main` 保存经过检查、能够运行的阶段版本；
- 每项工作在独立任务分支完成，通过合并请求进入 `main`；
- 开始工作前同步远程更新，当天完成的有效工作当天推送；
- 一个提交只完成一个可以说明清楚的改动；
- 数据接口、固定划分和评价规则的修改必须由另一名成员复核；
- 原始数据、模型权重、缓存和个人环境不直接提交到普通 Git 历史；
- 每个实验脚本产生的表格与图必须能由仓库内的脚本加配置复现，禁止只提交结果。

推荐提交说明：

```text
feat(data): add similarity-aware dataset split
feat(eval): add independent strength predictor
feat(cvae): implement conditional decoder
feat(ar): add temperature sampling
fix(split): prevent similar sequences crossing datasets
docs(m2): record EDA findings and reproduction commands
```
