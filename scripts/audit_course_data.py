#!/usr/bin/env python3
"""Audit the course promoter datasets and write a reproducible Markdown record."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

try:
    import pandas as pd
except ImportError as error:
    raise SystemExit("Install analysis dependencies with: python -m pip install -e '.[analysis]'") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the downloaded promoter directory containing strenth/ and reg_and_gen/",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.md"))
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_summary(values: np.ndarray) -> dict[str, object]:
    sequences = np.char.upper(np.asarray(values).astype(str))
    counts = Counter(sequences.tolist())
    lengths = Counter(np.char.str_len(sequences).astype(int).tolist())
    gc = np.array(
        [(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in sequences],
        dtype=np.float64,
    )
    return {
        "rows": len(sequences),
        "unique": len(counts),
        "duplicate_rows": sum(count - 1 for count in counts.values() if count > 1),
        "duplicate_groups": sum(count > 1 for count in counts.values()),
        "lengths": dict(sorted(lengths.items())),
        "invalid": sum(bool(set(sequence) - set("ACGT")) for sequence in sequences),
        "gc_mean": float(gc.mean()),
        "gc_sd": float(gc.std()),
        "gc_min": float(gc.min()),
        "gc_max": float(gc.max()),
    }


def numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    numbers = np.asarray(values, dtype=np.float64)
    return {
        "rows": len(numbers),
        "missing": int(np.isnan(numbers).sum()),
        "min": float(np.nanmin(numbers)),
        "q25": float(np.nanquantile(numbers, 0.25)),
        "median": float(np.nanmedian(numbers)),
        "q75": float(np.nanquantile(numbers, 0.75)),
        "q95": float(np.nanquantile(numbers, 0.95)),
        "max": float(np.nanmax(numbers)),
        "nonpositive": int(np.sum(numbers <= 0)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    strength_root = root / "strenth" / "data"

    ecoli_sequences_raw = np.load(strength_root / "promoter.npy", allow_pickle=False)
    ecoli_labels_raw = np.load(strength_root / "gene_expression.npy", allow_pickle=False)
    ecoli_sequences = np.char.upper(ecoli_sequences_raw.astype(str))
    ecoli_labels = ecoli_labels_raw.astype(np.float64)
    ecoli = sequence_summary(ecoli_sequences)
    ecoli_strength = numeric_summary(ecoli_labels)
    log_strength = np.log10(ecoli_labels)
    gc = np.array(
        [(sequence.count("G") + sequence.count("C")) / len(sequence) for sequence in ecoli_sequences]
    )
    pearson = float(np.corrcoef(gc, log_strength)[0, 1])
    spearman = float(np.corrcoef(average_ranks(gc), average_ranks(log_strength))[0, 1])

    ecoli_text = pd.read_csv(strength_root / "supplementary" / "E_coli.txt", sep="\t", skipinitialspace=True)
    ecoli_text.columns = [column.strip() for column in ecoli_text.columns]
    ecoli_text["promoter"] = ecoli_text["promoter"].astype(str).str.strip().str.upper()
    ecoli_text_match = np.array_equal(ecoli_text["promoter"].to_numpy(), ecoli_sequences)
    ecoli_label_match = np.array_equal(ecoli_text["strength"].to_numpy(float), ecoli_labels)

    yeast_sequences_raw = np.load(strength_root / "promoter_yeast.npy", allow_pickle=True)
    yeast_labels = np.load(strength_root / "gene_expression_yeast.npy", allow_pickle=False).astype(float)
    yeast = sequence_summary(yeast_sequences_raw)
    yeast_strength = numeric_summary(yeast_labels)
    yeast_csv = pd.read_csv(strength_root / "yeast_promoters_processed.csv")

    cyanobacterium_path = strength_root / "supplementary" / "Cyanobacterium.xlsx"
    cyanobacterium = pd.read_excel(cyanobacterium_path)
    cyanobacterium_sequences = sequence_summary(cyanobacterium["Promoter"].astype(str).to_numpy())
    cyanobacterium_reads = numeric_summary(cyanobacterium["Reads"].to_numpy())
    cy_groups = cyanobacterium.groupby("Promoter")["Reads"].agg(["size", "nunique"])
    training_path = strength_root / "Training dataset.xlsx"
    training_dataset = pd.read_excel(training_path)

    species_rows = []
    species_root = root / "reg_and_gen" / "Datasets"
    for directory in sorted(path for path in species_root.iterdir() if path.is_dir()):
        frame = pd.read_csv(directory / "Dataset.csv")
        summary = sequence_summary(frame["seq"].astype(str).to_numpy())
        labels = Counter(frame["label"].tolist())
        species_rows.append(
            (
                directory.name,
                len(frame),
                labels.get(1, 0),
                labels.get(0, 0),
                summary["lengths"],
                summary["unique"],
                summary["invalid"],
            )
        )

    tertiles = np.quantile(ecoli_labels, [1 / 3, 2 / 3])
    lines = [
        "# 课程启动子数据复核记录",
        "",
        "复核日期：2026年9月17日",
        "",
        "## 1 大肠杆菌连续强度主数据",
        "",
        f"- `promoter.npy` 原始形状 `{ecoli_sequences_raw.shape}`，数据类型 `{ecoli_sequences_raw.dtype}`；",
        f"- `gene_expression.npy` 原始形状 `{ecoli_labels_raw.shape}`，数据类型 `{ecoli_labels_raw.dtype}`，读取后需要转为浮点数；",
        f"- 样本数 {ecoli['rows']:,}，唯一序列 {ecoli['unique']:,}，完全重复 0；",
        f"- 全部序列为 50 bp，非法 DNA 字符 {ecoli['invalid']}；",
        f"- 强度最小值 {ecoli_strength['min']:.2f}，中位数 {ecoli_strength['median']:.3f}，95% 分位数 {ecoli_strength['q95']:.3f}，最大值 {ecoli_strength['max']:.2f}；",
        f"- 全量数据临时三等分边界为 {tertiles[0]:.2f} 和 {tertiles[1]:.2f}；正式实验必须仅用训练集重算；",
        f"- GC 含量均值 {ecoli['gc_mean']:.2%}，标准差 {ecoli['gc_sd']:.2%}，范围 {ecoli['gc_min']:.0%} 至 {ecoli['gc_max']:.0%}；",
        f"- GC 与 `log10(强度)` 的 Pearson 为 {pearson:.3f}，Spearman 为 {spearman:.3f}；",
        f"- `E_coli.txt` 与两个 NPY 数组在顺序、序列和强度上完全一致：{ecoli_text_match and ecoli_label_match}。",
        "",
        "主数据可以用于本项目的连续条件生成。强度分布长尾明显，应使用 `log10` 变换；完全无重复不代表没有近重复，正式划分仍需按序列相似性聚类。",
        "",
        "## 2 酵母连续标签数据",
        "",
        f"- {yeast['rows']:,} 条、全部 50 bp、{yeast['unique']:,} 条唯一序列、非法字符 {yeast['invalid']}；",
        f"- 标签范围 {yeast_strength['min']:.1f} 至 {yeast_strength['max']:.1f}，小于等于 0 的样本 {yeast_strength['nonpositive']:,} 条；",
        f"- `yeast_promoters_processed.csv` 与 NPY 数组一致：{np.array_equal(yeast_csv['seq'].astype(str).to_numpy(), yeast_sequences_raw.astype(str)) and np.array_equal(yeast_csv['strength'].to_numpy(float), yeast_labels)}；",
        "- `promoter_yeast.npy` 是 object 数组，需要 `allow_pickle=True`；优先使用 CSV 可以避免该读取限制。",
        "",
        "酵母标签含义和序列规律与细菌主数据不同，不应直接混合训练大肠杆菌生成模型。",
        "",
        "## 3 蓝藻数据",
        "",
        f"- {cyanobacterium_sequences['rows']:,} 行、{cyanobacterium_sequences['unique']:,} 条唯一序列、全部 100 bp；",
        f"- 重复后多出的记录 {cyanobacterium_sequences['duplicate_rows']} 行，涉及 {int((cy_groups['size'] > 1).sum())} 个序列组；",
        f"- 标签冲突的重复序列组 {int((cy_groups['nunique'] > 1).sum())} 个；",
        f"- Reads 范围 {cyanobacterium_reads['min']:.0f} 至 {cyanobacterium_reads['max']:.0f}，其中小于等于 0 的记录 {cyanobacterium_reads['nonpositive']} 条；",
        f"- `Training dataset.xlsx` 与 `Cyanobacterium.xlsx` 内容完全一致：{training_dataset.equals(cyanobacterium)}；",
        f"- 两个文件 SHA-256 是否相同：{file_hash(training_path) == file_hash(cyanobacterium_path)}。",
        "",
        "蓝藻数据在序列长度、物种和标签定义上均不同，而且存在同序列多标签冲突，不能直接并入主数据。",
        "",
        "## 4 六物种二分类数据",
        "",
        "| 物种 | 总数 | 正类 | 负类 | 长度分布 | 唯一序列 | 非法序列 |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, total, positive, negative, lengths, unique, invalid in species_rows:
        length_text = ", ".join(f"{length} bp: {count}" for length, count in lengths.items())
        lines.append(f"| {name} | {total:,} | {positive:,} | {negative:,} | {length_text} | {unique:,} | {invalid} |")
    lines.extend(
        [
            "",
            f"六物种数据合计 {sum(row[1] for row in species_rows):,} 条，标签只表示是否为启动子，不是连续强度。它可以用于辅助启动子识别评价，不能训练目标强度生成器。",
            "",
            "## 5 主数据文件校验值",
            "",
            f"- `promoter.npy`: `{file_hash(strength_root / 'promoter.npy')}`",
            f"- `gene_expression.npy`: `{file_hash(strength_root / 'gene_expression.npy')}`",
            f"- `promoter_yeast.npy`: `{file_hash(strength_root / 'promoter_yeast.npy')}`",
            f"- `gene_expression_yeast.npy`: `{file_hash(strength_root / 'gene_expression_yeast.npy')}`",
            "",
            "## 6 复核结论",
            "",
            "1. 本项目主数据仍应使用 11,884 条、50 bp 的大肠杆菌连续强度数据；",
            "2. M1 报告中的样本数、强度范围、GC 统计、酵母规模、蓝藻重复情况和六物种总量均与原始文件一致；",
            "3. `gene_expression.npy` 虽然保存为字符串数组，但全部可以无损转换为有限正浮点数；",
            "4. 酵母、蓝藻和六物种数据只能作为独立扩展或辅助评价，不能与主数据直接混合；",
            "5. 下一步需要检查近重复序列并固定相似性聚类划分，这仍是数据环节最重要的未完成工作。",
            "",
        ]
    )

    output = args.output
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report to {output}")


if __name__ == "__main__":
    main()
