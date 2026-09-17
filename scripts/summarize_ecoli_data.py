#!/usr/bin/env python3
"""Summarize every E. coli dataset in the supplied course data package."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the promoter directory containing strenth/ and reg_and_gen/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ecoli_data_overview.md"),
    )
    return parser.parse_args()


def read_strength_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t", skipinitialspace=True)
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in reader
        ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def length_text(rows: list[dict[str, str]]) -> str:
    counts = Counter(len(row["seq"]) for row in rows)
    return ", ".join(f"{length} bp：{count:,} 条" for length, count in sorted(counts.items()))


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()

    strength_rows = read_strength_rows(root / "strenth/data/supplementary/E_coli.txt")
    strengths = [float(row["strength"]) for row in strength_rows]
    strength_sequences = [row["promoter"].upper() for row in strength_rows]

    binary_root = root / "reg_and_gen/Datasets/Escherichia coli"
    binary_rows = read_csv(binary_root / "Dataset.csv")
    split_rows = {
        name: read_csv(binary_root / f"{name}.csv")
        for name in ("train", "dev", "test")
    }
    binary_sequences = [row["seq"].upper() for row in binary_rows]
    binary_counts = Counter(binary_sequences)
    labels_by_sequence: dict[str, set[str]] = defaultdict(set)
    for row in binary_rows:
        labels_by_sequence[row["seq"].upper()].add(row["label"])

    split_sets = {
        name: {row["seq"].upper() for row in rows}
        for name, rows in split_rows.items()
    }
    split_pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    overlaps = {(a, b): split_sets[a] & split_sets[b] for a, b in split_pairs}
    nonstandard = [row for row in binary_rows if len(row["seq"]) != 81]

    lines = [
        "# 课程数据包中的大肠杆菌数据",
        "",
        "复核日期：2026年9月17日",
        "",
        "> 结论：老师说的“数据不全是 50 bp”是正确的。课程包中有两套用途不同的大肠杆菌数据。11,884 条连续强度主数据全部为 50 bp；另一套启动子二分类数据共有 3,350 条，其中 3,349 条为 81 bp，1 条为 80 bp。两套数据不能混为一个数据集。",
        "",
        "## 1. 两套数据的区别",
        "",
        "| 数据集 | 样本数 | 长度分布 | 标签 | 适合的任务 | 在本项目中的用途 |",
        "|---|---:|---|---|---|---|",
        f"| 连续强度数据 | {len(strength_rows):,} | 50 bp：{len(strength_rows):,} 条 | 实数强度 {min(strengths):.2f}–{max(strengths):.2f} | 根据目标强度生成启动子 | **主数据集** |",
        f"| 启动子二分类数据 | {len(binary_rows):,} | {length_text(binary_rows)} | 0/1，表示是否为启动子 | 判断一条序列是否像启动子 | 辅助识别评价 |",
        "",
        "因此，我们在 M1 报告中说“全部为 50 bp”时，必须明确限定为**连续强度主数据**，不能泛指课程包里的全部大肠杆菌数据。",
        "",
        "## 2. 连续强度主数据",
        "",
        f"- 文件：`strenth/data/supplementary/E_coli.txt`，并与 `promoter.npy`、`gene_expression.npy` 一致；",
        f"- 样本数：{len(strength_rows):,}；",
        f"- 序列长度：全部 50 bp；",
        f"- 唯一序列：{len(set(strength_sequences)):,}，没有完全重复；",
        f"- 强度范围：{min(strengths):.2f} 至 {max(strengths):.2f}，中位数 {median(strengths):.3f}；",
        "- 标签含义：连续表达强度，可以作为条件生成模型的目标条件。",
        "",
        "前 5 条原始记录：",
        "",
        "| 序号 | 50 bp 启动子序列 | 强度 |",
        "|---:|---|---:|",
    ]
    for index, row in enumerate(strength_rows[:5], start=1):
        lines.append(f"| {index} | `{row['promoter'].upper()}` | {float(row['strength']):.2f} |")

    labels = Counter(row["label"] for row in binary_rows)
    lines.extend(
        [
            "",
            "## 3. 大肠杆菌启动子二分类数据",
            "",
            f"- 文件：`reg_and_gen/Datasets/Escherichia coli/Dataset.csv`；",
            f"- 样本数：{len(binary_rows):,}；",
            f"- 长度分布：{length_text(binary_rows)}；",
            f"- 标签分布：正类 `1` 共 {labels['1']:,} 条，负类 `0` 共 {labels['0']:,} 条；",
            f"- 唯一序列：{len(binary_counts):,}；完全重复多出的记录 {sum(count - 1 for count in binary_counts.values()):,} 条，涉及 {sum(count > 1 for count in binary_counts.values()):,} 个序列组；",
            f"- 重复序列的标签冲突：{sum(len(values) > 1 for values in labels_by_sequence.values())} 组。",
            "",
            "其中唯一的 80 bp 记录为：",
            "",
            "| seq_id | 类型 | 长度 | 标签 | 序列 |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in nonstandard:
        lines.append(
            f"| {row['seq_id']} | {row['seq_type']} | {len(row['seq'])} | {row['label']} | `{row['seq'].upper()}` |"
        )

    lines.extend(
        [
            "",
            "课程包给出的拆分情况：",
            "",
            "| 子集 | 样本数 | 长度分布 | 正类 | 负类 | 唯一序列 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for name in ("train", "dev", "test"):
        rows = split_rows[name]
        split_labels = Counter(row["label"] for row in rows)
        lines.append(
            f"| {name} | {len(rows):,} | {length_text(rows)} | {split_labels['1']:,} | {split_labels['0']:,} | {len(split_sets[name]):,} |"
        )

    lines.extend(
        [
            "",
            "原始 `train/dev/test` 之间各有 1 条完全相同的序列交叉出现：",
            "",
        ]
    )
    for (a, b), common in overlaps.items():
        for sequence in sorted(common):
            lines.append(f"- `{a}` 与 `{b}`：`{sequence}`")

    lines.extend(
        [
            "",
            "这套数据只能回答“是否为启动子”，不能回答“启动子强度是多少”。若以后用它训练辅助识别器，应先按序列去重并重新划分，避免相同序列跨训练集、验证集和测试集。",
            "",
            "## 4. 对第 25 组项目的直接影响",
            "",
            "1. 条件生成模型继续使用 11,884 条 50 bp 连续强度数据，因为我们的目标是按指定表达强度生成序列。",
            "2. 3,350 条 80/81 bp 二分类数据不直接并入生成模型训练；长度不同，标签含义也不同。",
            "3. 二分类数据可在后期训练一个辅助启动子识别器，但使用前需要处理重复和跨集合重叠。",
            "4. 汇报时建议表述为：**课程包中的序列并非全部为 50 bp；本项目选用的连续强度主数据全部为 50 bp，另一套大肠杆菌二分类数据主要为 81 bp。**",
            "5. `kucao_test.csv` 在课程代码中被明确用于 *Bacillus subtilis*（枯草芽孢杆菌）强度预测，因此不计入大肠杆菌数据。",
            "",
        ]
    )

    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote E. coli overview to {output}")


if __name__ == "__main__":
    main()
