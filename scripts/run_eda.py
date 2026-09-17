#!/usr/bin/env python3
"""Create the formal M2 exploratory analysis for the E. coli main dataset."""

from __future__ import annotations

import argparse
import html
from collections import Counter
from itertools import product
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices, load_promoter_arrays


COLORS = {"train": "#2563eb", "validation": "#d97706", "test": "#059669"}
BASE_COLORS = {"A": "#1d4ed8", "C": "#059669", "G": "#d97706", "T": "#dc2626"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="reports/eda")
    return parser.parse_args()


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def svg_document(width: int, height: int, content: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}'
        '.label{font-size:12px}.small{font-size:10px}.title{font-size:16px;font-weight:600}'
        '.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#dbe3ee;stroke-width:1}</style>'
        f"{content}</svg>"
    )


def write_strength_distribution(path: Path, values: dict[str, np.ndarray], boundaries: np.ndarray) -> None:
    width, height = 820, 380
    left, right, top, bottom = 62, 28, 48, 55
    x_min = min(float(data.min()) for data in values.values())
    x_max = max(float(data.max()) for data in values.values())
    bins = np.linspace(x_min, x_max, 31)
    max_count = max(int(np.histogram(data, bins=bins)[0].max()) for data in values.values())
    plot_width, plot_height = width - left - right, height - top - bottom

    pieces = [f'<text x="{left}" y="24" class="title">log10(强度)分布：固定划分三个集合</text>']
    for tick in range(5):
        y = top + plot_height * tick / 4
        value = max_count * (1 - tick / 4)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        pieces.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.0f}</text>')
    for tick in range(6):
        x = left + plot_width * tick / 5
        value = x_min + (x_max - x_min) * tick / 5
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="middle" class="small">{value:.1f}</text>')
    pieces.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{left+plot_width/2:.1f}" y="{height-12}" text-anchor="middle" class="label">log10(表达强度)</text>',
        f'<text x="17" y="{top+plot_height/2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 17 {top+plot_height/2:.1f})">样本数</text>',
    ])
    for boundary in boundaries:
        x = left + (boundary - x_min) / (x_max - x_min) * plot_width
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#475569" stroke-width="1.5" stroke-dasharray="5 4"/>')
    legend_x = width - 250
    for row, (name, data) in enumerate(values.items()):
        counts, _ = np.histogram(data, bins=bins)
        points = []
        for index, count in enumerate(counts):
            x = left + ((bins[index] + bins[index + 1]) / 2 - x_min) / (x_max - x_min) * plot_width
            y = top + plot_height * (1 - count / max_count)
            points.append(f"{x:.1f},{y:.1f}")
        pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[name]}" stroke-width="2"/>')
        pieces.append(f'<line x1="{legend_x}" y1="{31 + row * 17}" x2="{legend_x+15}" y2="{31 + row * 17}" stroke="{COLORS[name]}" stroke-width="2"/>')
        pieces.append(f'<text x="{legend_x+20}" y="{35 + row * 17}" class="small">{html.escape(name)}</text>')
    path.write_text(svg_document(width, height, "".join(pieces)), encoding="utf-8")


def write_position_heatmap(path: Path, sequences: np.ndarray) -> None:
    width, height = 900, 270
    left, top, cell_w, cell_h = 65, 52, 15, 32
    pieces = ['<text x="20" y="24" class="title">各位置碱基频率（全体 11,884 条主数据）</text>']
    frequencies = {base: np.array([np.mean([seq[pos] == base for seq in sequences]) for pos in range(50)]) for base in "ACGT"}
    for row, base in enumerate("ACGT"):
        y = top + row * cell_h
        pieces.append(f'<text x="{left-12}" y="{y+21}" text-anchor="end" class="label">{base}</text>')
        for position, value in enumerate(frequencies[base]):
            intensity = int(245 - 165 * float(value))
            color = f"rgb({intensity},{intensity + min(10, 255-intensity)},{255})"
            x = left + position * cell_w
            pieces.append(f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-1}" fill="{color}"/>')
            if value >= 0.40:
                pieces.append(f'<text x="{x+(cell_w-1)/2:.1f}" y="{y+20}" text-anchor="middle" class="small">{value:.2f}</text>')
    for position in range(0, 50, 5):
        x = left + position * cell_w + 7
        pieces.append(f'<text x="{x}" y="{top+4*cell_h+18}" text-anchor="middle" class="small">{position+1}</text>')
    pieces.append(f'<text x="{left+375}" y="{top+4*cell_h+40}" text-anchor="middle" class="label">序列位置（bp）</text>')
    path.write_text(svg_document(width, height, "".join(pieces)), encoding="utf-8")


def write_gc_by_group(path: Path, group_gcs: dict[str, np.ndarray]) -> None:
    width, height = 620, 350
    left, right, top, bottom = 65, 25, 48, 55
    plot_width, plot_height = width-left-right, height-top-bottom
    names = list(group_gcs)
    means = [float(group_gcs[name].mean()) for name in names]
    max_value = max(means) * 1.18
    pieces = ['<text x="20" y="24" class="title">不同强度组的平均 GC 含量</text>']
    for tick in range(5):
        y = top + plot_height * tick / 4
        value = max_value * (1 - tick / 4)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        pieces.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.0%}</text>')
    bar_width = 90
    for index, (name, mean) in enumerate(zip(names, means)):
        center = left + plot_width * (index + 0.5) / len(names)
        bar_height = mean / max_value * plot_height
        y = top + plot_height - bar_height
        pieces.append(f'<rect x="{center-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{BASE_COLORS["G"]}"/>')
        pieces.append(f'<text x="{center}" y="{y-7:.1f}" text-anchor="middle" class="label">{mean:.2%}</text>')
        pieces.append(f'<text x="{center}" y="{height-bottom+20}" text-anchor="middle" class="label">{html.escape(name)}</text>')
        pieces.append(f'<text x="{center}" y="{height-bottom+37}" text-anchor="middle" class="small">n={len(group_gcs[name]):,}</text>')
    pieces.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
    ])
    path.write_text(svg_document(width, height, "".join(pieces)), encoding="utf-8")


def top_kmers(sequences: np.ndarray, k: int = 4, limit: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(sequence[position : position + k] for position in range(len(sequence) - k + 1))
    return counts.most_common(limit)


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(
        args.data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    split_setting = args.split_file or data_config.get("split_file")
    if not split_setting:
        raise ValueError("Formal EDA requires a fixed split file")
    split_path = Path(split_setting)
    if not split_path.is_absolute():
        split_path = REPOSITORY_ROOT / split_path
    train_idx, validation_idx, test_idx = load_fixed_split_indices(split_path, len(sequences))
    split_indices = {"train": train_idx, "validation": validation_idx, "test": test_idx}
    log_strength = np.log10(strengths)
    train_boundaries = np.quantile(log_strength[train_idx], [1 / 3, 2 / 3])
    gc = np.array([gc_fraction(sequence) for sequence in sequences])
    group_labels = np.where(log_strength < train_boundaries[0], "弱强度", np.where(log_strength < train_boundaries[1], "中强度", "强强度"))

    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_strength_distribution(output_dir / "01_log_strength_distribution.svg", {name: log_strength[index] for name, index in split_indices.items()}, train_boundaries)
    write_position_heatmap(output_dir / "02_position_base_frequency.svg", sequences)
    group_gcs = {name: gc[group_labels == name] for name in ("弱强度", "中强度", "强强度")}
    write_gc_by_group(output_dir / "03_gc_by_strength_group.svg", group_gcs)

    lines = [
        "# M2 正式 EDA：大肠杆菌连续强度主数据",
        "",
        "分析对象为 11,884 条、长度均为 50 bp 的大肠杆菌启动子序列。所有统计基于版本化的 90% 相似性聚类划分；强度分档边界仅根据训练集计算。",
        "",
        "## 1 数据字典",
        "",
        "| 字段 | 含义 | 处理方式 |",
        "|---|---|---|",
        "| `sample_index` | 原始 NPY 数组中的行号 | 用于固定划分和复现 |",
        "| `sequence` | 50 bp DNA 启动子序列 | 大写，仅包含 A/T/C/G |",
        "| `strength` | 原始连续表达强度 | 正数，范围跨度大 |",
        "| `log10_strength` | `log10(strength)` | 预测和生成的连续条件 |",
        "| `split` | train / validation / test | 以相似性簇为单位固定 |",
        "",
        "## 2 质量与划分检查",
        "",
        "- 样本数：11,884；序列长度：全部 50 bp；完全重复：0；非法 DNA 字符：0。",
        "- 固定划分：训练集 8,318 条，验证集 1,783 条，测试集 1,783 条。",
        f"- 训练集 `log10(强度)` 分档边界：{train_boundaries[0]:.4f}、{train_boundaries[1]:.4f}。",
        "- 近重复保护：90% 相似性连通簇不会跨集合；详细簇统计见 `../m2_split_and_strength_baseline_2026-09-17.md`。",
        "",
        "| 集合 | 样本数 | log10 强度均值 ± 标准差 | GC 均值 ± 标准差 |",
        "|---|---:|---:|---:|",
    ]
    for name, index in split_indices.items():
        lines.append(f"| {name} | {len(index):,} | {log_strength[index].mean():.4f} ± {log_strength[index].std():.4f} | {gc[index].mean():.2%} ± {gc[index].std():.2%} |")
    lines.extend([
        "",
        "![三个集合的对数强度分布](01_log_strength_distribution.svg)",
        "",
        "三个集合的对数强度分布整体接近；后续模型调参只能查看训练和验证集，测试集只用于最终评价。",
        "",
        "## 3 强度与序列组成",
        "",
        f"原始强度范围为 {strengths.min():.2f} 至 {strengths.max():.2f}；`log10(强度)` 中位数为 {np.median(log_strength):.4f}。原始标签具有明显长尾，因此使用对数变换。",
        "",
        "| 强度组 | 样本数 | 平均 GC |",
        "|---|---:|---:|",
    ])
    for name in ("弱强度", "中强度", "强强度"):
        lines.append(f"| {name} | {len(group_gcs[name]):,} | {group_gcs[name].mean():.2%} |")
    lines.extend([
        "",
        "![不同强度组的 GC 含量](03_gc_by_strength_group.svg)",
        "",
        "GC 含量适合作为生成序列的分布检查指标，但不能单独预测强度；后续预测器需要利用位置与局部序列模式。",
        "",
        "## 4 位置规律与局部片段",
        "",
        "![各位置碱基频率](02_position_base_frequency.svg)",
        "",
        "位置碱基频率并不均匀，说明启动子不是简单的全局碱基比例集合。后续位置感知预测器将同时使用碱基身份和位置。",
        "",
        "全体序列中出现次数最多的 4-mer：",
        "",
        "| 4-mer | 出现次数 |",
        "|---|---:|",
    ])
    for token, count in top_kmers(sequences):
        lines.append(f"| `{token}` | {count:,} |")
    lines.extend([
        "",
        "## 5 对 M3 的直接约束",
        "",
        "1. 生成器输出必须为 50 bp、仅含 A/T/C/G 的序列。",
        "2. 生成条件使用训练集统计量标准化后的 `log10(强度)`，不能使用测试集信息。",
        "3. 评价时同时报告目标强度误差、GC、位置碱基频率、局部 k-mer、新颖性和多样性。",
        "4. 岭回归基线相关性较弱，因此下一步必须训练位置感知强度预测器，再将其冻结为独立评价器。",
        "",
    ])
    (output_dir / "ecoli_formal_eda.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote formal EDA report and figures to {output_dir}")


if __name__ == "__main__":
    main()
