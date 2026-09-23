#!/usr/bin/env python3
"""Render presentation-ready SVG figures from frozen M3 result files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563eb"
ORANGE = "#ea580c"
INK = "#172033"
MUTED = "#64748b"
GRID = "#dbe3ee"


def load_json(path: str) -> dict:
    return json.loads((REPOSITORY_ROOT / path).read_text(encoding="utf-8"))


def svg_document(width: int, height: int, content: str) -> str:
    style = """<style>
    text { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif; fill: #172033; }
    .title { font-size: 22px; font-weight: 700; }
    .subtitle { font-size: 13px; fill: #64748b; }
    .label { font-size: 13px; }
    .small { font-size: 11px; fill: #64748b; }
    .value { font-size: 15px; font-weight: 700; }
    .note { font-size: 12px; fill: #475569; }
    </style>"""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{style}{content}</svg>'


def paired_bar(x: int, y: int, width: int, height: int, title: str, subtitle: str, ar_value: float, vae_value: float, formatter, lower_better: bool) -> str:
    maximum = max(ar_value, vae_value) * 1.15 or 1.0
    ar_height = height * ar_value / maximum
    vae_height = height * vae_value / maximum
    winner = "条件自回归更好" if (ar_value < vae_value) == lower_better else "连续条件 VAE 更好"
    return f'''<text x="{x}" y="{y}" class="label" font-weight="700">{html.escape(title)}</text>
    <text x="{x}" y="{y + 19}" class="small">{html.escape(subtitle)}</text>
    <line x1="{x}" y1="{y + height + 42}" x2="{x + width}" y2="{y + height + 42}" stroke="{GRID}"/>
    <rect x="{x + 28}" y="{y + height + 42 - ar_height}" width="56" height="{ar_height}" fill="{BLUE}" rx="4"/>
    <rect x="{x + 116}" y="{y + height + 42 - vae_height}" width="56" height="{vae_height}" fill="{ORANGE}" rx="4"/>
    <text x="{x + 56}" y="{y + height + 59}" text-anchor="middle" class="small">自回归</text>
    <text x="{x + 144}" y="{y + height + 59}" text-anchor="middle" class="small">VAE</text>
    <text x="{x + 56}" y="{y + height + 32 - ar_height}" text-anchor="middle" class="value">{formatter(ar_value)}</text>
    <text x="{x + 144}" y="{y + height + 32 - vae_height}" text-anchor="middle" class="value">{formatter(vae_value)}</text>
    <text x="{x}" y="{y + height + 88}" class="note">{winner}</text>'''


def generator_figure(comparison: dict) -> str:
    ar = comparison["models"]["conditional_autoregressive"]
    vae = comparison["models"]["continuous_condition_vae"]
    mean = lambda model, key: float(np.mean([model[group][key] for group in ("weak", "medium", "strong")]))
    js = (mean(ar, "kmer_3_js_divergence"), mean(vae, "kmer_3_js_divergence"))
    motif_error = (mean(ar, "top_reference_motif_mean_abs_frequency_error"), mean(vae, "top_reference_motif_mean_abs_frequency_error"))
    overlap = (mean(ar, "top_motif_overlap_fraction"), mean(vae, "top_motif_overlap_fraction"))
    content = '''<rect width="1000" height="560" fill="#ffffff"/>
    <text x="54" y="52" class="title">M3 两类条件生成器：局部序列质量对比</text>
    <text x="54" y="77" class="subtitle">相同训练集、相同弱/中/强目标组、每组各生成 500 条候选；蓝色为条件自回归，橙色为连续条件 VAE</text>
    <rect x="54" y="104" width="892" height="336" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
    '''
    content += paired_bar(92, 145, 200, 168, "3-mer JS 散度", "越低越接近真实局部序列", *js, lambda value: f"{value:.5f}", True)
    content += paired_bar(388, 145, 200, 168, "高频 5-mer 误差", "越低越好", *motif_error, lambda value: f"{value:.5f}", True)
    content += paired_bar(684, 145, 200, 168, "高频 5-mer 重合率", "越高越好", *overlap, lambda value: f"{value:.2f}", False)
    content += '''<rect x="54" y="462" width="892" height="55" rx="8" fill="#eff6ff"/>
    <text x="74" y="486" class="label" font-weight="700">结论：</text><text x="125" y="486" class="label">条件自回归更擅长保留局部 k-mer 与候选 motif，因此是当前 M3 主生成基线。</text>
    <text x="74" y="507" class="note">两类模型的合法比例、新颖比例均为 1.0000；VAE 保留为支持连续强度输入的生成候选。</text>'''
    return svg_document(1000, 560, content)


def width_figure(width_result: dict) -> str:
    rows = sorted(width_result["summary"], key=lambda row: row["hidden_size"])
    content = '''<rect width="1000" height="560" fill="#ffffff"/>
    <text x="54" y="52" class="title">连续条件 VAE：网络宽度稳定性选择</text>
    <text x="54" y="77" class="subtitle">固定 β=0.20、潜变量维度=16；每种宽度采用 3 个随机种子、7 个连续目标强度</text>
    <rect x="54" y="108" width="892" height="315" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="96" y="151" class="small">隐藏层宽度</text><text x="297" y="151" class="small">平均 3-mer JS ↓</text><text x="502" y="151" class="small">平均目标—GC Pearson</text><text x="749" y="151" class="small">Pearson 标准差 ↓</text>'''
    y_values = [207, 305]
    maximum_js = max(row["mean_kmer_3_js"] for row in rows)
    maximum_std = max(row["std_target_gc_pearson"] for row in rows)
    for row, y in zip(rows, y_values):
        selected = row["hidden_size"] == 128
        fill = "#dbeafe" if selected else "#ffffff"
        stroke = BLUE if selected else GRID
        content += f'<rect x="78" y="{y - 42}" width="844" height="70" rx="8" fill="{fill}" stroke="{stroke}"/>'
        content += f'<text x="122" y="{y}" class="value">{row["hidden_size"]}</text>'
        js_width = 140 * row["mean_kmer_3_js"] / maximum_js
        std_width = 120 * row["std_target_gc_pearson"] / maximum_std
        color = BLUE if selected else "#94a3b8"
        content += f'<rect x="292" y="{y - 15}" width="{js_width}" height="18" rx="4" fill="{color}"/><text x="{292 + js_width + 8}" y="{y}" class="value">{row["mean_kmer_3_js"]:.5f}</text>'
        content += f'<text x="551" y="{y}" class="value">{row["mean_target_gc_pearson"]:.3f}</text>'
        content += f'<rect x="746" y="{y - 15}" width="{std_width}" height="18" rx="4" fill="{color}"/><text x="{746 + std_width + 8}" y="{y}" class="value">{row["std_target_gc_pearson"]:.3f}</text>'
        if selected:
            content += f'<text x="122" y="{y + 20}" class="small">当前选定配置</text>'
    content += '''<rect x="54" y="452" width="892" height="65" rx="8" fill="#eff6ff"/>
    <text x="74" y="478" class="label" font-weight="700">选择理由：</text><text x="145" y="478" class="label">128 宽度的局部 3-mer 偏差更低，连续条件响应更强，且不同随机种子的波动更小。</text>
    <text x="74" y="502" class="note">当前配置：hidden_size=128，latent_size=16，β=0.20。GC 响应用于观察分布变化，不等同于真实表达强度验证。</text>'''
    return svg_document(1000, 560, content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/figures/m3_midterm")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_json("reports/m3_conditional_generator_comparison_selected.json")
    widths = load_json("reports/m3_vae_width_stability.json")
    outputs = {
        "m3_generator_comparison.svg": generator_figure(comparison),
        "m3_vae_width_selection.svg": width_figure(widths),
    }
    for name, svg in outputs.items():
        (output_dir / name).write_text(svg, encoding="utf-8")
    print("\n".join(str(output_dir / name) for name in outputs))


if __name__ == "__main__":
    main()
