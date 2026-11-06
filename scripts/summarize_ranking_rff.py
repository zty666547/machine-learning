#!/usr/bin/env python3
"""Summarize the three pre-registered seeds without selecting a winner."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
seeds = (20260912, 20260913, 20260914)
rows = [json.loads((ROOT / f'reports/m4_ranking_rff_seed_{s}.json').read_text()) for s in seeds]
for seed, row in zip(seeds, rows):
    assert row['seed'] == seed
    for k in ('fold_seed', 'train_count', 'validation_count', 'data_sha256', 'protocol_sha256', 'numpy_version', 'training_boundaries'):
        assert row[k] == rows[0][k], k
metrics = ('mae', 'rmse', 'pearson', 'spearman', 'prediction_to_target_std_ratio')
summary = {'seeds': seeds, 'fold_seed': rows[0]['fold_seed'], 'data_sha256': rows[0]['data_sha256'], 'protocol_sha256': rows[0]['protocol_sha256'], 'note': 'Same grouped folds across seeds; seed variation is not independent test-set uncertainty. Linear ridge is deterministic across seeds. Ridge uses more fit rows than early-stopped neural models.', 'models': {}}
for group, name in [('references',n) for n in ('cnn','rnn_mse','rnn_huber','gru_mse')] + [('models',n) for n in ('ranking_rnn','linear_ridge','fourier_ridge')]:
    summary['models'][name] = {}
    for split in ('training_oof','development_validation'):
        items = [r[group][name][split] for r in rows]
        out = {}
        for metric in metrics:
            v = np.array([i[metric] for i in items]); out[metric] = {'mean':float(v.mean()), 'std':float(v.std()), 'values':v.tolist()}
        out['by_true_strength_group'] = {}
        for g in ('weak','medium','strong'):
            out['by_true_strength_group'][g] = {}
            for metric in ('mae','mean_signed_error'):
                v = np.array([i['by_true_strength_group'][g][metric] for i in items]); out['by_true_strength_group'][g][metric] = {'mean':float(v.mean()), 'std':float(v.std()), 'values':v.tolist()}
        summary['models'][name][split] = out
outpath = ROOT / 'reports/m4_ranking_rff_summary_2026-11-06.json'
outpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
for name, obj in summary['models'].items():
    s=obj['training_oof']; print(name, *(f'{k}={s[k]["mean"]:.5f}±{s[k]["std"]:.5f}' for k in ('mae','pearson','spearman')), *(f'{g}={s["by_true_strength_group"][g]["mae"]["mean"]:.5f}' for g in ('weak','strong')))
