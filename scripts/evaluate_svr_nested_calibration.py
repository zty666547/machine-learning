#!/usr/bin/env python3
"""Evaluate training-only, cluster-nested affine calibration of position SVR."""
import argparse
import csv
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from promoter_ml.data import PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from evaluate_strength_calibration import cluster_ids, group_folds
from evaluate_cnn_position_bins import fold_layout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictor(parameters):
    return make_pipeline(StandardScaler(), SVR(**parameters))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    config_path = ROOT / 'configs/m4_svr_nested_calibration_2026-11-15.toml'
    config = tomllib.loads(config_path.read_text())['experiment']
    svr_path = ROOT / 'configs/m4_rbf_svr_2026-11-09.toml'
    params = tomllib.loads(svr_path.read_text())['svr']
    data = Path(args.data_dir)
    seq, strength = load_promoter_arrays(data, 'promoter.npy', 'gene_expression.npy', 50)
    split = ROOT / 'data/splits/ecoli_similarity_split.csv'
    train, _, _ = load_fixed_split_indices(split, len(seq))
    clusters = cluster_ids(split, len(seq))[train]
    y = np.log10(strength[train])
    x = PositionKmerFeaturizer(sequence_length=50).transform(seq[train])
    layouts, _ = fold_layout(clusters, config['outer_fold_seed'])
    pred_rows, folds = [], []
    previous = json.loads((ROOT / 'reports/m4_target_oof_audit_2026-11-15.json').read_text())
    hashes = {'sequence': digest(data / 'promoter.npy'), 'strength': digest(data / 'gene_expression.npy'), 'split': digest(split)}
    if previous['data_sha256'] != hashes or len(layouts) != config['outer_folds']:
        raise ValueError('Data or outer fold mismatch')
    for i, (fit, early, hold) in enumerate(layouts):
        ref = np.sort(np.r_[fit, early])
        assert not set(clusters[ref]) & set(clusters[hold])
        inner = group_folds(clusters[ref], config['inner_folds'], config['inner_fold_seed_base'] + i)
        inner_pred = np.full(len(ref), np.nan)
        for j in range(config['inner_folds']):
            a, b = ref[inner != j], ref[inner == j]
            assert not set(clusters[a]) & set(clusters[b])
            model = predictor(params).fit(x[a], y[a])
            inner_pred[inner == j] = model.predict(x[b])
            print(f'outer {i+1}: inner {j+1} complete', flush=True)
        if not np.isfinite(inner_pred).all() or np.var(inner_pred) < 1e-12:
            raise ValueError('Invalid calibration predictions')
        slope = float(np.mean((inner_pred-inner_pred.mean()) * (y[ref]-y[ref].mean())) / np.var(inner_pred))
        intercept = float(y[ref].mean() - slope * inner_pred.mean())
        raw = predictor(params).fit(x[ref], y[ref]).predict(x[hold])
        calibrated = intercept + slope * raw
        boundaries = np.quantile(y[ref], [1/3, 2/3])
        train_groups = np.digitize(y[ref], boundaries)
        hold_groups = np.digitize(y[hold], boundaries)
        record = {'fold': i+1, 'calibration_intercept': intercept, 'calibration_slope': slope,
                  'inner_train_sizes': [int(np.sum(inner != j)) for j in range(config['inner_folds'])],
                  'holdout_count': len(hold), 'groups': [], 'mae_raw': float(np.mean(abs(raw-y[hold]))),
                  'mae_calibrated': float(np.mean(abs(calibrated-y[hold])))}
        for k, name in enumerate(('weak','medium','strong')):
            target = float(y[ref][train_groups == k].mean())
            item = {'group': name, 'target': target, 'true_group_count': int(sum(hold_groups == k))}
            for label, p in [('raw', raw), ('calibrated', calibrated)]:
                selected = np.argsort(abs(p-target), kind='stable')[:config['top_k']]
                item[label] = {'selection_true_distance': float(np.mean(abs(y[hold][selected]-target))),
                               'selection_predicted_distance': float(np.mean(abs(p[selected]-target))),
                               'true_group_mae': float(np.mean(abs(p[hold_groups == k]-y[hold][hold_groups == k])))}
            old = previous['folds'][i]['groups'][k]
            if abs(old['models']['svr']['mean_true_distance_to_target'] - item['raw']['selection_true_distance']) > 1e-8:
                raise ValueError('Raw SVR does not reproduce prior selection')
            record['groups'].append(item)
        for idx, a, b in zip(hold, raw, calibrated):
            pred_rows.append([i+1, int(train[idx]), float(y[idx]), float(a), float(b)])
        folds.append(record)
        print(f'outer {i+1}: complete, slope={slope:.5f}', flush=True)
    path = Path(args.predictions); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['fold','sample_index','observed_log10','raw','calibrated']); writer.writerows(pred_rows)
    summary = {'mae_raw': float(sum(f['mae_raw'] * f['holdout_count'] for f in folds) / len(train)),
               'mae_calibrated': float(sum(f['mae_calibrated'] * f['holdout_count'] for f in folds) / len(train)), 'groups': []}
    # Sample-weight all-fold MAE; average each target's three fold-level quantities equally.
    for k, name in enumerate(('weak','medium','strong')):
        summary['groups'].append({'group': name, **{label: {metric: float(np.mean([f['groups'][k][label][metric] for f in folds])) for metric in ('selection_true_distance','selection_predicted_distance','true_group_mae')} for label in ('raw','calibrated')}})
    pred_array = np.asarray(pred_rows, dtype=float)
    for label, column in [('raw', 3), ('calibrated', 4)]:
        summary[label + '_pearson'] = float(np.corrcoef(pred_array[:, 2], pred_array[:, column])[0, 1])
        summary[label + '_spearman'] = float(np.corrcoef(rankdata(pred_array[:, 2]), rankdata(pred_array[:, column]))[0, 1])
    result = {'protocol': 'Nested similarity-cluster OOF; inner 2-fold affine calibration within each outer training set; three outer held-out folds; prior development reused these data; no fresh independent test.',
              'data_sha256': hashes, 'config_sha256': digest(config_path), 'svr_config_sha256': digest(svr_path),
              'predictions_sha256': digest(path), 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
              'folds': folds, 'summary': summary}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
