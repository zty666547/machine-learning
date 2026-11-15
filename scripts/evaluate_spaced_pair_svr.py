#!/usr/bin/env python3
"""Compare fixed spaced-pair and existing-feature RBF SVR on cluster OOF samples."""
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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from promoter_ml.data import PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.spaced_pairs import SpacedPairFeaturizer
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return {'mae': float(np.mean(abs(y-p))),
            'pearson': float(np.corrcoef(y,p)[0,1]),
            'spearman': float(np.corrcoef(rankdata(y),rankdata(p))[0,1]),
            'prediction_std': float(np.std(p))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--svr-oof', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    config_path = ROOT / 'configs/m4_spaced_pair_svr_2026-11-15.toml'
    config = tomllib.loads(config_path.read_text())['experiment']
    data = Path(args.data_dir)
    split = ROOT / 'data/splits/ecoli_similarity_split.csv'
    prior = json.loads((ROOT / 'reports/m4_svr_nested_calibration_2026-11-15.json').read_text())
    hashes = {'sequence': sha(data/'promoter.npy'), 'strength': sha(data/'gene_expression.npy'), 'split': sha(split)}
    if hashes != prior['data_sha256'] or sha(args.svr_oof) != prior['predictions_sha256']:
        raise ValueError('SVR OOF or data provenance mismatch')
    if np.__version__ != prior['numpy_version'] or sklearn.__version__ != prior['sklearn_version']:
        raise ValueError('Different numeric environment than frozen SVR baseline')
    seq, strength = load_promoter_arrays(data, 'promoter.npy', 'gene_expression.npy', 50)
    train, _, _ = load_fixed_split_indices(split, len(seq))
    y = np.log10(strength[train])
    clusters = cluster_ids(split, len(seq))[train]
    base_x = PositionKmerFeaturizer(sequence_length=50).transform(seq[train])
    pair_x = SpacedPairFeaturizer(sequence_length=50, spacings=tuple(config['spacings'])).transform(seq[train])
    x = np.concatenate([base_x, pair_x], axis=1)
    svr_cfg_path = ROOT / 'configs/m4_rbf_svr_2026-11-09.toml'
    if sha(svr_cfg_path) != prior['svr_config_sha256']:
        raise ValueError('SVR model configuration differs from baseline')
    svr_cfg = tomllib.loads(svr_cfg_path.read_text())['svr']
    layouts, _ = fold_layout(clusters, config['outer_fold_seed'])
    if len(layouts) != config['outer_folds']:
        raise ValueError('Unexpected fold count')
    with open(args.svr_oof, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    baseline = {int(row['sample_index']): row for row in rows}
    if len(baseline) != len(train) or set(baseline) != set(train):
        raise ValueError('Baseline sample indices do not match')
    predictions = np.empty(len(train)); svr = np.empty(len(train)); records = []
    prior_target = json.loads((ROOT/'reports/m4_target_oof_audit_2026-11-15.json').read_text())
    for i, (fit, early, hold) in enumerate(layouts):
        ref = np.sort(np.r_[fit,early])
        assert not set(clusters[ref]) & set(clusters[hold])
        for ix in hold:
            row = baseline[int(train[ix])]
            if int(row['fold']) != i+1 or not np.isclose(float(row['observed_log10']), y[ix], rtol=0, atol=1e-12):
                raise ValueError('Baseline fold or observed label mismatch')
            svr[ix] = float(row['raw'])
        model = make_pipeline(StandardScaler(), SVR(**svr_cfg))
        model.fit(x[ref], y[ref]); predictions[hold] = model.predict(x[hold])
        gamma = float(model.named_steps['svr']._gamma)
        bounds = np.quantile(y[ref],[1/3,2/3])
        ref_groups = np.digitize(y[ref], bounds); hold_groups = np.digitize(y[hold],bounds)
        record = {'fold': i+1, 'train_count': len(ref), 'holdout_count': len(hold), 'gamma': gamma, 'groups': []}
        for k, name in enumerate(('weak','medium','strong')):
            target = float(np.mean(y[ref][ref_groups==k]))
            result = {'group': name, 'target_log10': target, 'true_group_count': int(np.sum(hold_groups==k))}
            for label, values in [('svr',svr[hold]), ('spaced_pair_svr',predictions[hold])]:
                selected = np.argsort(abs(values-target),kind='stable')[:config['top_k']]
                result[label] = {'true_group_mae': float(np.mean(abs(values[hold_groups==k]-y[hold][hold_groups==k]))),
                    'selected_true_distance': float(np.mean(abs(y[hold][selected]-target))),
                    'selected_predicted_distance': float(np.mean(abs(values[selected]-target)))}
            if abs(result['svr']['selected_true_distance']-prior_target['folds'][i]['groups'][k]['models']['svr']['mean_true_distance_to_target'])>1e-8:
                raise ValueError('SVR target selection does not reproduce')
            record['groups'].append(result)
        records.append(record)
        print(f'fold {i+1}: train {len(ref)}, holdout {len(hold)}, gamma {gamma:.8g}',flush=True)
    summary = {'svr': metrics(y,svr), 'spaced_pair_svr': metrics(y,predictions), 'label_std': float(np.std(y)), 'groups': []}
    for k,name in enumerate(('weak','medium','strong')):
        summary['groups'].append({'group':name, **{label:{metric:float(np.mean([f['groups'][k][label][metric] for f in records])) for metric in ('true_group_mae','selected_true_distance','selected_predicted_distance')} for label in ('svr','spaced_pair_svr')}})
    path = Path(args.predictions); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['sample_index','observed_log10','svr','spaced_pair_svr']); writer.writerows(zip(train.tolist(),y.tolist(),svr.tolist(),predictions.tolist()))
    output = {'protocol':'One fixed training-only cluster OOF split. SVR models use the same outer rows and hyperparameters but different features and gamma scale; model choices were developed using the same project data. Not an independent test or biological validation.',
              'data_sha256':hashes,'config_sha256':sha(config_path),'svr_input_sha256':sha(args.svr_oof),'predictions_sha256':sha(path),
              'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'feature_count':x.shape[1], 'baseline_feature_count':base_x.shape[1], 'spaced_pair_feature_count':pair_x.shape[1], 'folds':records,'summary':summary}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
