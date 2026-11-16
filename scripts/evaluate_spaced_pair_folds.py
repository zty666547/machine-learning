#!/usr/bin/env python3
"""Paired SVR feature comparisons on additional similarity-cluster folds."""
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
from promoter_ml.spaced_pairs import SpacedPairFeaturizer
from evaluate_strength_calibration import cluster_ids, group_folds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return {'mae': float(np.mean(abs(p-y))),
            'pearson': float(np.corrcoef(y,p)[0,1]),
            'spearman': float(np.corrcoef(rankdata(y),rankdata(p))[0,1]),
            'prediction_std': float(np.std(p))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--fold-seed', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    cfgpath = ROOT/'configs/m4_spaced_pair_fold_check_2026-11-16.toml'
    cfg = tomllib.loads(cfgpath.read_text())['experiment']
    modelpath = ROOT/'configs/m4_rbf_svr_2026-11-09.toml'
    pairpath = ROOT/'configs/m4_spaced_pair_svr_2026-11-15.toml'
    if args.fold_seed not in cfg['new_fold_seeds'] or cfg['outer_folds'] != 3:
        raise ValueError('Unregistered fold seed or fold count')
    if cfg['spacings'] != tomllib.loads(pairpath.read_text())['experiment']['spacings']:
        raise ValueError('Feature spacing differs from prior experiment')
    params = tomllib.loads(modelpath.read_text())['svr']
    previous = json.loads((ROOT/f'reports/m4_svr_fold_{args.fold_seed}.json').read_text())
    prev_pair = json.loads((ROOT/'reports/m4_spaced_pair_svr_2026-11-15.json').read_text())
    data = Path(args.data_dir); split = ROOT/'data/splits/ecoli_similarity_split.csv'
    hashes = {'sequence': sha(data/'promoter.npy'), 'strength': sha(data/'gene_expression.npy'), 'split': sha(split)}
    if hashes != previous['data_sha256'] or hashes != prev_pair['data_sha256']:
        raise ValueError('Data differ from earlier experiments')
    if sha(modelpath) != previous['prior_protocol_sha256'] or sha(pairpath) != prev_pair['config_sha256']:
        raise ValueError('Prior model or feature protocol differs')
    if np.__version__ != previous['numpy_version'] or sklearn.__version__ != previous['sklearn_version']:
        raise ValueError('Numeric versions differ from baseline')
    seq, strength = load_promoter_arrays(data, 'promoter.npy','gene_expression.npy',50)
    train, _, _ = load_fixed_split_indices(split, len(seq))
    y = np.log10(strength[train]); groups = cluster_ids(split, len(seq))[train]
    folds = group_folds(groups, 3, args.fold_seed)
    fold_hash = hashlib.sha256(folds.astype('<i8').tobytes()).hexdigest()
    if fold_hash != previous['fold_sha256']:
        raise ValueError('Fold assignment differs from existing SVR experiment')
    base = PositionKmerFeaturizer(sequence_length=50).transform(seq[train])
    pair = SpacedPairFeaturizer(sequence_length=50,spacings=tuple(cfg['spacings'])).transform(seq[train])
    expanded = np.concatenate([base,pair], axis=1)
    preds = {key: np.empty(len(train)) for key in ('svr','spaced_pair_svr')}
    fold_results = []
    for fold in range(3):
        fit = np.flatnonzero(folds != fold); hold = np.flatnonzero(folds == fold)
        if set(groups[fit]) & set(groups[hold]):
            raise ValueError('Cluster crosses fold boundary')
        record = {'fold':fold+1,'fit_count':len(fit),'holdout_count':len(hold),'gamma':{},'groups':[]}
        for name, x in [('svr',base),('spaced_pair_svr',expanded)]:
            model=make_pipeline(StandardScaler(),SVR(**params));model.fit(x[fit],y[fit]);preds[name][hold]=model.predict(x[hold])
            record['gamma'][name]=float(model[-1]._gamma)
            print(args.fold_seed,name,'fold',fold+1,'done',flush=True)
        bounds=np.quantile(y[fit],[1/3,2/3]); train_group=np.digitize(y[fit],bounds);hold_group=np.digitize(y[hold],bounds)
        for k, group in enumerate(('weak','medium','strong')):
            target=float(np.mean(y[fit][train_group==k])); g={'group':group,'target_log10':target,'holdout_group_count':int(np.sum(hold_group==k))}
            for name in preds:
                p=preds[name][hold];selected=np.argsort(abs(p-target),kind='stable')[:cfg['top_k']]
                g[name]={'true_group_mae':float(np.mean(abs(p[hold_group==k]-y[hold][hold_group==k]))),
                         'selected_true_distance':float(np.mean(abs(y[hold][selected]-target))),
                         'selected_predicted_distance':float(np.mean(abs(p[selected]-target)))}
            record['groups'].append(g)
        fold_results.append(record)
    summary={name:metrics(y,p) for name,p in preds.items()};summary['groups']=[]
    for k,name in enumerate(('weak','medium','strong')):
        summary['groups'].append({'group':name,**{model:{metric:float(np.mean([r['groups'][k][model][metric] for r in fold_results])) for metric in ('true_group_mae','selected_true_distance','selected_predicted_distance')} for model in preds}})
    if abs(summary['svr']['mae']-previous['models']['svr']['training_oof']['mae'])>1e-8:
        raise ValueError('SVR prior OOF MAE not reproduced')
    out=Path(args.predictions);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['fold','sample_index','observed_log10','svr','spaced_pair_svr']);writer.writerows(zip((folds+1).tolist(),train.tolist(),y.tolist(),preds['svr'].tolist(),preds['spaced_pair_svr'].tolist()))
    result={'protocol':'Paired fixed SVR features on new cluster folds chosen after prior result. Shared training data and prior development; not independent test or biological validation.',
            'fold_seed':args.fold_seed,'fold_sha256':fold_hash,'data_sha256':hashes,'config_sha256':sha(cfgpath),'svr_config_sha256':sha(modelpath),'feature_config_sha256':sha(pairpath),'predictions_sha256':sha(out),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
            'feature_counts':{'baseline':base.shape[1],'added':pair.shape[1],'expanded':expanded.shape[1]},'folds':fold_results,'summary':summary}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
