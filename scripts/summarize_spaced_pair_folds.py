#!/usr/bin/env python3
"""Summarize original and pre-registered additional cluster-fold comparisons."""
import hashlib
import json
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
config=ROOT/'configs/m4_spaced_pair_fold_check_2026-11-16.toml'
cfg=tomllib.loads(config.read_text())['experiment']
paths=[ROOT/'reports/m4_spaced_pair_svr_2026-11-15.json']+[ROOT/f'reports/m4_spaced_pair_fold_{seed}.json' for seed in cfg['new_fold_seeds']]
results=[json.loads(p.read_text()) for p in paths]
if any(r['data_sha256']!=results[0]['data_sha256'] for r in results):
    raise ValueError('Input data identities differ')
expected=hashlib.sha256(config.read_bytes()).hexdigest()
if any(r['config_sha256']!=expected for r in results[1:]) or [r['fold_seed'] for r in results[1:]]!=cfg['new_fold_seeds']:
    raise ValueError('Fold comparison provenance differs')
summary=[]
for seed,result in zip([cfg['reference_fold_seed']]+cfg['new_fold_seeds'],results):
    summary.append({'fold_seed':seed,'svr':result['summary']['svr'],'spaced_pair_svr':result['summary']['spaced_pair_svr'],'groups':result['summary']['groups'],
                    'strong_fold_distances':[{'fold':f['fold'],'svr':f['groups'][2]['svr']['selected_true_distance'],'spaced_pair_svr':f['groups'][2]['spaced_pair_svr']['selected_true_distance']} for f in result['folds']]})
out={'protocol':'One observed reference fold assignment plus two pre-registered follow-up fold assignments, all on shared previously developed training data; not independent tests.',
     'config_sha256':expected,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'summaries':summary}
(ROOT/'reports/m4_spaced_pair_fold_check_2026-11-16.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for x in summary:
    print(x['fold_seed'],'MAE',round(x['svr']['mae'],5),round(x['spaced_pair_svr']['mae'],5),'strong',[round(x['groups'][2][n]['selected_true_distance'],5) for n in ('svr','spaced_pair_svr')])
