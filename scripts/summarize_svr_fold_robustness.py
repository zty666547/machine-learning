#!/usr/bin/env python3
"""Summarize fixed and new correlated cluster-fold evaluations."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
reference=json.loads((ROOT/'reports/m4_rbf_svr_2026-11-09.json').read_text()); new=[json.loads((ROOT/f'reports/m4_svr_fold_{s}.json').read_text()) for s in (20261110,20261111)];runs=[reference,*new]
for x in new:
 if x['data_sha256']!=reference['data_sha256'] or x['numpy_version']!=reference['numpy_version'] or x['sklearn_version']!=reference['sklearn_version'] or x['prior_protocol_sha256']!=reference['protocol_sha256']:raise ValueError('provenance mismatch')
if len({x['fold_sha256'] for x in new})!=2:raise ValueError('new fold seeds produced identical assignments')
r={'note':'Same samples reused across cluster partitions; no independent test or confidence interval. Original and two new deterministic folds.','fold_seeds':[x['fold_seed'] for x in runs],'data_sha256':reference['data_sha256'],'models':{}}
for name in ('ridge','svr'):
 values=[x['models'][name]['training_oof'] for x in runs];out={}
 for metric in ('mae','pearson','spearman','prediction_to_target_std_ratio'):
  a=np.array([z[metric] for z in values]);out[metric]={'values':a.tolist(),'mean':float(a.mean()),'std':float(a.std())}
 out['groups']={}
 for group in ('weak','medium','strong'):
  out['groups'][group]={}
  for metric in ('mae','mean_signed_error'):
   a=np.array([z['by_true_strength_group'][group][metric] for z in values]);out['groups'][group][metric]={'values':a.tolist(),'mean':float(a.mean()),'std':float(a.std())}
 r['models'][name]=out
(ROOT/'reports/m4_svr_fold_robustness_2026-11-12.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for name,v in r['models'].items():print(name,[(m,round(v[m]['mean'],5),[round(x,5) for x in v[m]['values']]) for m in ('mae','pearson','spearman')],[(g,round(v['groups'][g]['mae']['mean'],5)) for g in ('weak','medium','strong')])
