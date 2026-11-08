#!/usr/bin/env python3
"""Summarize paired baseline and tail-weighted RNN results."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; seeds=(20260912,20260913,20260914)
rows=[json.loads((ROOT/f'reports/m4_tail_weight_seed_{s}.json').read_text()) for s in seeds]
for s,r in zip(seeds,rows):
 assert r['seed']==s
 for k in ('fold_seed','data_sha256','protocol_sha256','numpy_version','training_boundaries'): assert r[k]==rows[0][k]
summary={'seeds':seeds,'fold_seed':rows[0]['fold_seed'],'protocol_sha256':rows[0]['protocol_sha256'],'data_sha256':rows[0]['data_sha256'],'note':'Paired seeds and same grouped folds. Seed variation does not represent independent test uncertainty. Development validation is previously reused; test was not evaluated.','models':{}}
for name, get in [('rnn_mse',lambda r:r['baseline_rnn']),('tail_weighted_rnn',lambda r:r['model'])]:
 summary['models'][name]={}
 for split in ('training_oof','development_validation'):
  items=[get(r)[split] for r in rows]; out={}
  for m in ('mae','rmse','pearson','spearman','prediction_to_target_std_ratio'):
   a=np.array([x[m] for x in items]); out[m]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
  out['groups']={}
  for g in ('weak','medium','strong'):
   out['groups'][g]={}
   for m in ('mae','mean_signed_error'):
    a=np.array([x['by_true_strength_group'][g][m] for x in items]);out['groups'][g][m]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
  summary['models'][name][split]=out
(ROOT/'reports/m4_tail_weight_summary_2026-11-08.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for n,v in summary['models'].items():
 x=v['training_oof'];print(n,*(f'{m}={x[m]["mean"]:.5f}±{x[m]["std"]:.5f}' for m in ('mae','pearson','spearman','prediction_to_target_std_ratio')),*(f'{g}_mae={x["groups"][g]["mae"]["mean"]:.5f}' for g in ('weak','medium','strong')))
 print('  dev',*(f'{m}={v["development_validation"][m]["mean"]:.5f}±{v["development_validation"][m]["std"]:.5f}' for m in ('mae','pearson','spearman')))
