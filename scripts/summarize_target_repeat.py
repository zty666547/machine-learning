#!/usr/bin/env python3
"""Summarize previously defined same-condition target-distance rankings."""
import argparse,hashlib,json,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_target_repeat_2026-11-13.toml';cfg=tomllib.loads(cp.read_text())['experiment'];previous=ROOT/'reports/m4_target_ranking_2026-11-13.json';p0=json.loads(previous.read_text());firstgen=json.loads((ROOT/'reports/m4_generated_svr_compare_2026-11-12.json').read_text());new=[json.loads((ROOT/f'reports/m4_target_repeat_run_{i}.json').read_text()) for i in range(2)]
 if p0['candidate_csv_sha256']!=firstgen['candidate_csv_sha256']:raise ValueError('reference provenance mismatch')
 for i,z in enumerate(new):
  if z['vae_seed']!=cfg['run_pairs'][i][0] or z['sample_seed']!=cfg['run_pairs'][i][1] or z['data_sha256']!=firstgen['data_sha256'] or z['svr_checkpoint_sha256']!=cfg['svr_checkpoint_sha256'] or z['protocol_sha256']!=sha(cp) or z['numpy_version']!=firstgen['numpy_version']:raise ValueError('new-run provenance mismatch')
 out={'protocol':'Previously explored target-distance ranking repeated on two pre-existing seed pairs. Three runs share source data and fixed scoring models; not independent biological validation. Autoregressive training deterministic; VAE seeds differ.','protocol_sha256':sha(cp),'reference_result_sha256':sha(previous),'new_result_sha256':[sha(ROOT/f'reports/m4_target_repeat_run_{i}.json') for i in range(2)],'runs':[cfg['reference_run'],*cfg['run_pairs']],'top_k':cfg['top_k'],'groups':[]}
 for gen in cfg['generators']:
  for group in cfg['targets']:
   old=next(x for x in p0['groups'] if x['generator']==gen and x['group']==group);z=[next(x for x in t['groups'] if x['generator']==gen and x['group']==group) for t in new];counts=[old['target_top_k_overlap_count']]+[x['target_top_k_overlap_count'] for x in z];cors=[old['target_distance_spearman']]+[x['target_distance_spearman'] for x in z];
   out['groups'].append({'generator':gen,'group':group,'target_top_k_overlap_counts':counts,'target_top_k_overlap_mean':float(np.mean(counts)),'target_distance_spearman_values':cors,'target_distance_spearman_mean':float(np.mean(cors)),'unique_counts':[next(x for x in firstgen['groups'] if x['generator']==gen and x['group']==group)['unique_sequence_count']]+[x['unique_sequence_count'] for x in z]})
 output=Path(a.output);output=output if output.is_absolute() else ROOT/output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for x in out['groups']:print(x['generator'],x['group'],'overlap',x['target_top_k_overlap_counts'],'mean',round(x['target_top_k_overlap_mean'],2),'rho',round(x['target_distance_spearman_mean'],4),'unique',x['unique_counts'])
if __name__=='__main__':main()
