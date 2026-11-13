#!/usr/bin/env python3
"""Export exploratory target ranks for three fixed generated sample runs."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_target_ranking import rank_target

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-csv',required=True);p.add_argument('--repeat-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();base=json.loads((ROOT/'reports/m4_generated_svr_compare_2026-11-12.json').read_text());previous=json.loads((ROOT/'reports/m4_target_ranking_2026-11-13.json').read_text());run=[(20260912,Path(a.reference_csv),base)]
 for i in range(2):
  report=json.loads((ROOT/f'reports/m4_target_repeat_run_{i}.json').read_text());run.append((report['vae_seed'],Path(a.repeat_dir)/f'candidates_{i}.csv',report))
 output=Path(a.output);output=output if output.is_absolute() else ROOT/output;output.parent.mkdir(parents=True,exist_ok=True);allrows=[]
 for seed,path,report in run:
  if sha(path)!=report['candidate_csv_sha256']:raise ValueError('candidate hash mismatch')
  with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
  if len(rows)!=3000:raise ValueError('candidate count mismatch')
  for group in report['groups']:
   gen,condition=group['generator'],group['group'];items=[x for x in rows if x['generator']==gen and x['input_group']==condition];items.sort(key=lambda x:int(x['sample_index']));target=group['target_log_strength']
   if len(items)!=500 or [int(x['sample_index']) for x in items]!=list(range(500)) or any(abs(float(x['target_log10'])-target)>1e-12 for x in items):raise ValueError('group mismatch')
   rm=rank_target([float(x['m3_score']) for x in items],target);rs=rank_target([float(x['svr_score']) for x in items],target)
   expected=next(x for x in previous['groups'] if x['generator']==gen and x['group']==condition)['target_top_k_overlap_count'] if seed==20260912 else group['target_top_k_overlap_count']
   if int(((rm<=50)&(rs<=50)).sum())!=expected:raise ValueError('rank overlap mismatch')
   for i,x in enumerate(items):allrows.append([seed,gen,condition,x['sample_index'],x['sequence'],target,x['m3_score'],x['svr_score'],int(rm[i]),int(rs[i]),int(rm[i]<=50),int(rs[i]<=50)])
 with output.open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['vae_seed','generator','input_group','sample_index','sequence','target_log10','m3_score','svr_score','m3_target_rank','svr_target_rank','m3_top50','svr_top50']);w.writerows(allrows)
 print('Exported',len(allrows),'exploratory rows; SHA-256:',sha(output))
if __name__=='__main__':main()
