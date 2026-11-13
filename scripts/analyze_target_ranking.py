#!/usr/bin/env python3
"""Compare target-distance rankings on fixed exploratory generated candidates."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.metrics import _rankdata

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank_target(scores,target):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or len(scores)==0 or not np.isfinite(scores).all() or not np.isfinite(target):raise ValueError('invalid scores or target')
    order=np.argsort(np.abs(scores-target),kind='stable');ranks=np.empty(len(order),dtype=int);ranks[order]=np.arange(1,len(order)+1);return ranks

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--ranking-csv',required=True);a=p.parse_args();cp=ROOT/'configs/m4_target_ranking_2026-11-13.toml';cfg=tomllib.loads(cp.read_text())['experiment'];sourcepath=ROOT/cfg['source_report'];source=json.loads(sourcepath.read_text());path=Path(a.candidate_csv)
 if sha(path)!=cfg['candidate_sha256'] or sha(path)!=source['candidate_csv_sha256']:raise ValueError('candidate file differs from fixed prior')
 with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=3000:raise ValueError('unexpected total candidate count')
 summaries=[];ranking=[]
 for gen in cfg['generators']:
  for group in cfg['groups']:
   items=[r for r in rows if r['generator']==gen and r['input_group']==group];items.sort(key=lambda r:int(r['sample_index']));old=next(z for z in source['groups'] if z['generator']==gen and z['group']==group)
   if len(items)!=old['count'] or [int(z['sample_index']) for z in items]!=list(range(old['count'])):raise ValueError('sample count/index mismatch')
   target=old['target_log_strength'];seq=[z['sequence'] for z in items];h=hashlib.sha256(('\n'.join(seq)+'\n').encode()).hexdigest()
   if h!=old['sequence_sha256'] or any(abs(float(z['target_log10'])-target)>1e-12 for z in items):raise ValueError('sequences or target mismatch')
   m=np.array([float(z['m3_score']) for z in items]);s=np.array([float(z['svr_score']) for z in items]);
   if abs(m.mean()-old['m3_mean'])>1e-10 or abs(s.mean()-old['svr_mean'])>1e-10:raise ValueError('prior score mean mismatch')
   rm=rank_target(m,target);rs=rank_target(s,target);k=cfg['top_k'];mt=rm<=k;st=rs<=k;intersection=int(np.sum(mt&st));dm=np.abs(m-target);ds=np.abs(s-target);corr=float(np.corrcoef(_rankdata(dm),_rankdata(ds))[0,1]);
   summaries.append({'generator':gen,'group':group,'target_log_strength':target,'count':len(items),'m3_above_target_fraction':float(np.mean(m>target)),'svr_above_target_fraction':float(np.mean(s>target)),'target_distance_spearman':corr,'target_top_k_overlap_count':intersection,'prior_high_score_top_k_overlap_count':old['top_k_overlap_count'],'m3_target_top_mean_distance':float(dm[mt].mean()),'svr_target_top_mean_distance':float(ds[st].mean()),'m3_target_vs_high_overlap':int(np.sum(mt & (np.argsort(-m,kind='stable').argsort()<k))),'svr_target_vs_high_overlap':int(np.sum(st & (np.argsort(-s,kind='stable').argsort()<k)))})
   for i,z in enumerate(items):ranking.append([gen,group,i,seq[i],target,m[i],s[i],dm[i],ds[i],int(rm[i]),int(rs[i]),int(mt[i]),int(st[i])])
 ranked=Path(a.ranking_csv);ranked=ranked if ranked.is_absolute() else ROOT/ranked;ranked.parent.mkdir(parents=True,exist_ok=True)
 with ranked.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['generator','input_group','sample_index','sequence','target_log10','m3_score','svr_score','m3_target_distance','svr_target_distance','m3_target_rank','svr_target_rank','m3_top50','svr_top50']);w.writerows(ranking)
 result={'protocol':'Posthoc exploratory re-ranking of previously inspected fixed candidates and scores; no new model training, labels, test metrics or independent validation. Target-distance is predicted score difference, not measured error. Top-k ties by original index.','candidate_csv_sha256':sha(path),'source_report_sha256':sha(sourcepath),'protocol_sha256':sha(cp),'ranking_csv_sha256':sha(ranked),'numpy_version':np.__version__,'top_k':cfg['top_k'],'groups':summaries}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for z in summaries:print(z['generator'],z['group'],'target_overlap',z['target_top_k_overlap_count'],'prior_high_overlap',z['prior_high_score_top_k_overlap_count'],'rank_corr',round(z['target_distance_spearman'],4),'above target',round(z['m3_above_target_fraction'],3),round(z['svr_above_target_fraction'],3),'target_vs_high',z['m3_target_vs_high_overlap'],z['svr_target_vs_high_overlap'])
if __name__=='__main__':main()
