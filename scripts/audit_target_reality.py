#!/usr/bin/env python3
"""Retrospective development-data audit of target-distance selection."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import MotifCNNStrengthPredictor,RidgeStrengthPredictor
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
from evaluate_strength_calibration import cluster_ids
from evaluate_m4_condition_response import score
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences
from analyze_target_ranking import rank_target

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--svr-model',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-csv',required=True);a=p.parse_args();cp=ROOT/'configs/m4_target_reality_audit_2026-11-15.toml';ex=tomllib.loads(cp.read_text())['experiment'];m3=tomllib.loads((ROOT/'configs/m3_screening_evaluator.toml').read_text())['screening_evaluator'];cfg=load_config(ROOT/'configs/m2_motif_cnn_selected.toml');data=cfg['data'];old=json.loads((ROOT/'reports/m4_condition_response_2026-10-30.json').read_text());svrold=json.loads((ROOT/'reports/m4_rbf_svr_2026-11-09.json').read_text());modelpath=Path(a.svr_model)
 if sha(modelpath)!=ex['svr_checkpoint_sha256'] or np.__version__!=old['numpy_version']:raise ValueError('model or NumPy version mismatch')
 seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 if hashes!=old['dataset_sha256'] or hashes!=svrold['data_sha256']:raise ValueError('dataset differs')
 train,validation=seq[tr],seq[dev];yt,yv=np.log10(strength[tr]),np.log10(strength[dev]);labels,_=assign_strength_groups(yt);targets={g:float(yt[labels==g].mean()) for g in ex['groups']};print('retrain fixed M3 and score development pool',flush=True)
 cnn=MotifCNNStrengthPredictor(seed=int(cfg['project']['seed']),**cfg['motif_cnn']);cnn.fit(one_hot_sequences(train),yt,validation=(one_hot_sequences(validation),yv));gm=RidgeStrengthPredictor(alpha=10).fit(KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(train),yt);pm=RidgeStrengthPredictor(alpha=10).fit(PositionKmerFeaturizer(sequence_length=50).transform(train),yt);weights=np.array([m3['motif_cnn_weight'],m3['global_kmer_ridge_weight'],m3['position_ridge_weight']]);mscore,_=score((cnn,gm,pm,weights),validation);svr=PortableRBFSVR.load(modelpath);sscore=svr.predict(PositionKmerFeaturizer(sequence_length=50).transform(validation));m3pearson=regression_metrics(yv,mscore)['pearson'];svrmae=float(np.mean(abs(yv-sscore)))
 if abs(m3pearson-m3['selection_value'])>1e-8 or abs(svrmae-svrold['models']['svr']['development_validation']['mae'])>1e-8:raise ValueError('reference predictor mismatch')
 rows=[];results=[];k=ex['top_k'];
 if k<1 or k>len(dev):raise ValueError('invalid top_k')
 for group in ex['groups']:
  t=targets[group];true_gap=abs(yv-t);random=float(true_gap.mean());oracle_idx=np.argsort(true_gap,kind='stable')[:k];oracle=float(true_gap[oracle_idx].mean());item={'group':group,'target_log10':t,'pool_count':len(dev),'top_k':k,'random_pool_mean_true_distance':random,'oracle_topk_true_distance':oracle,'models':{}}
  selected={}
  for name,sc in [('m3',mscore),('svr',sscore)]:
   r=rank_target(sc,t);idx=np.flatnonzero(r<=k);selected[name]=set(idx.tolist());item['models'][name]={'mean_true_distance_to_target':float(true_gap[idx].mean()),'median_true_distance_to_target':float(np.median(true_gap[idx])),'mean_predicted_distance_to_target':float(np.mean(abs(sc[idx]-t))),'selected_true_mean':float(yv[idx].mean()),'selected_true_min':float(yv[idx].min()),'selected_true_max':float(yv[idx].max()),'target_distance_spearman':regression_metrics(true_gap,abs(sc-t))['spearman']}
   for i in idx:rows.append([group,name,int(dev[i]),float(t),float(yv[i]),float(sc[i]),float(true_gap[i]),int(r[i])])
  item['top_k_overlap_count']=len(selected['m3']&selected['svr']);results.append(item)
 outcsv=Path(a.selected_csv);outcsv=outcsv if outcsv.is_absolute() else ROOT/outcsv;outcsv.parent.mkdir(parents=True,exist_ok=True)
 with outcsv.open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['target_group','scorer','course_sample_index','target_log10','observed_log10','predicted_log10','true_absolute_target_distance','target_rank']);w.writerows(rows)
 result={'protocol':'Retrospective diagnostic on reused development pool; target values from training labels. M3 used same dev labels for early stop and past weight selection. SVR training excludes dev but dev used in development. Oracle uses dev truth only as hindsight lower bound; no test metrics or independent biological validation.','data_sha256':hashes,'protocol_sha256':sha(cp),'svr_checkpoint_sha256':sha(modelpath),'selected_csv_sha256':sha(outcsv),'numpy_version':np.__version__,'training_count':len(tr),'development_count':len(dev),'m3_validation_pearson_reproduced':m3pearson,'svr_validation_mae_reproduced':svrmae,'groups':results}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for z in results:print(z['group'],'random',round(z['random_pool_mean_true_distance'],4),'oracle',round(z['oracle_topk_true_distance'],4),'M3',round(z['models']['m3']['mean_true_distance_to_target'],4),'SVR',round(z['models']['svr']['mean_true_distance_to_target'],4),'overlap',z['top_k_overlap_count'])
if __name__=='__main__':main()
