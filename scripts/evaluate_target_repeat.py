#!/usr/bin/env python3
"""Compare fixed M3 and fixed SVR scores on previously reproducible generated samples."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import ConditionalSequenceVAE,MotifCNNStrengthPredictor,RidgeStrengthPredictor
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
from generate_conditional_position_baseline import assign_strength_groups
from train_motif_cnn_predictor import one_hot_sequences
from evaluate_m4_condition_response import score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--svr-model',required=True);p.add_argument('--output',required=True);p.add_argument('--candidate-csv',required=True);p.add_argument('--run-index',type=int,required=True,choices=(0,1));a=p.parse_args();cp=ROOT/'configs/m4_target_repeat_2026-11-13.toml';ex=tomllib.loads(cp.read_text())['experiment'];vae_seed,sample_seed=ex['run_pairs'][a.run_index];oldpath=ROOT/'reports/m4_condition_response_2026-10-30.json';old=json.loads(oldpath.read_text());cfg=load_config(ROOT/'configs/m2_motif_cnn_selected.toml');data=cfg['data'];m3=tomllib.loads((ROOT/'configs/m3_screening_evaluator.toml').read_text())['screening_evaluator'];weights=np.array([m3['motif_cnn_weight'],m3['global_kmer_ridge_weight'],m3['position_ridge_weight']]);path=Path(a.svr_model)
 if sha(path)!=ex['svr_checkpoint_sha256'] or np.__version__!=old['numpy_version'] or not np.isclose(weights.sum(),1):raise ValueError('checkpoint/version/config mismatch')
 seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 if hashes!=old['dataset_sha256']:raise ValueError('dataset mismatch')
 train,validation=seq[tr],seq[dev];yt,yv=np.log10(strength[tr]),np.log10(strength[dev]);groups,bounds=assign_strength_groups(yt);targets={g:float(yt[groups==g].mean()) for g in ex['targets']}; run=next(r for r in old['runs'] if r['vae_seed']==vae_seed and r['sample_seed']==sample_seed);print('train fixed screening CNN and generators',flush=True)
 cnn=MotifCNNStrengthPredictor(seed=int(cfg['project']['seed']),**cfg['motif_cnn']);cnn.fit(one_hot_sequences(train),yt,validation=(one_hot_sequences(validation),yv));global_model=RidgeStrengthPredictor(alpha=10).fit(KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(train),yt);pos=RidgeStrengthPredictor(alpha=10).fit(PositionKmerFeaturizer(sequence_length=50).transform(train),yt);predictor=(cnn,global_model,pos,weights);v,_=score(predictor,validation)
 if abs(regression_metrics(yv,v)['pearson']-m3['selection_value'])>1e-8:raise ValueError('M3 baseline differs')
 auto=ContinuousConditionalAutoregressiveGenerator(order=3,condition_bins=3).fit(train,yt);vae=ConditionalSequenceVAE(hidden_size=128,latent_size=16,beta=0.2,epochs=120,seed=vae_seed);vae.fit(one_hot_sequences(train),yt,validation=(one_hot_sequences(validation),yv));svr=PortableRBFSVR.load(path);feature=PositionKmerFeaturizer(sequence_length=50);rows=[];results=[]
 for offset,g in enumerate(ex['targets']):
  for name in ex['generators']:
   print('score',name,g,flush=True);seed=sample_seed+offset;samples=auto.sample(targets[g],ex['sample_count'],seed) if name=='autoregressive' else vae.sample(np.full(ex['sample_count'],targets[g]),seed);seqhash=hashlib.sha256(('\n'.join(samples.tolist())+'\n').encode()).hexdigest();reference=run['groups'][g][name]
   if seqhash!=reference['sample_sequence_sha256']:raise ValueError('generated sample hash differs')
   m3score,_=score(predictor,samples);svrscore=svr.predict(feature.transform(samples));
   if abs(m3score.mean()-reference['screening_score']['mean'])>1e-8:raise ValueError('M3 generated mean differs')
   agree=regression_metrics(m3score,svrscore);k=ex['top_k'];dm=np.abs(m3score-targets[g]);ds=np.abs(svrscore-targets[g]);tm=np.argsort(dm,kind='stable')[:k];ts=np.argsort(ds,kind='stable')[:k];target_overlap=len(set(tm)&set(ts));target_corr=regression_metrics(dm,ds)['spearman'];itop=np.argsort(-m3score,kind='stable')[:k];jtop=np.argsort(-svrscore,kind='stable')[:k];intersection=len(set(itop)&set(jtop));results.append({'generator':name,'group':g,'target_log_strength':targets[g],'count':len(samples),'unique_sequence_count':len(set(samples.tolist())),'sequence_sha256':seqhash,'m3_mean':float(m3score.mean()),'m3_std':float(m3score.std()),'svr_mean':float(svrscore.mean()),'svr_std':float(svrscore.std()),'pearson_between_scores':agree['pearson'],'spearman_between_scores':agree['spearman'],'target_distance_spearman':target_corr,'target_top_k_overlap_count':target_overlap,'m3_above_target_fraction':float(np.mean(m3score>targets[g])),'svr_above_target_fraction':float(np.mean(svrscore>targets[g])),'top_k':k,'top_k_overlap_count':intersection,'top_k_overlap_fraction':intersection/k});rows.extend((name,g,targets[g],i,s,float(q),float(r)) for i,(s,q,r) in enumerate(zip(samples,m3score,svrscore)))
 outcsv=Path(a.candidate_csv);outcsv=outcsv if outcsv.is_absolute() else ROOT/outcsv;outcsv.parent.mkdir(parents=True,exist_ok=True)
 with outcsv.open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['generator','input_group','target_log10','sample_index','sequence','m3_score','svr_score']);writer.writerows(rows)
 result={'protocol':'One existing run, three conditions, 500 samples per model and group. Fixed M3 and fixed SVR trained on overlapping course data, M3 previously used dev for early stopping. No selection or wet-lab validation. Top-50 by highest score within each group; overlap is row-based including duplicate sequences.','data_sha256':hashes,'candidate_csv_sha256':sha(outcsv),'protocol_sha256':sha(cp),'previous_response_sha256':sha(oldpath),'svr_checkpoint_sha256':sha(path),'numpy_version':np.__version__,'sample_seed':sample_seed,'vae_seed':vae_seed,'groups':results}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for r in results:print(r['generator'],r['group'],'m3',round(r['m3_mean'],4),'svr',round(r['svr_mean'],4),'spearman',round(r['spearman_between_scores'],4),'target_overlap',r['target_top_k_overlap_count'])
if __name__=='__main__':main()
