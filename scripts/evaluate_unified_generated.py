#!/usr/bin/env python3
"""Recheck fixed generated records against train and consolidate M4 metrics."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.structure_analysis import encode_sequences,position_js,position_kmer_frequency
from train_motif_cnn_predictor import one_hot_sequences
from generate_conditional_position_baseline import assign_strength_groups
from audit_sequence_neighbors import nearest

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pairwise_diversity(x):
 x=np.asarray(x,dtype=float)
 if x.ndim!=3 or x.shape[2]!=4 or len(x)<2:raise ValueError('need at least two one-hot sequences')
 flat=x.reshape(len(x),-1);distance=1-(flat@flat.T)/x.shape[1];i,j=np.triu_indices(len(x),1);return float(distance[i,j].mean())

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--ranking-csv',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_unified_generated_2026-11-13.toml';cfg=tomllib.loads(cp.read_text())['experiment'];path=Path(a.ranking_csv);previous=ROOT/'reports/m4_condition_response_2026-10-30.json';old=json.loads(previous.read_text());score_reports=[json.loads((ROOT/'reports/m4_generated_svr_compare_2026-11-12.json').read_text())]+[json.loads((ROOT/f'reports/m4_target_repeat_run_{i}.json').read_text()) for i in (0,1)]
 if sha(path)!=cfg['ranking_csv_sha256'] or np.__version__!=old['numpy_version']:raise ValueError('candidate file or NumPy version mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,_,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 if hashes!=old['dataset_sha256'] or any(r['data_sha256']!=hashes for r in score_reports):raise ValueError('prior data mismatch')
 train=seq[tr];y=np.log10(strength[tr]);labels,_=assign_strength_groups(y);profiles={g:position_kmer_frequency(encode_sequences(train[labels==g]),3) for g in cfg['groups']};refx=one_hot_sequences(train);trainset=set(train.tolist());results=[]
 with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=len(cfg['seeds'])*len(cfg['groups'])*len(cfg['generators'])*cfg['per_group']:raise ValueError('candidate count mismatch')
 for si,seed in enumerate(cfg['seeds']):
  run=next(r for r in old['runs'] if r['vae_seed']==seed);sr=score_reports[si]
  if sr['vae_seed']!=seed:raise ValueError('score run mismatch')
  for model in cfg['generators']:
   for group in cfg['groups']:
    items=[r for r in rows if int(r['vae_seed'])==seed and r['generator']==model and r['input_group']==group];items.sort(key=lambda r:int(r['sample_index']));ref=run['groups'][group][model];score_ref=next(z for z in sr['groups'] if z['generator']==model and z['group']==group)
    if len(items)!=cfg['per_group'] or [int(z['sample_index']) for z in items]!=list(range(cfg['per_group'])):raise ValueError('group indices mismatch')
    samples=np.array([z['sequence'] for z in items]);seqhash=hashlib.sha256(('\n'.join(samples.tolist())+'\n').encode()).hexdigest()
    if seqhash!=ref['sample_sequence_sha256'] or seqhash!=score_ref['sequence_sha256'] or any(len(s)!=50 or set(s)-set('ACGT') for s in samples):raise ValueError('sequence hash or validity mismatch')
    m3=np.array([float(z['m3_score']) for z in items]);svr=np.array([float(z['svr_score']) for z in items]);
    if abs(m3.mean()-ref['screening_score']['mean'])>1e-8 or abs(svr.mean()-score_ref['svr_mean'])>1e-8 or any(abs(float(z['target_log10'])-ref['target_log_strength'])>1e-12 for z in items):raise ValueError('score or target mismatch')
    encoded=encode_sequences(samples);query=one_hot_sequences(samples);gc=float(np.mean((encoded==1)|(encoded==2)));js=float(position_js(profiles[group],position_kmer_frequency(encoded,3)).mean());_,distance,_=nearest(refx,np.zeros(len(train)),query)
    if abs(gc-ref['gc_mean'])>1e-12 or abs(js-ref['position_3mer_js_against_training_groups'][group])>1e-12:raise ValueError('previous structure metrics differ')
    results.append({'seed':seed,'generator':model,'group':group,'target_log_strength':ref['target_log_strength'],'count':len(samples),'sequence_sha256':seqhash,'valid_fraction':1.0,'unique_fraction':len(set(samples.tolist()))/len(samples),'novel_vs_train_fraction':float(np.mean([s not in trainset for s in samples])),'pairwise_hamming_fraction':pairwise_diversity(query),'nearest_train_similarity_mean':float(np.mean(1-distance/50)),'gc_mean':gc,'position_3mer_js_to_train_group':js,'m3_score_mean':float(m3.mean()),'svr_score_mean':float(svr.mean())})
    print(seed,model,group,'JS',round(js,5),'diversity',round(results[-1]['pairwise_hamming_fraction'],4),flush=True)
 metrics=['valid_fraction','unique_fraction','novel_vs_train_fraction','pairwise_hamming_fraction','nearest_train_similarity_mean','gc_mean','position_3mer_js_to_train_group','m3_score_mean','svr_score_mean'];summary=[]
 for model in cfg['generators']:
  for group in cfg['groups']:
   matching=[r for r in results if r['generator']==model and r['group']==group];d={'generator':model,'group':group,'target_log_strength':matching[0]['target_log_strength']}
   for name in metrics:
    values=np.array([r[name] for r in matching]);d[name]={'mean':float(values.mean()),'std_across_runs':float(values.std()),'values':values.tolist()}
   summary.append(d)
 allseq=[r['sequence'] for r in rows];result={'protocol':'Consolidation of previously inspected candidate records; 3 paired runs, 3 conditions, 500 per group; training data reference only. Same-data scoring is not biological validation.','data_sha256':hashes,'candidate_csv_sha256':sha(path),'prior_condition_sha256':sha(previous),'protocol_sha256':sha(cp),'numpy_version':np.__version__,'training_count':len(train),'generated_record_count':len(rows),'unique_across_records':len(set(allseq)),'runs':results,'summary':summary}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
