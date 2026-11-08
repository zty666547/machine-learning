#!/usr/bin/env python3
"""Measure strength agreement among nearest cross-cluster sequences."""
from pathlib import Path
import argparse,hashlib,json,sys,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences

def nearest(fit_x, fit_y, query_x):
    # All one-hot coordinates are 0 or 1; dot product counts matching positions.
    reference=fit_x.reshape(len(fit_x),-1).astype(np.float32)
    y=np.empty(len(query_x)); dist=np.empty(len(query_x),dtype=int); ties=np.empty(len(query_x),dtype=int)
    for start in range(0,len(query_x),128):
        q=query_x[start:start+128].reshape(-1,reference.shape[1]).astype(np.float32)
        similarity=q@reference.T; best=similarity.max(axis=1); mask=similarity==best[:,None]
        n=mask.sum(axis=1); stop=start+len(q); y[start:stop]=(mask@fit_y)/n;dist[start:stop]=50-best.astype(int);ties[start:stop]=n
    return y,dist,ties

def detail(y,pred,dist,ties,boundaries):
    table=[]
    for lo,hi in ((0,4),(5,9),(10,14),(15,19),(20,24),(25,50)):
        m=(dist>=lo)&(dist<=hi)
        table.append({'distance_range':[lo,hi],'count':int(m.sum()),'mae':float(np.mean(abs(y[m]-pred[m]))) if m.any() else None})
    return {'metrics':split_diagnostic(y,pred,boundaries),'distance_median':float(np.median(dist)), 'distance_q10_q90':np.quantile(dist,[.1,.9]).tolist(),'distance_bins':table,'nearest_ties_mean':float(ties.mean()),'fraction_with_ties':float(np.mean(ties>1))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfgpath=ROOT/'configs/m4_neighbor_audit_2026-11-08.toml';cfg=tomllib.loads(cfgpath.read_text())['experiment'];data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data']
    seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));cl=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(cl[tr])&set(cl[dev])
    idx=np.concatenate([tr,dev]);n=len(tr);x=one_hot_sequences(seq[idx]); y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);folds,_=fold_layout(cl[tr],cfg['fold_seed'])
    pred=np.empty(n);mean=np.empty(n);dist=np.empty(n,dtype=int);ties=np.empty(n,dtype=int); sizes=[]
    for i,(fit,early,hold) in enumerate(folds):
        ref=np.sort(np.concatenate([fit,early]));assert not set(cl[tr][ref])&set(cl[tr][hold]);print('fold',i+1,flush=True)
        pred[hold],dist[hold],ties[hold]=nearest(x[ref],y[ref],x[hold]);mean[hold]=y[ref].mean();sizes.append({'fit_count':len(ref),'holdout_count':len(hold)})
    print('development',flush=True);dp,dd,dt=nearest(x[:n],y[:n],x[n:]);dm=np.full(len(dev),y[:n].mean())
    r={'protocol':'Nearest fit sequence(s) by aligned 50 bp Hamming distance, average all tied nearest labels. Cross-cluster outer folds; no test features or metrics. Similarity is not proof of biological mechanism or independent measurement.','data_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]},'protocol_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'numpy_version':np.__version__,'fold_seed':cfg['fold_seed'],'train_count':n,'validation_count':len(dev),'folds':sizes,'training_oof':detail(y[:n],pred,dist,ties,bounds),'training_oof_fit_mean':{'mae':float(np.mean(abs(y[:n]-mean)))},'development_validation':detail(y[n:],dp,dd,dt,bounds),'development_fit_mean':{'mae':float(np.mean(abs(y[n:]-dm)))}}
    out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'oof_mae':r['training_oof']['metrics']['mae'],'mean_mae':r['training_oof_fit_mean']['mae'],'median_distance':r['training_oof']['distance_median'],'bins':r['training_oof']['distance_bins']},indent=2))
if __name__=='__main__':main()
