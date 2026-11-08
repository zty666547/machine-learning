#!/usr/bin/env python3
"""Compare uniform and training-derived position-weighted 10-neighbor predictors."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences
from audit_weighted_neighbors import position_weights

def knn(fit_x,fit_y,query_x,weights,k,tolerance=1e-10):
    if k<1 or k>len(fit_x):raise ValueError('invalid k')
    ref=fit_x.reshape(len(fit_x),-1);w=np.repeat(weights,4);pred=np.empty(len(query_x));ess=np.empty(len(query_x));contributors=np.empty(len(query_x),int);ties=np.empty(len(query_x),int)
    for start in range(0,len(query_x),128):
        q=query_x[start:start+128].reshape(-1,len(w));sim=(q*w)@ref.T;boundary=np.partition(sim,sim.shape[1]-k,axis=1)[:,sim.shape[1]-k];closer=sim>boundary[:,None]+tolerance;equal=np.abs(sim-boundary[:,None])<=tolerance
        cnt=closer.sum(axis=1);ntie=equal.sum(axis=1);fraction=(k-cnt)/ntie
        if np.any(ntie==0) or np.any(fraction<=0) or np.any(fraction>1):raise ValueError('invalid tie resolution')
        a=closer.astype(float)+equal*fraction[:,None];stop=start+len(q);pred[start:stop]=(a@fit_y)/k;ess[start:stop]=k*k/(a*a).sum(axis=1);contributors[start:stop]=(a>0).sum(axis=1);ties[start:stop]=ntie
    return pred,ess,contributors,ties

def knn_fixed_count(fit_x,fit_y,query_x,weights,k):
    if k<1 or k>len(fit_x):raise ValueError("invalid k")
    ref=fit_x.reshape(len(fit_x),-1);w=np.repeat(weights,4);pred=np.empty(len(query_x))
    for start in range(0,len(query_x),128):
        q=query_x[start:start+128].reshape(-1,len(w));sim=(q*w)@ref.T
        indices=np.argsort(-sim,axis=1,kind="stable")[:,:k];pred[start:start+len(q)]=fit_y[indices].mean(axis=1)
    return pred

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cp=ROOT/'configs/m4_knn_position_2026-11-08.toml';cfg=tomllib.loads(cp.read_text())['experiment'];data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));cl=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(cl[tr])&set(cl[dev]);idx=np.r_[tr,dev];n=len(tr);x=one_hot_sequences(seq[idx]);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);folds,_=fold_layout(cl[tr],cfg['fold_seed']);k=cfg['neighbors'];tol=cfg['tie_tolerance']
    posthoc={z:np.empty(n) for z in ('uniform','weighted')}
    results={z:{'pred':np.empty(n),'ess':np.empty(n),'contributors':np.empty(n,int),'ties':np.empty(n,int)} for z in ('uniform','weighted')}; details=[]
    for i,(fit,early,hold) in enumerate(folds):
        ref=np.sort(np.r_[fit,early]);assert not set(cl[tr][ref])&set(cl[tr][hold]);w=position_weights(x[ref],y[ref]);print('fold',i+1,flush=True)
        for name,weights in [('uniform',np.ones(50)),('weighted',w)]:
            posthoc[name][hold]=knn_fixed_count(x[ref],y[ref],x[hold],weights,k)
            pred,ess,contributors,ties=knn(x[ref],y[ref],x[hold],weights,k,tol);r=results[name];r['pred'][hold]=pred;r['ess'][hold]=ess;r['contributors'][hold]=contributors;r['ties'][hold]=ties
        details.append({'fit_count':len(ref),'holdout_count':len(hold),'weights':w.tolist()})
    def summarize(y0,pred,ess,contributors,ties):return {'metrics':split_diagnostic(y0,pred,bounds),'effective_sample_size_mean':float(ess.mean()),'contributors_mean':float(contributors.mean()),'fraction_boundary_tied':float(np.mean(ties>1))}
    out={'protocol':'10 total neighbor weight, fractional averaging across all kth-distance ties; actual contributors and effective sample size may exceed 10. Position weights fit only on outer training data. Cross-cluster folds; test format-checked only.','data_sha256':{a:hashlib.sha256(v.read_bytes()).hexdigest() for a,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]},'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'fold_seed':cfg['fold_seed'],'train_count':n,'validation_count':len(dev),'folds':details,'training_oof':{},'development_validation':{}}
    w=position_weights(x[:n],y[:n]);postpath=ROOT/'configs/m4_knn_position_posthoc_2026-11-08.toml';postcfg=tomllib.loads(postpath.read_text())['experiment'];assert postcfg['neighbors']==k
    out['posthoc']={'protocol':'After observing fractional results, retain exactly k neighbors using fit order to resolve exact ties; tie choice does not use labels.','protocol_sha256':hashlib.sha256(postpath.read_bytes()).hexdigest(),'training_oof':{},'development_validation':{}}
    for name,weights in [('uniform',np.ones(50)),('weighted',w)]:
        out['posthoc']['training_oof'][name]=split_diagnostic(y[:n],posthoc[name],bounds);dev_fixed=knn_fixed_count(x[:n],y[:n],x[n:],weights,k);out['posthoc']['development_validation'][name]=split_diagnostic(y[n:],dev_fixed,bounds)
        r=results[name];out['training_oof'][name]=summarize(y[:n],r['pred'],r['ess'],r['contributors'],r['ties']);p,e,c,t=knn(x[:n],y[:n],x[n:],weights,k,tol);out['development_validation'][name]=summarize(y[n:],p,e,c,t)
    path=Path(a.output);path=path if path.is_absolute() else ROOT/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for split in ('training_oof','development_validation'):
        print('posthoc',split,[(name,round(out['posthoc'][split][name]['mae'],5),round(out['posthoc'][split][name]['pearson'],5),round(out['posthoc'][split][name]['spearman'],5)) for name in ('uniform','weighted')])
        print(split,[(name,round(out[split][name]['metrics']['mae'],5),round(out[split][name]['metrics']['pearson'],5),round(out[split][name]['metrics']['spearman'],5),round(out[split][name]['effective_sample_size_mean'],2),round(out[split][name]['contributors_mean'],2)) for name in ('uniform','weighted')])
if __name__=='__main__':main()
