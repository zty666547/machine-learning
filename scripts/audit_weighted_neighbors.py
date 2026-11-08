#!/usr/bin/env python3
"""Compare training-derived positional weights against uniform sequence distance."""
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
from audit_sequence_neighbors import nearest

def position_weights(x,y):
    counts=x.sum(axis=0); sums=np.einsum('npl,n->pl',x,y); mu=float(y.mean()); var=float(y.var())
    means=np.divide(sums,counts,out=np.full_like(sums,mu),where=counts>0)
    score=np.sum(counts*(means-mu)**2,axis=1)/len(y)/(var if var>1e-12 else 1.0)
    return 1.0+10.0*score

def weighted_nearest(fit_x,fit_y,query_x,weights,tolerance=1e-10):
    ref=fit_x.reshape(len(fit_x),-1);w=np.repeat(weights,4);pred=np.empty(len(query_x));ties=np.empty(len(query_x),int)
    for start in range(0,len(query_x),128):
        q=query_x[start:start+128].reshape(-1,len(w)); sim=(q*w)@ref.T; best=sim.max(axis=1); mask=np.abs(sim-best[:,None])<=tolerance; n=mask.sum(axis=1); stop=start+len(q);pred[start:stop]=(mask@fit_y)/n;ties[start:stop]=n
    return pred,ties

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cp=ROOT/'configs/m4_weighted_neighbor_2026-11-08.toml';cfg=tomllib.loads(cp.read_text())['experiment'];data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data']
    seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));cl=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(cl[tr])&set(cl[dev]);idx=np.r_[tr,dev];n=len(tr);x=one_hot_sequences(seq[idx]);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);folds,_=fold_layout(cl[tr],cfg['fold_seed'])
    uniform=np.empty(n);weighted=np.empty(n);uniform_ties=np.empty(n,int);fold_details=[]
    for i,(fit,early,hold) in enumerate(folds):
        ref=np.sort(np.r_[fit,early]);assert not set(cl[tr][ref])&set(cl[tr][hold]);w=position_weights(x[ref],y[ref]);print('fold',i+1,flush=True)
        uniform[hold],_,uniform_ties[hold]=nearest(x[ref],y[ref],x[hold]);weighted[hold],ties=weighted_nearest(x[ref],y[ref],x[hold],w,cfg['tie_tolerance']);fold_details.append({'fit_count':len(ref),'holdout_count':len(hold),'weights':w.tolist(),'fraction_ties':float(np.mean(ties>1))})
    w=position_weights(x[:n],y[:n]);du,_,dev_ties=nearest(x[:n],y[:n],x[n:]);dw,dt=weighted_nearest(x[:n],y[:n],x[n:],w,cfg['tie_tolerance'])
    def compare(y0,u,v): return {'uniform':split_diagnostic(y0,u,bounds),'weighted':split_diagnostic(y0,v,bounds),'fraction_different_prediction':float(np.mean(abs(u-v)>1e-10))}
    r={'protocol':'Position weights from fit-only single-position explained label variance. Cross-cluster folds, all fit and early rows available for fitting distance; fixed tie tolerance. Holdout/development labels never enter weights. Test rows format-checked only.','data_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]},'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'fold_seed':cfg['fold_seed'],'train_count':n,'validation_count':len(dev),'folds':fold_details,'development_weights':w.tolist(),'development_fraction_ties':float(np.mean(dt>1)),'training_oof':compare(y[:n],uniform,weighted),'development_validation':compare(y[n:],du,dw)}
    for key,u,v,t in (("training_oof",uniform,weighted,uniform_ties),("development_validation",du,dw,dev_ties)):
        changed=abs(u-v)>1e-10; r[key]["uniform_tied_count"]=int(np.sum(t>1));r[key]["changed_with_unique_uniform_neighbor"]=int(np.sum(changed & (t==1)));r[key]["changed_with_tied_uniform_neighbor"]=int(np.sum(changed & (t>1)))
    out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for k in ('training_oof','development_validation'):
        print(k,[(method,round(r[k][method]['mae'],5),round(r[k][method]['pearson'],5),round(r[k][method]['spearman'],5)) for method in ('uniform','weighted')], 'changed',r[k]['fraction_different_prediction'])
if __name__=='__main__':main()
