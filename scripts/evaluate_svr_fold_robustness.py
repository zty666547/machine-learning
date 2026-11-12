#!/usr/bin/env python3
"""Evaluate previously fixed ridge and SVR across new cluster folds."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import RidgeStrengthPredictor
from evaluate_strength_calibration import cluster_ids,group_folds
from analyze_frozen_motif_cnn import split_diagnostic

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--fold-seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_svr_fold_robustness_2026-11-12.toml';cfg=tomllib.loads(cp.read_text())['experiment'];priorcfg=ROOT/'configs/m4_rbf_svr_2026-11-09.toml';priorparams=tomllib.loads(priorcfg.read_text());
 if a.fold_seed not in cfg['fold_seeds'] or cfg['outer_folds']!=3:raise ValueError('unregistered fold seed')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,_,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq))[tr];x=PositionKmerFeaturizer(sequence_length=50).transform(seq[tr]);y=np.log10(strength[tr]);bounds=np.quantile(y,[1/3,2/3]);fold=group_folds(clusters,cfg['outer_folds'],a.fold_seed);priorpath=ROOT/'reports/m4_rbf_svr_2026-11-09.json';prior=json.loads(priorpath.read_text());hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 if prior['data_sha256']!=hashes or prior['fold_seed']!=cfg['reference_fold_seed'] or prior['numpy_version']!=np.__version__ or prior['sklearn_version']!=sklearn.__version__ or prior['protocol_sha256']!=sha(priorcfg):raise ValueError('previous protocol/data/environment mismatch')
 result={'protocol':'New fixed cluster-fold seeds, same features/outer training rows/model settings as original. No dev or test metrics. Repeated folds share data and are not independent samples.','fold_seed':a.fold_seed,'fold_sha256':hashlib.sha256(fold.astype('<i8').tobytes()).hexdigest(),'data_sha256':hashes,'protocol_sha256':sha(cp),'prior_protocol_sha256':sha(priorcfg),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'train_count':len(tr),'models':{}}
 predictions={}
 for name in ('ridge','svr'):
  pred=np.empty(len(tr));info=[]
  for i in range(cfg['outer_folds']):
   fit=np.flatnonzero(fold!=i);hold=np.flatnonzero(fold==i);assert not set(clusters[fit])&set(clusters[hold]);print(a.fold_seed,name,i+1,flush=True)
   model=RidgeStrengthPredictor(alpha=priorparams['ridge']['alpha']) if name=='ridge' else make_pipeline(StandardScaler(),SVR(**priorparams['svr']))
   model.fit(x[fit],y[fit]);pred[hold]=model.predict(x[hold]);info.append({'fit_count':len(fit),'holdout_count':len(hold),'support_vectors':int(model[-1].support_.size) if name=='svr' else None})
  predictions[name]=pred;result['models'][name]={'training_oof':split_diagnostic(y,pred,bounds),'folds':info}
 result['paired_fraction_svr_lower_absolute_error']=float(np.mean(abs(predictions['svr']-y)<abs(predictions['ridge']-y)))
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for n,v in result['models'].items():print(n,[(m,round(v['training_oof'][m],5)) for m in ('mae','pearson','spearman')])
if __name__=='__main__':main()
