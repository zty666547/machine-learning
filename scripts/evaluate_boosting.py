#!/usr/bin/env python3
"""Cluster-fold comparison of linear and gradient boosting strength regressors."""
import argparse, hashlib, json, sys, tomllib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import PositionKmerFeaturizer, load_fixed_split_indices, load_promoter_arrays
from promoter_ml.models import RidgeStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
 cfgpath=ROOT/'configs/m4_boosting_2026-11-08.toml'; cfg=tomllib.loads(cfgpath.read_text()); ex=cfg['experiment']
 if a.seed not in ex['seeds'] or ex['outer_folds']!=3: raise ValueError('Protocol mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data']; seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50)
 tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq)); clusters=cluster_ids(ROOT/data['split_file'],len(seq)); assert not set(clusters[tr])&set(clusters[dev])
 idx=np.concatenate([tr,dev]); n=len(tr); x=PositionKmerFeaturizer(sequence_length=50).transform(seq[idx]); y=np.log10(strength[idx]); bounds=np.quantile(y[:n],[1/3,2/3]); folds,_=fold_layout(clusters[tr],ex['fold_seed'])
 r={'seed':a.seed,'fold_seed':ex['fold_seed'],'train_count':n,'validation_count':len(dev),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'data_sha256':{k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]},'protocol_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'models':{}}
 for name in ('linear','boosting'):
  def make():
   return RidgeStrengthPredictor(alpha=cfg['linear']['alpha']) if name=='linear' else HistGradientBoostingRegressor(random_state=a.seed,**cfg['boosting'])
  pred=np.empty(n); details=[]
  for i,(fit,early,hold) in enumerate(folds):
   print(a.seed,name,i+1,flush=True); fitted=np.sort(np.concatenate([fit,early])); model=make(); model.fit(x[fitted],y[fitted]);pred[hold]=model.predict(x[hold]);details.append({'fit_count':len(fitted),'holdout_count':len(hold)})
  model=make();model.fit(x[:n],y[:n]);r['models'][name]={'training_oof':split_diagnostic(y[:n],pred,bounds),'development_validation':split_diagnostic(y[n:],model.predict(x[n:]),bounds),'folds':details}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print({k:round(v['training_oof']['mae'],5) for k,v in r['models'].items()})
if __name__=='__main__':main()
