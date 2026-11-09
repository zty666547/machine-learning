#!/usr/bin/env python3
"""Compare RBF support vector regression and ridge under identical cluster folds."""
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
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_rbf_svr_2026-11-09.toml';cfg=tomllib.loads(cp.read_text());ex=cfg['experiment'];
 if ex['outer_folds']!=3:raise ValueError('protocol mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);idx=np.r_[tr,dev];n=len(tr);x=PositionKmerFeaturizer(sequence_length=50).transform(seq[idx]);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);layouts,_=fold_layout(clusters[tr],ex['fold_seed']);hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 prior=json.loads((ROOT/'reports/m4_extra_trees_seed_20260912.json').read_text());
 if prior['data_sha256']!=hashes or prior['fold_seed']!=ex['fold_seed'] or prior['numpy_version']!=np.__version__ or prior['sklearn_version']!=sklearn.__version__:raise ValueError('Prior data or environment mismatch')
 result={'protocol':'Same position/global features and all outer-training rows for RBF SVR and ridge. Standardization and gamma fit only on each training fold. SVR deterministic; dev previously reused and test loaded for format checking only.','fold_seed':ex['fold_seed'],'data_sha256':hashes,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'train_count':n,'validation_count':len(dev),'models':{}}
 for name in ('ridge','svr'):
  pred=np.empty(n);folds=[]
  def make():return RidgeStrengthPredictor(alpha=cfg['ridge']['alpha']) if name=='ridge' else make_pipeline(StandardScaler(),SVR(**cfg['svr']))
  for i,(fit,early,hold) in enumerate(layouts):
   print(name,i+1,flush=True);ref=np.sort(np.r_[fit,early]);assert not set(clusters[tr][ref])&set(clusters[tr][hold]);model=make();model.fit(x[ref],y[ref]);pred[hold]=model.predict(x[hold]);folds.append({'fit_count':len(ref),'holdout_count':len(hold),'support_vectors':int(model[-1].support_.size) if name=='svr' else None})
  model=make();print(name,'development',flush=True);model.fit(x[:n],y[:n]);v=model.predict(x[n:]);result['models'][name]={'training_oof':split_diagnostic(y[:n],pred,bounds),'development_validation':split_diagnostic(y[n:],v,bounds),'folds':folds,'final_support_vectors':int(model[-1].support_.size) if name=='svr' else None}
  if name=='ridge' and (abs(result['models'][name]['training_oof']['mae']-prior['models']['linear']['training_oof']['mae'])>1e-8 or abs(result['models'][name]['development_validation']['mae']-prior['models']['linear']['development_validation']['mae'])>1e-8):raise ValueError('Prior ridge results differ')
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for n,m in result['models'].items():print(n,[(k,round(m['training_oof'][k],5)) for k in ('mae','pearson','spearman','prediction_to_target_std_ratio')])
if __name__=='__main__':main()
