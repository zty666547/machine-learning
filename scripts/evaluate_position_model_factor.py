#!/usr/bin/env python3
"""Compare global and position-plus-global features across ridge and RBF SVR."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import RidgeStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_position_model_factor_2026-11-10.toml';cfg=tomllib.loads(cp.read_text())['experiment'];data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));cl=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(cl[tr])&set(cl[dev]);idx=np.r_[tr,dev];n=len(tr);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);folds,_=fold_layout(cl[tr],cfg['fold_seed']);
 if cfg['outer_folds']!=3:raise ValueError('protocol mismatch')
 hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};gpath=ROOT/'reports/m4_global_svr_2026-11-09.json';ppath=ROOT/'reports/m4_rbf_svr_2026-11-09.json';g=json.loads(gpath.read_text());p0=json.loads(ppath.read_text());gcfg=ROOT/'configs/m4_global_svr_2026-11-09.toml';pcfg=ROOT/'configs/m4_rbf_svr_2026-11-09.toml';
 if any(z['data_sha256']!=hashes or z['fold_seed']!=cfg['fold_seed'] or z['numpy_version']!=np.__version__ for z in (g,p0)) or g['sklearn_version']!=p0['sklearn_version'] or g['protocol_sha256']!=sha(gcfg) or p0['protocol_sha256']!=sha(pcfg) or g['svr_protocol_sha256']!=sha(pcfg):raise ValueError('prior data or protocol mismatch')
 result={'protocol':'Two feature sets x two model types on same fixed cluster folds and outer training rows; ridge alpha 10 both; reuse fixed prior deterministic SVR aggregate results and rerun ridge. SVR gamma=scale changes with feature dimension and fit data. No independent validation; no sample-level paired differences for prior SVR.','data_sha256':hashes,'prior_result_sha256':{'global_svr':sha(gpath),'position_svr':sha(ppath)},'protocol_sha256':sha(cp),'svr_protocol_sha256':sha(pcfg),'fold_seed':cfg['fold_seed'],'numpy_version':np.__version__,'sklearn_version_in_prior':g['sklearn_version'],'train_count':n,'validation_count':len(dev),'models':{}}
 for name,featurizer in [('ridge_global',KmerFeaturizer(k_min=1,k_max=3,include_gc=True)),('ridge_position_global',PositionKmerFeaturizer(sequence_length=50))]:
  x=featurizer.transform(seq[idx]);pred=np.empty(n);details=[]
  for i,(fit,early,hold) in enumerate(folds):
   ref=np.sort(np.r_[fit,early]);assert not set(cl[tr][ref])&set(cl[tr][hold]);print(name,i+1,flush=True);model=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(x[ref],y[ref]);pred[hold]=model.predict(x[hold]);details.append({'fit_count':len(ref),'holdout_count':len(hold)})
  model=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(x[:n],y[:n]);devpred=model.predict(x[n:]);result['models'][name]={'training_oof':split_diagnostic(y[:n],pred,bounds),'development_validation':split_diagnostic(y[n:],devpred,bounds),'folds':details,'feature_count':x.shape[1]}
 for split in ('training_oof','development_validation'):
  if abs(result['models']['ridge_global'][split]['mae']-g[split]['standalone_global_ridge']['mae'])>1e-8:raise ValueError('previous global ridge mismatch')
 result['models']['svr_global']={split:g[split]['standalone_global_svr'] for split in ('training_oof','development_validation')}
 result['models']['svr_position_global']={split:p0['models']['svr'][split] for split in ('training_oof','development_validation')}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for name,v in result['models'].items():
  x=v['training_oof'];print(name,[(m,round(x[m],5)) for m in ('mae','pearson','spearman','prediction_to_target_std_ratio')],[(g,round(z['mae'],5)) for g,z in x['by_true_strength_group'].items()])
if __name__=='__main__':main()
