#!/usr/bin/env python3
"""Paired substitution of an RBF SVR component in the fixed fusion."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import RidgeStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--cnn-member',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_global_svr_2026-11-09.toml';cfg=tomllib.loads(cp.read_text())['experiment'];svrp=ROOT/'configs/m4_rbf_svr_2026-11-09.toml';svrcfg=tomllib.loads(svrp.read_text());ecp=ROOT/'configs/m4_cnn_ensemble_2026-11-09.toml';m3=tomllib.loads((ROOT/'configs/m3_screening_evaluator.toml').read_text())['screening_evaluator'];weights=np.array(cfg['weights']);
 if cfg['outer_folds']!=3 or cfg['fold_seed']!=svrcfg['experiment']['fold_seed'] or not np.allclose(weights,[m3['motif_cnn_weight'],m3['global_kmer_ridge_weight'],m3['position_ridge_weight']]):raise ValueError('protocol mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);idx=np.r_[tr,dev];n=len(tr);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);layouts,_=fold_layout(clusters[tr],cfg['fold_seed']);hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 previous=json.loads((ROOT/'reports/m4_rbf_svr_2026-11-09.json').read_text());cnnprior=json.loads((ROOT/f"reports/m4_rnn_vs_cnn_seed_{cfg['cnn_seed']}.json").read_text());member=Path(a.cnn_member)
 if previous['data_sha256']!=hashes or previous['fold_seed']!=cfg['fold_seed'] or previous['numpy_version']!=np.__version__ or previous['sklearn_version']!=sklearn.__version__ or previous['protocol_sha256']!=hashlib.sha256(svrp.read_bytes()).hexdigest():raise ValueError('SVR prior or environment mismatch')
 with np.load(member,allow_pickle=False) as f:
  if int(f['seed'])!=cfg['cnn_seed'] or str(f['protocol_sha256'])!=hashlib.sha256(ecp.read_bytes()).hexdigest() or str(f['numpy_version'])!=cnnprior['numpy_version'] or not np.array_equal(f['train_idx'],tr) or not np.array_equal(f['dev_idx'],dev) or any(str(f[k+'_sha256'])!=hashes[k] for k in hashes):raise ValueError('CNN member mismatch')
  cnn=f['oof'].copy();dcnn=f['dev'].copy()
 if abs(np.mean(abs(cnn-y[:n]))-cnnprior['models']['cnn']['training_oof']['mae'])>1e-8 or abs(np.mean(abs(dcnn-y[n:]))-cnnprior['models']['cnn']['development_validation']['mae'])>1e-8:raise ValueError('CNN member metrics differ')
 gx=KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(seq[idx]);px=PositionKmerFeaturizer(sequence_length=50).transform(seq[idx]);oof=np.empty((3,n));dv=np.empty((3,len(dev)));folds=[]
 for i,(fit,early,hold) in enumerate(layouts):
  ref=np.sort(np.r_[fit,early]);assert not set(clusters[tr][ref])&set(clusters[tr][hold]);print('fold',i+1,flush=True);g=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(gx[ref],y[ref]);pos=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(px[ref],y[ref]);svr=make_pipeline(StandardScaler(),SVR(**svrcfg['svr']));svr.fit(gx[ref],y[ref]);oof[0,hold]=g.predict(gx[hold]);oof[1,hold]=pos.predict(px[hold]);oof[2,hold]=svr.predict(gx[hold]);folds.append({'fit_count':len(ref),'holdout_count':len(hold),'support_vectors':int(svr[-1].support_.size)})
 print('development',flush=True);g=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(gx[:n],y[:n]);pos=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(px[:n],y[:n]);svr=make_pipeline(StandardScaler(),SVR(**svrcfg['svr']));svr.fit(gx[:n],y[:n]);dv[0]=g.predict(gx[n:]);dv[1]=pos.predict(px[n:]);dv[2]=svr.predict(gx[n:]);
 result={'protocol':'Same fixed fusion weights and shared CNN and position ridge; global ridge replaced with RBF SVR using the same global features. CNN train-internal early stop differs from original M3 dev-stop. Dev reused; test format checked only.','data_sha256':hashes,'cnn_member_sha256':hashlib.sha256(member.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'svr_protocol_sha256':hashlib.sha256(svrp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'cnn_numpy_version':cnnprior['numpy_version'],'fold_seed':cfg['fold_seed'],'cnn_seed':cfg['cnn_seed'],'train_count':n,'validation_count':len(dev),'folds':folds}
 for name,yy,c,r in [('training_oof',y[:n],cnn,oof),('development_validation',y[n:],dcnn,dv)]:
  base=weights[0]*c+weights[1]*r[0]+weights[2]*r[1];candidate=weights[0]*c+weights[1]*r[2]+weights[2]*r[1];result[name]={'original_components':split_diagnostic(yy,base,bounds),'svr_substitution':split_diagnostic(yy,candidate,bounds),'fraction_absolute_error_improved':float(np.mean(abs(candidate-yy)<abs(base-yy))),'mean_absolute_score_change':float(np.mean(abs(candidate-base))),'standalone_global_ridge':split_diagnostic(yy,r[0],bounds),'standalone_global_svr':split_diagnostic(yy,r[2],bounds)}
 previous_fusion=json.loads((ROOT/'reports/m4_fusion_svr_2026-11-09.json').read_text())
 for split in ('training_oof','development_validation'):
  if abs(result[split]['original_components']['mae']-previous_fusion[split]['original_components']['mae'])>1e-8: raise ValueError('prior fusion baseline mismatch')
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for s in ('training_oof','development_validation'):
  print(s,[(k,round(result[s][k]['mae'],5),round(result[s][k]['pearson'],5),round(result[s][k]['spearman'],5)) for k in ('original_components','svr_substitution')])
if __name__=='__main__':main()
