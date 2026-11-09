#!/usr/bin/env python3
"""Compare frozen fusion with one CNN versus the three-seed CNN mean."""
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

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--members-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_fusion_cnn_ensemble_2026-11-09.toml';cfg=tomllib.loads(cp.read_text())['experiment']; prevconfig=ROOT/'configs/m4_cnn_ensemble_2026-11-09.toml';prev=tomllib.loads(prevconfig.read_text())['experiment'];m3=tomllib.loads((ROOT/'configs/m3_screening_evaluator.toml').read_text())['screening_evaluator'];weights=np.array(cfg['weights']);
 if cfg['seeds']!=prev['seeds'] or cfg['fold_seed']!=prev['fold_seed'] or cfg['outer_folds']!=3 or not np.allclose(weights,[m3['motif_cnn_weight'],m3['global_kmer_ridge_weight'],m3['position_ridge_weight']]):raise ValueError('frozen protocol mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);selected=np.r_[tr,dev];n=len(tr);y=np.log10(strength[selected]);bounds=np.quantile(y[:n],[1/3,2/3]);layouts,_=fold_layout(clusters[tr],cfg['fold_seed']);hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};ph=hashlib.sha256(prevconfig.read_bytes()).hexdigest();members=[];devmembers=[];member_hashes={}
 for seed in cfg['seeds']:
  path=Path(a.members_dir)/f'member_{seed}.npz';member_hashes[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest();prior=json.loads((ROOT/f'reports/m4_rnn_vs_cnn_seed_{seed}.json').read_text())
  with np.load(path,allow_pickle=False) as f:
   if int(f['seed'])!=seed or str(f['protocol_sha256'])!=ph or str(f['numpy_version'])!=np.__version__ or not np.array_equal(f['train_idx'],tr) or not np.array_equal(f['dev_idx'],dev) or any(str(f[k+'_sha256'])!=hashes[k] for k in hashes):raise ValueError('CNN member provenance mismatch')
   q=f['oof'].copy();v=f['dev'].copy()
  if prior['data_sha256']!=hashes or prior['fold_seed']!=cfg['fold_seed'] or abs(np.mean(abs(q-y[:n]))-prior['models']['cnn']['training_oof']['mae'])>1e-8 or abs(np.mean(abs(v-y[n:]))-prior['models']['cnn']['development_validation']['mae'])>1e-8:raise ValueError('CNN member baseline mismatch')
  members.append(q);devmembers.append(v)
 members=np.stack(members);devmembers=np.stack(devmembers);global_x=KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(seq[selected]);position_x=PositionKmerFeaturizer(sequence_length=50).transform(seq[selected]);ridge=np.empty((2,n));dridge=np.empty((2,len(dev)))
 for j,features in enumerate((global_x,position_x)):
  for i,(fit,early,hold) in enumerate(layouts):
   ref=np.sort(np.r_[fit,early]);assert not set(clusters[tr][ref])&set(clusters[tr][hold]);model=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(features[ref],y[ref]);ridge[j,hold]=model.predict(features[hold])
  model=RidgeStrengthPredictor(alpha=cfg['ridge_alpha']).fit(features[:n],y[:n]);dridge[j]=model.predict(features[n:])
 result={'protocol':'Same M3 weights; compare seed 20260912 CNN vs equal mean of three seeds, identical ridge OOF predictions. CNN uses train-internal early stop unlike deployed M3 dev-stop; paired diagnostic, not exact deployed M3 rerun. Dev repeatedly reused. Test rows format-checked only.','data_sha256':hashes,'member_file_sha256':member_hashes,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'train_count':n,'validation_count':len(dev),'fold_seed':cfg['fold_seed'],'seeds':cfg['seeds']}
 for name,yy,cs,rs in [('training_oof',y[:n],members,ridge),('development_validation',y[n:],devmembers,dridge)]:
  baseline=weights[0]*cs[cfg['seeds'].index(cfg['baseline_seed'])]+weights[1]*rs[0]+weights[2]*rs[1];ensemble=weights[0]*cs.mean(axis=0)+weights[1]*rs[0]+weights[2]*rs[1]
  result[name]={'single_cnn_fusion':split_diagnostic(yy,baseline,bounds),'ensemble_cnn_fusion':split_diagnostic(yy,ensemble,bounds),'paired_mean_absolute_change':float(np.mean(abs(ensemble-baseline))),'fraction_improved_absolute_error':float(np.mean(abs(ensemble-yy)<abs(baseline-yy)))}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for s in ('training_oof','development_validation'):
  print(s,[(k,round(result[s][k]['mae'],5),round(result[s][k]['pearson'],5),round(result[s][k]['spearman'],5)) for k in ('single_cnn_fusion','ensemble_cnn_fusion')])
if __name__=='__main__':main()
