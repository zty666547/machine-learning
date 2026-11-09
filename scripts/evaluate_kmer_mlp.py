#!/usr/bin/env python3
"""Paired fit-only ridge versus k-mer MLP with cluster-aware folds."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import PositionKmerFeaturizer,load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import PositionMLPStrengthPredictor,RidgeStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_kmer_mlp_2026-11-09.toml';cfg=tomllib.loads(cp.read_text());ex=cfg['experiment'];
 if a.seed not in ex['seeds'] or ex['outer_folds']!=3:raise ValueError('protocol mismatch')
 data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);idx=np.r_[tr,dev];n=len(tr);x=PositionKmerFeaturizer(sequence_length=50).transform(seq[idx]);y=np.log10(strength[idx]);bounds=np.quantile(y[:n],[1/3,2/3]);layouts,final=fold_layout(clusters[tr],ex['fold_seed'])
 hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};prior=json.loads((ROOT/f'reports/m4_rnn_vs_cnn_seed_{a.seed}.json').read_text())
 if prior['data_sha256']!=hashes or prior['fold_seed']!=ex['fold_seed'] or prior['numpy_version']!=np.__version__:raise ValueError('prior protocol differs')
 result={'protocol':'Same features and fit-only rows for ridge and MLP; MLP uses inner early-stop, ridge has no early-stop. Fold-aware feature/label scaling fitted only on fit rows. Development previously reused. Test loaded for format checks only.','seed':a.seed,'fold_seed':ex['fold_seed'],'data_sha256':hashes,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'train_count':n,'validation_count':len(dev),'models':{}}
 for name in ('ridge','mlp'):
  pred=np.empty(n);folds=[]
  def make():return RidgeStrengthPredictor(alpha=cfg['ridge']['alpha']) if name=='ridge' else PositionMLPStrengthPredictor(seed=a.seed,**cfg['mlp'])
  for i,(fit,early,hold) in enumerate(layouts):
   print(a.seed,name,i+1,flush=True);assert not set(clusters[tr][fit])&set(clusters[tr][hold]);model=make();model.fit(x[fit],y[fit],validation=(x[early],y[early])) if name=='mlp' else model.fit(x[fit],y[fit]);pred[hold]=model.predict(x[hold]);folds.append({'fit_count':len(fit),'early_stop_count':len(early) if name=='mlp' else 0,'holdout_count':len(hold),'epochs_run':len(model.history) if name=='mlp' else None})
  fit,early=final;model=make();model.fit(x[fit],y[fit],validation=(x[early],y[early])) if name=='mlp' else model.fit(x[fit],y[fit]);result['models'][name]={'training_oof':split_diagnostic(y[:n],pred,bounds),'development_validation':split_diagnostic(y[n:],model.predict(x[n:]),bounds),'folds':folds}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:round(v['training_oof']['mae'],5) for k,v in result['models'].items()})
if __name__=='__main__':main()
