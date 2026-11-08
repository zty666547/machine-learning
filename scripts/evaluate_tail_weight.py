#!/usr/bin/env python3
"""Evaluate tail-weighted RNN using the frozen cluster-aware folds."""
from __future__ import annotations
import argparse, hashlib, json, sys, tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import RNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from analyze_frozen_motif_cnn import split_diagnostic
from train_motif_cnn_predictor import one_hot_sequences

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint-dir');a=p.parse_args()
 cp=ROOT/'configs/m4_tail_weight_2026-11-08.toml'; protocol=tomllib.loads(cp.read_text()); e=protocol['experiment']; model_cfg=protocol['tail_weighted_rnn']
 if a.seed not in e['seeds'] or e['outer_folds']!=3: raise ValueError('Seed/folds differ from frozen protocol')
 cfg=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data']; seq,strength=load_promoter_arrays(a.data_dir,cfg['sequence_file'],cfg['label_file'],50)
 train,val,_=load_fixed_split_indices(ROOT/cfg['split_file'],len(seq)); clusters=cluster_ids(ROOT/cfg['split_file'],len(seq))
 if set(clusters[train])&set(clusters[val]): raise ValueError('Cluster crosses fixed train/validation split')
 selected=np.concatenate([train,val]); n=len(train); x=one_hot_sequences(seq[selected]); y=np.log10(strength[selected]); layouts,final_layout=fold_layout(clusters[train],e['fold_seed']); bounds=np.quantile(y[:n],[1/3,2/3])
 data_sha={'sequence':hashlib.sha256((Path(a.data_dir)/cfg['sequence_file']).read_bytes()).hexdigest(),'strength':hashlib.sha256((Path(a.data_dir)/cfg['label_file']).read_bytes()).hexdigest(),'split':hashlib.sha256((ROOT/cfg['split_file']).read_bytes()).hexdigest()}
 previous=ROOT/f'reports/m4_rnn_vs_cnn_seed_{a.seed}.json'; baseline=json.loads(previous.read_text())
 if baseline['fold_seed']!=e['fold_seed'] or baseline['seed']!=a.seed or baseline['data_sha256']!=data_sha: raise ValueError('Baseline data/seed/fold mismatch')
 result={'protocol':'Only training loss changes from the paired vanilla RNN: MSE is weighted by 1+0.75*absolute standardized log target. Same architecture, cluster folds, optimizer, seeds and MAE-based early stopping. Development validation was previously reused. Test rows are format-checked by the data loader only; no test metrics/features.','seed':a.seed,'fold_seed':e['fold_seed'],'train_count':n,'validation_count':len(val),'data_sha256':data_sha,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'training_boundaries':bounds.tolist(),'baseline_rnn':baseline['models']['rnn'],'model':{}}
 def create(): return RNNStrengthPredictor(seed=a.seed,**{k:v for k,v in model_cfg.items() if k!='weight_rule'})
 oof=np.empty(n);folds=[]
 for k,(fit,early,holdout) in enumerate(layouts):
  print(f'seed={a.seed} weighted_rnn fold={k+1}/3',flush=True);m=create();m.fit(x[fit],y[fit],validation=(x[early],y[early]));oof[holdout]=m.predict(x[holdout]);folds.append({'fit_count':len(fit),'early_stop_count':len(early),'holdout_count':len(holdout),'epochs_run':len(m.history),'parameter_count':sum(v.size for v in m.parameters.values()),'clipped_batches':sum(v['clipped_batches'] for v in m.history)})
 fit,early=final_layout;m=create();print(f'seed={a.seed} weighted_rnn final/development',flush=True);m.fit(x[fit],y[fit],validation=(x[early],y[early]))
 result['model']={'training_oof':split_diagnostic(y[:n],oof,bounds),'folds':folds,'development_validation':split_diagnostic(y[n:],m.predict(x[n:]),bounds),'final_fit':{'fit_count':len(fit),'early_stop_count':len(early),'epochs_run':len(m.history)}}
 if a.checkpoint_dir:
  d=Path(a.checkpoint_dir); d=d if d.is_absolute() else ROOT/d;d.mkdir(parents=True,exist_ok=True);m.save(d/f'tail_weighted_rnn_seed_{a.seed}.npz')
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'mae':result['model']['training_oof']['mae'],'spearman':result['model']['training_oof']['spearman'],'weak_mae':result['model']['training_oof']['by_true_strength_group']['weak']['mae'],'strong_mae':result['model']['training_oof']['by_true_strength_group']['strong']['mae']},indent=2))
if __name__=='__main__': main()
