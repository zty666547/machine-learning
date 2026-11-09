#!/usr/bin/env python3
"""Reproduce one CNN member for an equal-weight ensemble experiment."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import MotifCNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from train_motif_cnn_predictor import one_hot_sequences

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();cp=ROOT/'configs/m4_cnn_ensemble_2026-11-09.toml';cfg=tomllib.loads(cp.read_text())['experiment'];
 if a.seed not in cfg['seeds'] or cfg['outer_folds']!=3:raise ValueError('seed/fold mismatch')
 c=load_config(ROOT/'configs/m2_motif_cnn_selected.toml');data=c['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(clusters[tr])&set(clusters[dev]);idx=np.r_[tr,dev];n=len(tr);x=one_hot_sequences(seq[idx]);y=np.log10(strength[idx]);layouts,final_layout=fold_layout(clusters[tr],cfg['fold_seed']);oof=np.empty(n)
 hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};prior=json.loads((ROOT/f'reports/m4_rnn_vs_cnn_seed_{a.seed}.json').read_text())
 if prior['data_sha256']!=hashes or prior['fold_seed']!=cfg['fold_seed'] or prior['numpy_version']!=np.__version__:raise ValueError('data or environment differs from prior')
 for i,(fit,early,hold) in enumerate(layouts):
  print('seed',a.seed,'fold',i+1,flush=True);model=MotifCNNStrengthPredictor(seed=a.seed,**c['motif_cnn']);model.fit(x[fit],y[fit],validation=(x[early],y[early]));oof[hold]=model.predict(x[hold])
 fit,early=final_layout;model=MotifCNNStrengthPredictor(seed=a.seed,**c['motif_cnn']);print('seed',a.seed,'final',flush=True);model.fit(x[fit],y[fit],validation=(x[early],y[early]));dp=model.predict(x[n:]);mae=float(np.mean(abs(oof-y[:n])));dmae=float(np.mean(abs(dp-y[n:])))
 if abs(mae-prior['models']['cnn']['training_oof']['mae'])>1e-8 or abs(dmae-prior['models']['cnn']['development_validation']['mae'])>1e-8:raise ValueError('CNN prior MAE mismatch')
 out=Path(a.output_dir);out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f'member_{a.seed}.npz',oof=oof,dev=dp,train_idx=tr,dev_idx=dev,sequence_sha256=hashes['sequence'],strength_sha256=hashes['strength'],split_sha256=hashes['split'],seed=a.seed,protocol_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),numpy_version=np.__version__);model.save(out/f'cnn_{a.seed}.npz');print('baseline reproduced',mae,dmae)
if __name__=='__main__':main()
