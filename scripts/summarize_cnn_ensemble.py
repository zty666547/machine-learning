#!/usr/bin/env python3
"""Combine verified paired CNN predictions without selecting on development data."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from analyze_frozen_motif_cnn import split_diagnostic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_cnn_ensemble_2026-11-09.toml';cfg=tomllib.loads(cp.read_text())['experiment'];data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));y=np.log10(strength);bounds=np.quantile(y[tr],[1/3,2/3]);preds=[];devpred=[];directory=Path(a.input_dir);directory=directory if directory.is_absolute() else ROOT/directory
 hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};ph=hashlib.sha256(cp.read_bytes()).hexdigest()
 for seed in cfg['seeds']:
  with np.load(directory/f'member_{seed}.npz',allow_pickle=False) as f:
   if int(f['seed'])!=seed or str(f['protocol_sha256'])!=ph or str(f['numpy_version'])!=np.__version__ or not np.array_equal(f['train_idx'],tr) or not np.array_equal(f['dev_idx'],dev) or any(str(f[k+'_sha256'])!=hashes[k] for k in hashes):raise ValueError('member provenance mismatch')
   preds.append(f['oof'].copy());devpred.append(f['dev'].copy())
 preds=np.stack(preds);devpred=np.stack(devpred);r={'protocol':'Fixed equal mean of three same-fold CNNs in log10 label scale. Training internal folds and early stopping match member baselines. Development repeatedly reused; test only format-checked. No evaluator replacement.','data_sha256':hashes,'protocol_sha256':ph,'numpy_version':np.__version__,'seeds':cfg['seeds'],'fold_seed':cfg['fold_seed'],'train_count':len(tr),'validation_count':len(dev),'training_oof':{},'development_validation':{}}
 for split,ys,pred in [('training_oof',y[tr],preds),('development_validation',y[dev],devpred)]:
  members=[split_diagnostic(ys,z,bounds) for z in pred];combined=split_diagnostic(ys,pred.mean(axis=0),bounds);r[split]={'ensemble':combined,'individual':members,'mean_member_mae':float(np.mean([m['mae'] for m in members])),'mean_member_pearson':float(np.mean([m['pearson'] for m in members])),'mean_member_spearman':float(np.mean([m['spearman'] for m in members])),'mean_member_prediction_std_ratio':float(np.mean([m['prediction_to_target_std_ratio'] for m in members])),'mean_member_disagreement_std':float(pred.std(axis=0).mean())}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');
 for split in ('training_oof','development_validation'):
  v=r[split];e=v['ensemble'];print(split,'members',v['mean_member_mae'],v['mean_member_pearson'],v['mean_member_spearman'],'ensemble',e['mae'],e['pearson'],e['spearman'],'disagreement',v['mean_member_disagreement_std'])
if __name__=='__main__':main()
