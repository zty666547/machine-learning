#!/usr/bin/env python3
"""Audit CNN OOF errors versus distance to outer training sequences."""
import argparse,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.config import load_config
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.models import MotifCNNStrengthPredictor
from evaluate_strength_calibration import cluster_ids
from evaluate_cnn_position_bins import fold_layout
from audit_sequence_neighbors import nearest
from train_motif_cnn_predictor import one_hot_sequences

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();cp=ROOT/'configs/m4_distance_error_2026-11-09.toml';cfg=tomllib.loads(cp.read_text())['experiment']
 if a.seed not in cfg['seeds'] or cfg['outer_folds']!=3:raise ValueError('seed/folds mismatch')
 c=load_config(ROOT/'configs/m2_motif_cnn_selected.toml');data=c['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,_,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq))[tr];x=one_hot_sequences(seq[tr]);y=np.log10(strength[tr]);bounds=np.quantile(y,[1/3,2/3]);groups=np.digitize(y,bounds);layouts,_=fold_layout(clusters,cfg['fold_seed']);pred=np.empty(len(tr));dist=np.empty(len(tr),int)
 hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};priorpath=ROOT/f'reports/m4_rnn_vs_cnn_seed_{a.seed}.json';prior=json.loads(priorpath.read_text())
 if prior['data_sha256']!=hashes or prior['fold_seed']!=cfg['fold_seed'] or prior['numpy_version']!=np.__version__:raise ValueError('prior data or environment mismatch')
 for i,(fit,early,hold) in enumerate(layouts):
  print('seed',a.seed,'fold',i+1,flush=True);ref=np.sort(np.r_[fit,early]);assert not set(clusters[ref])&set(clusters[hold]);_,dist[hold],_=nearest(x[ref],y[ref],x[hold]);model=MotifCNNStrengthPredictor(seed=a.seed,**c['motif_cnn']);model.fit(x[fit],y[fit],validation=(x[early],y[early]));pred[hold]=model.predict(x[hold])
 err=np.abs(pred-y);mae=float(err.mean());old=prior['models']['cnn']['training_oof']['mae'];
 if abs(mae-old)>1e-8:raise ValueError(f'CNN baseline MAE differs: {mae} vs {old}')
 bins=[]
 for lo,hi in cfg['distance_bins']:
  m=(dist>=lo)&(dist<=hi);count=int(m.sum());g=[]
  for j,name in enumerate(('weak','medium','strong')):
   z=m&(groups==j);g.append({'group':name,'count':int(z.sum()),'mae':float(err[z].mean()) if z.any() else None})
  bins.append({'distance':[lo,hi],'count':count,'mae':float(err[m].mean()) if count else None,'mean_target':float(y[m].mean()) if count else None,'strong_fraction':float(np.mean(groups[m]==2)) if count else None,'by_strength':g})
 r={'protocol':'OOF CNN rerun, nearest aligned 50 bp distance to entire outer training fold including fit/early; distance is not causality or biological similarity. Development/test labels unused.','seed':a.seed,'fold_seed':cfg['fold_seed'],'data_sha256':hashes,'protocol_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'numpy_version':np.__version__,'train_count':len(tr),'baseline_mae_reproduced':True,'oof_mae':mae,'distance_absolute_error_pearson':float(np.corrcoef(dist,err)[0,1]),'bins':bins}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'seed':a.seed,'mae':mae,'corr':r['distance_absolute_error_pearson'],'bins':bins},indent=2))
if __name__=='__main__':main()
