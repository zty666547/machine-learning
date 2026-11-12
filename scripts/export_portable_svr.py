#!/usr/bin/env python3
"""Fit the fixed RBF SVR on course training rows and verify portable export."""
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
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
from evaluate_strength_calibration import cluster_ids

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--model-output',required=True);p.add_argument('--report',required=True);a=p.parse_args();cp=ROOT/'configs/m4_portable_svr_2026-11-12.toml';ec=tomllib.loads(cp.read_text())['export'];modelpath=ROOT/ec['model_protocol'];priorpath=ROOT/ec['reference_report'];cfg=tomllib.loads(modelpath.read_text());prior=json.loads(priorpath.read_text());data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data'];seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));cl=cluster_ids(ROOT/data['split_file'],len(seq));assert not set(cl[tr])&set(cl[dev]);hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]}
 if hashes!=prior['data_sha256'] or sha(modelpath)!=prior['protocol_sha256'] or np.__version__!=prior['numpy_version'] or sklearn.__version__!=prior['sklearn_version']:raise ValueError('prior data, config or versions differ')
 selected=np.r_[tr,dev];n=len(tr);x=PositionKmerFeaturizer(sequence_length=50).transform(seq[selected]);y=np.log10(strength[selected]);print('Training fixed SVR on',n,'rows',flush=True);pipeline=make_pipeline(StandardScaler(),SVR(**cfg['svr']));pipeline.fit(x[:n],y[:n]);reference=pipeline.predict(x[n:]);mae=float(np.mean(abs(reference-y[n:])));old=prior['models']['svr']['development_validation']['mae'];
 if abs(mae-old)>float(ec['verify_absolute_tolerance']):raise ValueError('Prior SVR development MAE not reproduced')
 portable=PortableRBFSVR.from_pipeline(pipeline);out=Path(a.model_output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);portable.save(out);loaded=PortableRBFSVR.load(out);scores=loaded.predict(x[n:]);diff=float(np.max(abs(scores-reference)));
 if diff>float(ec['verify_absolute_tolerance']):raise ValueError('Portable prediction differs from sklearn reference')
 report={'purpose':'Reproduce fixed SVR and verify portable export; dev reused for consistency only, no selection. Test rows loaded for format checks only. Scores do not measure generated sequence activity.','data_sha256':hashes,'model_protocol_sha256':sha(modelpath),'export_protocol_sha256':sha(cp),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'training_count':n,'verification_count':len(dev),'support_vectors':int(pipeline[-1].support_.size),'feature_count':int(x.shape[1]),'checkpoint_sha256':sha(out),'reference_development_mae':mae,'previous_development_mae':old,'portable_max_absolute_difference':diff,'tolerance':float(ec['verify_absolute_tolerance'])}
 rp=Path(a.report);rp=rp if rp.is_absolute() else ROOT/rp;rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
