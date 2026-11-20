#!/usr/bin/env python3
"""Replay saved reference checkpoints and compare retraining to archived results."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.models import ConditionalSequenceVAE,MotifCNNStrengthPredictor,RidgeStrengthPredictor
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
from promoter_ml.data import PositionKmerFeaturizer
from evaluate_m4_condition_response import score


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.run_dir);cp=ROOT/'configs/mainline_retrain_reference_2026-11-20.toml';cfg=tomllib.loads(cp.read_text())['reproduction'];tol=cfg['model_absolute_tolerance']
    classical=json.loads((folder/'svr_retrained.json').read_text());result=json.loads((folder/'generated_retrained.json').read_text());reference_path=ROOT/'reports/m4_generated_svr_compare_2026-11-12.json';reference=json.loads(reference_path.read_text());candidate=folder/'candidates_retrained.csv';checkpoint=folder/'checkpoints';manifest=json.loads((checkpoint/'manifest.json').read_text());history=json.loads((checkpoint/'training_history.json').read_text())
    if sha(folder/'svr.npz')!=cfg['reference_svr_checkpoint_sha256'] or classical['checkpoint_sha256']!=cfg['reference_svr_checkpoint_sha256']:raise ValueError('Rebuilt SVR checkpoint differs from historical reference')
    if sha(candidate)!=cfg['reference_candidate_sha256'] or result!=reference:raise ValueError('Fresh reference run differs from archived candidates or metrics')
    for name,digest in manifest['file_sha256'].items():
        if Path(name).name!=name or sha(checkpoint/name)!=digest:raise ValueError('Checkpoint manifest mismatch')
    with candidate.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=cfg['records']:raise ValueError('Unexpected record count')
    cnn=MotifCNNStrengthPredictor.load(checkpoint/'motif_cnn.npz');global_model=RidgeStrengthPredictor.load(checkpoint/'global_ridge.npz');position=RidgeStrengthPredictor.load(checkpoint/'position_ridge.npz');weights=np.array([.5,.4,.1]);predictor=(cnn,global_model,position,weights);svr=PortableRBFSVR.load(folder/'svr.npz')
    # Restore inference parameters only. Optimizer state is not saved or resumed.
    with np.load(checkpoint/'conditional_vae.npz',allow_pickle=False) as f:
        vae=ConditionalSequenceVAE(sequence_length=int(f['sequence_length']),hidden_size=int(f['hidden_size']),latent_size=int(f['latent_size']),beta=.2);vae.condition_mean=float(f['condition_mean']);vae.condition_scale=float(f['condition_scale']);metadata={'sequence_length','hidden_size','latent_size','condition_mean','condition_scale'};vae.parameters={k:f[k].copy() for k in f.files if k not in metadata}
    with np.load(checkpoint/'autoregressive_state.npz',allow_pickle=False) as f:
        auto=ContinuousConditionalAutoregressiveGenerator(order=int(f['order']),condition_bins=int(f['condition_bins']),smoothing=float(f['smoothing']));auto.sequence_length=int(f['sequence_length']);auto.bin_centres=f['bin_centres'].copy();auto.fallback_probabilities=f['fallback_probabilities'].copy();auto.context_probabilities={(int(b),int(pos),str(context)):prob.copy() for b,pos,context,prob in zip(f['bin_ids'],f['positions'],f['contexts'],f['context_probabilities'])}
    replay=[];feature=PositionKmerFeaturizer(sequence_length=50)
    for offset,group in enumerate(('weak','medium','strong')):
        for name,model in [('autoregressive',auto),('vae',vae)]:
            records=sorted((r for r in rows if r['generator']==name and r['input_group']==group),key=lambda r:int(r['sample_index']));target=float(records[0]['target_log10']);seed=cfg['reference_sampling_seed']+offset
            samples=model.sample(target,500,seed) if name=='autoregressive' else model.sample(np.full(500,target),seed)
            if samples.tolist()!=[r['sequence'] for r in records]:raise ValueError('Saved generator does not reproduce samples')
            m3,_=score(predictor,samples);sp=svr.predict(feature.transform(samples));m3diff=float(np.max(abs(m3-np.array([float(r['m3_score']) for r in records]))));svrdiff=float(np.max(abs(sp-np.array([float(r['svr_score']) for r in records]))))
            if max(m3diff,svrdiff)>tol:raise ValueError('Saved scorers differ from fresh-run scores')
            replay.append({'generator':name,'group':group,'count':len(records),'sequence_replay_exact':True,'m3_replay_max_absolute_difference':m3diff,'svr_replay_max_absolute_difference':svrdiff});print(name,group,'saved-state replay verified',flush=True)
    files={str(path.relative_to(folder)):sha(path) for path in sorted(folder.rglob('*')) if path.is_file() and path.name!='verification.json'}
    output={'scope':'One fixed historical reference run freshly retrained using original reused-development protocol; inference checkpoints replayed. Not an independent test, new model selection, three-run reproduction or biological validation.', 'config_sha256':sha(cp),'reference_report_sha256':sha(reference_path),'data_sha256':result['data_sha256'],'numpy_replay':np.__version__,'svr_training_numpy':classical['numpy_version'],'svr_training_sklearn':classical['sklearn_version'],'records':len(rows),'candidate_csv_byte_identical':True,'generated_report_exact_match':True,'svr_checkpoint_byte_identical':True,'svr_portable_max_absolute_difference':classical['portable_max_absolute_difference'],'cnn_epochs':len(history['cnn']),'vae_epochs':len(history['vae']),'model_seed':cfg['reference_run_seed'],'sampling_seed':cfg['reference_sampling_seed'],'checkpoint_optimizer_state_saved':False,'replayed_groups':replay,'local_file_sha256':files}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print('Reference training and all saved-state replays match.')


if __name__=='__main__':main()
