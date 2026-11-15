#!/usr/bin/env python3
"""Retrospective target-selection audit on cluster-separated training folds."""
import argparse,csv,hashlib,json,sys,tomllib
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
from generate_conditional_position_baseline import assign_strength_groups
from analyze_target_ranking import rank_target

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--cnn-member',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-csv',required=True);a=p.parse_args();cp=ROOT/'configs/m4_target_oof_audit_2026-11-15.toml';ex=tomllib.loads(cp.read_text())['experiment'];scfgpath=ROOT/'configs/m4_rbf_svr_2026-11-09.toml';scfg=tomllib.loads(scfgpath.read_text());ecp=ROOT/'configs/m4_cnn_ensemble_2026-11-09.toml';m3=tomllib.loads((ROOT/'configs/m3_screening_evaluator.toml').read_text())['screening_evaluator'];w=np.array([m3['motif_cnn_weight'],m3['global_kmer_ridge_weight'],m3['position_ridge_weight']]);data=load_config(ROOT/'configs/m2_motif_cnn_selected.toml')['data']
 if ex['outer_folds']!=3 or ex['fold_seed']!=scfg['experiment']['fold_seed'] or not np.allclose(w,[.5,.4,.1]):raise ValueError('protocol mismatch')
 seq,strength=load_promoter_arrays(a.data_dir,data['sequence_file'],data['label_file'],50);tr,dev,_=load_fixed_split_indices(ROOT/data['split_file'],len(seq));clusters=cluster_ids(ROOT/data['split_file'],len(seq))[tr];xseq=seq[tr];y=np.log10(strength[tr]);layouts,_=fold_layout(clusters,ex['fold_seed']);hashes={k:sha(v) for k,v in [('sequence',Path(a.data_dir)/data['sequence_file']),('strength',Path(a.data_dir)/data['label_file']),('split',ROOT/data['split_file'])]};prior=json.loads((ROOT/'reports/m4_rbf_svr_2026-11-09.json').read_text());cnnprior=json.loads((ROOT/f"reports/m4_rnn_vs_cnn_seed_{ex['cnn_seed']}.json").read_text());fusionprior=json.loads((ROOT/'reports/m4_fusion_cnn_ensemble_2026-11-09.json').read_text());member=Path(a.cnn_member)
 if hashes!=prior['data_sha256'] or hashes!=cnnprior['data_sha256'] or prior['protocol_sha256']!=sha(scfgpath) or prior['numpy_version']!=np.__version__ or prior['sklearn_version']!=sklearn.__version__:raise ValueError('prior provenance or versions mismatch')
 with np.load(member,allow_pickle=False) as f:
  if int(f['seed'])!=ex['cnn_seed'] or str(f['protocol_sha256'])!=sha(ecp) or str(f['numpy_version'])!=cnnprior['numpy_version'] or not np.array_equal(f['train_idx'],tr) or not np.array_equal(f['dev_idx'],dev) or any(str(f[k+'_sha256'])!=hashes[k] for k in hashes):raise ValueError('CNN member mismatch')
  cnn=f['oof'].copy()
 if abs(np.mean(abs(cnn-y))-cnnprior['models']['cnn']['training_oof']['mae'])>1e-8:raise ValueError('prior CNN OOF differs')
 gx=KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(xseq);px=PositionKmerFeaturizer(sequence_length=50).transform(xseq);m3=np.empty(len(tr));sv=np.empty(len(tr));folds=[];selections=[];k=ex['top_k']
 for fold,(fit,early,hold) in enumerate(layouts):
  ref=np.sort(np.r_[fit,early]);assert not set(clusters[ref])&set(clusters[hold]);print('fold',fold+1,flush=True);global_model=RidgeStrengthPredictor(alpha=10).fit(gx[ref],y[ref]);position_model=RidgeStrengthPredictor(alpha=10).fit(px[ref],y[ref]);model=make_pipeline(StandardScaler(),SVR(**scfg['svr']));model.fit(px[ref],y[ref]);m3[hold]=w[0]*cnn[hold]+w[1]*global_model.predict(gx[hold])+w[2]*position_model.predict(px[hold]);sv[hold]=model.predict(px[hold]);labels,_=assign_strength_groups(y[ref]);targets={g:float(y[ref][labels==g].mean()) for g in ex['groups']};summary={'fold':fold+1,'fit_count':len(fit),'early_stop_count':len(early),'holdout_count':len(hold),'groups':[]}
  for group in ex['groups']:
   target=targets[group];true_gap=abs(y[hold]-target);item={'group':group,'target_log10':target,'random_pool_mean_true_distance':float(true_gap.mean()),'oracle_topk_true_distance':float(np.sort(true_gap,kind='stable')[:k].mean()),'models':{}};selected={}
   for name,scores in [('m3',m3[hold]),('svr',sv[hold])]:
    ranks=rank_target(scores,target);chosen=np.flatnonzero(ranks<=k);selected[name]=set(chosen.tolist());item['models'][name]={'mean_true_distance_to_target':float(true_gap[chosen].mean()),'median_true_distance_to_target':float(np.median(true_gap[chosen])),'mean_predicted_distance_to_target':float(np.mean(abs(scores[chosen]-target))),'selected_true_mean':float(y[hold][chosen].mean())}
    for i in chosen:selections.append([fold+1,group,name,int(tr[hold[i]]),target,float(y[hold[i]]),float(scores[i]),float(true_gap[i]),int(ranks[i])])
   item['top_k_overlap_count']=len(selected['m3']&selected['svr']);summary['groups'].append(item)
  folds.append(summary)
 if abs(np.mean(abs(sv-y))-prior['models']['svr']['training_oof']['mae'])>1e-8 or abs(np.mean(abs(m3-y))-fusionprior['training_oof']['single_cnn_fusion']['mae'])>1e-8:raise ValueError('previous OOF predictor metrics differ')
 selected_path=Path(a.selected_csv);selected_path=selected_path if selected_path.is_absolute() else ROOT/selected_path;selected_path.parent.mkdir(parents=True,exist_ok=True)
 with selected_path.open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f);writer.writerow(['fold','target_group','scorer','course_sample_index','target_log10','observed_log10','predicted_log10','true_absolute_target_distance','target_rank']);writer.writerows(selections)
 summary=[]
 for group in ex['groups']:
  g=[next(z for z in f['groups'] if z['group']==group) for f in folds];summary.append({'group':group,'targets':[z['target_log10'] for z in g],'random_mean':float(np.mean([z['random_pool_mean_true_distance'] for z in g])),'oracle_mean':float(np.mean([z['oracle_topk_true_distance'] for z in g])),'m3_mean':float(np.mean([z['models']['m3']['mean_true_distance_to_target'] for z in g])),'svr_mean':float(np.mean([z['models']['svr']['mean_true_distance_to_target'] for z in g])),'m3_predicted_gap_mean':float(np.mean([z['models']['m3']['mean_predicted_distance_to_target'] for z in g])),'svr_predicted_gap_mean':float(np.mean([z['models']['svr']['mean_predicted_distance_to_target'] for z in g])),'top_k_overlap_counts':[z['top_k_overlap_count'] for z in g]})
 result={'protocol':'Training-only cluster-separated OOF; targets estimated only on outer training labels; CNN fit/early within outer train; ridge/SVR fit full outer train. Model and data used in earlier development, so not independent final testing. Test/dev only loaded for format checks; oracle hindsight only.','data_sha256':hashes,'protocol_sha256':sha(cp),'svr_protocol_sha256':sha(scfgpath),'cnn_member_sha256':sha(member),'selected_csv_sha256':sha(selected_path),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'cnn_numpy_version':cnnprior['numpy_version'],'train_count':len(tr),'folds':folds,'summary':summary,'oof_m3_mae_reproduced':float(np.mean(abs(m3-y))),'oof_svr_mae_reproduced':float(np.mean(abs(sv-y)))}
 out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for z in summary:print(z['group'],'random',round(z['random_mean'],4),'oracle',round(z['oracle_mean'],4),'M3',round(z['m3_mean'],4),'SVR',round(z['svr_mean'],4),'pred gaps',round(z['m3_predicted_gap_mean'],4),round(z['svr_predicted_gap_mean'],4),'overlap',z['top_k_overlap_counts'])
if __name__=='__main__':main()
