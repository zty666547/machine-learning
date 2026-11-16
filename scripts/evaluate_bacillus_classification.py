#!/usr/bin/env python3
"""B. subtilis 81 bp promoter classification baseline, validation only."""
import argparse
import csv
import hashlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.bacillus import aligned_hamming_components,reverse_complement
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--split-output',required=True);p.add_argument('--validation-predictions',required=True);a=p.parse_args()
    configpath=ROOT/'configs/bacillus_classification_baseline_2026-11-16.toml';cfg=tomllib.loads(configpath.read_text())['experiment'];directory=Path(a.data_dir)
    cpath=directory/'classification.csv';spath=directory/'strength.csv';auditpath=directory/'audit.json'
    audit=json.loads(auditpath.read_text());rows=read(cpath);strength=read(spath)
    if audit['classification']['sha256']!='87cae44c1e69668866869511f2a58c930a3b61d6a275c8bca718d0fcdb7ddcf5' or audit['strength']['sha256']!='3cff8a17718cac375a2bf180447fc84e0ff101c304788d625861ea0348b6adea':raise ValueError('Course raw data provenance differs')
    sequences=np.array([r['sequence'] for r in rows]);y=np.array([int(r['label']) for r in rows],dtype=int);source_rows=[int(r['source_row']) for r in rows]
    if len(rows)!=1500 or len(set(sequences.tolist()))!=1500 or set(y)!={0,1} or any(len(s)!=81 or set(s)-set('ACGT') for s in sequences):raise ValueError('Classification data mismatch')
    if len(strength)!=60 or any(len(r['sequence'])!=61 or set(r['sequence'])-set('ACGT') for r in strength):raise ValueError('Strength data mismatch')
    clusters,edges=aligned_hamming_components(sequences,cfg['max_mismatches']);cluster_sizes=np.bincount(clusters)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=cfg['seed']);trainval,test=next(cv.split(np.zeros(len(y)),y,clusters))
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=cfg['seed']+1);tr_rel,dev_rel=next(inner.split(np.zeros(len(trainval)),y[trainval],clusters[trainval]));train=trainval[tr_rel];dev=trainval[dev_rel]
    if set(clusters[train])&set(clusters[dev]) or set(clusters[train])&set(clusters[test]) or set(clusters[dev])&set(clusters[test]):raise ValueError('Clusters cross split')
    split=np.full(len(rows),'',dtype=object);split[train]='train';split[dev]='validation';split[test]='test'
    if any(not x for x in split):raise ValueError('Unassigned row')
    splitpath=Path(a.split_output);splitpath.parent.mkdir(parents=True,exist_ok=True)
    with splitpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['source_row','cluster_id','split']);writer.writerows(zip(source_rows,clusters.tolist(),split.tolist()))
    overlap=[]
    for s in strength:
        query=s['sequence'];reverse=reverse_complement(query)
        hits=[{'source_row':source_rows[i],'split':str(split[i]),'orientation':'forward' if query in x else 'reverse_complement'} for i,x in enumerate(sequences) if query in x or reverse in x]
        if hits:overlap.append({'strength_source_row':int(s['source_row']),'matches':hits})
    model_data={};pred_rows=[]
    for name in cfg['models']:
        if name=='majority_class':
            model=DummyClassifier(strategy='most_frequent');xtrain=np.zeros((len(train),1));xdev=np.zeros((len(dev),1))
        else:
            feat=KmerFeaturizer(1,3,True) if name=='global_kmer_logistic' else PositionKmerFeaturizer(sequence_length=81)
            xtrain=feat.transform(sequences[train]);xdev=feat.transform(sequences[dev]);model=make_pipeline(StandardScaler(),LogisticRegression(C=cfg['logistic_C'],solver=cfg['logistic_solver'],max_iter=cfg['logistic_max_iter'],random_state=cfg['seed']))
        model.fit(xtrain,y[train]);prob=model.predict_proba(xdev)[:,1];pred=(prob>=cfg['threshold']).astype(int);truth=y[dev]
        cm=confusion_matrix(truth,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
        model_data[name]={'validation_count':len(dev),'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'precision':float(precision_score(truth,pred,zero_division=0)),'recall':float(recall_score(truth,pred,zero_division=0)),'specificity':float(tn/(tn+fp)),'f1':float(f1_score(truth,pred,zero_division=0)),'auroc':float(roc_auc_score(truth,prob)),'average_precision':float(average_precision_score(truth,prob)),'confusion_matrix_0_1':cm.tolist(),'feature_count':xtrain.shape[1]}
        pred_rows.extend((source_rows[idx],name,int(y[idx]),float(prob[k]),int(pred[k])) for k,idx in enumerate(dev))
        print(name,json.dumps(model_data[name]),flush=True)
    predpath=Path(a.validation_predictions);predpath.parent.mkdir(parents=True,exist_ok=True)
    with predpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['source_row','model','true_label','positive_probability','predicted_label']);writer.writerows(pred_rows)
    sizes={name:{'count':int(sum(split==name)),'label_0':int(sum((split==name)&(y==0))),'label_1':int(sum((split==name)&(y==1)))} for name in ('train','validation','test')}
    conflict_groups=Counter(r['sequence'] for r in strength)
    result={'protocol':'Species-specific promoter classification; 81 bp aligned-Hamming cluster groups; stratified-group split uses all class labels for stratification, test labels counted and test sequences inspected only for label-free overlap. No test model predictions/metrics. Strength data not used for fitting, labels or evaluation.',
            'classification_csv_sha256':sha(cpath),'strength_csv_sha256':sha(spath),'raw_classification_sha256':audit['classification']['sha256'],'raw_strength_sha256':audit['strength']['sha256'],'config_sha256':sha(configpath),'split_sha256':sha(splitpath),'validation_predictions_sha256':sha(predpath),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
            'clustering':{'count':len(cluster_sizes),'singleton_count':int(sum(cluster_sizes==1)),'largest_size':int(max(cluster_sizes)),'edge_count':edges,'mixed_label_components':int(sum(len(set(y[clusters==i]))>1 for i in range(len(cluster_sizes))))},'splits':sizes,'source_sequence_types':dict(Counter(r['source_seq_type'] for r in rows)),
            'strength_data_audit':{'rows':len(strength),'unique_sequences':len(conflict_groups),'repeated_sequence_count':sum(n>1 for n in conflict_groups.values()),'exact_full_61bp_subsequence_matches':overlap,'note':'Exact containment only, either orientation; not an approximate or coordinate-based independence check.'},'models':model_data}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('clusters',result['clustering'],'splits',sizes,'overlap',len(overlap))


if __name__=='__main__':main()
