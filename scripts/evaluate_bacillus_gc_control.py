#!/usr/bin/env python3
"""Post-hoc GC-only control on the existing Bacillus validation split."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--validation-predictions',required=True);a=p.parse_args()
    config=ROOT/'configs/bacillus_gc_control_2026-11-16.toml';cfg=tomllib.loads(config.read_text())['experiment'];source=Path(a.data_dir)/'classification.csv';split=ROOT/'data/splits/bacillus_classification_split_2026-11-16.csv';prior=json.loads((ROOT/'reports/bacillus_classification_baseline_2026-11-16.json').read_text())
    if sha(source)!=prior['classification_csv_sha256'] or sha(split)!=prior['split_sha256']:raise ValueError('Data or split mismatch')
    data={int(r['source_row']):r for r in read(source)};sr=read(split);train=[data[int(z['source_row'])] for z in sr if z['split']=='train'];dev=[data[int(z['source_row'])] for z in sr if z['split']=='validation']
    def features(rows):return np.array([[(r['sequence'].count('G')+r['sequence'].count('C'))/len(r['sequence'])] for r in rows])
    xt=features(train);xv=features(dev);yt=np.array([int(r['label']) for r in train]);yv=np.array([int(r['label']) for r in dev]);model=make_pipeline(StandardScaler(),LogisticRegression(C=cfg['logistic_C'],solver=cfg['logistic_solver'],max_iter=cfg['logistic_max_iter'],random_state=cfg['seed']));model.fit(xt,yt);probs=model.predict_proba(xv)[:,1];pred=(probs>=cfg['threshold']).astype(int);cm=confusion_matrix(yv,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    metrics={'accuracy':float(accuracy_score(yv,pred)),'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'precision':float(precision_score(yv,pred,zero_division=0)),'recall':float(recall_score(yv,pred,zero_division=0)),'specificity':float(tn/(tn+fp)),'f1':float(f1_score(yv,pred,zero_division=0)),'auroc':float(roc_auc_score(yv,probs)),'average_precision':float(average_precision_score(yv,probs)),'confusion_matrix_0_1':cm.tolist()}
    predpath=Path(a.validation_predictions);predpath.parent.mkdir(parents=True,exist_ok=True)
    with predpath.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['source_row','true_label','gc_fraction','positive_probability','predicted_label']);w.writerows((int(row['source_row']),int(yv[i]),float(xv[i,0]),float(probs[i]),int(pred[i])) for i,row in enumerate(dev))
    result={'protocol':'Post-hoc GC-only control; fit train only, evaluate reused validation only, no test GC or predictions. Not evidence for cause or external generalization.', 'config_sha256':sha(config),'classification_sha256':sha(source),'split_sha256':sha(split),'predictions_sha256':sha(predpath),'validation_metrics':metrics,'gc_by_split_and_label':{name:{str(k):{'count':int(sum(y==k)),'mean':float(x[y==k,0].mean())} for k in (0,1)} for name,x,y in [('train',xt,yt),('validation',xv,yv)]}}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
