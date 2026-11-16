#!/usr/bin/env python3
"""Compare uniform and position-based priors for continuous autoregression."""
import argparse
import csv
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.generation_position_prior import PositionPriorAutoregressiveGenerator
from promoter_ml.structure_analysis import encode_sequences,position_js,position_kmer_frequency


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_nll(model,sequences,targets):
    """Teacher-forced sequence NLL (bits/base), with each known target as condition."""
    index={base:i for i,base in enumerate('ACGT')};results=[]
    for sequence,target in zip(sequences,targets):
        lower,upper,weight=model._nearest_bins(float(target));padded='^'*model.order+sequence;total=0.0
        for pos,base in enumerate(sequence):
            context=padded[pos:pos+model.order]
            a=model.context_probabilities.get((lower,pos,context),model.fallback_probabilities[lower,pos])
            b=model.context_probabilities.get((upper,pos,context),model.fallback_probabilities[upper,pos])
            probability=(1-weight)*a[index[base]]+weight*b[index[base]]
            total-=np.log2(probability)
        results.append(total/len(sequence))
    return np.asarray(results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--output',required=True);parser.add_argument('--samples',required=True);parser.add_argument('--nll-csv',required=True);args=parser.parse_args()
    configpath=ROOT/'configs/m4_ar_position_prior_2026-11-16.toml';cfg=tomllib.loads(configpath.read_text())['experiment'];data=Path(args.data_dir);split=ROOT/'data/splits/ecoli_similarity_split.csv'
    if abs(cfg['prior_total_mass']-4*cfg['smoothing_per_base'])>1e-12:
        raise ValueError('Prior masses differ')
    # The source files and split are recorded; match the already validated project dataset.
    hashes={'sequence':sha(data/'promoter.npy'),'strength':sha(data/'gene_expression.npy'),'split':sha(split)}
    existing=json.loads((ROOT/'reports/m4_rbf_svr_2026-11-09.json').read_text())
    if hashes != existing['data_sha256']:
        raise ValueError('Course data or split mismatch')
    seq,strength=load_promoter_arrays(data,'promoter.npy','gene_expression.npy',50);train,val,_=load_fixed_split_indices(split,len(seq));y=np.log10(strength);train_y=y[train];val_y=y[val]
    bounds=np.quantile(train_y,[1/3,2/3]);train_groups=np.digitize(train_y,bounds);val_groups=np.digitize(val_y,bounds)
    models={}
    for name,cls in [('uniform',ContinuousConditionalAutoregressiveGenerator),('position_prior',PositionPriorAutoregressiveGenerator)]:
        model=cls(order=cfg['order'],condition_bins=cfg['condition_bins'],smoothing=cfg['smoothing_per_base']).fit(seq[train],train_y);models[name]=model
        print('fit',name,flush=True)
    if not np.array_equal(models['uniform'].bin_centres,models['position_prior'].bin_centres) or not np.array_equal(models['uniform'].fallback_probabilities,models['position_prior'].fallback_probabilities):
        raise ValueError('Conditions or fallback changed')
    nll={name:conditional_nll(model,seq[val],val_y) for name,model in models.items()}
    groups=('weak','medium','strong');summary={'nll_bits_per_base':{name:float(v.mean()) for name,v in nll.items()},'nll_by_group':{},'target_log10':{},'sampling':[]}
    for k,g in enumerate(groups):
        summary['nll_by_group'][g]={name:float(v[val_groups==k].mean()) for name,v in nll.items()}
        summary['target_log10'][g]=float(train_y[train_groups==k].mean())
    output_rows=[];run_results=[];train_set=set(seq[train].tolist())
    for name,model in models.items():
        for k,g in enumerate(groups):
            target=summary['target_log10'][g];reference=position_kmer_frequency(encode_sequences(seq[val][val_groups==k]),3)
            for seed in cfg['sampling_seeds']:
                generated=model.sample(target,cfg['samples_per_group'],seed+k)
                enc=encode_sequences(generated);gc=float(np.mean((enc==1)|(enc==2)))
                js=float(position_js(reference,position_kmer_frequency(enc,3)).mean())
                result={'model':name,'group':g,'seed':seed,'position_3mer_js_to_validation':js,'gc_fraction':gc,'unique_fraction':len(set(generated.tolist()))/len(generated),'exact_novel_fraction':float(np.mean([s not in train_set for s in generated]))}
                run_results.append(result);output_rows.extend((name,g,seed,i,s) for i,s in enumerate(generated.tolist()))
                print('sample',name,g,seed,'JS',round(js,6),flush=True)
    for name in models:
        for g in groups:
            group=[r for r in run_results if r['model']==name and r['group']==g]
            summary['sampling'].append({'model':name,'group':g,**{key:float(np.mean([r[key] for r in group])) for key in ('position_3mer_js_to_validation','gc_fraction','unique_fraction','exact_novel_fraction')}})
    samples=Path(args.samples);samples.parent.mkdir(parents=True,exist_ok=True)
    with samples.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['model','group','seed','sample_index','sequence']);writer.writerows(output_rows)
    nllpath=Path(args.nll_csv);nllpath.parent.mkdir(parents=True,exist_ok=True)
    with nllpath.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_index','uniform_nll_bits_per_base','position_prior_nll_bits_per_base']);writer.writerows(zip(val.tolist(),nll['uniform'].tolist(),nll['position_prior'].tolist()))
    result={'protocol':'Conditional NLL and sampled sequence structure on repeatedly used validation data; no new independent test or biological validation. Training-derived target used as condition; observed validation target used for teacher-forced NLL only.',
            'data_sha256':hashes,'config_sha256':sha(configpath),'samples_sha256':sha(samples),'nll_csv_sha256':sha(nllpath),'numpy_version':np.__version__,'train_count':len(train),'validation_count':len(val),'validation_outside_bin_centres_count':int(np.sum((val_y<models['uniform'].bin_centres[0])|(val_y>models['uniform'].bin_centres[-1]))),'bin_centres':models['uniform'].bin_centres.tolist(),'run_results':run_results,'summary':summary}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
