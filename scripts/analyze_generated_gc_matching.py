#!/usr/bin/env python3
"""Compare generated weak/strong scores after exact GC-count matching."""
import argparse
import csv
import hashlib
import json
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--selected-csv', required=True)
    args = parser.parse_args()
    config_path = ROOT/'configs/m4_generated_gc_matching_2026-11-16.toml'
    cfg = tomllib.loads(config_path.read_text())['experiment']
    if sha(args.ranking_csv) != cfg['input_sha256']:
        raise ValueError('Generated sequence/score file checksum differs from fixed input')
    previous_path = ROOT/'reports/m4_unified_generated_2026-11-13.json'
    prior = json.loads(previous_path.read_text())
    if prior['candidate_csv_sha256'] != cfg['input_sha256']:
        raise ValueError('Prior generated-data summary input mismatch')
    with open(args.ranking_csv, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9000:
        raise ValueError('Expected 9000 previously generated records')
    selections = []; runs = []
    for generator_id, generator in enumerate(cfg['generators']):
        for seed in cfg['seeds']:
            groups = {}
            for group in cfg['groups']:
                chosen = sorted((r for r in rows if r['generator']==generator and int(r['vae_seed'])==seed and r['input_group']==group), key=lambda r:int(r['sample_index']))
                if len(chosen) != cfg['expected_per_group'] or [int(r['sample_index']) for r in chosen] != list(range(cfg['expected_per_group'])):
                    raise ValueError('Wrong group size or duplicate/missing sample index')
                for row in chosen:
                    seq = row['sequence']
                    if len(seq)!=50 or set(seq)-set('ACGT') or not all(np.isfinite(float(row[s+'_score'])) for s in ('m3','svr')):
                        raise ValueError('Invalid sequence or score')
                    row['gc_count'] = seq.count('G') + seq.count('C')
                ref = next(r for r in prior['runs'] if r['generator']==generator and r['seed']==seed and r['group']==group)
                for scorer in ('m3','svr'):
                    if abs(np.mean([float(r[scorer+'_score']) for r in chosen])-ref[scorer+'_score_mean']) > 1e-8:
                        raise ValueError('Existing unadjusted group score does not reproduce')
                groups[group] = chosen
            selected = {'weak':[], 'strong':[]}; strata = []
            for gc in range(51):
                pools = {g:[r for r in groups[g] if r['gc_count']==gc] for g in cfg['groups']}
                n = min(len(pools['weak']),len(pools['strong']))
                if n == 0:
                    continue
                strata.append({'gc_count':gc, 'weak_available':len(pools['weak']), 'strong_available':len(pools['strong']), 'selected_per_group':n})
                for group_id, group in enumerate(cfg['groups']):
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['match_random_seed'],seed,generator_id,gc,group_id]))
                    indices=np.sort(rng.choice(len(pools[group]),size=n,replace=False))
                    selected[group].extend(pools[group][int(i)] for i in indices)
            if not selected['weak'] or Counter(r['gc_count'] for r in selected['weak']) != Counter(r['gc_count'] for r in selected['strong']):
                raise ValueError('GC matching failed')
            run={'generator':generator,'seed':seed,'available_per_group':cfg['expected_per_group'],
                 'matched_per_group':len(selected['weak']),'gc_strata':strata,'scores':{}}
            for scorer in ('m3','svr'):
                result={}
                for prefix, records in [('all',groups),('matched',selected)]:
                    for group in cfg['groups']:
                        result[prefix+'_'+group+'_mean'] = float(np.mean([float(r[scorer+'_score']) for r in records[group]]))
                    result[prefix+'_strong_minus_weak']=result[prefix+'_strong_mean']-result[prefix+'_weak_mean']
                run['scores'][scorer]=result
            for group in cfg['groups']:
                for r in selected[group]:
                    selections.append([generator,seed,group,int(r['sample_index']),r['gc_count'],float(r['m3_score']),float(r['svr_score'])])
            runs.append(run)
            print(generator,seed,'matched',len(selected['weak']), 'M3',round(run['scores']['m3']['matched_strong_minus_weak'],5),'SVR',round(run['scores']['svr']['matched_strong_minus_weak'],5))
    path=Path(args.selected_csv);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['generator','seed','group','sample_index','gc_count','m3_score','svr_score']);writer.writerows(selections)
    summary=[]
    for generator in cfg['generators']:
        matching=[r for r in runs if r['generator']==generator]
        summary.append({'generator':generator,'matched_counts':[r['matched_per_group'] for r in matching],
                        'scores':{s:{k:float(np.mean([r['scores'][s][k] for r in matching])) for k in ('all_strong_minus_weak','matched_strong_minus_weak')} for s in ('m3','svr')}})
    output={'protocol':'Post-hoc GC-matched descriptive comparison of previously inspected synthetic candidates and same-data-trained scores; no real labels or causal inference. Samples matched without using scores.',
            'config_sha256':sha(config_path),'candidate_csv_sha256':sha(args.ranking_csv),'prior_summary_sha256':sha(previous_path),'selected_csv_sha256':sha(path),'numpy_version':np.__version__,'runs':runs,'summary':summary}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
