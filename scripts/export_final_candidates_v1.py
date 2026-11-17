#!/usr/bin/env python3
"""Export a frozen, auditable computational shortlist from existing candidates."""
import argparse,csv,hashlib,json,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.candidate_selection import greedy_target_selection,longest_homopolymer
from promoter_ml.data import load_fixed_split_indices,load_promoter_arrays
from promoter_ml.structure_analysis import encode_sequences,match_window


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)


def nearest_training(query,reference):
    distances=np.empty(len(query),dtype=int);indices=np.empty(len(query),dtype=int)
    for start in range(0,len(query),32):
        d=np.count_nonzero(query[start:start+32,None,:]!=reference[None,:,:],axis=2)
        indices[start:start+len(d)]=d.argmin(axis=1);distances[start:start+len(d)]=d.min(axis=1)
    return distances,indices


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--ranking-csv',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    cp=ROOT/'configs/final_candidates_v1_2026-11-17.toml';cfg=tomllib.loads(cp.read_text())['experiment'];inp=Path(args.ranking_csv);data=Path(args.data_dir);split=ROOT/'data/splits/ecoli_similarity_split.csv'
    if sha(inp)!=cfg['input_sha256']:raise ValueError('Candidate input checksum mismatch')
    priorpath=ROOT/'reports/m4_unified_generated_2026-11-13.json';prior=json.loads(priorpath.read_text());hashes={'sequence':sha(data/'promoter.npy'),'strength':sha(data/'gene_expression.npy'),'split':sha(split)}
    if hashes!=prior['data_sha256'] or prior['candidate_csv_sha256']!=sha(inp):raise ValueError('Source or split mismatch')
    seq,strength=load_promoter_arrays(data,'promoter.npy','gene_expression.npy',50);train,_,_=load_fixed_split_indices(split,len(seq));ty=np.log10(strength[train]);bins=np.digitize(ty,np.quantile(ty,[1/3,2/3]));targets={g:float(ty[bins==i].mean()) for i,g in enumerate(cfg['groups'])}
    with inp.open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    records=[]
    for i,r in enumerate(raw):
        seq0=r['sequence'];record={'generator':r['generator'],'group':r['input_group'],'run_seed':int(r['vae_seed']),'sample_index':int(r['sample_index']),'source_csv_row':i+2,'sequence':seq0,'target_log10':float(r['target_log10']),'m3_score':float(r['m3_score']),'svr_score':float(r['svr_score'])}
        if len(seq0)!=50 or set(seq0)-set('ACGT') or not all(np.isfinite(record[k]) for k in ('target_log10','m3_score','svr_score')):raise ValueError('Invalid sequence or score')
        if record['group'] not in targets or abs(record['target_log10']-targets[record['group']])>1e-12:raise ValueError('Target differs from training definition')
        record['longest_homopolymer']=longest_homopolymer(seq0);record['gc_fraction']=(seq0.count('G')+seq0.count('C'))/50;records.append(record)
    if len(records)!=9000 or len({(r['generator'],r['group'],r['run_seed'],r['sample_index']) for r in records})!=9000:raise ValueError('Missing or repeated records')
    print('Compute nearest training distances for',len(records),'records',flush=True)
    distance,nearest=nearest_training(encode_sequences(np.array([r['sequence'] for r in records])),encode_sequences(seq[train]))
    for i,r in enumerate(records):
        r['nearest_training_sample_index']=int(train[nearest[i]]);r['training_hamming_distance']=int(distance[i]);r['training_identity']=1-distance[i]/50;r['m3_target_gap']=abs(r['m3_score']-r['target_log10']);r['svr_target_gap']=abs(r['svr_score']-r['target_log10']);r['scorer_difference']=abs(r['m3_score']-r['svr_score']);r['eligible']=int(distance[i]>=cfg['minimum_training_hamming_distance'] and r['longest_homopolymer']<=cfg['max_homopolymer_run'])
    selected=[];buckets=[]
    pairs=tomllib.loads((ROOT/'configs/m4_target_repeat_2026-11-13.toml').read_text())['experiment'];sample_seeds=dict([pairs['reference_run']]+pairs['run_pairs'])
    for generator in cfg['generators']:
        for group_index,group in enumerate(cfg['groups']):
            for seed in cfg['seeds']:
                bucket=[r for r in records if (r['generator'],r['group'],r['run_seed'])==(generator,group,seed)]
                if len(bucket)!=cfg['expected_records_per_bucket'] or sorted(r['sample_index'] for r in bucket)!=list(range(500)):raise ValueError('Bucket mismatch')
                eligible=[r for r in bucket if r['eligible']];chosen,skipped=greedy_target_selection(eligible,[r['sequence'] for r in selected],cfg['quota_per_bucket'],cfg['minimum_selected_hamming_distance'])
                for rank,r in enumerate(chosen,1):
                    out=dict(r);out['candidate_id']=f"G25_{'AR' if generator=='autoregressive' else 'VAE'}_{group}_R{seed}_I{r['sample_index']:03d}";out['bucket_selection_rank']=rank;out['sampling_seed']=int(sample_seeds[seed]+group_index);selected.append(out)
                buckets.append({'generator':generator,'group':group,'run_seed':seed,'input_count':len(bucket),'eligible_count':len(eligible),'homopolymer_failed_count':sum(r['longest_homopolymer']>cfg['max_homopolymer_run'] for r in bucket),'training_similarity_failed_count':sum(r['training_hamming_distance']<cfg['minimum_training_hamming_distance'] for r in bucket),'diversity_skipped_before_quota':skipped,'selected_count':len(chosen)})
    encoded=encode_sequences(np.array([r['sequence'] for r in selected]));a=match_window(encoded,'TTGACA',[14,15,16]);b=match_window(encoded,'TATAAT',[37,38,39])
    for i,r in enumerate(selected):r['sigma70_minus35_hypothesis_hit']=int(a[i]);r['sigma70_minus10_hypothesis_hit']=int(b[i]);r['status']='computational_candidate_unmeasured'
    d=np.count_nonzero(encoded[:,None,:]!=encoded[None,:,:],axis=2);np.fill_diagonal(d,51);minimum=int(d.min())
    if minimum<cfg['minimum_selected_hamming_distance']:raise ValueError('Global diversity check failed')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    fields=['candidate_id','generator','group','run_seed','sampling_seed','sample_index','source_csv_row','bucket_selection_rank','sequence','target_log10','m3_score','svr_score','m3_target_gap','svr_target_gap','scorer_difference','gc_fraction','longest_homopolymer','nearest_training_sample_index','training_hamming_distance','training_identity','sigma70_minus35_hypothesis_hit','sigma70_minus10_hypothesis_hit','status']
    write_csv(out/'candidates.csv',selected,fields)
    write_csv(out/'pool_audit.csv',records,list(records[0]))
    with (out/'candidates.fasta').open('w',encoding='utf-8') as f:
        for r in selected:f.write(f">{r['candidate_id']} target_log10={r['target_log10']:.6f} status=unmeasured\n{r['sequence']}\n")
    (out/'README.md').write_text('''# 第25组：计算候选清单 v1

90 条大肠杆菌 50 bp 候选：两种生成器 × 三个目标 × 三次既有运行 × 每次 5 条。

- candidates.csv：候选序列、来源编号、两个预测器分数、到目标的预测距离、GC、同聚碱基与按位训练相似度。
- candidates.fasta：与 CSV 完全相同的候选序列。
- pool_audit.csv：9,000 条输入的质量检查记录，便于追溯筛选。
- manifest.json：规则与输入摘要、产物校验值和结果概览。

分数均为模型预测 log10 尺度，不是实测表达强度；规则没有设置目标命中阈值，入选不代表达到目标。M3 用于主排序，SVR 仅作附带复核和并列排序；两评分器共享项目数据，不是独立生物实验。各运行和各目标采用固定配额，未根据本清单验证或优化配额。规则及所用数据此前已经过开发探索。

motif 字段只在假定序列为 TSS 上游 -50..-1 的窗口下标记 σ70 候选模式（指定起始位置 ±1、至多一个碱基不匹配），不表示实验确认或适用于全部序列。训练相似度只按同向对齐 Hamming 比较，不能证明与基因组或其他数据库独立。

完整解释、阈值和复现命令见仓库 reports/final_candidates_v1_2026-11-17.md 和 configs/final_candidates_v1_2026-11-17.toml。
''',encoding='utf-8')
    summary=[]
    for generator in cfg['generators']:
        for group in cfg['groups']:
            r=[x for x in selected if x['generator']==generator and x['group']==group];summary.append({'generator':generator,'group':group,'count':len(r),'target_log10':r[0]['target_log10'],**{k:float(np.mean([x[k] for x in r])) for k in ('m3_score','svr_score','m3_target_gap','svr_target_gap','training_identity','gc_fraction')}})
    files={name:sha(out/name) for name in ('candidates.csv','candidates.fasta','pool_audit.csv','README.md')}
    manifest={'version':cfg['version'],'protocol':'Post-development computational shortlist; no measured candidate strength, no target-hit threshold, no biological validation. Per-run quotas and heuristic QC thresholds fixed before this export.', 'config_sha256':sha(cp),'input_sha256':sha(inp),'data_sha256':hashes,'prior_summary_sha256':sha(priorpath),'numpy_version':np.__version__,'file_sha256':files,'input_count':len(records),'selected_count':len(selected),'selected_minimum_pair_hamming_distance':minimum,'buckets':buckets,'summary':summary}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8');report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));print('selected',len(selected),'minimum pair distance',minimum)


if __name__=='__main__':main()
