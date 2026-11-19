#!/usr/bin/env python3
"""Recompute mainline sequence statistics and audit the existing v1 shortlist.

Archived model metrics are read, not retrained or independently revalidated.
"""
import argparse,csv,hashlib,json,re,subprocess,sys,tomllib
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.data import load_promoter_arrays,load_fixed_split_indices
from promoter_ml.structure_analysis import encode_sequences,position_kmer_frequency,position_js


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def check(value,message):
    if not value:raise ValueError(message)
def close(a,b):return abs(float(a)-float(b))<1e-10


def render(e):
    names={'autoregressive':'连续自回归','vae':'连续条件 VAE'};groups={'weak':'弱','medium':'中','strong':'强'}
    lines=['# 第25组：条件启动子生成主线结果与结论 v1','',
    '整理日期：2026-11-19。本稿依据仓库既有模型结果及本轮重新核验的序列统计编写，可用于后续期末报告整合。它不代表新增训练实验、独立测试成绩或生成候选的实测强度。','',
    '## 1. 任务与数据','',
    '本项目研究根据目标表达强度生成 50 bp 大肠杆菌启动子候选序列，并比较连续条件自回归与连续条件 VAE 的条件响应、序列分布、新颖性和多样性。主数据共 11,884 条，按相似性簇固定为训练 8,318 条、开发验证 1,783 条、测试 1,783 条。强度输入使用原始标签的 log10 变换，弱、中、强分组及目标均值由训练部分确定。','',
    '来源核验支持将标签解释为自然启动子相关转录水平的代理指标；逐条测量条件、归一化单位与 σ 因子注释仍不完整，不能把标签直接理解为跨实验通用的绝对表达强度。开发验证集反复参与模型开发，原测试集在早期曾被查看。当前结果均按相应开发口径解释。','',
    '## 2. 模型与评价方法','',
    '连续自回归采用位置相关的三阶 Markov 上下文、三个条件区间和区间间概率插值，支持连续条件输入；当前实现不是 RNN/LSTM。连续条件 VAE 使用 128 维隐藏层、16 维潜变量及 KL 权重 0.2；这些设置来自项目开发期对照。两种模型都在课程训练数据上学习，当前统一比较的是三个既有训练/采样组合、三个目标点、每组 500 条，共 9,000 条生成记录。','',
    'M3 筛选器由 motif CNN、全局 k-mer 岭回归和位置岭回归组成，权重 0.5/0.4/0.1；SVR 作为另一计算评分参考。两评分器与生成器共享项目数据，CNN 早停及融合权重曾使用开发验证集。评分不是实测强度，也不能把两个模型当作两份独立生物验证。','',
    '局部结构采用逐起始位置的 3-mer Jensen–Shannon 散度（JS），参考为对应强度训练组，越低表示所统计分布越接近。多样性是组内不同序列对平均 Hamming 距离除以 50。训练新颖性仅指不完全复制课程训练序列。','',
    '## 3. 两种生成器的统一结果','',
    '| 模型 | 目标 | 逐位置 3-mer JS | 组内平均 Hamming/50 | 平均 GC | M3 均分 | SVR 均分 |',
    '|---|---|---:|---:|---:|---:|---:|']
    for r in e['generated_recomputed']:
        lines.append(f"| {names[r['generator']]} | {groups[r['group']]} | {r['position_3mer_js']:.5f} | {r['diversity']:.5f} | {r['gc_fraction']:.5f} | {r['m3_score']:.4f} | {r['svr_score']:.4f} |")
    lines += ['', '表中为三个既有运行的简单平均。9,000 条序列均为合法 50 bp、批次内互不完全相同，且均未完全复制训练序列。本批自回归在三个目标组的逐位置 3-mer JS 均低于 VAE，组内平均差异相近。这支持当前自回归配置在所测局部分布上的保真度较好，不能推出真实功能更强或对所有指标普遍更优。训练参考也参与了拟合，这不是独立分布泛化证明。','',
    '## 4. 条件响应与预测器限制','',
    '既有五目标响应实验显示两模型的平均筛选分数随输入目标上升，但变化范围远小于目标范围。GC 匹配后，强目标减弱目标的预测均分差仍为正：','',
    '| 模型 | 匹配后 M3 强减弱 | 匹配后 SVR 强减弱 |','|---|---:|---:|']
    for r in e['gc_matching_archived']:
        lines.append(f"| {names[r['generator']]} | {r['scores']['m3']['matched_strong_minus_weak']:.4f} | {r['scores']['svr']['matched_strong_minus_weak']:.4f} |")
    b=e['screening_development_archived']
    lines += ['', 'GC 匹配复用了已观察样本，只保持两组 GC 总量分布一致，没有控制位置、其他短片段及其交互；此结果是评分层面的描述，不能解释为真实强度变化或 GC 的因果贡献。','',
    f"原 M3 融合器的开发验证 MAE 为 {b['mae']:.5f}，Pearson 为 {b['pearson']:.5f}，Spearman 为 {b['spearman']:.5f}。这些数值来自原有开发验证记录；模型的强弱端预测收缩尚未解决。不同分折、早停方式、数据量和目标尺度的模型成绩不能混在一起排优劣。",
    '', '训练集相似性簇折外的已标注样本检查还显示：按预测分数接近目标选出的中目标序列，预测差距很小，已有真实标签到目标的平均距离却仍较大：','',
    '| 评分器 | 选中记录预测值到中目标的距离 | 选中记录已有标签到中目标的距离 |','|---|---:|---:|']
    t=next(r for r in e['target_oof_archived'] if r['group']=='medium')
    for name,label in [('m3','M3'),('svr','SVR')]:lines.append(f"| {label} | {t[name+'_predicted_gap_mean']:.4f} | {t[name+'_mean']:.4f} |")
    x=e['external_archived']
    lines += ['', '上述距离是三个折各自目标下的均值，不是生成候选的实测误差。该检查也参与了开发，不能当作独立测试。', '',
    f"外部报告基因数据曾做跨实验探索：{x['count']} 条记录上融合分数与连续实验值的 Spearman 为 {x['spearman']:.4f}，AUROC 为 {x['auroc']:.4f}。完整外部实验片段为 150 bp，模型只看对应 50 bp；测量尺度和潜在序列关联仍有差异。AUROC 不是分类准确率，该结果不能用作生成序列的生物验证。",
    '', '## 5. 固定规则下的 90 条计算候选','',
    '候选 v1 保留两种生成器、三个目标和三个运行，每个桶选 5 条，共 90 条。质量规则包括长度/字符合法、同聚碱基最长不超过 8、与训练序列同向对齐距离至少 6；按 M3 到目标的距离排序，以 SVR 距离及原编号处理并列，并要求入选序列之间至少相差 10 个位置。motif 只作假设性注释，GC 不作硬性过滤。这些阈值和配额是计算约定，未证明为最优。','',
    f"本轮复核清单共有 {e['candidate_audit']['count']} 条不同序列，入选序列两两最小 Hamming 距离为 {e['candidate_audit']['minimum_pair_hamming']}，最高训练同向按位相似度为 {e['candidate_audit']['maximum_training_identity']:.0%}。输入池中 21 条因同聚碱基段过长排除，训练相似性和候选间距离规则在本批前排候选中没有实际剔除记录。",
    '', '| 模型 | 目标 | 平均 M3 目标距离 | 平均 SVR 目标距离 |','|---|---|---:|---:|']
    for r in e['candidate_summary_archived']:
        lines.append(f"| {names[r['generator']]} | {groups[r['group']]} | {r['m3_target_gap']:.4f} | {r['svr_target_gap']:.4f} |")
    lines += ['', '该清单按 M3 筛选，较小的 M3 距离是选择规则的预期结果，不能用于证明生成器性能提高。规则没有设置目标命中阈值，入选不表示真实强度达到目标。CSV、FASTA、全池核查和校验清单位于本地 `outputs/final_candidates_v1/`，具体来源字段和重建命令见[候选说明](final_candidates_v1_2026-11-17.md)。','',
    '## 6. 对研究问题的回答','',
    '**RQ1：能否按目标强度生成序列？** 已实现连续条件输入，并观察到序列组成与计算评分的响应；尚无证据证明真实表达强度准确达到目标。当前结论是在所测样本中观察到计算条件响应，精确强度控制尚待独立测量。','',
    '**RQ2：两种生成策略有何差异？** 当前自回归在逐位置 3-mer 分布上更接近训练参考；两者在本批次的合法性、精确新颖性和平均多样性相近。评分器局限和缺少实测标签，使我们不能宣布某一模型在生物学强度控制上胜出。','',
    '**RQ3：结构与新颖性是否合理？** 已完成局部分布、GC、候选 motif、新颖性及多样性计算检查。按位不相同或局部模式合理不等于生物功能成立；TSS 窗口只有数据集级文献依据，缺少逐序列 σ 因子与元件注释，固定 σ70 扫描不能代表所有样本。','',
    '## 7. 工作范围与剩余事项','',
    '目前完成两种生成器、计算评分器、主要开发期对照、90 条候选 v1 以及本稿的结论整理。多轮开发尝试尚未形成足以全面替换主方案的证据，当前继续保留已有主方案。枯草芽孢杆菌分类实验作为独立扩展，其分类准确率不纳入本主线的表达强度结论。','',
    '仍需结合课程正式要求完成最终提交版报告和答辩材料；若能获得条件匹配的新外部数据或候选实测结果，应作为进一步验证。已有开发验证集与早期查看过的测试集不能重新包装为全新独立测试。未实现的扩散生成器不计入已完成模型。','',
    '## 8. 本轮复核与复现','',
    f"本轮重新计算 9,000 条序列的 GC、组均分、逐位置 3-mer JS、精确训练新颖性及组内平均 Hamming 距离，并与归档汇总一致性核对；对候选文件校验摘要、输入来源、配额、FASTA、训练最近匹配和两两距离。项目 {e['unit_tests']['count']} 项单元测试通过。模型训练成绩、GC 匹配和外部评价来自已有记录，本轮没有重训模型或重做外部实验。",
    '', '```sh',
    'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\',
    'python scripts/audit_mainline_evidence_v1.py --data-dir "$DATA_DIR" \\',
    '  --ranking-csv outputs/m4_target_repeats/rankings_all.csv \\',
    '  --candidate-dir outputs/final_candidates_v1 \\',
    '  --json-output reports/mainline_evidence_v1_2026-11-19.json \\',
    '  --markdown-output reports/mainline_results_v1_2026-11-19.md','```','',
    '原始课程数据和本地生成文件按仓库规则不上传，需按原记录重建。JSON 包含本轮检查结果与证据文件 SHA-256；内容校验通过不表示所有历史模型在新环境下均已完整重训复现。','', '主要证据：','']
    for name in e['source_sha256']:lines.append(f'- [{name}]({name})')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--ranking-csv',required=True);p.add_argument('--candidate-dir',required=True);p.add_argument('--json-output',required=True);p.add_argument('--markdown-output',required=True);a=p.parse_args()
    config=ROOT/'configs/mainline_evidence_v1_2026-11-19.toml';cfg=tomllib.loads(config.read_text())['evidence'];sources={name:json.loads((ROOT/'reports'/name).read_text()) for name in cfg['reports']};unified=sources['m4_unified_generated_2026-11-13.json'];short=sources['final_candidates_v1_2026-11-17.json'];data=Path(a.data_dir);split=ROOT/'data/splits/ecoli_similarity_split.csv'
    hashes={'sequence':sha(data/'promoter.npy'),'strength':sha(data/'gene_expression.npy'),'split':sha(split)}
    for name,report in sources.items():
        if 'data_sha256' in report:check(report['data_sha256']==hashes,'Data mismatch in '+name)
    check(sha(a.ranking_csv)==unified['candidate_csv_sha256']==short['input_sha256'],'Input ranking mismatch')
    seq,strength=load_promoter_arrays(data,'promoter.npy','gene_expression.npy',50);train,val,test=load_fixed_split_indices(split,len(seq));check((len(train),len(val),len(test))==(cfg['training_count'],cfg['validation_count'],cfg['test_count']),'Split counts mismatch')
    y=np.log10(strength[train]);labels=np.digitize(y,np.quantile(y,[1/3,2/3]));trainset=set(seq[train]);reference={g:position_kmer_frequency(encode_sequences(seq[train][labels==i]),3) for i,g in enumerate(('weak','medium','strong'))}
    rows=read_csv(a.ranking_csv);check(len(rows)==cfg['input_count'],'Wrong input count');check(len({r['sequence'] for r in rows})==len(rows),'Unexpected duplicate generated input')
    per_run=[]
    for old in unified['runs']:
        selected=[r for r in rows if r['generator']==old['generator'] and r['input_group']==old['group'] and int(r['vae_seed'])==old['seed']];check(len(selected)==old['count'],'Run count mismatch');encoded=encode_sequences(np.array([r['sequence'] for r in selected]));check(encoded.shape[1]==50,'Length mismatch');n=len(selected)
        counts=position_kmer_frequency(encoded,1)*n;diversity=float(np.mean((n*n-(counts**2).sum(axis=1))/(n*(n-1))))
        r={'generator':old['generator'],'group':old['group'],'seed':old['seed'],'position_3mer_js':float(position_js(reference[old['group']],position_kmer_frequency(encoded,3)).mean()),'diversity':diversity,'gc_fraction':float(np.mean((encoded==1)|(encoded==2))),'m3_score':float(np.mean([float(z['m3_score']) for z in selected])),'svr_score':float(np.mean([float(z['svr_score']) for z in selected]))}
        for new,key in [('position_3mer_js','position_3mer_js_to_train_group'),('diversity','pairwise_hamming_fraction'),('gc_fraction','gc_mean'),('m3_score','m3_score_mean'),('svr_score','svr_score_mean')]:check(close(r[new],old[key]),'Generated metric differs: '+key)
        check(all(z['sequence'] not in trainset for z in selected),'Exact training duplicate');per_run.append(r)
    aggregate=[]
    for generator in ('autoregressive','vae'):
        for group in ('weak','medium','strong'):
            rr=[r for r in per_run if r['generator']==generator and r['group']==group];aggregate.append({'generator':generator,'group':group,**{k:float(np.mean([r[k] for r in rr])) for k in ('position_3mer_js','diversity','gc_fraction','m3_score','svr_score')}})
    folder=Path(a.candidate_dir);manifest=json.loads((folder/'manifest.json').read_text());check(manifest==short,'Manifest and archived export report differ');allowed={'candidates.csv','candidates.fasta','pool_audit.csv','README.md'};check(set(manifest['file_sha256'])==allowed,'Unexpected artifact manifest')
    for name,digest in manifest['file_sha256'].items():check(sha(folder/name)==digest,'Artifact checksum mismatch: '+name)
    candidates=read_csv(folder/'candidates.csv');check(len(candidates)==cfg['selected_count'],'Candidate count mismatch');check(len({r['candidate_id'] for r in candidates})==len(candidates) and len({r['sequence'] for r in candidates})==len(candidates),'Duplicate candidates');check(set(Counter((r['generator'],r['group'],r['run_seed']) for r in candidates).values())=={5},'Quota differs')
    minimum=50;max_identity=0.0;trainencoded=encode_sequences(seq[train]);verified=[]
    for i,r in enumerate(candidates):
        raw=rows[int(r['source_csv_row'])-2];check(raw['sequence']==r['sequence'] and close(raw['m3_score'],r['m3_score']) and close(raw['svr_score'],r['svr_score']),'Candidate source mismatch');x=encode_sequences(np.array([r['sequence']]))[0];d=np.count_nonzero(trainencoded!=x,axis=1);near=int(d.argmin());check(int(d[near])==int(r['training_hamming_distance']) and int(train[near])==int(r['nearest_training_sample_index']),'Nearest training match differs');max_identity=max(max_identity,1-float(d[near])/50)
        for previous in verified:minimum=min(minimum,int(np.count_nonzero(x!=previous)))
        verified.append(x)
    check(minimum==manifest['selected_minimum_pair_hamming_distance'],'Pairwise distance differs');fasta=(folder/'candidates.fasta').read_text().splitlines();check(len(fasta)==2*len(candidates),'FASTA count mismatch')
    for i,r in enumerate(candidates):check(fasta[2*i].startswith('>'+r['candidate_id']+' ') and fasta[2*i+1]==r['sequence'],'FASTA does not match CSV')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,text=True,capture_output=True);check(tests.returncode==0,'Unit tests failed: '+tests.stderr);count=re.search(r'Ran (\d+) tests',tests.stderr+tests.stdout);check(count is not None,'Cannot read test count')
    result={'scope':'Existing-result consolidation plus generated-statistic and artifact recomputation. No predictive model retraining, new independent testing, or wet experiment.', 'config_sha256':sha(config),'audit_script_sha256':sha(Path(__file__)),'source_sha256':{name:sha(ROOT/'reports'/name) for name in cfg['reports']},'data_sha256':hashes,'numpy_version':np.__version__,'split_counts':{'train':len(train),'validation':len(val),'test':len(test)},'generated_record_count':len(rows),'generated_recomputed':aggregate,'candidate_audit':{'count':len(candidates),'minimum_pair_hamming':minimum,'maximum_training_identity':max_identity,'all_manifest_files_match':True,'all_nearest_training_matches_match':True,'fasta_matches_csv':True},'screening_development_archived':sources['m3_screening_evaluator_tuning.json']['best_blend'],'gc_matching_archived':sources['m4_generated_gc_matching_2026-11-16.json']['summary'],'target_oof_archived':sources['m4_target_oof_audit_2026-11-15.json']['summary'],'external_archived':sources['m4_external_screening_results_2026-11-03.json']['record_level_primary']['blend'],'candidate_summary_archived':short['summary'],'unit_tests':{'count':int(count.group(1)),'passed':True}}
    output=Path(a.json_output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');markdown=Path(a.markdown_output);markdown.parent.mkdir(parents=True,exist_ok=True);markdown.write_text(render(result));print('Verified',len(rows),'generated records,',len(candidates),'candidates and',count.group(1),'tests; wrote evidence JSON and conclusion draft.')


if __name__=='__main__':main()
