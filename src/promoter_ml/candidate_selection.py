"""Deterministic computational candidate selection helpers."""


def longest_homopolymer(sequence):
    if not sequence or set(sequence)-set('ACGT'):
        raise ValueError('Expected nonempty A/C/G/T sequence')
    longest=run=1
    for a,b in zip(sequence,sequence[1:]):
        run=run+1 if a==b else 1
        longest=max(longest,run)
    return longest


def greedy_target_selection(records, existing_sequences, quota, minimum_distance):
    """Choose by predicted target distance subject to global aligned diversity."""
    ordered=sorted(records,key=lambda r:(abs(r['m3_score']-r['target_log10']),abs(r['svr_score']-r['target_log10']),r['sample_index']))
    selected=[];seen=list(existing_sequences);skipped=0
    for record in ordered:
        sequence=record['sequence']
        if any(len(s)!=len(sequence) for s in seen):
            raise ValueError('Aligned comparison requires equal lengths')
        if any(sum(a!=b for a,b in zip(sequence,s))<minimum_distance for s in seen):
            skipped+=1;continue
        selected.append(record);seen.append(sequence)
        if len(selected)==quota:break
    if len(selected)!=quota:
        raise ValueError('Not enough eligible and diverse records for fixed quota')
    return selected,skipped
