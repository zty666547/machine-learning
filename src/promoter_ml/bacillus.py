"""Small-data sequence checks for separate B. subtilis experiments."""
import numpy as np
from promoter_ml.data import _UnionFind


def aligned_hamming_components(sequences, max_mismatches):
    """Exhaustively connect same-length sequences within a Hamming threshold."""
    seq = np.asarray(sequences).astype(str)
    if seq.ndim != 1 or not len(seq):
        raise ValueError('Expected nonempty one-dimensional sequences')
    length = len(seq[0])
    if not length or any(len(s)!=length or set(s)-set('ACGT') for s in seq):
        raise ValueError('Sequences must have the same length and use A/C/G/T')
    if not isinstance(max_mismatches,int) or not 0<=max_mismatches<=length:
        raise ValueError('Invalid Hamming threshold')
    chars=np.frombuffer(''.join(seq).encode('ascii'),dtype=np.uint8).reshape(len(seq),length)
    union=_UnionFind(len(seq));edges=0
    for left in range(len(seq)-1):
        distance=np.count_nonzero(chars[left+1:]!=chars[left],axis=1)
        for right in np.flatnonzero(distance<=max_mismatches)+left+1:
            edges+=1;union.union(left,int(right))
    mapping={};groups=np.empty(len(seq),dtype=np.int64)
    for i in range(len(seq)):
        root=union.find(i)
        if root not in mapping:mapping[root]=len(mapping)
        groups[i]=mapping[root]
    return groups,edges


def reverse_complement(sequence):
    if not sequence or set(sequence)-set('ACGT'):
        raise ValueError('Expected a nonempty A/C/G/T sequence')
    return sequence.translate(str.maketrans('ACGT','TGCA'))[::-1]
