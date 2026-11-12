#!/usr/bin/env python3
"""Score 50 bp candidate sequences using a NumPy-only RBF SVR checkpoint."""
import argparse,csv,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.data import PositionKmerFeaturizer
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
from score_m3_screening_evaluator import load_candidates

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--candidate-file',required=True);p.add_argument('--output',required=True);a=p.parse_args();model=PortableRBFSVR.load(a.model);seq=load_candidates(a.candidate_file);x=PositionKmerFeaturizer(sequence_length=50).transform(seq);pred=model.predict(x);out=Path(a.output);out=out if out.is_absolute() else ROOT/out;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['sequence','predicted_log10_strength']);writer.writerows(zip(seq,pred))
 print('Saved',len(seq),'predictions to',out);print('Checkpoint SHA-256:',hashlib.sha256(Path(a.model).read_bytes()).hexdigest());print('Exploratory predictions; not measured expression strength.')
if __name__=='__main__':main()
