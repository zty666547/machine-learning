#!/usr/bin/env python3
"""Score 50 bp sequences with three preselected CNN inference checkpoints."""
import argparse,csv,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.models import MotifCNNStrengthPredictor
from score_m3_screening_evaluator import load_candidates
from train_motif_cnn_predictor import one_hot_sequences

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs=3,required=True,metavar=('CNN1','CNN2','CNN3'));p.add_argument('--candidate-file',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 seq=load_candidates(a.candidate_file);features=one_hot_sequences(seq);values=np.stack([MotifCNNStrengthPredictor.load(path).predict(features) for path in a.models]);average=values.mean(axis=0)
 if not np.isfinite(average).all():raise ValueError('nonfinite prediction')
 output=Path(a.output);output=output if output.is_absolute() else ROOT/output;output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f);writer.writerow(['sequence','ensemble_log10_strength','cnn1','cnn2','cnn3']);writer.writerows(zip(seq,average,*values))
 print('Saved',len(seq),'predicted log10 strengths to',output)
 for name in a.models: print(Path(name).name,hashlib.sha256(Path(name).read_bytes()).hexdigest())
 print('Exploratory predictions on the course label scale, not experimental measurements.')
if __name__=='__main__':main()
