import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from promoter_ml.candidate_selection import greedy_target_selection,longest_homopolymer


class CandidateSelectionTests(unittest.TestCase):
    def test_target_distance_is_used_instead_of_highest_score(self):
        rows=[{'sequence':'ACGT','target_log10':1.,'m3_score':s,'svr_score':s,'sample_index':i} for i,s in enumerate([3.,1.1,2.])]
        selected,_=greedy_target_selection(rows,[],1,1)
        self.assertEqual(selected[0]['sample_index'],1)

    def test_global_diversity_and_score_ties(self):
        rows=[{'sequence':s,'target_log10':1.,'m3_score':1.,'svr_score':1.,'sample_index':i} for i,s in enumerate(['AAAC','CCCC','GGGG'])]
        selected,skipped=greedy_target_selection(rows,['AAAA'],2,2)
        self.assertEqual([r['sample_index'] for r in selected],[1,2])
        self.assertEqual(skipped,1)
        with self.assertRaises(ValueError):greedy_target_selection(rows,['AAAA'],3,2)

    def test_run_length(self):
        self.assertEqual(longest_homopolymer('AAACCCCCGT'),5)
        with self.assertRaises(ValueError):longest_homopolymer('AN')


if __name__=='__main__':unittest.main()
