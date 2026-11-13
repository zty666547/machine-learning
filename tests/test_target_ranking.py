import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_target_ranking import rank_target
class TargetRankingTest(unittest.TestCase):
 def test_closeness_both_sides_and_stable_ties(self):
  np.testing.assert_array_equal(rank_target([1.,3.,2.,2.],2.),[3,4,1,2])
  np.testing.assert_array_equal(rank_target([2.,3.,4.],1.),[1,2,3])
 def test_invalid_scores(self):
  with self.assertRaises(ValueError):rank_target([np.nan],1.)
  with self.assertRaises(ValueError):rank_target([],1.)
if __name__=='__main__':unittest.main()
