import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_unified_generated import pairwise_diversity
class TestDiversity(unittest.TestCase):
 def test_normalized_pairwise_distance(self):
  x=np.eye(4)[np.array([[0,0],[0,1],[1,1]])]
  self.assertAlmostEqual(pairwise_diversity(x),2/3)
 def test_needs_two(self):
  with self.assertRaises(ValueError):pairwise_diversity(np.ones((1,50,4)))
if __name__=='__main__':unittest.main()
