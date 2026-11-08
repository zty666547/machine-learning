import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_sequence_neighbors import nearest

class NearestAuditTest(unittest.TestCase):
 def test_aligned_distance_and_tie_average(self):
  refs=np.zeros((3,50,4));query=np.zeros((2,50,4));refs[:,:,0]=1;query[:,:,0]=1
  refs[1,0]=[0,1,0,0];refs[2,1]=[0,0,1,0];query[1,2]=[0,0,0,1]
  score,distance,ties=nearest(refs,np.array([1.,2.,4.]),query)
  np.testing.assert_array_equal(distance,[0,1]);np.testing.assert_array_equal(ties,[1,1]);np.testing.assert_allclose(score,[1.,1.])
  score,distance,ties=nearest(refs[1:],np.array([2.,4.]),query[:1]);np.testing.assert_array_equal(distance,[1]);np.testing.assert_array_equal(ties,[2]);np.testing.assert_allclose(score,[3.])
if __name__=='__main__':unittest.main()
