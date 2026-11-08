import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_knn_position import knn,knn_fixed_count
class TestKNNPosition(unittest.TestCase):
 def test_boundary_ties_fractional_and_fixed(self):
  x=np.zeros((4,50,4));x[:,:,0]=1;q=x[:1].copy()
  for i in range(1,4):x[i,i-1]=[0,1,0,0]
  y=np.array([2.,3.,6.,9.]);p,e,c,t=knn(x,y,q,np.ones(50),2)
  np.testing.assert_allclose(p,[4.]);np.testing.assert_allclose(e,[3.]);np.testing.assert_array_equal(c,[4]);np.testing.assert_array_equal(t,[3]);np.testing.assert_allclose(knn_fixed_count(x,y,q,np.ones(50),2),[2.5])
 def test_invalid_k(self):
  x=np.zeros((1,50,4));
  with self.assertRaises(ValueError):knn(x,np.array([0.]),x,np.ones(50),2)
if __name__=='__main__':unittest.main()
