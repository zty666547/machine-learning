import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_weighted_neighbors import position_weights,weighted_nearest
class TestWeightedNeighbors(unittest.TestCase):
 def test_weight_training_and_tie_breaking(self):
  x=np.zeros((4,50,4));x[:,:,0]=1;x[2:,0,:]=[0,1,0,0]; y=np.array([0.,0.,2.,2.]);w=position_weights(x,y)
  self.assertAlmostEqual(w[0],11.);np.testing.assert_allclose(w[1:],1.)
  refs=np.zeros((2,50,4));refs[:,:,0]=1;refs[0,0]=[0,1,0,0];refs[1,1]=[0,1,0,0];q=np.zeros((1,50,4));q[:,:,0]=1
  score,ties=weighted_nearest(refs,np.array([1.,3.]),q,w);np.testing.assert_allclose(score,[3.]);np.testing.assert_array_equal(ties,[1])
  score,ties=weighted_nearest(refs,np.array([1.,3.]),q,np.ones(50));np.testing.assert_allclose(score,[2.]);np.testing.assert_array_equal(ties,[2])
if __name__=='__main__':unittest.main()
