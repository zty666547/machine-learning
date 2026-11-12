import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from promoter_ml.models.portable_rbf_svr import PortableRBFSVR
class PortableSVRTest(unittest.TestCase):
 def test_inference_roundtrip_matches_direct_rbf(self):
  mean=np.array([1.,2.]);scale=np.array([2.,.5]);sv=np.array([[0.,1.],[2.,-1.]]);dual=np.array([.3,-.8]);x=np.array([[1.,2.],[3.,1.],[2.,4.]])
  model=PortableRBFSVR(mean,scale,sv,dual,.7,.2,PortableRBFSVR.schema);z=(x-mean)/scale;expected=np.exp(-.2*np.sum((z[:,None,:]-sv[None,:,:])**2,axis=2))@dual+.7
  np.testing.assert_allclose(model.predict(x,batch_size=2),expected,atol=1e-12,rtol=0)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'model.npz';model.save(path);loaded=PortableRBFSVR.load(path);np.testing.assert_allclose(loaded.predict(x),expected,atol=1e-12,rtol=0)
 def test_invalid_checkpoint_and_input(self):
  with self.assertRaises(ValueError):PortableRBFSVR([0],[0],[[0]],[1],0,1,PortableRBFSVR.schema)
  with self.assertRaises(ValueError):PortableRBFSVR([0],[1],[[0]],[1],0,1,'unexpected')
  m=PortableRBFSVR([0],[1],[[0]],[1],0,1,PortableRBFSVR.schema)
  with self.assertRaises(ValueError):m.predict(np.zeros((1,2)))
  with self.assertRaises(ValueError):m.predict(np.array([[np.nan]]))
if __name__=='__main__':unittest.main()
