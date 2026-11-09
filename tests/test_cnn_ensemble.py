import sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from promoter_ml.models import MotifCNNStrengthPredictor
class CNNCheckpointTest(unittest.TestCase):
 def test_roundtrip_and_invalid_metadata(self):
  m=MotifCNNStrengthPredictor(filters=2,kernel_size=2,location_bins=2,hidden_size=3);m.target_mean=1.;m.target_scale=.5;rng=np.random.default_rng(1);m.parameters={'filters':rng.normal(size=(2,2,4)),'filter_bias':np.zeros(2),'head_weight':rng.normal(size=(4,3)),'head_bias':np.zeros(3),'output_weight':rng.normal(size=(3,1)),'output_bias':np.zeros(1)};x=np.eye(4)[np.array([[0,1,2,3,0],[3,2,1,0,3]])]
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.npz';m.save(path);loaded=MotifCNNStrengthPredictor.load(path);np.testing.assert_array_equal(loaded.predict(x),m.predict(x));m.target_scale=0.;m.save(path)
   with self.assertRaises(ValueError):MotifCNNStrengthPredictor.load(path)
if __name__=='__main__':unittest.main()
