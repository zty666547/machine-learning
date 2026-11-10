import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from promoter_ml.data import KmerFeaturizer,PositionKmerFeaturizer
class FeatureFactorTest(unittest.TestCase):
 def test_position_features_include_same_global_features(self):
  seq=np.array(['ACGT'*12+'AC','TGCA'*12+'TG']);g=KmerFeaturizer(k_min=1,k_max=3,include_gc=True).transform(seq);f=PositionKmerFeaturizer(sequence_length=50);x=f.transform(seq)
  np.testing.assert_array_equal(x[:,f.position_feature_count:],g)
  self.assertGreater(f.position_feature_count,0)
if __name__=='__main__':unittest.main()
