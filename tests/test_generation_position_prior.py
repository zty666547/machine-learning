import unittest
import numpy as np
from promoter_ml.generation import ContinuousConditionalAutoregressiveGenerator
from promoter_ml.generation_position_prior import PositionPriorAutoregressiveGenerator


class PositionPriorTests(unittest.TestCase):
    def test_sparse_context_uses_position_prior_with_same_total_mass(self):
        x = np.array(['AA', 'CA', 'GC', 'TC'])
        y = np.array([0., 1., 2., 3.])
        base = ContinuousConditionalAutoregressiveGenerator(order=1, condition_bins=2, smoothing=.1).fit(x,y)
        model = PositionPriorAutoregressiveGenerator(order=1, condition_bins=2, smoothing=.1).fit(x,y)
        key = (0, 1, 'A')
        np.testing.assert_allclose(model.fallback_probabilities,base.fallback_probabilities)
        np.testing.assert_allclose(base.context_probabilities[key], np.array([1.1,.1,.1,.1])/1.4)
        np.testing.assert_allclose(model.context_probabilities[key], (np.array([1.,0.,0.,0.])+.4*np.array([2.1,.1,.1,.1])/2.4)/1.4)
        self.assertGreater(model.context_probabilities[key][0],base.context_probabilities[key][0])

    def test_unseen_context_uses_fallback_and_sampling_is_valid(self):
        model = PositionPriorAutoregressiveGenerator(order=1,condition_bins=2).fit(np.array(['AA','CA','GC','TC']),np.array([0.,1.,2.,3.]))
        self.assertNotIn((0,1,'G'),model.context_probabilities)
        a=model.sample(1.2,50,42);b=model.sample(1.2,50,42)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(all(len(s)==2 and not set(s)-set('ACGT') for s in a))
        self.assertTrue(all(np.isclose(p.sum(),1.0) and (p>0).all() for p in model.context_probabilities.values()))


if __name__=='__main__':
    unittest.main()
