from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_m4_condition_response import describe, scores_summary


class ConditionResponseTest(unittest.TestCase):
    def test_score_groups_use_training_boundaries_and_do_not_imply_accuracy(self):
        result = scores_summary(np.array([1.0, 2.0, 2.5, 3.0]), np.array([2.0, 3.0]))
        self.assertEqual(result["screening_group_fraction"], {"weak": 0.25, "medium": 0.5, "strong": 0.25})
        self.assertAlmostEqual(result["mean"], 2.125)

    def test_rejects_empty_or_nonfinite_values(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                describe(np.asarray(values))


if __name__ == "__main__":
    unittest.main()
