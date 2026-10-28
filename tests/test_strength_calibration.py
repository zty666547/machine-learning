import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_strength_calibration import group_folds


class GroupFoldTest(unittest.TestCase):
    def test_reproducible_and_keeps_clusters_together(self):
        groups = np.array([4, 4, 4, 9, 9, 13, 13, 20, 20, 30, 40, 50])
        folds = group_folds(groups, 3, seed=77)
        np.testing.assert_array_equal(folds, group_folds(groups, 3, seed=77))
        self.assertEqual(set(folds), {0, 1, 2})
        for group in np.unique(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)

    def test_rejects_insufficient_groups(self):
        with self.assertRaises(ValueError):
            group_folds(np.array([1, 1, 2]), 3, seed=1)


if __name__ == "__main__":
    unittest.main()
