from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate_ecoli_external_screening import ranking_metrics, unique_sequences


class ExternalMetricsTest(unittest.TestCase):
    def test_ties_auroc_and_grouped_average_precision(self):
        result = ranking_metrics([1, 2, 3, 4], [.9, .8, .8, .1], [1, 0, 1, 0])
        self.assertAlmostEqual(result["auroc"], .875)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["active_fraction"], .5)

    def test_spearman_uses_average_ranks_and_handles_negative_values(self):
        result = ranking_metrics([-10, 0, 0, 7], [1, 2, 2, 3], [0, 0, 1, 1])
        self.assertAlmostEqual(result["spearman"], 1.0)
        with self.assertRaises(ValueError): ranking_metrics([1, 2], [1, 2], [0, 0])

    def test_duplicate_median_and_conflicting_labels(self):
        rows = [{"sequence_50bp": "AAA", "active": "active", "mpra_expression_scaled": "-1"},
                {"sequence_50bp": "AAA", "active": "active", "mpra_expression_scaled": "3"}]
        unique = unique_sequences(rows)
        self.assertEqual(unique[0]["mpra_expression_scaled"], 1)
        with self.assertRaises(ValueError):
            unique_sequences([rows[0], {**rows[1], "active": "inactive"}])


if __name__ == "__main__":
    unittest.main()
