from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_lafleur_tss_positions import classify_peak, peak_summary


class TssDiagnosticTest(unittest.TestCase):
    def test_position_interpretations_and_boundaries(self):
        result = classify_peak(52, 52, 73)
        self.assertEqual(result["zero_based"]["difference_from_nominal_itr_start"], 0)
        self.assertEqual(result["one_based"]["difference_from_nominal_itr_start"], -1)
        self.assertFalse(classify_peak(50, 52, 73)["one_based"]["can_extract_preceding_50bp"])
        self.assertFalse(classify_peak(74, 52, 73)["one_based"]["within_sequence"])

    def test_ties_and_invalid(self):
        self.assertEqual(peak_summary(["TSS_pos_3", "TSS_pos_2"], [5, 5]), (5.0, [2, 3]))
        self.assertEqual(peak_summary(["TSS_pos_3", "TSS_pos_2"], [None, 2]), (2.0, [2]))
        with self.assertRaises(ValueError): peak_summary(["TSS_pos_3"], [-1])


if __name__ == "__main__": unittest.main()
