from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_lafleur_source import nominal_window


class SyntheticSourceTest(unittest.TestCase):
    def test_window_uses_design_time_itr_start(self):
        row = {"UP": "A"*16, "h35": "C"*6, "spacs": "G"*17,
               "h10": "T"*6, "disc": "A"*7, "ITR": "C"*21}
        window, start, length = nominal_window(row)
        self.assertEqual((start, length), (52, 73))
        self.assertEqual(window, ("A"*16+"C"*6+"G"*17+"T"*6+"A"*7)[2:])
        self.assertIsNone(nominal_window(row, -3)[0])
        self.assertEqual(len(nominal_window(row, 5)[0]), 50)

    def test_invalid_module_rejected(self):
        row = {"UP": "N", "h35": "C", "spacs": "G", "h10": "T", "disc": "A", "ITR": "C"}
        with self.assertRaises(ValueError): nominal_window(row)


if __name__ == "__main__":
    unittest.main()
