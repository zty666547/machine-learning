import random
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from promoter_ml.external_overlap import AlignedNeighborIndex, reverse_complement
from audit_ecoli_mpra_external import tss_window, GenomeMismatchError, GenomeBoundsError


class ExternalOverlapTest(unittest.TestCase):
    def test_matches_brute_force_at_threshold_and_rejects_above(self):
        random.seed(4)
        base = "".join(random.choices("ACGT", k=50))
        refs = [base]
        for positions in ([0, 9, 18, 27, 36], [0, 9, 18, 27, 36, 45]):
            sequence = list(base)
            for i in positions: sequence[i] = "A" if sequence[i] != "A" else "C"
            refs.append("".join(sequence))
        refs += ["".join(random.choices("ACGT", k=50)) for _ in range(100)]
        actual = AlignedNeighborIndex(refs, 5).find(base)
        expected = {i: d for i, s in enumerate(refs)
                    if (d := sum(a != b for a, b in zip(base, s))) <= 5}
        self.assertEqual(actual, expected)
        self.assertEqual(actual[1], 5)
        self.assertNotIn(2, actual)

    def test_reverse_complement_and_invalid_query(self):
        self.assertEqual(reverse_complement("ACGTAA"), "TTACGT")
        index = AlignedNeighborIndex(["ACGTAA"], 1)
        self.assertEqual(index.find(reverse_complement("TTACGT")), {0: 0})
        with self.assertRaises(ValueError): index.find("NNNNNN")

    def test_tss_window_checks_both_strands_and_out_of_bounds(self):
        genome = "ACGTTGCATGCA" * 30
        segment = genome[30:180]
        plus = {"category": "tss", "variant": segment, "start": "30", "end": "180",
                "tss_position": "150", "strand": "+"}
        minus = {"category": "tss", "variant": reverse_complement(segment), "start": "30", "end": "180",
                 "tss_position": "60", "strand": "-"}
        self.assertEqual(tss_window(plus, genome), segment[70:120])
        self.assertEqual(tss_window(minus, genome), reverse_complement(segment)[70:120])
        with self.assertRaises(GenomeMismatchError): tss_window({**plus, "variant": "A" * 150}, genome)
        with self.assertRaises(GenomeBoundsError): tss_window({**plus, "start": "-1", "end": "149"}, genome)


if __name__ == "__main__":
    unittest.main()
