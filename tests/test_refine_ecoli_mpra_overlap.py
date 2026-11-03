from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.external_overlap import AlignedNeighborIndex, reverse_complement
from refine_ecoli_mpra_overlap import shifted_neighbours, duplicate_summary


class RefinedOverlapTest(unittest.TestCase):
    def test_shifted_mismatch_and_reverse_complement(self):
        rng = random.Random(77)
        variant = "".join(rng.choices("ACGT", k=150))
        reference = list(variant[72:122])
        for i in (0, 10, 20):
            reference[i] = "A" if reference[i] != "A" else "C"
        reference = "".join(reference)
        index = AlignedNeighborIndex([reference], 5)
        best, split = shifted_neighbours(variant, index, {reference: {"train"}})
        self.assertEqual((best[0], best[2], best[3]), (3, 2, "forward"))
        self.assertEqual(split["train"][0], 3)
        reverse = reverse_complement(reference)
        index2 = AlignedNeighborIndex([reverse], 5)
        best2, _ = shifted_neighbours(variant, index2, {reverse: {"test"}})
        self.assertEqual((best2[0], best2[2], best2[3]), (3, 2, "reverse_complement"))

    def test_duplicate_counts_track_contexts_and_conflicting_labels(self):
        common = {"sequence_50bp": "A" * 50, "active": "active", "mpra_expression_scaled": "1", "_variant": "A" * 150}
        rows = [common, {**common, "active": "inactive", "mpra_expression_scaled": "2", "_variant": "C" * 150},
                {**common, "sequence_50bp": "C" * 50}]
        result = duplicate_summary(rows)
        self.assertEqual(result["duplicate_50bp_groups"], 1)
        self.assertEqual(result["rows_in_duplicate_groups"], 2)
        self.assertEqual(result["groups_with_both_activity_labels"], 1)
        self.assertEqual(result["groups_with_different_150bp_context"], 1)


if __name__ == "__main__":
    unittest.main()
