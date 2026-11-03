from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from audit_ecoli_genome_proximity import exact_genome_hits, nearest_gap, external_start
from promoter_ml.external_overlap import reverse_complement


class GenomeProximityTest(unittest.TestCase):
    def test_circular_mapping_and_both_orientations(self):
        genome = "AAAACCCCGGGT"
        self.assertIn((10, "+"), exact_genome_hits(genome, "GTAA"))
        self.assertIn((10, "-"), exact_genome_hits(genome, "TTAC"))

    def test_nearest_gap_on_circle(self):
        self.assertEqual(nearest_gap(5, [100, 970], 1000, 20), 15)
        self.assertEqual(nearest_gap(110, [100, 970], 1000, 20), 0)
        self.assertEqual(nearest_gap(5, [], 1000, 20), None)

    def test_nearest_gap_matches_brute_force(self):
        rng = random.Random(404)
        for _ in range(100):
            starts = sorted(rng.sample(range(1000), 30))
            query = rng.randrange(1000)
            expected = min(max(0, min(abs(query - x), 1000 - abs(query - x)) - 50) for x in starts)
            self.assertEqual(nearest_gap(query, starts, 1000), expected)

    def test_external_window_coordinates(self):
        genome = "ACGT" * 100
        pos = 120
        self.assertEqual(external_start({"tss_position": str(pos), "strand": "+",
                                        "sequence_50bp": genome[pos-50:pos]}, genome), 70)
        self.assertEqual(external_start({"tss_position": str(pos), "strand": "-",
                                        "sequence_50bp": reverse_complement(genome[pos:pos+50])}, genome), 120)


if __name__ == "__main__":
    unittest.main()
