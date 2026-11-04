from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promoter_ml.edit_distance_overlap import bounded_edit_distance, EditDistanceIndex


class EditDistanceTest(unittest.TestCase):
    def test_insert_delete_and_threshold(self):
        self.assertEqual(bounded_edit_distance("ACGTT", "ACGT", 1, force_python=True), 1)
        self.assertEqual(bounded_edit_distance("ACGT", "AGGT", 1, force_python=True), 1)
        self.assertIsNone(bounded_edit_distance("AAAA", "CCCC", 3, force_python=True))
        self.assertIsNone(bounded_edit_distance("AAAA", "AAAC", 0))

    def test_index_matches_brute_force_for_indels(self):
        rng = random.Random(774)
        base = "".join(rng.choices("ACGT", k=50))
        refs = [base]
        for _ in range(45):
            variant = list(base)
            i, j = sorted(rng.sample(range(5, 45), 2))
            variant.insert(i, rng.choice("ACGT"))
            del variant[j + 1]
            for k in rng.sample(range(50), rng.randrange(4)):
                variant[k] = rng.choice("ACGT")
            refs.append("".join(variant))
        refs += ["".join(rng.choices("ACGT", k=50)) for _ in range(35)]
        index = EditDistanceIndex(refs, 5)
        for query in [base, *refs[1:10]]:
            actual = index.find(query)
            expected = {i: d for i, r in enumerate(refs)
                        if (d := bounded_edit_distance(query, r, 5, force_python=True)) is not None}
            self.assertEqual(actual, expected)

    def test_acceleration_matches_python(self):
        rng = random.Random(88)
        for _ in range(25):
            a = "".join(rng.choices("ACGT", k=50))
            b = list(a)
            for i in rng.sample(range(50), rng.randrange(7)): b[i] = rng.choice("ACGT")
            b = "".join(b)
            self.assertEqual(bounded_edit_distance(a, b, 5), bounded_edit_distance(a, b, 5, force_python=True))


if __name__ == "__main__":
    unittest.main()
