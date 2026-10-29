import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ecoli_provenance import compare_text, file_hashes


class ProvenanceAuditTest(unittest.TestCase):
    def test_git_blob_hash_known_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            path.write_bytes(b"")
            self.assertEqual(file_hashes(path)["git_blob_sha1"], "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

    def test_text_sequence_and_label_mismatches_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text("strength\t promoter\n1.5\t acg\n2\t ccc\n", encoding="utf-8")
            result = compare_text(path, np.array(["ACG", "CCA"]), np.array([1.5, 3.0]))
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["sequence_mismatches"], 1)
            self.assertEqual(result["strength_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
