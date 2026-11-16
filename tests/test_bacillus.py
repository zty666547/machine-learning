import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from promoter_ml.bacillus import aligned_hamming_components, reverse_complement


class BacillusSequenceTests(unittest.TestCase):
    def test_hamming_components_are_transitive(self):
        groups,edges=aligned_hamming_components(np.array(['AAA','AAT','ATT','CCC']),1)
        self.assertEqual(groups.tolist(),[0,0,0,1])
        self.assertEqual(edges,2)

    def test_validation_and_reverse_complement(self):
        self.assertEqual(reverse_complement('AACG'),'CGTT')
        for values in (['AC','A'],['AX','AC']):
            with self.assertRaises(ValueError):aligned_hamming_components(values,1)
        with self.assertRaises(ValueError):reverse_complement('AN')


if __name__=='__main__':unittest.main()
