import sys
import unittest
import tempfile
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_cnn_position_bins import fold_layout
import summarize_cnn_position_bins


class CNNPositionSplitTest(unittest.TestCase):
    def test_inner_and_outer_splits_respect_groups(self):
        groups = np.repeat(np.arange(30), [1, 2, 3] * 10)
        layouts, final = fold_layout(groups, fold_seed=42)
        holdouts = []
        for fit, early, holdout in layouts:
            self.assertFalse(set(groups[fit]) & set(groups[early]))
            self.assertFalse(set(groups[fit]) & set(groups[holdout]))
            self.assertFalse(set(groups[early]) & set(groups[holdout]))
            holdouts.extend(holdout.tolist())
        self.assertEqual(sorted(holdouts), list(range(len(groups))))
        self.assertFalse(set(groups[final[0]]) & set(groups[final[1]]))

    def test_bin_selection_uses_oof_not_validation(self):
        def metrics(mae):
            return {"mae": mae, "rmse": 1.0, "pearson": 0.2, "spearman": 0.1,
                    "prediction_to_target_std_ratio": 0.3,
                    "by_true_strength_group": {g: {"mae": mae} for g in ("weak", "medium", "strong")}}

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (1, 2):
                run = {"model_seed": seed, "fold_seed": 9, "bins": [5, 10],
                       "model_config": {"filters": 32}, "train_count": 10, "validation_count": 5,
                       "variants": {
                           "5": {"training_oof": metrics(0.4), "validation": metrics(0.9),
                                 "final_fit": {"parameter_count": 1}},
                           "10": {"training_oof": metrics(0.5), "validation": metrics(0.1),
                                  "final_fit": {"parameter_count": 2}},
                       }}
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps(run), encoding="utf-8")
                paths.append(str(path))
            output = Path(directory) / "summary.json"
            with patch.object(sys, "argv", ["summary", *paths, "--output", str(output)]), redirect_stdout(io.StringIO()):
                summarize_cnn_position_bins.main()
            self.assertEqual(json.loads(output.read_text())["selected_bins"], 5)


if __name__ == "__main__":
    unittest.main()
