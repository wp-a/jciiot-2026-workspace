import unittest
from pathlib import Path

import numpy as np

from scripts import evaluate_bc_rnn_offline as module


class ErrorMetricsTests(unittest.TestCase):
    def test_reports_whole_action_and_group_errors(self):
        targets = np.zeros((2, 4), dtype=float)
        predictions = np.array(
            [
                [1.0, -1.0, 2.0, 0.0],
                [1.0, -1.0, 0.0, 2.0],
            ]
        )

        metrics = module.error_metrics(
            predictions,
            targets,
            action_groups={"arm": (0, 2), "gripper": (2, 4)},
        )

        self.assertAlmostEqual(metrics["mse"], 1.5)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["groups"]["arm"]["mse"], 1.0)
        self.assertAlmostEqual(metrics["groups"]["gripper"]["mse"], 2.0)
        self.assertAlmostEqual(metrics["out_of_range_fraction"], 0.25)

    def test_rejects_prediction_target_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            module.error_metrics(
                np.zeros((2, 3)),
                np.zeros((2, 4)),
                action_groups={},
            )


class CheckpointSelectionTests(unittest.TestCase):
    def test_selects_lowest_validation_loss_not_latest_epoch(self):
        paths = [
            Path("model_epoch_9_best_validation_0.0138.pth"),
            Path("model_epoch_11_best_validation_0.0126.pth"),
            Path("model_epoch_300.pth"),
        ]

        selected = module.select_best_validation_checkpoint(paths)

        self.assertEqual(selected.name, "model_epoch_11_best_validation_0.0126.pth")

    def test_rejects_missing_validation_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "best-validation"):
            module.select_best_validation_checkpoint([Path("model_epoch_300.pth")])


if __name__ == "__main__":
    unittest.main()
