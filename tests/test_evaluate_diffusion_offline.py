import unittest
from pathlib import Path

from scripts import evaluate_diffusion_offline as module


class TrialSummaryTests(unittest.TestCase):
    def test_uses_median_trial_and_passes_only_when_outputs_are_valid(self):
        trials = [
            {
                "sampling_seed": 3,
                "all_finite": True,
                "action_dim": 20,
                "metrics": {"mse": 0.30, "groups": {}},
            },
            {
                "sampling_seed": 1,
                "all_finite": True,
                "action_dim": 20,
                "metrics": {"mse": 0.10, "groups": {}},
            },
            {
                "sampling_seed": 2,
                "all_finite": True,
                "action_dim": 20,
                "metrics": {"mse": 0.20, "groups": {}},
            },
        ]

        summary = module.summarize_trials(
            trials,
            constant_baseline_mse=0.25,
            expected_action_dim=20,
        )

        self.assertEqual(summary["median_trial_seed"], 2)
        self.assertAlmostEqual(summary["median_heldout_mse"], 0.20)
        self.assertAlmostEqual(summary["relative_improvement"], 0.20)
        self.assertTrue(summary["offline_gate_passed"])

    def test_rejects_any_nonfinite_trial_even_when_median_beats_baseline(self):
        trials = [
            {
                "sampling_seed": seed,
                "all_finite": seed != 2,
                "action_dim": 20,
                "metrics": {"mse": mse, "groups": {}},
            }
            for seed, mse in ((1, 0.10), (2, 0.20), (3, 0.30))
        ]

        summary = module.summarize_trials(
            trials,
            constant_baseline_mse=0.40,
            expected_action_dim=20,
        )

        self.assertFalse(summary["all_trials_finite"])
        self.assertFalse(summary["offline_gate_passed"])

    def test_requires_an_odd_number_of_trials(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            module.summarize_trials(
                [],
                constant_baseline_mse=0.40,
                expected_action_dim=20,
            )


class PeriodicCheckpointSelectionTests(unittest.TestCase):
    def test_selects_saved_epoch_with_lowest_logged_validation_loss(self):
        paths = [
            Path("model_epoch_1.pth"),
            Path("model_epoch_10.pth"),
            Path("model_epoch_20.pth"),
        ]
        validation_losses = {1: 0.8, 10: 0.2, 20: 0.3}

        selected = module.select_periodic_checkpoint_by_validation(
            paths,
            validation_losses,
        )

        self.assertEqual(selected.name, "model_epoch_10.pth")

    def test_parses_validation_losses_from_robomimic_log(self):
        log = """
Validation Epoch 1
{
    "Loss": 1.25,
    "Time_Epoch": 0.1
}
Validation Epoch 2
{
    "Loss": 2.5e-02,
    "Time_Epoch": 0.1
}
"""

        self.assertEqual(
            module.parse_validation_losses(log),
            {1: 1.25, 2: 0.025},
        )


if __name__ == "__main__":
    unittest.main()
