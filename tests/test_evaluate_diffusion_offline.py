import unittest

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


if __name__ == "__main__":
    unittest.main()
