import math
import unittest

from scripts.perturbation_protocol import (
    sample_perturbation,
    tier_limits,
)


class PerturbationProtocolTests(unittest.TestCase):
    def test_small_tier_matches_locked_design(self):
        limits = tier_limits("small")

        self.assertAlmostEqual(limits.object_xy_m, 0.02)
        self.assertAlmostEqual(limits.object_yaw_rad, math.radians(5.0))
        self.assertAlmostEqual(limits.base_xy_m, 0.01)
        self.assertAlmostEqual(limits.base_yaw_rad, math.radians(2.0))
        self.assertEqual(limits.mass_fraction, 0.0)
        self.assertEqual(limits.friction_fraction, 0.0)

    def test_same_identity_produces_identical_sample(self):
        kwargs = {
            "tier": "small",
            "seed": 20260728,
            "task_index": 0,
            "object_name": "line_5_container_h01_near",
        }

        first = sample_perturbation(**kwargs)
        second = sample_perturbation(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_different_seed_changes_sample(self):
        common = {
            "tier": "small",
            "task_index": 0,
            "object_name": "line_5_container_h01_near",
        }

        first = sample_perturbation(seed=20260728, **common)
        second = sample_perturbation(seed=20260729, **common)

        self.assertNotEqual(first.generator_digest, second.generator_digest)
        self.assertNotEqual(first.numeric_values(), second.numeric_values())

    def test_nominal_is_exact_noop(self):
        sample = sample_perturbation(
            tier="nominal",
            seed=99,
            task_index=4,
            object_name="white_tote_b01_left_center",
        )

        self.assertEqual(
            sample.numeric_values(),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        )

    def test_samples_stay_inside_each_tier(self):
        for tier in ("nominal", "small", "medium", "stress"):
            limits = tier_limits(tier)
            for seed in range(20):
                sample = sample_perturbation(
                    tier=tier,
                    seed=seed,
                    task_index=3,
                    object_name="blue_container_h01_back_upper",
                )
                self.assertLessEqual(abs(sample.object_dx_m), limits.object_xy_m)
                self.assertLessEqual(abs(sample.object_dy_m), limits.object_xy_m)
                self.assertLessEqual(abs(sample.object_dyaw_rad), limits.object_yaw_rad)
                self.assertLessEqual(abs(sample.base_dx_m), limits.base_xy_m)
                self.assertLessEqual(abs(sample.base_dy_m), limits.base_xy_m)
                self.assertLessEqual(abs(sample.base_dyaw_rad), limits.base_yaw_rad)
                self.assertGreaterEqual(sample.mass_scale, 1.0 - limits.mass_fraction)
                self.assertLessEqual(sample.mass_scale, 1.0 + limits.mass_fraction)
                self.assertGreaterEqual(
                    sample.friction_scale,
                    1.0 - limits.friction_fraction,
                )
                self.assertLessEqual(
                    sample.friction_scale,
                    1.0 + limits.friction_fraction,
                )

    def test_unknown_tier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown perturbation tier"):
            tier_limits("extreme")


if __name__ == "__main__":
    unittest.main()
