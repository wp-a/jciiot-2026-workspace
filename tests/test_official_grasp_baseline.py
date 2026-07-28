import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_official_grasp_baseline import (
    build_jobs,
    has_required_grasp_sites,
    initialized_object_center,
    physical_grasp_success,
    summarize,
    validate_materialized_checkpoint,
)


TASKS = [
    {
        "level": "L1",
        "scene_prefix": "factory_sorting_1",
        "env_name": "FactorySorting1",
        "source": "input_5",
        "object": ["near", "far"],
    },
    {
        "level": "L5",
        "scene_prefix": "factory_sorting_9",
        "env_name": "FactorySorting9",
        "source": "input_1",
        "object": ["center", "front", "back"],
    },
]


class OfficialGraspBaselineTests(unittest.TestCase):
    def test_grasp_site_check_requires_both_arms(self):
        class Model:
            def site_name2id(self, name):
                if name.endswith("_left_grasp_site"):
                    return 1
                raise KeyError(name)

        raw = type("Raw", (), {"sim": type("Sim", (), {"model": Model()})()})()

        self.assertFalse(has_required_grasp_sites(raw, "crate"))

    def test_object_center_is_sampled_only_after_wrapped_env_initialization(self):
        events = []

        class Sim:
            def forward(self):
                events.append("forward")

        raw = type("Raw", (), {"sim": Sim()})()
        env = object()

        def current_obs(received_env):
            self.assertIs(received_env, env)
            events.append("idle_step")

        def object_center(received_raw, object_name):
            self.assertIs(received_raw, raw)
            self.assertEqual(object_name, "crate")
            self.assertEqual(events, ["idle_step", "forward"])
            return [7.25, 4.6, 1.25]

        result = initialized_object_center(
            {
                "base_robosuite_env": lambda received_env: raw,
                "current_wrapped_policy_obs": current_obs,
                "object_center_pos": object_center,
            },
            env,
            "crate",
        )

        self.assertEqual(result, [7.25, 4.6, 1.25])

    def test_build_jobs_expands_every_scored_object_and_seed(self):
        jobs = build_jobs(TASKS, seeds=[3, 7])

        self.assertEqual(len(jobs), 10)
        self.assertEqual(
            [(job.level, job.object_name, job.seed) for job in jobs[:5]],
            [
                ("L1", "near", 3),
                ("L1", "far", 3),
                ("L5", "center", 3),
                ("L5", "front", 3),
                ("L5", "back", 3),
            ],
        )

    def test_physical_success_requires_bilateral_grasp_and_lift(self):
        valid = {
            "grasp_status": {"left": True, "right": True},
            "lifted_m": 0.131,
            "collision": False,
            "infrastructure_error": None,
        }
        self.assertTrue(physical_grasp_success(valid))

        for patch in (
            {"grasp_status": {"left": False, "right": True}},
            {"lifted_m": 0.129},
            {"collision": True},
            {"infrastructure_error": "environment failed"},
        ):
            record = dict(valid)
            record.update(patch)
            self.assertFalse(physical_grasp_success(record), patch)

    def test_summary_keeps_failed_and_missing_attempts_in_denominator(self):
        records = [
            {
                "level": "L1",
                "object_name": "near",
                "physical_success": True,
                "collision": False,
                "infrastructure_error": None,
            },
            {
                "level": "L1",
                "object_name": "far",
                "physical_success": False,
                "collision": True,
                "infrastructure_error": None,
            },
            {
                "level": "L2",
                "object_name": "upper",
                "physical_success": False,
                "collision": False,
                "infrastructure_error": "crash",
            },
        ]

        summary = summarize(records, planned_runs=5)

        self.assertEqual(summary["planned_runs"], 5)
        self.assertEqual(summary["recorded_runs"], 3)
        self.assertEqual(summary["successful_runs"], 1)
        self.assertEqual(summary["missing_runs"], 2)
        self.assertEqual(summary["success_rate"], 0.2)
        self.assertEqual(summary["collision_runs"], 1)
        self.assertEqual(summary["infrastructure_errors"], 1)

    def test_checkpoint_validator_rejects_git_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pth"
            checkpoint.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 139543773\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                validate_materialized_checkpoint(checkpoint)

    def test_checkpoint_validator_returns_sha256_for_real_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pth"
            checkpoint.write_bytes(b"materialized checkpoint")

            result = validate_materialized_checkpoint(checkpoint)

        self.assertEqual(result["size_bytes"], 23)
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
