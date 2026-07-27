import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_official_experiment import (
    acceptance_met,
    audit_trajectory,
    write_json_atomic,
)


class OfficialExperimentTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "level": "L1",
            "env_name": "FactorySorting1_EXAMPLE",
            "source": "input_5",
            "target": "output_4",
            "object": ["box_near", "box_far"],
            "max_score": 10,
        }
        self.trajectory = {
            "events": [
                {
                    "name": "grasp_end",
                    "frame": 1,
                    "source": "input_5",
                    "object_name": "box_near",
                    "success": True,
                }
            ],
            "frames": [
                {
                    "object_positions": {"box_near": [7.0, 4.6, 1.1]},
                    "has_collision": False,
                },
                {
                    "object_positions": {"box_near": [-0.05, -7.32, 1.0]},
                    "has_collision": False,
                },
            ],
        }

    def test_audit_records_score_grasp_collision_and_distance(self):
        manifest = audit_trajectory(
            task_index=0,
            task=self.task,
            trajectory=self.trajectory,
            trajectory_path="trajectory.json",
            score_details={"total": 10, "items": []},
            target_center_xy=[0.0, -7.2],
            official_commit="a" * 40,
            workspace_commit="b" * 40,
            seed=20260727,
            elapsed_s=12.5,
            execution_result={"success": True},
        )

        self.assertEqual(manifest["official_score"], 10)
        self.assertEqual(manifest["successful_grasp_events"], 1)
        self.assertEqual(manifest["collision_frames"], 0)
        self.assertAlmostEqual(manifest["final_target_distance_m"], 0.13, places=6)
        self.assertEqual(manifest["trajectory_frames"], 2)
        self.assertTrue(acceptance_met(manifest, required_score=10))

    def test_collision_or_missing_grasp_rejects_full_score_gate(self):
        self.trajectory["events"][0]["success"] = False
        self.trajectory["frames"][1]["has_collision"] = True
        manifest = audit_trajectory(
            task_index=0,
            task=self.task,
            trajectory=self.trajectory,
            trajectory_path="trajectory.json",
            score_details={"total": 10, "items": []},
            target_center_xy=[0.0, -7.2],
            official_commit="a" * 40,
            workspace_commit="b" * 40,
            seed=20260727,
            elapsed_s=12.5,
            execution_result={"success": True},
        )

        self.assertEqual(manifest["successful_grasp_events"], 0)
        self.assertEqual(manifest["collision_frames"], 1)
        self.assertFalse(acceptance_met(manifest, required_score=10))

    def test_atomic_writer_replaces_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            write_json_atomic(path, {"status": "complete", "score": 10})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete", "score": 10},
            )
            self.assertFalse(path.with_name("result.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
