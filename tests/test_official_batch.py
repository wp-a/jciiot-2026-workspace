import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_official_batch import (
    BatchJob,
    build_jobs,
    is_terminal_manifest,
    summarize_manifests,
)


def _manifest(*, level: str, score: int, max_score: int, collision_frames: int = 0):
    required_grasps = 3 if level == "L5" else 1
    return {
        "status": "complete",
        "level": level,
        "seed": 20260727,
        "official_score": score,
        "max_score": max_score,
        "successful_grasp_events": required_grasps,
        "required_grasp_events": required_grasps,
        "collision_frames": collision_frames,
        "final_target_distance_m": 0.2,
        "elapsed_s": 10.0,
        "execution_result": {"success": True},
    }


class OfficialBatchTests(unittest.TestCase):
    def test_script_can_be_invoked_directly(self):
        repository_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/run_official_batch.py", "--help"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--candidate-root", result.stdout)

    def test_build_jobs_assigns_unique_paths_for_each_task_and_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = build_jobs(
                task_indices=[1, 4],
                seeds=[20260727, 20260728],
                output_dir=Path(temp_dir),
            )

            self.assertEqual(len(jobs), 4)
            self.assertTrue(all(isinstance(job, BatchJob) for job in jobs))
            self.assertEqual(len({job.manifest_path for job in jobs}), 4)
            self.assertEqual(len({job.trajectory_path for job in jobs}), 4)
            self.assertEqual(jobs[0].label, "l2-seed-20260727")
            self.assertEqual(jobs[-1].label, "l5-seed-20260728")

    def test_terminal_manifest_requires_a_valid_finished_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            self.assertFalse(is_terminal_manifest(path))

            path.write_text("not json", encoding="utf-8")
            self.assertFalse(is_terminal_manifest(path))

            path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            self.assertFalse(is_terminal_manifest(path))

            path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            self.assertTrue(is_terminal_manifest(path))

            path.write_text(json.dumps({"status": "error"}), encoding="utf-8")
            self.assertTrue(is_terminal_manifest(path))

    def test_summary_reports_full_score_rate_collision_rate_and_wilson_interval(self):
        manifests = [
            _manifest(level="L2", score=15, max_score=15),
            _manifest(level="L2", score=10, max_score=15, collision_frames=1),
        ]
        manifests[1]["execution_result"] = {"success": False}

        summary = summarize_manifests(manifests, planned_runs=2)
        level = summary["levels"]["L2"]

        self.assertEqual(summary["completed_runs"], 2)
        self.assertEqual(level["full_score_runs"], 1)
        self.assertEqual(level["full_score_rate"], 0.5)
        self.assertEqual(level["collision_runs"], 1)
        self.assertEqual(level["collision_rate"], 0.5)
        self.assertLess(level["full_score_rate_wilson_95"][0], 0.5)
        self.assertGreater(level["full_score_rate_wilson_95"][1], 0.5)

    def test_summary_counts_error_manifests_as_completed_failures(self):
        error = {
            "status": "error",
            "level": "L5",
            "seed": 20260729,
            "official_score": 0,
            "collision_frames": None,
            "successful_grasp_events": 0,
            "required_grasp_events": 3,
            "elapsed_s": 2.0,
        }

        summary = summarize_manifests([error], planned_runs=3)

        self.assertEqual(summary["completed_runs"], 1)
        self.assertEqual(summary["remaining_runs"], 2)
        self.assertEqual(summary["levels"]["L5"]["error_runs"], 1)
        self.assertEqual(summary["levels"]["L5"]["full_score_runs"], 0)


if __name__ == "__main__":
    unittest.main()
