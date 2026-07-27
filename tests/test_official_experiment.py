import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_official_experiment as experiment_module
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

    def test_parser_defaults_to_flow_and_accepts_agent_mode(self):
        required = [
            "--candidate-root", "/tmp/candidate",
            "--expected-official-commit", "a" * 40,
            "--workspace-commit", "b" * 40,
            "--trajectory", "/tmp/trajectory.json",
            "--output", "/tmp/manifest.json",
        ]

        self.assertEqual(
            experiment_module.build_parser().parse_args(required).execution_mode,
            "flow",
        )
        self.assertEqual(
            experiment_module.build_parser().parse_args(
                required + ["--execution-mode", "agent"]
            ).execution_mode,
            "agent",
        )

    def test_execute_task_agent_mode_calls_official_robot_agent(self):
        captured = {}

        class FakeOutput:
            def as_dict(self):
                return {
                    "skill_name": "competition_task",
                    "success": True,
                    "payload": {"workflow": {"success": True}},
                }

        class FakeRobotAgent:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def run(self, prompt):
                captured["prompt"] = prompt
                return FakeOutput()

        robot_agent = types.ModuleType("robot_agent")
        core = types.ModuleType("robot_agent.core")
        agent_module = types.ModuleType("robot_agent.core.agent")
        agent_module.RobotAgent = FakeRobotAgent
        modules = {
            "robot_agent": robot_agent,
            "robot_agent.core": core,
            "robot_agent.core.agent": agent_module,
        }
        backend = object()
        scene = SimpleNamespace(scene_name="factory", map_name="factory_map")
        grid = object()
        task = dict(self.task, scene_prefix="factory_sorting_1_example")

        with mock.patch.dict(sys.modules, modules):
            result = experiment_module.execute_task(
                execution_mode="agent",
                backend=backend,
                scene_context=scene,
                grid=grid,
                task=task,
                task_index=0,
                max_attempts=1,
            )

        self.assertEqual(result["skill_name"], "competition_task")
        self.assertTrue(captured["prompt"].strip())
        self.assertIs(captured["kwargs"]["backend"], backend)
        self.assertIs(captured["kwargs"]["scene_context"], scene)
        self.assertIs(captured["kwargs"]["grid"], grid)
        self.assertEqual(
            captured["kwargs"]["scene_metadata"]["task_index"],
            0,
        )
        self.assertEqual(
            captured["kwargs"]["scene_metadata"]["input_object_map"],
            {"input_5": "box_near"},
        )

    def test_execute_task_flow_mode_preserves_direct_workflow(self):
        captured = {}

        def run_official_task(**kwargs):
            captured.update(kwargs)
            return {"success": True, "states": {"box_near": "verified"}}

        robot_agent = types.ModuleType("robot_agent")
        workflows = types.ModuleType("robot_agent.workflows")
        flow_module = types.ModuleType("robot_agent.workflows.competition_flow")
        flow_module.run_official_task = run_official_task
        modules = {
            "robot_agent": robot_agent,
            "robot_agent.workflows": workflows,
            "robot_agent.workflows.competition_flow": flow_module,
        }

        with mock.patch.dict(sys.modules, modules):
            result = experiment_module.execute_task(
                execution_mode="flow",
                backend="backend",
                scene_context="scene",
                grid="grid",
                task=self.task,
                task_index=0,
                max_attempts=2,
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["max_attempts"], 2)
        self.assertEqual(captured["task"], self.task)

    def test_main_captures_started_at_before_running_experiment(self):
        clock_calls = []

        def fake_clock():
            clock_calls.append("start")
            return "2026-07-27T12:00:00+00:00"

        def fake_run(_args):
            self.assertEqual(clock_calls, ["start"])
            return {
                "status": "complete",
                "official_score": 10,
                "successful_grasp_events": 1,
                "required_grasp_events": 1,
                "collision_frames": 0,
                "final_target_distance_m": 0.1,
                "execution_result": {"success": True},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "manifest.json"
            with (
                mock.patch.object(experiment_module, "_utc_now", side_effect=fake_clock),
                mock.patch.object(experiment_module, "run_experiment", side_effect=fake_run),
                mock.patch("builtins.print"),
            ):
                exit_code = experiment_module.main(
                    [
                        "--candidate-root", temp_dir,
                        "--expected-official-commit", "a" * 40,
                        "--workspace-commit", "b" * 40,
                        "--trajectory", str(Path(temp_dir) / "trajectory.json"),
                        "--output", str(output),
                        "--required-score", "10",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["runner"]["started_at"],
                "2026-07-27T12:00:00+00:00",
            )
            self.assertEqual(manifest["runner"]["execution_mode"], "flow")


if __name__ == "__main__":
    unittest.main()
