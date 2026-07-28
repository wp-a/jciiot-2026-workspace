import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import run_official_experiment as experiment_module
from scripts.perturbation_protocol import PerturbationSample
from scripts.run_official_experiment import (
    acceptance_met,
    apply_perturbation,
    audit_trajectory,
    resolve_scored_object,
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
            experiment_module.build_parser().parse_args(required).perturbation_tier,
            "nominal",
        )
        self.assertIsNone(
            experiment_module.build_parser().parse_args(required).perturbation_object,
        )
        self.assertEqual(
            experiment_module.build_parser().parse_args(
                required
                + [
                    "--execution-mode", "agent",
                    "--perturbation-tier", "small",
                    "--perturbation-object", "box_far",
                ]
            ).execution_mode,
            "agent",
        )

    def test_resolve_scored_object_uses_requested_or_first_candidate(self):
        self.assertEqual(resolve_scored_object(self.task), "box_near")
        self.assertEqual(
            resolve_scored_object(self.task, requested_name="box_far"),
            "box_far",
        )
        with self.assertRaisesRegex(ValueError, "not a scored candidate"):
            resolve_scored_object(self.task, requested_name="unknown")

    def test_audit_preserves_requested_and_measured_perturbation(self):
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
            perturbation={"tier": "small", "object_dx_m": 0.01},
            perturbation_application={"valid": True, "measured_object_dx_m": 0.01},
        )

        self.assertEqual(manifest["perturbation"]["tier"], "small")
        self.assertTrue(manifest["perturbation_application"]["valid"])

    def test_apply_perturbation_changes_only_target_object_and_delegates_base(self):
        class FakeData:
            def __init__(self):
                self.qpos = np.array([1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0])
                self.qvel = np.ones(6)
                self.set_qpos_calls = 0
                self.set_qvel_calls = 0

            def get_joint_qpos(self, _joint_name):
                return self.qpos.copy()

            def set_joint_qpos(self, _joint_name, value):
                self.set_qpos_calls += 1
                self.qpos = np.asarray(value, dtype=float).copy()

            def set_joint_qvel(self, _joint_name, value):
                self.set_qvel_calls += 1
                self.qvel = np.asarray(value, dtype=float).copy()

        class FakeModel:
            def __init__(self):
                self.body_parentid = np.array([0, 0, 1, 0])
                self.body_mass = np.array([0.0, 2.0, 1.0, 4.0])
                self.geom_bodyid = np.array([1, 2, 3])
                self.geom_friction = np.array(
                    [[1.0, 0.2, 0.1], [0.5, 0.1, 0.05], [3.0, 0.3, 0.2]],
                    dtype=float,
                )

        class FakeSim:
            def __init__(self):
                self.model = FakeModel()
                self.data = FakeData()
                self.forward_calls = 0

            def forward(self):
                self.forward_calls += 1

        raw_env = SimpleNamespace(
            sim=FakeSim(),
            obj_body_id={"box_near": 1, "box_far": 3},
            material_metadata={"box_near": {"joint_name": "box_near_free"}},
        )

        class FakeBackend:
            def __init__(self):
                self.env = raw_env
                self.base_xy = np.array([5.0, 6.0], dtype=float)
                self.base_yaw = 0.25

            def get_base_pose(self):
                return self.base_xy.copy(), self.base_yaw

        backend = FakeBackend()
        base_calls = []

        def set_base_pose(target_backend, xy, yaw):
            base_calls.append((np.asarray(xy, dtype=float).copy(), float(yaw)))
            target_backend.base_xy = np.asarray(xy, dtype=float).copy()
            target_backend.base_yaw = float(yaw)

        sample = PerturbationSample(
            tier="medium",
            seed=7,
            task_index=0,
            object_name="box_near",
            object_dx_m=0.02,
            object_dy_m=-0.01,
            object_dyaw_rad=0.10,
            base_dx_m=-0.03,
            base_dy_m=0.02,
            base_dyaw_rad=-0.04,
            mass_scale=1.10,
            friction_scale=0.90,
            generator_digest="c" * 64,
        )

        audit = apply_perturbation(
            backend,
            self.task,
            sample,
            base_pose_setter=set_base_pose,
        )

        np.testing.assert_allclose(raw_env.sim.data.qpos[:3], [1.02, 1.99, 0.8])
        np.testing.assert_allclose(
            raw_env.sim.data.qpos[3:7],
            [np.cos(0.05), 0.0, 0.0, np.sin(0.05)],
        )
        np.testing.assert_allclose(raw_env.sim.data.qvel, np.zeros(6))
        np.testing.assert_allclose(raw_env.sim.model.body_mass, [0.0, 2.2, 1.1, 4.0])
        np.testing.assert_allclose(
            raw_env.sim.model.geom_friction,
            [[0.9, 0.18, 0.09], [0.45, 0.09, 0.045], [3.0, 0.3, 0.2]],
        )
        self.assertEqual(len(base_calls), 1)
        np.testing.assert_allclose(base_calls[0][0], [4.97, 6.02])
        self.assertAlmostEqual(base_calls[0][1], 0.21)
        self.assertEqual(raw_env.sim.forward_calls, 1)
        self.assertTrue(audit["valid"])
        self.assertAlmostEqual(audit["measured_object_dx_m"], 0.02)
        self.assertAlmostEqual(audit["measured_base_dy_m"], 0.02)

    def test_nominal_perturbation_is_a_measured_noop(self):
        class FakeData:
            def __init__(self):
                self.qpos = np.array([1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0])

            def get_joint_qpos(self, _joint_name):
                return self.qpos.copy()

            def set_joint_qpos(self, *_args):
                raise AssertionError("nominal sample must not write object state")

            def set_joint_qvel(self, *_args):
                raise AssertionError("nominal sample must not write object velocity")

        sim = SimpleNamespace(
            data=FakeData(),
            model=SimpleNamespace(
                body_parentid=np.array([0, 0]),
                body_mass=np.array([0.0, 1.0]),
                geom_bodyid=np.array([1]),
                geom_friction=np.array([[1.0, 0.1, 0.01]]),
            ),
            forward=lambda: None,
        )
        backend = SimpleNamespace(
            env=SimpleNamespace(
                sim=sim,
                obj_body_id={"box_near": 1},
                material_metadata={"box_near": {"joint_name": "box_near_free"}},
            ),
            get_base_pose=lambda: (np.array([5.0, 6.0]), 0.25),
        )
        sample = PerturbationSample(
            tier="nominal",
            seed=7,
            task_index=0,
            object_name="box_near",
            object_dx_m=0.0,
            object_dy_m=0.0,
            object_dyaw_rad=0.0,
            base_dx_m=0.0,
            base_dy_m=0.0,
            base_dyaw_rad=0.0,
            mass_scale=1.0,
            friction_scale=1.0,
            generator_digest="d" * 64,
        )

        audit = apply_perturbation(
            backend,
            self.task,
            sample,
            base_pose_setter=lambda *_args: (_ for _ in ()).throw(
                AssertionError("nominal sample must not write base state")
            ),
        )

        self.assertTrue(audit["valid"])
        self.assertTrue(audit["nominal_noop"])

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
