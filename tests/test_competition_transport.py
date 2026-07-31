import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "skills"
    / "competition_transport.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("competition_transport", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PhysicalTransportGeometryTests(unittest.TestCase):
    def test_module_exists_in_allowed_skills_directory(self):
        self.assertTrue(MODULE_PATH.is_file())

    def test_world_velocity_rotates_into_base_frame(self):
        module = load_module()

        actual = module.world_velocity_to_base_frame(
            np.array([1.0, 0.0]),
            math.pi / 2.0,
        )

        np.testing.assert_allclose(actual, [0.0, -1.0], atol=1e-9)

    def test_slew_limited_command_bounds_each_control_dimension(self):
        module = load_module()

        actual = module.slew_limited_command(
            np.array([0.1, -0.1, 0.0]),
            np.array([0.5, -0.5, 0.2]),
            max_delta=np.array([0.05, 0.10, 0.02]),
        )

        np.testing.assert_allclose(actual, [0.15, -0.20, 0.02], atol=1e-9)

    def test_transport_base_goal_preserves_measured_hold_offset(self):
        module = load_module()
        base_xy = np.array([1.0, 2.0])
        base_yaw = math.pi / 2.0
        object_xy = np.array([0.4, 2.3])
        object_target_xy = np.array([7.0, -1.0])

        goal = module.transport_base_goal(
            object_target_xy=object_target_xy,
            base_xy=base_xy,
            base_yaw=base_yaw,
            object_xy=object_xy,
        )

        measured_offset_world = object_xy - base_xy
        np.testing.assert_allclose(
            goal + measured_offset_world,
            object_target_xy,
            atol=1e-9,
        )

    def test_contact_stability_resets_when_either_gripper_loses_contact(self):
        module = load_module()

        self.assertEqual(
            module.next_contact_stability(
                {"right": True, "left": True},
                stable_steps=4,
            ),
            5,
        )
        self.assertEqual(
            module.next_contact_stability(
                {"right": True, "left": False},
                stable_steps=4,
            ),
            0,
        )

    def test_planar_grasp_drift_is_invariant_to_rigid_base_rotation(self):
        module = load_module()
        start = {
            "base_yaw": 0.0,
            "object_pos": np.array([0.5, 0.0, 1.0]),
            "gripper_positions": {
                "right": np.array([0.5, -0.2, 1.0]),
                "left": np.array([0.5, 0.2, 1.0]),
            },
        }
        rotated = {
            "base_yaw": math.pi / 2.0,
            "object_pos": np.array([0.0, 0.5, 1.0]),
            "gripper_positions": {
                "right": np.array([0.2, 0.5, 1.0]),
                "left": np.array([-0.2, 0.5, 1.0]),
            },
        }

        self.assertAlmostEqual(module.planar_grasp_drift(start, rotated), 0.0)

    def test_planar_reseat_moves_only_farther_gripper_toward_object(self):
        module = load_module()

        deltas = module.unilateral_planar_reseat_deltas(
            {
                "right": np.array([7.4, 4.62, 1.3]),
                "left": np.array([7.4, 4.40, 1.3]),
            },
            object_position=np.array([7.4, 4.58, 1.3]),
            inward_delta=0.002,
        )

        np.testing.assert_allclose(deltas["right"], np.zeros(2))
        np.testing.assert_allclose(deltas["left"], [0.0, 0.002])

    def test_vertical_hold_delta_adds_feedforward_and_corrects_height_error(self):
        module = load_module()

        steady = module.vertical_hold_delta(
            current_z=1.30,
            target_z=1.30,
            feedforward=0.0004,
            gain=0.8,
            max_delta=0.003,
        )
        correcting = module.vertical_hold_delta(
            current_z=1.298,
            target_z=1.30,
            feedforward=0.0004,
            gain=0.8,
            max_delta=0.003,
        )

        self.assertAlmostEqual(steady, 0.0004)
        self.assertAlmostEqual(correcting, 0.0020)

    def test_direct_base_step_rotates_command_and_bounds_displacement(self):
        module = load_module()

        target = module.direct_base_step_target(
            base_xy=np.array([1.0, 2.0]),
            base_yaw=math.pi / 2.0,
            base_command=np.array([0.4, 0.0, 0.0]),
            control_dt=0.05,
        )

        np.testing.assert_allclose(target, [1.0, 2.02], atol=1e-9)

    def test_default_direct_base_step_is_no_more_than_six_millimetres(self):
        module = load_module()
        config = module.PhysicalCarryConfig()

        target = module.direct_base_step_target(
            base_xy=np.zeros(2),
            base_yaw=0.0,
            base_command=np.array([config.max_linear, 0.0, 0.0]),
            control_dt=config.base_control_dt,
        )

        self.assertLessEqual(float(np.linalg.norm(target)), 0.006 + 1e-12)

    def test_pivot_compensation_preserves_object_offset_during_turn(self):
        module = load_module()
        base_xy = np.array([0.0, 0.0])
        pivot_xy = np.array([1.0, 0.0])

        base_velocity = module.pivot_compensated_base_velocity(
            base_xy=base_xy,
            base_yaw=0.0,
            pivot_xy=pivot_xy,
            angular_velocity=math.pi / 2.0,
            control_dt=1.0,
        )
        target_base = module.direct_base_step_target(
            base_xy=base_xy,
            base_yaw=0.0,
            base_command=np.array(
                [base_velocity[0], base_velocity[1], math.pi / 2.0]
            ),
            control_dt=1.0,
        )

        np.testing.assert_allclose(target_base, [1.0, -1.0], atol=1e-9)
        rotated_offset = module.world_velocity_to_base_frame(
            pivot_xy - target_base,
            math.pi / 2.0,
        )
        np.testing.assert_allclose(rotated_offset, [1.0, 0.0], atol=1e-9)

    def test_grasp_pivot_is_midpoint_of_bilateral_grippers(self):
        module = load_module()

        pivot = module.bilateral_grasp_pivot_xy(
            {
                "right": np.array([1.2, -0.3, 0.9]),
                "left": np.array([1.2, 0.5, 0.9]),
            }
        )

        np.testing.assert_allclose(pivot, [1.2, 0.1])

    def test_long_route_step_budget_covers_slow_physical_carry(self):
        module = load_module()

        budget = module.physical_carry_step_budget(
            [np.array([1.0, 0.0]), np.array([1.0, 2.0])],
            start_xy=np.array([0.0, 0.0]),
            max_linear=0.04,
            control_dt=0.05,
        )

        nominal_steps = math.ceil(3.0 / (0.04 * 0.05))
        self.assertGreaterEqual(budget, nominal_steps * 2)


class PostureLockedPhysicalCarryDriverTests(unittest.TestCase):
    def test_declares_that_locked_posture_skips_arm_recenter(self):
        module = load_module()

        self.assertFalse(
            module.PostureLockedPhysicalCarryDriver.requires_height_recenter
        )

    def test_restores_only_non_gripper_robot_posture_after_actuated_step(self):
        module = load_module()
        forward_calls = []

        class Delegate:
            def capture_hold_targets(self, _backend):
                return {"torso": np.array([0.2])}

            def step(self, backend, **_kwargs):
                backend.env.sim.data.qpos[:] += 10.0
                backend.env.sim.data.qvel[:] += 20.0
                return {"collision": False}

        addresses = {"arm": 0, "torso": 1, "head": 2, "gripper": 3}
        model = SimpleNamespace(
            get_joint_qpos_addr=addresses.__getitem__,
            get_joint_qvel_addr=addresses.__getitem__,
        )
        robot = SimpleNamespace(
            robot_arm_joints=["arm"],
            robot_model=SimpleNamespace(
                torso_joints=["torso"],
                head_joints=["head"],
            ),
        )
        backend = SimpleNamespace(
            env=SimpleNamespace(
                robots=[robot],
                sim=SimpleNamespace(
                    model=model,
                    data=SimpleNamespace(
                        qpos=np.array([1.0, 2.0, 3.0, 4.0]),
                        qvel=np.array([5.0, 6.0, 7.0, 8.0]),
                    ),
                    forward=lambda: forward_calls.append(True),
                ),
            ),
            _record_trajectory_frame=lambda **_kwargs: None,
        )
        driver = module.PostureLockedPhysicalCarryDriver(Delegate())

        driver.capture_hold_targets(backend)
        result = driver.step(backend)

        self.assertFalse(result["collision"])
        np.testing.assert_allclose(backend.env.sim.data.qpos, [1, 2, 3, 14])
        np.testing.assert_allclose(backend.env.sim.data.qvel, [5, 6, 7, 28])
        self.assertEqual(forward_calls, [True])

    def test_partial_height_recovery_refreshes_locked_posture(self):
        module = load_module()

        class Delegate:
            def recover_height(self, _backend, **_kwargs):
                return False

        driver = module.PostureLockedPhysicalCarryDriver(Delegate())
        refreshed = {"qpos": np.array([2.0])}
        driver._posture = {"qpos": np.array([1.0])}
        driver._capture_robot_posture = lambda _backend: refreshed

        result = driver.recover_height(object())

        self.assertFalse(result)
        self.assertIs(driver._posture, refreshed)

    def test_planar_reseat_refreshes_locked_posture(self):
        module = load_module()

        class Delegate:
            def step(self, _backend, **_kwargs):
                return {"collision": False}

        driver = module.PostureLockedPhysicalCarryDriver(Delegate())
        refreshed = {"qpos": np.array([2.0])}
        driver._posture = {"qpos": np.array([1.0])}
        driver._capture_robot_posture = lambda _backend: refreshed

        result = driver.recover_planar(object(), base_command=np.zeros(3))

        self.assertFalse(result["collision"])
        self.assertIs(driver._posture, refreshed)

    def test_single_arm_under_support_target_descends_and_moves_toward_midpoint(self):
        module = load_module()
        current = {
            "right": np.array([7.25, 4.82, 1.36]),
            "left": np.array([7.25, 4.42, 1.36]),
        }

        right_targets = module.single_arm_under_support_targets(
            current,
            moving_arm="right",
            separation_axis=np.array([0.0, 1.0, 0.0]),
            descent_m=0.12,
            inset_m=0.06,
        )
        left_targets = module.single_arm_under_support_targets(
            current,
            moving_arm="left",
            separation_axis=np.array([0.0, 1.0, 0.0]),
            descent_m=0.12,
            inset_m=0.06,
        )

        np.testing.assert_allclose(right_targets["right"], [7.25, 4.76, 1.24])
        np.testing.assert_allclose(right_targets["left"], current["left"])
        np.testing.assert_allclose(left_targets["left"], [7.25, 4.48, 1.24])
        np.testing.assert_allclose(left_targets["right"], current["right"])

    def test_single_arm_under_support_target_rejects_invalid_geometry(self):
        module = load_module()
        current = {"right": np.ones(3), "left": np.zeros(3)}

        invalid_cases = (
            {"moving_arm": "middle"},
            {"separation_axis": np.zeros(3)},
            {"descent_m": -0.01},
            {"inset_m": -0.01},
            {"separation_axis": np.array([0.0, np.nan, 0.0])},
        )
        defaults = {
            "moving_arm": "right",
            "separation_axis": np.array([0.0, 1.0, 0.0]),
            "descent_m": 0.12,
            "inset_m": 0.06,
        }
        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                module.single_arm_under_support_targets(
                    current,
                    **{**defaults, **override},
                )


class FakePhysicalTransportDriver:
    def __init__(
        self,
        *,
        contacts=None,
        object_heights=None,
        collision_step=None,
        advance=True,
        recover_success=True,
        recovered_height_result=None,
        planar_slip_per_step=0.0,
        gripper_xy_offsets=None,
    ):
        self.base_xy = np.zeros(2, dtype=float)
        self.object_xy = np.array([0.5, 0.0], dtype=float)
        self.yaw = 0.0
        self.contacts = list(contacts or [{"right": True, "left": True}])
        self.object_heights = list(object_heights or [1.0])
        self.collision_step = collision_step
        self.advance = bool(advance)
        self.recover_success = bool(recover_success)
        self.recovered_height_result = recovered_height_result
        self.planar_slip_per_step = float(planar_slip_per_step)
        self.gripper_xy_offsets = {
            arm: np.asarray(offset, dtype=float).copy()
            for arm, offset in (
                gripper_xy_offsets
                or {"right": np.zeros(2), "left": np.zeros(2)}
            ).items()
        }
        self.recovered_height = None
        self.recover_calls = []
        self.planar_recoveries = []
        self.gripper_z = {"right": 1.01, "left": 1.01}
        self.steps = []
        self.events = []

    def capture_hold_targets(self, _backend):
        return {"torso": np.array([0.3]), "head": np.array([0.1, -0.1])}

    def observe(self, _backend, _object_name):
        index = min(len(self.steps), len(self.contacts) - 1)
        height_index = min(len(self.steps), len(self.object_heights) - 1)
        object_z = (
            self.recovered_height
            if self.recovered_height is not None
            else self.object_heights[height_index]
        )
        return {
            "base_xy": self.base_xy.copy(),
            "base_yaw": self.yaw,
            "object_pos": np.array(
                [self.object_xy[0], self.object_xy[1], object_z],
                dtype=float,
            ),
            "contacts": dict(self.contacts[index]),
            "gripper_positions": {
                arm: np.array(
                    [
                        self.object_xy[0]
                        + self.gripper_xy_offsets[arm][0]
                        + len(self.steps) * self.planar_slip_per_step,
                        self.object_xy[1] + self.gripper_xy_offsets[arm][1],
                        self.gripper_z[arm],
                    ],
                    dtype=float,
                )
                for arm in ("right", "left")
            },
        }

    def step(
        self,
        _backend,
        *,
        object_name,
        base_command,
        hold_targets,
        arm_world_deltas=None,
        gripper_value=1.0,
        base_control_dt=0.05,
    ):
        command = np.asarray(base_command, dtype=float).copy()
        self.steps.append(
            {
                "object_name": object_name,
                "base_command": command,
                "hold_targets": hold_targets,
                "arm_world_deltas": arm_world_deltas,
                "gripper_value": float(gripper_value),
                "base_control_dt": float(base_control_dt),
            }
        )
        if self.advance:
            self.base_xy += command[:2]
            self.yaw += command[2]
        if arm_world_deltas:
            planar_deltas = [
                np.asarray(arm_world_deltas[arm], dtype=float)[:2]
                for arm in ("right", "left")
            ]
            self.object_xy += np.mean(planar_deltas, axis=0)
            for arm in ("right", "left"):
                self.gripper_z[arm] += float(arm_world_deltas[arm][2])
        return {"collision": self.collision_step == len(self.steps)}

    def record_event(self, _backend, event, **payload):
        self.events.append((event, payload))
        return (event, payload)

    def recover_height(
        self,
        _backend,
        *,
        object_name,
        lift_height,
        max_steps,
        max_action,
    ):
        self.recover_calls.append(
            {
                "object_name": object_name,
                "lift_height": float(lift_height),
                "max_steps": int(max_steps),
                "max_action": float(max_action),
            }
        )
        if self.recovered_height_result is not None:
            self.recovered_height = float(self.recovered_height_result)
        elif self.recover_success:
            self.recovered_height = 1.0
        return self.recover_success

    def recover_planar(self, backend, **kwargs):
        self.planar_recoveries.append(kwargs)
        return self.step(backend, **kwargs)


class FakeHeadingAlignedTransportDriver(FakePhysicalTransportDriver):
    def __init__(self):
        super().__init__()
        self.relative_object = np.array([0.5, 0.0], dtype=float)
        self.gripper_offsets = {
            "right": np.array([0.5, -0.2], dtype=float),
            "left": np.array([0.5, 0.2], dtype=float),
        }

    def _rotation(self):
        return np.array(
            [
                [math.cos(self.yaw), -math.sin(self.yaw)],
                [math.sin(self.yaw), math.cos(self.yaw)],
            ]
        )

    def observe(self, _backend, _object_name):
        rotation = self._rotation()
        object_xy = self.base_xy + rotation @ self.relative_object
        return {
            "base_xy": self.base_xy.copy(),
            "base_yaw": self.yaw,
            "object_pos": np.array([*object_xy, 1.0]),
            "contacts": {"right": True, "left": True},
            "gripper_positions": {
                arm: np.array(
                    [*(self.base_xy + rotation @ offset), 1.01]
                )
                for arm, offset in self.gripper_offsets.items()
            },
        }

    def step(self, _backend, **kwargs):
        command = np.asarray(kwargs["base_command"], dtype=float)
        dt = float(kwargs.get("base_control_dt", 0.05))
        rotation = self._rotation()
        self.base_xy += rotation @ command[:2] * dt
        self.yaw += command[2] * dt
        self.steps.append(
            {
                "base_command": command.copy(),
                "arm_world_deltas": kwargs["arm_world_deltas"],
            }
        )
        return {"collision": False}


class FakeInchwormDriver:
    def __init__(self):
        self.base_xy = np.zeros(2, dtype=float)
        self.object_pos = np.array([0.5, 0.0, 1.0], dtype=float)
        self.grippers = {
            "right": np.array([0.5, -0.2, 1.01], dtype=float),
            "left": np.array([0.5, 0.2, 1.01], dtype=float),
        }
        self.steps = []

    def capture_hold_targets(self, _backend):
        return {}

    def observe(self, _backend, _object_name):
        return {
            "base_xy": self.base_xy.copy(),
            "base_yaw": 0.0,
            "object_pos": self.object_pos.copy(),
            "contacts": {"right": True, "left": True},
            "gripper_positions": {
                arm: position.copy() for arm, position in self.grippers.items()
            },
        }

    def step(
        self,
        _backend,
        *,
        object_name,
        base_command,
        hold_targets,
        arm_world_deltas=None,
        gripper_value=1.0,
        base_control_dt=0.05,
    ):
        del object_name, hold_targets, gripper_value
        command = np.asarray(base_command, dtype=float)
        base_step = command[:2] * float(base_control_dt)
        deltas = {
            arm: np.asarray((arm_world_deltas or {}).get(arm, np.zeros(3)), dtype=float)
            for arm in ("right", "left")
        }
        self.steps.append({"base_step": base_step.copy(), "arm_deltas": deltas})
        self.base_xy += base_step
        for arm in ("right", "left"):
            self.grippers[arm][:2] += base_step + deltas[arm][:2]
            self.grippers[arm][2] += deltas[arm][2]
        if not np.any(np.abs(base_step) > 0.0):
            self.object_pos += np.mean(np.stack(list(deltas.values())), axis=0)
        return {"collision": False}

    def record_event(self, _backend, event, **payload):
        return event, payload


class PhysicalTransportRunnerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            k_angular=1.0,
            max_linear=0.10,
            max_angular=0.05,
            max_linear_delta=0.10,
            max_angular_delta=0.05,
            object_drop_tolerance=0.02,
        )

    def run_transport(
        self,
        driver,
        *,
        path=None,
        config=None,
        minimum_object_z=0.98,
    ):
        self.assertTrue(hasattr(self.module, "run_physical_transport"))
        return self.module.run_physical_transport(
            object(),
            path=path or [np.array([0.18, 0.0])],
            object_name="box",
            hold_yaw=0.0,
            minimum_object_z=minimum_object_z,
            config=config or self.config,
            driver=driver,
        )

    def test_success_reaches_path_with_only_physics_steps(self):
        driver = FakePhysicalTransportDriver()

        result = self.run_transport(driver)

        self.assertTrue(result["success"])
        self.assertIsNone(result["failure_stage"])
        self.assertGreaterEqual(len(driver.steps), 2)
        self.assertLess(float(result["final_distance"]), 0.02)
        self.assertTrue(
            all(
                step["arm_world_deltas"]["right"][2] == 0.0
                and step["arm_world_deltas"]["left"][2] == 0.0
                for step in driver.steps
            )
        )

    def test_single_gripper_contact_loss_fails_immediately(self):
        driver = FakePhysicalTransportDriver(
            contacts=[
                {"right": True, "left": True},
                {"right": True, "left": False},
            ]
        )

        result = self.run_transport(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "contact")
        self.assertEqual(len(driver.steps), 1)

    def test_transport_end_event_records_terminal_physical_state(self):
        driver = FakePhysicalTransportDriver(
            contacts=[
                {"right": True, "left": True},
                {"right": True, "left": False},
            ]
        )

        self.run_transport(driver)

        event, payload = driver.events[-1]
        self.assertEqual(event, "physical_transport_end")
        self.assertEqual(
            payload["contacts"], {"right": True, "left": False}
        )
        self.assertEqual(payload["final_object_pos"], [0.5, 0.0, 1.0])
        self.assertEqual(
            sorted(payload["final_gripper_positions"]), ["left", "right"]
        )

    def test_planar_grasp_drift_fails_even_while_contacts_remain_true(self):
        driver = FakePhysicalTransportDriver(planar_slip_per_step=0.03)
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            max_planar_grasp_drift=0.02,
        )

        result = self.run_transport(driver, config=config)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "planar_grasp_drift")
        self.assertGreater(result["max_planar_grasp_drift_m"], 0.02)
        self.assertEqual(result["contacts"], {"right": True, "left": True})

    def test_planar_drift_triggers_bounded_physical_reseat(self):
        driver = FakePhysicalTransportDriver(
            planar_slip_per_step=0.015,
            gripper_xy_offsets={
                "right": np.array([0.0, 0.05]),
                "left": np.array([0.0, -0.2]),
            },
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=30,
            k_linear=1.0,
            max_linear=0.05,
            max_linear_delta=0.05,
            max_planar_grasp_drift=0.50,
            planar_recovery_trigger=0.015,
            planar_recovery_steps=1,
            planar_recovery_inward_delta=0.002,
        )

        result = self.run_transport(driver, config=config)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(driver.planar_recoveries), 1)
        recovery = driver.planar_recoveries[0]["arm_world_deltas"]
        np.testing.assert_allclose(recovery["right"], np.zeros(3))
        self.assertAlmostEqual(np.linalg.norm(recovery["left"][:2]), 0.002)

    def test_transport_result_records_start_and_final_physical_poses(self):
        driver = FakePhysicalTransportDriver()

        result = self.run_transport(driver)

        self.assertEqual(result["start_object_pos"], [0.5, 0.0, 1.0])
        self.assertEqual(result["final_object_pos"], [0.5, 0.0, 1.0])
        self.assertEqual(sorted(result["start_gripper_positions"]), ["left", "right"])
        self.assertEqual(sorted(result["final_gripper_positions"]), ["left", "right"])
        self.assertAlmostEqual(result["max_planar_grasp_drift_m"], 0.0)

    def test_object_drop_fails_immediately(self):
        driver = FakePhysicalTransportDriver(object_heights=[1.0, 0.97])

        result = self.run_transport(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "object_drop")
        self.assertEqual(len(driver.steps), 1)

    def test_collision_fails_immediately(self):
        driver = FakePhysicalTransportDriver(collision_step=1)

        result = self.run_transport(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "collision")
        self.assertEqual(len(driver.steps), 1)

    def test_step_budget_exhaustion_has_no_fallback(self):
        driver = FakePhysicalTransportDriver(advance=False)
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=3,
            max_linear_delta=0.10,
            max_angular_delta=0.05,
        )

        result = self.run_transport(driver, config=config)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "timeout")
        self.assertEqual(len(driver.steps), 3)

    def test_direct_base_step_does_not_add_planar_arm_lead(self):
        driver = FakePhysicalTransportDriver()
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=10,
            k_linear=1.0,
            max_linear=0.08,
            max_linear_delta=0.08,
        )

        result = self.run_transport(
            driver,
            path=[np.array([0.08, 0.0])],
            config=config,
        )

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.steps), 1)
        self.assertGreater(driver.steps[0]["base_command"][0], 0.0)
        np.testing.assert_allclose(
            driver.steps[0]["arm_world_deltas"]["right"][:2], np.zeros(2)
        )

    def test_heading_aligned_mode_pivots_around_object_before_path_motion(self):
        driver = FakeHeadingAlignedTransportDriver()
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.01,
            max_steps=500,
            max_linear=0.10,
            max_angular=0.50,
            max_linear_delta=0.10,
            max_angular_delta=0.50,
            base_control_dt=0.05,
            align_heading_to_path=True,
            heading_translation_tolerance=0.05,
        )

        result = self.run_transport(
            driver,
            path=[np.array([0.0, 0.10])],
            config=config,
        )

        self.assertTrue(result["success"])
        rotating = [
            step for step in driver.steps if abs(step["base_command"][2]) > 0.01
        ]
        translating = [
            step
            for step in driver.steps
            if np.linalg.norm(step["base_command"][:2]) > 0.01
        ]
        self.assertTrue(rotating)
        self.assertTrue(translating)
        first_translation = next(
            index
            for index, step in enumerate(driver.steps)
            if np.linalg.norm(step["base_command"][:2]) > 0.01
        )
        self.assertEqual(first_translation, 0)
        self.assertGreater(abs(driver.steps[0]["base_command"][2]), 0.01)

    def test_heading_aligned_mode_can_rotate_without_base_translation(self):
        driver = FakeHeadingAlignedTransportDriver()
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.01,
            max_steps=500,
            max_linear=0.10,
            max_angular=0.50,
            max_linear_delta=0.10,
            max_angular_delta=0.50,
            base_control_dt=0.05,
            align_heading_to_path=True,
            pivot_compensation_enabled=False,
            heading_translation_tolerance=0.05,
        )

        result = self.run_transport(
            driver,
            path=[np.array([0.0, 0.10])],
            config=config,
        )

        self.assertTrue(result["success"])
        self.assertTrue(
            any(abs(step["base_command"][2]) > 0.01 for step in driver.steps)
        )
        first_rotating = next(
            step
            for step in driver.steps
            if abs(step["base_command"][2]) > 0.01
        )
        self.assertLessEqual(
            np.linalg.norm(first_rotating["base_command"][:2]),
            1e-12,
        )

    def test_object_slip_triggers_a_physical_height_recovery(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.985, 0.985, 0.985]
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            height_recovery_trigger=0.01,
            height_recovery_steps=40,
        )

        result = self.run_transport(driver, config=config)

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.recover_calls), 1)
        self.assertGreater(driver.recover_calls[0]["lift_height"], 0.0)
        self.assertTrue(
            any(
                step["arm_world_deltas"]["right"][2] < 0.0
                and step["arm_world_deltas"]["left"][2] < 0.0
                for step in driver.steps
            )
        )

    def test_height_recovery_tracks_initial_hold_above_safety_floor(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.985, 0.985, 0.985]
        )

        result = self.run_transport(driver, minimum_object_z=0.90)

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.recover_calls), 1)
        self.assertAlmostEqual(driver.recover_calls[0]["lift_height"], 0.015)

    def test_height_recovery_allows_configured_post_lift_settling(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.975, 0.975, 0.975]
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            height_recovery_trigger=0.004,
            height_settle_allowance=0.02,
            height_safety_margin=0.01,
        )

        result = self.run_transport(
            driver,
            config=config,
            minimum_object_z=0.90,
        )

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.recover_calls), 1)
        self.assertAlmostEqual(driver.recover_calls[0]["lift_height"], 0.005)

    def test_locked_posture_skips_recenter_before_height_recovery(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.985, 0.985, 0.985]
        )
        driver.requires_height_recenter = False

        result = self.run_transport(driver)

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.recover_calls), 1)
        self.assertFalse(
            any(
                step["arm_world_deltas"]["right"][2] < 0.0
                or step["arm_world_deltas"]["left"][2] < 0.0
                for step in driver.steps
            )
        )

    def test_failed_height_recovery_stops_transport_without_fallback(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.985, 0.985, 0.985],
            recover_success=False,
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            height_recovery_trigger=0.01,
        )

        result = self.run_transport(driver, config=config)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "height_recovery")
        self.assertEqual(len(driver.recover_calls), 1)

    def test_disabled_height_recovery_preserves_physical_drop_gate(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.975, 0.975, 0.975]
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            height_recovery_enabled=False,
            height_recovery_trigger=0.01,
        )

        result = self.run_transport(
            driver,
            config=config,
            minimum_object_z=0.98,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "object_drop")
        self.assertEqual(driver.recover_calls, [])

    def test_measured_recovery_accepts_height_even_if_helper_target_was_stricter(self):
        driver = FakePhysicalTransportDriver(
            object_heights=[1.0, 0.985, 0.985, 0.985],
            recover_success=False,
            recovered_height_result=0.995,
        )
        config = self.module.PhysicalCarryConfig(
            waypoint_tolerance=0.02,
            max_steps=20,
            k_linear=1.0,
            max_linear=0.10,
            max_linear_delta=0.10,
            height_recovery_trigger=0.01,
        )

        result = self.run_transport(driver, config=config)

        self.assertTrue(result["success"])
        self.assertEqual(len(driver.recover_calls), 1)

    def test_physical_action_contains_every_controlled_part(self):
        self.assertTrue(hasattr(self.module, "physical_action_parts"))
        robot = SimpleNamespace(
            composite_controller=SimpleNamespace(
                _action_split_indexes={
                    "right": (0, 6),
                    "right_gripper": (6, 7),
                    "left": (7, 13),
                    "left_gripper": (13, 14),
                    "torso": (14, 15),
                    "head": (15, 17),
                    "base": (17, 20),
                }
            ),
            gripper={
                "right": SimpleNamespace(dof=1),
                "left": SimpleNamespace(dof=1),
            },
        )

        action = self.module.physical_action_parts(
            robot,
            base_command=np.array([0.1, -0.2, 0.03]),
            gripper_value=1.0,
            hold_targets={
                "torso": np.array([0.25]),
                "head": np.array([0.1, -0.1]),
            },
        )

        self.assertEqual(
            set(action),
            {
                "right",
                "right_gripper",
                "left",
                "left_gripper",
                "torso",
                "head",
                "base",
            },
        )
        np.testing.assert_allclose(action["right"], np.zeros(6))
        np.testing.assert_allclose(action["left"], np.zeros(6))
        np.testing.assert_allclose(action["base"], [0.1, -0.2, 0.03])
        np.testing.assert_allclose(action["right_gripper"], [1.0])
        np.testing.assert_allclose(action["left_gripper"], [1.0])


class InchwormTransportRunnerTests(unittest.TestCase):
    def test_default_reset_stays_below_observed_contact_loss_distance(self):
        module = load_module()

        self.assertAlmostEqual(module.InchwormCarryConfig().reset_distance, 0.06)

    def test_stops_after_arm_stroke_when_measured_object_progress_reaches_target(self):
        module = load_module()
        driver = FakeInchwormDriver()
        config = module.InchwormCarryConfig(
            stroke_distance=0.08,
            stroke_vertical_feedforward=0.0,
            reset_distance=0.08,
            reset_max_linear=0.04,
            max_cycles=2,
        )

        result = module.run_inchworm_transport(
            object(),
            object_name="box",
            travel_direction=np.array([1.0, 0.0]),
            travel_distance=0.05,
            minimum_object_z=0.90,
            config=config,
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["cycle_count"], 1)
        self.assertGreaterEqual(result["object_progress_m"], 0.05)
        self.assertAlmostEqual(result["base_translation_m"], 0.0, places=6)
        self.assertTrue(
            any(np.linalg.norm(step["base_step"]) == 0.0 for step in driver.steps)
        )
        self.assertFalse(
            any(np.linalg.norm(step["base_step"]) > 0.0 for step in driver.steps)
        )

    def test_repeats_macro_cycles_until_measured_object_progress_reaches_target(self):
        module = load_module()
        driver = FakeInchwormDriver()
        config = module.InchwormCarryConfig(
            stroke_distance=0.08,
            stroke_vertical_feedforward=0.0,
            reset_distance=0.08,
            reset_max_linear=0.04,
            max_cycles=3,
        )

        result = module.run_inchworm_transport(
            object(),
            object_name="box",
            travel_direction=np.array([1.0, 0.0]),
            travel_distance=0.12,
            minimum_object_z=0.90,
            config=config,
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["cycle_count"], 2)
        self.assertGreaterEqual(result["object_progress_m"], 0.12)
        self.assertGreaterEqual(result["cycles"][1]["total_progress_m"], 0.12)
        self.assertAlmostEqual(result["base_translation_m"], 0.08, places=6)
        self.assertEqual(result["cycles"][1]["reset_steps"], 0)


class CradleTransferTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def observation(
        self,
        *,
        base_xy=(0.0, 0.0),
        object_z=1.0,
        minimum_object_z=0.95,
        right_support=("robot0_arm_6_collision",),
        left_support=("robot0_arm_6_left_collision",),
        right_drift=0.01,
        left_drift=0.01,
        max_drift=0.04,
        collision=False,
    ):
        return self.module.CradleObservation(
            base_xy=base_xy,
            object_z=object_z,
            minimum_object_z=minimum_object_z,
            gripper_contacts={"right": True, "left": True},
            support_contacts={
                "right": tuple(right_support),
                "left": tuple(left_support),
            },
            object_to_wrist_drift_m={
                "right": right_drift,
                "left": left_drift,
            },
            max_drift_m=max_drift,
            judge_collision=collision,
        )

    def test_cradle_support_requires_real_wrist_or_forearm_contact(self):
        supported = self.observation(
            right_support=("gripper0_right_hand_collision",),
            left_support=("robot0_arm_5_left_collision",),
        )
        table_contact_only = self.observation(
            right_support=("input_5_table_collision",),
        )

        self.assertTrue(self.module.is_cradle_supported(supported))
        self.assertFalse(self.module.is_cradle_supported(table_contact_only))

    def test_cradle_support_does_not_assign_a_link_to_the_wrong_arm(self):
        swapped = self.observation(
            right_support=("robot0_arm_6_left_collision",),
            left_support=("robot0_arm_6_collision",),
        )

        self.assertFalse(self.module.is_cradle_supported(swapped))

    def test_fingerpad_only_contact_is_not_cradle_support(self):
        fingerpads = self.observation(
            right_support=("gripper0_right_right_fingerpad_collision",),
            left_support=("gripper0_left_left_fingerpad_collision",),
        )

        self.assertFalse(self.module.is_cradle_supported(fingerpads))

    def test_cradle_stability_resets_on_contact_or_height_loss(self):
        stable = self.module.next_cradle_stability(self.observation(), 4)
        contact_loss = self.module.next_cradle_stability(
            self.observation(left_support=()),
            stable,
        )
        height_loss = self.module.next_cradle_stability(
            self.observation(object_z=0.94),
            stable,
        )

        self.assertEqual(stable, 5)
        self.assertEqual(contact_loss, 0)
        self.assertEqual(height_loss, 0)

    def test_cradle_delta_is_bounded_and_symmetric(self):
        deltas = self.module.bounded_symmetric_cradle_deltas(
            center_delta=np.zeros(3),
            separation_axis=np.array([0.0, 1.0, 0.0]),
            inward_delta=0.03,
            max_delta=0.012,
        )

        np.testing.assert_allclose(deltas["right"], -deltas["left"])
        self.assertLessEqual(np.linalg.norm(deltas["right"]), 0.012 + 1e-12)
        self.assertLessEqual(np.linalg.norm(deltas["left"]), 0.012 + 1e-12)

    def test_cradle_transfer_stops_on_judge_collision(self):
        driver = FakeCradleDriver([self.observation(collision=True)])

        result = self.module.run_physical_cradle_transfer(
            object(),
            object_name="box",
            travel_direction=np.array([1.0, 0.0]),
            travel_distance=0.5,
            required_stable_steps=2,
            driver=driver,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "collision")
        self.assertEqual(driver.steps, [])

    def test_cradle_transfer_never_calls_attachment_or_object_pose_helpers(self):
        driver = FakeCradleDriver([self.observation()])

        result = self.module.run_physical_cradle_transfer(
            object(),
            object_name="box",
            travel_direction=np.array([1.0, 0.0]),
            travel_distance=0.0,
            required_stable_steps=1,
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(driver.forbidden_calls, 0)
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("transport_attachment", source)
        self.assertNotIn("set_object_pose", source)


class FakeCradleDriver:
    def __init__(self, observations):
        self.observations = list(observations)
        self.steps = []
        self.forbidden_calls = 0

    def observe_cradle(self, _backend, _object_name):
        index = min(len(self.steps), len(self.observations) - 1)
        return self.observations[index]

    def step_cradle(
        self,
        _backend,
        *,
        object_name,
        base_world_delta,
        arm_world_deltas,
    ):
        self.steps.append(
            {
                "object_name": object_name,
                "base_world_delta": np.asarray(base_world_delta, dtype=float),
                "arm_world_deltas": arm_world_deltas,
            }
        )
        return {"collision": False}

    def record_event(self, _backend, _event, **_payload):
        return None

    def capture_transport_attachment(self, *_args, **_kwargs):
        self.forbidden_calls += 1
        raise AssertionError("attachment helper must not be called")

    def set_object_pose(self, *_args, **_kwargs):
        self.forbidden_calls += 1
        raise AssertionError("object pose helper must not be called")


class FakePhysicalPlacementDriver:
    def __init__(
        self,
        *,
        object_heights,
        contacts=None,
        object_xy=(0.1, 0.1),
        collision_step=None,
    ):
        self.object_heights = list(object_heights)
        self.contacts = list(contacts or [{"right": True, "left": True}])
        self.object_xy = np.asarray(object_xy, dtype=float)
        self.collision_step = collision_step
        self.steps = []

    def capture_hold_targets(self, _backend):
        return {"torso": np.array([0.3]), "head": np.array([0.0, 0.0])}

    def observe(self, _backend, _object_name):
        index = min(len(self.steps), len(self.object_heights) - 1)
        contact_index = min(len(self.steps), len(self.contacts) - 1)
        contacts = dict(self.contacts[contact_index])
        if self.steps and self.steps[-1]["gripper_value"] < 0.0:
            contacts = {"right": False, "left": False}
        return {
            "base_xy": np.zeros(2, dtype=float),
            "base_yaw": 0.0,
            "object_pos": np.array(
                [self.object_xy[0], self.object_xy[1], self.object_heights[index]],
                dtype=float,
            ),
            "contacts": contacts,
        }

    def step(
        self,
        _backend,
        *,
        object_name,
        base_command,
        hold_targets,
        arm_world_deltas=None,
        gripper_value=1.0,
    ):
        self.steps.append(
            {
                "object_name": object_name,
                "base_command": np.asarray(base_command, dtype=float).copy(),
                "hold_targets": hold_targets,
                "arm_world_deltas": arm_world_deltas,
                "gripper_value": float(gripper_value),
            }
        )
        return {"collision": self.collision_step == len(self.steps)}

    def record_event(self, _backend, event, **payload):
        return (event, payload)


class FakePhysicalAlignmentDriver:
    def __init__(self, *, collision_step=None, contacts=None):
        self.object_pos = np.array([0.0, 0.0, 1.0], dtype=float)
        self.grippers = {
            "right": np.array([0.0, -0.2, 1.0], dtype=float),
            "left": np.array([0.0, 0.2, 1.0], dtype=float),
        }
        self.collision_step = collision_step
        self.contacts = contacts or {"right": True, "left": True}
        self.steps = []

    def capture_hold_targets(self, _backend):
        return {"torso": np.array([0.3]), "head": np.array([0.0, 0.0])}

    def observe(self, _backend, _object_name):
        return {
            "base_xy": np.array([0.0, 0.0]),
            "base_yaw": 0.0,
            "object_pos": self.object_pos.copy(),
            "contacts": dict(self.contacts),
            "gripper_positions": {
                arm: position.copy() for arm, position in self.grippers.items()
            },
        }

    def step(self, _backend, **kwargs):
        self.steps.append(kwargs)
        delta = np.mean(
            np.stack(list(kwargs["arm_world_deltas"].values())),
            axis=0,
        )
        self.object_pos += delta
        for arm in self.grippers:
            self.grippers[arm] += delta
        return {"collision": len(self.steps) == self.collision_step}

    def record_event(self, _backend, _event, **_payload):
        return None


class PhysicalTargetAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_moves_held_object_inside_scoring_margin_with_stationary_base(self):
        driver = FakePhysicalAlignmentDriver()

        result = self.module.run_physical_target_alignment(
            object(),
            object_name="box",
            target_xy=np.array([0.82, 0.0]),
            minimum_object_z=0.95,
            target_distance=0.70,
            max_translation=0.18,
            step_size=0.02,
            max_steps=20,
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertLessEqual(result["final_distance"], 0.70 + 1e-9)
        self.assertGreater(result["translation_m"], 0.0)
        self.assertTrue(
            all(np.allclose(step["base_command"], 0.0) for step in driver.steps)
        )
        self.assertTrue(all(step["gripper_value"] == 1.0 for step in driver.steps))

    def test_collision_stops_target_alignment_without_fallback(self):
        driver = FakePhysicalAlignmentDriver(collision_step=1)

        result = self.module.run_physical_target_alignment(
            object(),
            object_name="box",
            target_xy=np.array([1.0, 0.0]),
            minimum_object_z=0.95,
            driver=driver,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "collision")
        self.assertEqual(len(driver.steps), 1)


class PhysicalPlacementRunnerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = self.module.PhysicalCarryConfig(
            descent_step=0.002,
            max_descent=0.03,
            support_stability_steps=2,
            support_motion_tolerance=0.0002,
            release_steps=2,
            settle_steps=2,
        )
        self.config.minimum_descent_before_support = 0.008

    def run_place(self, driver, *, target_xy=(0.0, 0.0), config=None):
        self.assertTrue(hasattr(self.module, "run_physical_place"))
        return self.module.run_physical_place(
            object(),
            object_name="box",
            target_xy=np.asarray(target_xy, dtype=float),
            config=config or self.config,
            driver=driver,
        )

    def test_release_occurs_only_after_measured_descent_and_support_plateau(self):
        driver = FakePhysicalPlacementDriver(
            object_heights=[1.0, 0.995, 0.990, 0.990, 0.990, 0.990, 0.990]
        )

        result = self.run_place(driver)

        self.assertTrue(result["success"])
        gripper_commands = [step["gripper_value"] for step in driver.steps]
        first_release = gripper_commands.index(-1.0)
        self.assertGreaterEqual(first_release, 4)
        self.assertTrue(all(value == 1.0 for value in gripper_commands[:first_release]))
        self.assertTrue(all(value == -1.0 for value in gripper_commands[first_release:]))
        self.assertGreaterEqual(result["descent"], 0.008)
        self.assertTrue(result["support_detected"])

    def test_contact_loss_during_descent_fails_without_release(self):
        driver = FakePhysicalPlacementDriver(
            object_heights=[1.0, 0.995, 0.990],
            contacts=[
                {"right": True, "left": True},
                {"right": True, "left": False},
            ],
        )

        result = self.run_place(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "contact")
        self.assertNotIn(-1.0, [step["gripper_value"] for step in driver.steps])

    def test_descent_without_support_fails_without_release(self):
        driver = FakePhysicalPlacementDriver(
            object_heights=[1.0 - 0.002 * step for step in range(20)]
        )

        result = self.run_place(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "support")
        self.assertNotIn(-1.0, [step["gripper_value"] for step in driver.steps])

    def test_collision_during_descent_fails_without_release(self):
        driver = FakePhysicalPlacementDriver(
            object_heights=[1.0, 0.995, 0.990],
            collision_step=1,
        )

        result = self.run_place(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "collision")
        self.assertNotIn(-1.0, [step["gripper_value"] for step in driver.steps])

    def test_final_target_distance_is_measured_after_release(self):
        driver = FakePhysicalPlacementDriver(
            object_heights=[1.0, 0.990, 0.990, 0.990, 0.990, 0.990],
            object_xy=(1.0, 0.0),
        )

        result = self.run_place(driver)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "target_distance")
        self.assertAlmostEqual(result["final_distance"], 1.0)


if __name__ == "__main__":
    unittest.main()
