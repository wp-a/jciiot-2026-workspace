import importlib.util
import math
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


class FakePhysicalTransportDriver:
    def __init__(
        self,
        *,
        contacts=None,
        object_heights=None,
        collision_step=None,
        advance=True,
    ):
        self.base_xy = np.zeros(2, dtype=float)
        self.yaw = 0.0
        self.contacts = list(contacts or [{"right": True, "left": True}])
        self.object_heights = list(object_heights or [1.0])
        self.collision_step = collision_step
        self.advance = bool(advance)
        self.steps = []

    def capture_hold_targets(self, _backend):
        return {"torso": np.array([0.3]), "head": np.array([0.1, -0.1])}

    def observe(self, _backend, _object_name):
        index = min(len(self.steps), len(self.contacts) - 1)
        height_index = min(len(self.steps), len(self.object_heights) - 1)
        return {
            "base_xy": self.base_xy.copy(),
            "base_yaw": self.yaw,
            "object_pos": np.array(
                [self.base_xy[0] + 0.5, self.base_xy[1], self.object_heights[height_index]],
                dtype=float,
            ),
            "contacts": dict(self.contacts[index]),
        }

    def step(self, _backend, *, object_name, base_command, hold_targets):
        command = np.asarray(base_command, dtype=float).copy()
        self.steps.append(
            {
                "object_name": object_name,
                "base_command": command,
                "hold_targets": hold_targets,
            }
        )
        if self.advance:
            self.base_xy += command[:2]
            self.yaw += command[2]
        return {"collision": self.collision_step == len(self.steps)}

    def record_event(self, _backend, event, **payload):
        return (event, payload)


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

    def run_transport(self, driver, *, path=None, config=None):
        self.assertTrue(hasattr(self.module, "run_physical_transport"))
        return self.module.run_physical_transport(
            object(),
            path=path or [np.array([0.18, 0.0])],
            object_name="box",
            hold_yaw=0.0,
            minimum_object_z=0.98,
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
