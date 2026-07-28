import importlib.util
import math
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
