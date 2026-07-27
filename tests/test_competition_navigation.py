import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "skills"
    / "competition_navigation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("competition_navigation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompetitionNavigationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_l1_keeps_public_reference_pose_and_arm_assignment(self):
        pose = self.module.grasp_aligned_base_pose(
            object_xy=[7.059, 4.619],
            right_site_xy=[7.349, 4.739],
            left_site_xy=[7.349, 4.499],
            station_center=[7.186, 3.938],
            station_approach=[8.0, 4.619],
        )

        self.assertAlmostEqual(pose["base_xy"][0], 8.0, places=3)
        self.assertAlmostEqual(pose["base_xy"][1], 4.619, places=3)
        self.assertAlmostEqual(pose["staging_xy"][0], 8.0, places=3)
        self.assertAlmostEqual(pose["staging_xy"][1], 4.619, places=3)
        self.assertAlmostEqual(abs(pose["yaw"]), math.pi, places=3)
        self.assertFalse(pose["swap_arm_targets"])

    def test_l2_approaches_upper_tote_from_clear_side_and_swaps_arms(self):
        pose = self.module.grasp_aligned_base_pose(
            object_xy=[11.867624, 4.624856],
            right_site_xy=[12.032624, 4.409856],
            left_site_xy=[11.702624, 4.409856],
            station_center=[11.937, 3.932],
            station_approach=[13.0, 3.932],
        )

        self.assertAlmostEqual(pose["base_xy"][0], 11.867624, places=3)
        self.assertAlmostEqual(pose["base_xy"][1], 5.565857, places=3)
        self.assertAlmostEqual(pose["staging_xy"][0], 13.0, places=3)
        self.assertAlmostEqual(pose["staging_xy"][1], 5.565857, places=3)
        self.assertAlmostEqual(pose["yaw"], -math.pi / 2.0, places=3)
        self.assertTrue(pose["swap_arm_targets"])

    def test_bounded_yaw_step_uses_shortest_wrapped_rotation(self):
        next_yaw = self.module.bounded_yaw_step(
            current_yaw=-math.pi,
            target_yaw=-math.pi / 2.0,
            max_step=0.025,
        )

        self.assertAlmostEqual(next_yaw, -math.pi + 0.025)


if __name__ == "__main__":
    unittest.main()
