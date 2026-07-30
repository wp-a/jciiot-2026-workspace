import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


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

    def test_l2_pose_approaches_the_objects_grasp_face(self):
        pose = self.module.grasp_aligned_base_pose(
            object_xy=[11.867624, 4.624856],
            right_site_xy=[12.032624, 4.409856],
            left_site_xy=[11.702624, 4.409856],
            station_center=[11.937, 3.932],
            station_approach=[13.0, 3.932],
        )

        self.assertAlmostEqual(pose["base_xy"][0], 11.867624, places=3)
        self.assertAlmostEqual(pose["base_xy"][1], 3.758855, places=3)
        self.assertAlmostEqual(pose["staging_xy"][0], 13.0, places=3)
        self.assertAlmostEqual(pose["staging_xy"][1], 3.758855, places=3)
        self.assertAlmostEqual(pose["yaw"], math.pi / 2.0, places=3)
        self.assertFalse(pose["swap_arm_targets"])

    def test_candidate_selection_rejects_a_grasp_face_blocked_by_another_object(self):
        upper = self.module.grasp_aligned_base_pose(
            object_xy=[11.867624, 4.624856],
            right_site_xy=[12.032624, 4.409856],
            left_site_xy=[11.702624, 4.409856],
            station_center=[11.937, 3.932],
            station_approach=[13.0, 3.932],
        )
        lower = self.module.grasp_aligned_base_pose(
            object_xy=[11.867624, 3.1954],
            right_site_xy=[12.032624, 2.9804],
            left_site_xy=[11.702624, 2.9804],
            station_center=[11.937, 3.932],
            station_approach=[13.0, 3.932],
        )

        selected = self.module.select_grasp_candidate(
            [
                {
                    "name": "green_tote_b01_upper",
                    "base_xy": upper["base_xy"],
                    "object_xy": [11.867624, 4.624856],
                },
                {
                    "name": "green_tote_b01_lower",
                    "base_xy": lower["base_xy"],
                    "object_xy": [11.867624, 3.1954],
                },
            ],
            station_approach=[13.0, 3.932],
        )

        self.assertEqual(selected, "green_tote_b01_lower")

    def test_bounded_yaw_step_uses_shortest_wrapped_rotation(self):
        next_yaw = self.module.bounded_yaw_step(
            current_yaw=-math.pi,
            target_yaw=-math.pi / 2.0,
            max_step=0.025,
        )

        self.assertAlmostEqual(next_yaw, -math.pi + 0.025)

    def test_scanned_grasp_yaw_limit_stays_below_collision_boundary(self):
        next_yaw = self.module.bounded_yaw_step(
            current_yaw=1.5701,
            target_yaw=2.2585,
            max_step=self.module.SAFE_GRASP_YAW_CORRECTION,
        )

        self.assertAlmostEqual(next_yaw, 1.7201)

    def test_orient_base_preserves_world_xy_during_every_yaw_step(self):
        backend = SimpleNamespace(
            xy=np.array([8.034, 5.332], dtype=float),
            yaw=-math.pi,
        )
        raw_env = SimpleNamespace(
            robots=[object()],
            action_spec=(np.zeros(1), np.ones(1)),
        )
        backend.env = raw_env
        backend.get_base_pose = lambda: (backend.xy.copy(), backend.yaw)
        anchors = []

        def set_yaw(_env, _robot, yaw):
            backend.yaw = float(yaw)
            backend.xy += np.array([0.10, -0.10])

        def set_xy(_env, _robot, xy):
            anchors.append(np.asarray(xy, dtype=float).copy())
            backend.xy = np.asarray(xy, dtype=float).copy()

        raw_env.step = lambda _action: (None, None, None, {"has_judge_collision": False})
        private_backend = ModuleType("robot_agent.environments.robosuite_backend")
        private_backend._capture_upper_body_posture = lambda _env, _robot: {}
        private_backend._restore_upper_body_posture = lambda _env, _posture: None
        private_backend._shortest_angle = lambda angle: (
            float(angle) + math.pi
        ) % (2.0 * math.pi) - math.pi
        private_backend._set_base_world_yaw_direct = set_yaw
        private_backend._set_base_xy_direct = set_xy
        robot_agent = ModuleType("robot_agent")
        environments = ModuleType("robot_agent.environments")

        with patch.dict(
            sys.modules,
            {
                "robot_agent": robot_agent,
                "robot_agent.environments": environments,
                "robot_agent.environments.robosuite_backend": private_backend,
            },
        ):
            reached = self.module.orient_base(
                backend,
                -math.pi / 2.0,
                tolerance=0.001,
                max_steps=100,
                max_yaw_step=0.025,
            )

        self.assertTrue(reached)
        np.testing.assert_allclose(backend.xy, [8.034, 5.332])
        self.assertGreater(len(anchors), 5)
        for anchor in anchors:
            np.testing.assert_allclose(anchor, [8.034, 5.332])

    def test_reached_base_orientation_faces_grasp_center(self):
        orientation = self.module.grasp_orientation_from_base(
            base_xy=[12.4061, 2.353],
            right_site_xy=[12.032624, 2.9804],
            left_site_xy=[11.702624, 2.9804],
        )

        expected_yaw = math.atan2(2.9804 - 2.353, 11.867624 - 12.4061)
        self.assertAlmostEqual(orientation["yaw"], expected_yaw)
        self.assertFalse(orientation["swap_arm_targets"])

    def test_wall_side_station_pose_uses_reachable_approach_axis(self):
        pose = self.module.station_side_grasp_pose(
            grasp_center_xy=[-14.674088, 4.199868],
            right_site_xy=[-14.509088, 4.199868],
            left_site_xy=[-14.839088, 4.199868],
            station_center=[-14.544, 5.01],
            station_approach=[-13.1, 5.01],
            base_standoff=1.0,
        )

        self.assertAlmostEqual(pose["base_xy"][0], -13.674088)
        self.assertAlmostEqual(pose["base_xy"][1], 4.199868)
        self.assertAlmostEqual(pose["staging_xy"][0], -13.1)
        self.assertAlmostEqual(pose["staging_xy"][1], 4.199868)
        self.assertAlmostEqual(abs(pose["yaw"]), math.pi)
        self.assertFalse(pose["swap_arm_targets"])


if __name__ == "__main__":
    unittest.main()
