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

    def test_l2_upper_tote_uses_the_collision_free_station_side(self):
        pose = self.module.station_axis_grasp_pose(
            grasp_center_xy=[11.867624, 4.409856],
            right_site_xy=[12.032624, 4.409856],
            left_site_xy=[11.702624, 4.409856],
            station_center=[11.937, 3.932],
            station_approach=[13.0, 3.932],
        )

        np.testing.assert_allclose(pose["base_xy"], [12.518625, 4.409856])
        np.testing.assert_allclose(pose["staging_xy"], [13.0, 4.409856])
        self.assertAlmostEqual(abs(pose["yaw"]), math.pi)
        self.assertTrue(pose["precise_alignment"])

    def test_l3_far_right_blue_tote_uses_the_south_station_axis(self):
        standoff = self.module.station_axis_standoff_for_object(
            "blue_tote_b01_far_right"
        )
        pose = self.module.station_axis_grasp_pose(
            grasp_center_xy=[-0.000201, 8.473143],
            right_site_xy=[-0.000201, 8.638143],
            left_site_xy=[-0.000201, 8.308143],
            station_center=[0.144, 8.473],
            station_approach=[0.11, 7.55],
            base_standoff=standoff,
            facing_xy=[-0.215201, 8.473143],
        )

        self.assertAlmostEqual(standoff, 0.78)
        np.testing.assert_allclose(
            pose["base_xy"],
            [-0.000201, 7.693143],
            atol=1e-4,
        )
        self.assertTrue(pose["precise_alignment"])
        self.assertAlmostEqual(pose["yaw"], 1.839758, places=5)
        np.testing.assert_allclose(
            pose["orientation_target_xy"],
            [-0.215201, 8.473143],
        )

    def test_l4_blue_container_biases_grasp_for_safe_delivery_offset(self):
        pose = self.module.transport_biased_grasp_pose(
            {
                "base_xy": [-8.979311, 5.343394],
                "staging_xy": [-8.979311, 5.010000],
                "yaw": math.pi,
            },
            object_name="blue_container_h01_back_upper",
            object_xy=[-9.848312, 5.343394],
            station_center=[-9.761, 5.010],
            station_approach=[-8.300, 5.010],
        )

        np.testing.assert_allclose(
            pose["base_xy"],
            [-8.979311, 5.143394],
        )
        np.testing.assert_allclose(
            pose["staging_xy"],
            [-8.979311, 5.010000],
        )
        np.testing.assert_allclose(
            pose["orientation_target_xy"],
            [-9.848312, 5.343394],
        )
        self.assertAlmostEqual(pose["yaw"], 2.9153, places=3)

    def test_transport_grasp_bias_does_not_change_other_object_families(self):
        original = {
            "base_xy": [1.0, 2.0],
            "staging_xy": [1.5, 2.0],
            "yaw": 0.5,
        }

        pose = self.module.transport_biased_grasp_pose(
            original,
            object_name="green_tote_b01_upper",
            object_xy=[0.0, 2.0],
            station_center=[0.0, 0.0],
            station_approach=[1.0, 0.0],
        )

        self.assertEqual(pose, original)

    def test_nearly_cardinal_station_axis_snaps_to_its_dominant_axis(self):
        snapped = self.module.dominant_cardinal_axis(
            np.array([-0.034, -0.923]),
        )

        np.testing.assert_allclose(snapped, [0.0, -1.0])

    def test_other_objects_keep_their_existing_grasp_pose_family(self):
        self.assertIsNone(
            self.module.station_axis_standoff_for_object(
                "green_tote_b01_lower"
            )
        )
        self.assertAlmostEqual(
            self.module.station_axis_standoff_for_object(
                "green_tote_b01_upper"
            ),
            self.module.REFERENCE_BASE_TO_GRASP_CENTER,
        )

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

    def test_carried_object_alignment_yaw_reduces_l4_target_distance(self):
        base_xy = np.array([5.8, -8.2])
        object_xy = np.array([4.800908, -8.005454])
        target_xy = np.array([4.872, -7.261])
        current_yaw = 2.938242

        target_yaw = self.module.carried_object_alignment_yaw(
            base_xy=base_xy,
            base_yaw=current_yaw,
            object_xy=object_xy,
            target_xy=target_xy,
        )

        relative_world = object_xy - base_xy
        cosine = math.cos(-current_yaw)
        sine = math.sin(-current_yaw)
        relative_local = np.array(
            [
                cosine * relative_world[0] - sine * relative_world[1],
                sine * relative_world[0] + cosine * relative_world[1],
            ]
        )
        cosine = math.cos(target_yaw)
        sine = math.sin(target_yaw)
        aligned_object_xy = base_xy + np.array(
            [
                cosine * relative_local[0] - sine * relative_local[1],
                sine * relative_local[0] + cosine * relative_local[1],
            ]
        )

        self.assertAlmostEqual(target_yaw, 2.339, places=2)
        self.assertLess(float(np.linalg.norm(aligned_object_xy - target_xy)), 0.40)

    def test_grasp_base_alignment_uses_a_tight_bounded_final_move(self):
        calls = []
        backend = SimpleNamespace(xy=np.array([13.0, 4.41], dtype=float))
        backend.get_base_pose = lambda: (backend.xy.copy(), math.pi)

        def follow_path(path, *, waypoint_tolerance, max_steps):
            calls.append((path, waypoint_tolerance, max_steps))
            backend.xy = np.asarray(path[-1], dtype=float).copy()
            return True

        backend.follow_path = follow_path
        success = self.module.align_base_for_grasp(
            backend,
            [12.517624, 4.409856],
        )

        self.assertTrue(success)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(
            calls[0][1], self.module.PRECISE_GRASP_BASE_TOLERANCE
        )
        self.assertEqual(calls[0][2], self.module.PRECISE_GRASP_BASE_MAX_STEPS)

    def test_grasp_base_alignment_skips_motion_inside_tolerance(self):
        backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([1.0, 2.0]), 0.0),
            follow_path=lambda *_args, **_kwargs: self.fail(
                "already-aligned base should not move"
            ),
        )

        self.assertTrue(
            self.module.align_base_for_grasp(backend, [1.01, 2.0])
        )

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

    def test_orient_base_delegates_attached_turn_to_official_helper(self):
        raw_env = SimpleNamespace()
        frames = []
        backend = SimpleNamespace(
            env=raw_env,
            get_base_pose=lambda: (np.array([5.8, -8.2]), 2.938242),
            _record_trajectory_frame=lambda **kwargs: frames.append(kwargs),
        )
        calls = []
        turn_module = ModuleType(
            "robosuite.environments.factory_sorting.turn_to_station"
        )

        def turn_to_face_xy(**kwargs):
            calls.append(kwargs)
            kwargs["post_step_callback"]()
            return {"success": True}

        turn_module.turn_to_face_xy = turn_to_face_xy

        with patch.dict(
            sys.modules,
            {
                "robosuite": ModuleType("robosuite"),
                "robosuite.environments": ModuleType("robosuite.environments"),
                "robosuite.environments.factory_sorting": ModuleType(
                    "robosuite.environments.factory_sorting"
                ),
                "robosuite.environments.factory_sorting.turn_to_station": (
                    turn_module
                ),
            },
        ):
            reached = self.module.orient_base(
                backend,
                2.339,
                maintain_official_attachment=True,
            )

        self.assertTrue(reached)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["env"], raw_env)
        expected_target = np.array([5.8, -8.2]) + np.array(
            [math.cos(2.339), math.sin(2.339)]
        )
        np.testing.assert_allclose(calls[0]["target_xy"], expected_target)
        self.assertFalse(calls[0]["render"])
        self.assertEqual(frames, [{"_env": raw_env}])

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
