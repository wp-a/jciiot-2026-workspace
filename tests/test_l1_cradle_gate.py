import unittest
from types import SimpleNamespace

import numpy as np

from scripts.run_l1_cradle_gate import (
    closure_axis_error_degrees,
    cradle_gate_accepted,
    cradle_gate_failures,
    eef_site_pose,
    has_bilateral_object_contact,
    minimum_undirected_axis_rotation,
    next_orientation_alignment_state,
    normalized_osc_orientation_command,
    opposed_wall_clearance_targets,
    opposed_wall_squeeze_targets,
    orientation_alignment_failures,
    push_gate_accepted,
    push_gate_failures,
)


VALID_RECORD = {
    "physical_grasp": True,
    "lift_m": 0.131,
    "support_contact_steps": 20,
    "base_translation_m": 0.50,
    "attachment_calls": 0,
    "object_pose_writes": 0,
    "collision_frames": 0,
    "dropped": False,
    "infrastructure_error": None,
}

VALID_PUSH_RECORD = {
    "physical_contact_steps": 20,
    "object_translation_m": 0.50,
    "base_translation_m": 0.30,
    "attachment_calls": 0,
    "object_pose_writes": 0,
    "collision_frames": 0,
    "infrastructure_error": None,
}

VALID_ORIENTATION_RECORD = {
    "orientation_right_error_deg": 4.0,
    "orientation_left_error_deg": 3.5,
    "orientation_stable_steps": 5,
    "orientation_max_position_drift_m": 0.03,
    "orientation_collision_frames": 0,
    "infrastructure_error": None,
}


class L1CradleGateTests(unittest.TestCase):
    def test_gate_accepts_only_complete_physical_evidence(self):
        self.assertTrue(cradle_gate_accepted(VALID_RECORD))
        self.assertEqual(cradle_gate_failures(VALID_RECORD), [])

    def test_each_hard_condition_rejects_the_record(self):
        invalid_values = {
            "physical_grasp": False,
            "lift_m": 0.129,
            "support_contact_steps": 19,
            "base_translation_m": 0.299,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "dropped": True,
            "infrastructure_error": "simulator failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_RECORD)
                record[key] = value
                self.assertFalse(cradle_gate_accepted(record))
                self.assertIn(key, cradle_gate_failures(record))

    def test_missing_evidence_is_rejected(self):
        for key in VALID_RECORD:
            with self.subTest(key=key):
                record = dict(VALID_RECORD)
                del record[key]
                self.assertFalse(cradle_gate_accepted(record))
                self.assertIn(key, cradle_gate_failures(record))

    def test_non_finite_numeric_evidence_is_rejected(self):
        for key in (
            "lift_m",
            "support_contact_steps",
            "base_translation_m",
            "attachment_calls",
            "object_pose_writes",
            "collision_frames",
        ):
            with self.subTest(key=key):
                record = dict(VALID_RECORD)
                record[key] = float("nan")
                self.assertFalse(cradle_gate_accepted(record))
                self.assertIn(key, cradle_gate_failures(record))


class L1PhysicalPushGateTests(unittest.TestCase):
    def test_gate_accepts_complete_physical_push_evidence(self):
        self.assertTrue(push_gate_accepted(VALID_PUSH_RECORD))
        self.assertEqual(push_gate_failures(VALID_PUSH_RECORD), [])

    def test_each_hard_condition_rejects_the_record(self):
        invalid_values = {
            "physical_contact_steps": 19,
            "object_translation_m": 0.499,
            "base_translation_m": 0.299,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "infrastructure_error": "simulator failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_PUSH_RECORD)
                record[key] = value
                self.assertFalse(push_gate_accepted(record))
                self.assertIn(key, push_gate_failures(record))


class OrientationCommandTests(unittest.TestCase):
    @staticmethod
    def rotation_z(angle: float) -> np.ndarray:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )

    def test_osc_orientation_command_transforms_and_clips_rotation(self):
        command = normalized_osc_orientation_command(
            world_rotation_delta=self.rotation_z(np.pi / 2.0),
            controller_origin_rotation=self.rotation_z(np.pi),
            output_min=np.array([-0.05] * 3 + [-0.5] * 3),
            output_max=np.array([0.05] * 3 + [0.5] * 3),
            max_action=0.30,
        )

        np.testing.assert_allclose(command, [0.0, 0.0, 0.30], atol=1e-8)

    def test_osc_orientation_command_returns_zero_for_no_rotation(self):
        command = normalized_osc_orientation_command(
            world_rotation_delta=np.eye(3),
            controller_origin_rotation=np.eye(3),
            output_min=np.array([-0.05] * 3 + [-0.5] * 3),
            output_max=np.array([0.05] * 3 + [0.5] * 3),
            max_action=0.30,
        )

        np.testing.assert_allclose(command, np.zeros(3), atol=1e-12)

    def test_osc_orientation_command_rejects_invalid_scaling(self):
        common = {
            "world_rotation_delta": np.eye(3),
            "controller_origin_rotation": np.eye(3),
            "output_min": np.zeros(6),
            "output_max": np.zeros(6),
            "max_action": 0.30,
        }
        with self.assertRaises(ValueError):
            normalized_osc_orientation_command(**common)
        common["output_min"] = np.array([-0.05] * 3 + [-0.5] * 3)
        common["output_max"] = np.array([0.05] * 3 + [0.5] * 3)
        common["world_rotation_delta"] = np.full((3, 3), float("nan"))
        with self.assertRaises(ValueError):
            normalized_osc_orientation_command(**common)


class EefSitePoseTests(unittest.TestCase):
    def test_pose_uses_the_osc_grip_site_boundary(self):
        rotation = OrientationCommandTests.rotation_z(np.pi / 4.0)
        raw_env = SimpleNamespace(
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    site_xpos=np.array([[9.0, 8.0, 7.0], [1.0, 2.0, 3.0]]),
                    site_xmat=np.array([np.eye(3).reshape(-1), rotation.reshape(-1)]),
                )
            )
        )
        robot = SimpleNamespace(eef_site_id={"right": 1})

        position, orientation = eef_site_pose(raw_env, robot, "right")

        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(orientation, rotation)


class OrientationAlignmentGateTests(unittest.TestCase):
    def test_gate_accepts_complete_bounded_alignment_evidence(self):
        self.assertEqual(orientation_alignment_failures(VALID_ORIENTATION_RECORD), [])

    def test_gate_rejects_each_failed_alignment_condition(self):
        invalid_values = {
            "orientation_right_error_deg": 5.01,
            "orientation_left_error_deg": 5.01,
            "orientation_stable_steps": 4,
            "orientation_max_position_drift_m": 0.0301,
            "orientation_collision_frames": 1,
            "infrastructure_error": "simulator failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_ORIENTATION_RECORD)
                record[key] = value
                self.assertIn(key, orientation_alignment_failures(record))

    def test_gate_rejects_missing_and_non_finite_alignment_evidence(self):
        numeric_fields = tuple(VALID_ORIENTATION_RECORD)[:-1]
        for key in VALID_ORIENTATION_RECORD:
            with self.subTest(missing=key):
                record = dict(VALID_ORIENTATION_RECORD)
                del record[key]
                self.assertIn(key, orientation_alignment_failures(record))
        for key in numeric_fields:
            with self.subTest(non_finite=key):
                record = dict(VALID_ORIENTATION_RECORD)
                record[key] = float("nan")
                self.assertIn(key, orientation_alignment_failures(record))


class OrientationAlignmentStateTests(unittest.TestCase):
    def test_state_requires_consecutive_bounded_steps(self):
        state = {"stable_steps": 0, "max_position_drift_m": 0.0}
        for expected in range(1, 5):
            state = next_orientation_alignment_state(
                state,
                right_error_deg=4.0,
                left_error_deg=3.0,
                position_drift_m=0.02,
                collision=False,
            )
            self.assertEqual(state["stable_steps"], expected)
            self.assertFalse(state["aligned"])

        state = next_orientation_alignment_state(
            state,
            right_error_deg=4.0,
            left_error_deg=3.0,
            position_drift_m=0.02,
            collision=False,
        )

        self.assertEqual(state["stable_steps"], 5)
        self.assertTrue(state["aligned"])
        self.assertFalse(state["terminate"])

    def test_state_resets_stability_on_angular_error(self):
        state = {"stable_steps": 4, "max_position_drift_m": 0.01}

        state = next_orientation_alignment_state(
            state,
            right_error_deg=5.01,
            left_error_deg=3.0,
            position_drift_m=0.02,
            collision=False,
        )

        self.assertEqual(state["stable_steps"], 0)
        self.assertFalse(state["aligned"])
        self.assertFalse(state["terminate"])

    def test_state_terminates_on_collision_or_excessive_drift(self):
        for collision, drift, reason in (
            (True, 0.01, "collision"),
            (False, 0.0301, "position_drift"),
        ):
            with self.subTest(reason=reason):
                state = next_orientation_alignment_state(
                    {"stable_steps": 2, "max_position_drift_m": 0.01},
                    right_error_deg=3.0,
                    left_error_deg=3.0,
                    position_drift_m=drift,
                    collision=collision,
                )
                self.assertTrue(state["terminate"])
                self.assertEqual(state["failure"], reason)
                self.assertFalse(state["aligned"])
                self.assertEqual(state["stable_steps"], 0)

    def test_state_rejects_non_finite_measurements(self):
        with self.assertRaises(ValueError):
            next_orientation_alignment_state(
                {"stable_steps": 0, "max_position_drift_m": 0.0},
                right_error_deg=float("nan"),
                left_error_deg=3.0,
                position_drift_m=0.0,
                collision=False,
            )

class OpposedWallRegraspTests(unittest.TestCase):
    def test_axis_rotation_maps_tilted_closure_axis_to_wall_normal(self):
        source = np.array([0.0, 0.6, 0.8])
        target = np.array([0.0, 1.0, 0.0])

        rotation = minimum_undirected_axis_rotation(source, target)

        np.testing.assert_allclose(rotation @ source, target, atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_axis_rotation_uses_the_nearest_sign_of_an_undirected_axis(self):
        source = np.array([0.0, -0.8, 0.6])
        target = np.array([0.0, 1.0, 0.0])

        rotation = minimum_undirected_axis_rotation(source, target)

        np.testing.assert_allclose(rotation @ source, -target, atol=1e-8)
        self.assertAlmostEqual(closure_axis_error_degrees(rotation @ source, target), 0.0)

    def test_closure_axis_error_treats_parallel_and_antiparallel_as_aligned(self):
        self.assertAlmostEqual(
            closure_axis_error_degrees([0.0, 1.0, 0.0], [0.0, 2.0, 0.0]),
            0.0,
        )
        self.assertAlmostEqual(
            closure_axis_error_degrees([0.0, -1.0, 0.0], [0.0, 2.0, 0.0]),
            0.0,
        )

    def test_axis_rotation_rejects_invalid_vectors(self):
        invalid_axes = (
            np.zeros(3),
            np.array([1.0, 2.0]),
            np.array([1.0, float("nan"), 0.0]),
            np.array([1.0, float("inf"), 0.0]),
        )
        for axis in invalid_axes:
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    minimum_undirected_axis_rotation(axis, [0.0, 1.0, 0.0])
                with self.assertRaises(ValueError):
                    closure_axis_error_degrees(axis, [0.0, 1.0, 0.0])

    def test_bilateral_contact_accepts_any_real_contact_from_each_arm(self):
        self.assertTrue(
            has_bilateral_object_contact(
                {
                    "right": ("gripper0_right_left_fingertip_collision",),
                    "left": ("gripper0_left_left_fingertip_collision",),
                }
            )
        )
        self.assertFalse(
            has_bilateral_object_contact(
                {
                    "right": ("gripper0_right_left_fingertip_collision",),
                    "left": (),
                }
            )
        )

    def test_clearance_targets_move_both_arms_outside_the_long_walls(self):
        current = {
            "right": np.array([7.35, 4.73, 1.37]),
            "left": np.array([7.35, 4.50, 1.37]),
        }

        targets = opposed_wall_clearance_targets(
            current,
            separation_axis=np.array([0.0, 1.0, 0.0]),
            clearance_m=0.10,
        )

        np.testing.assert_allclose(targets["right"], [7.35, 4.83, 1.37])
        np.testing.assert_allclose(targets["left"], [7.35, 4.40, 1.37])

    def test_clearance_targets_orient_a_reversed_axis_from_left_to_right(self):
        current = {
            "right": np.array([7.35, 4.73, 1.37]),
            "left": np.array([7.35, 4.50, 1.37]),
        }

        targets = opposed_wall_clearance_targets(
            current,
            separation_axis=np.array([0.0, -2.0, 0.0]),
            clearance_m=0.10,
        )

        np.testing.assert_allclose(targets["right"], [7.35, 4.83, 1.37])
        np.testing.assert_allclose(targets["left"], [7.35, 4.40, 1.37])

    def test_clearance_rejects_invalid_distances_and_vertical_axes(self):
        current = {
            "right": np.array([7.35, 4.73, 1.37]),
            "left": np.array([7.35, 4.50, 1.37]),
        }
        for distance in (-0.01, float("nan"), float("inf")):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    opposed_wall_clearance_targets(
                        current,
                        separation_axis=np.array([0.0, 1.0, 0.0]),
                        clearance_m=distance,
                    )
        with self.assertRaises(ValueError):
            opposed_wall_clearance_targets(
                current,
                separation_axis=np.array([0.0, 0.0, 1.0]),
                clearance_m=0.10,
            )

    def test_squeeze_targets_move_both_arms_toward_the_container(self):
        current = {
            "right": np.array([7.12, 4.82, 1.24]),
            "left": np.array([7.12, 4.41, 1.24]),
        }

        targets = opposed_wall_squeeze_targets(
            current,
            separation_axis=np.array([0.0, 1.0, 0.0]),
            squeeze_m=0.025,
        )

        np.testing.assert_allclose(targets["right"], [7.12, 4.795, 1.24])
        np.testing.assert_allclose(targets["left"], [7.12, 4.435, 1.24])

    def test_missing_evidence_is_rejected(self):
        for key in VALID_PUSH_RECORD:
            with self.subTest(key=key):
                record = dict(VALID_PUSH_RECORD)
                del record[key]
                self.assertFalse(push_gate_accepted(record))
                self.assertIn(key, push_gate_failures(record))

if __name__ == "__main__":
    unittest.main()
