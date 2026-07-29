import unittest
from types import SimpleNamespace

import numpy as np

from scripts.run_l1_cradle_gate import (
    allocate_segment_steps,
    closure_axis_error_degrees,
    cradle_gate_accepted,
    cradle_gate_failures,
    eef_site_pose,
    has_bilateral_object_contact,
    interior_joint_bounds,
    interpolate_directed_axis,
    joint_seed_failures,
    joint_seed_node_failure,
    joint_seed_objective_residual,
    minimum_undirected_axis_rotation,
    nearest_directed_axis_target,
    next_joint_seed_path_state,
    next_orientation_alignment_state,
    normalized_osc_orientation_command,
    opposed_wall_clearance_targets,
    opposed_wall_squeeze_targets,
    orientation_alignment_failures,
    parse_args,
    push_gate_accepted,
    push_gate_failures,
    scheduled_orientation_action_limit,
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

VALID_JOINT_SEED_RECORD = {
    "joint_seed_success": True,
    "joint_seed_right_error_deg": 9.9,
    "joint_seed_left_error_deg": 9.8,
    "joint_seed_max_endpoint_position_error_m": 0.015,
    "joint_seed_max_path_position_drift_m": 0.03,
    "joint_seed_min_bound_margin_rad": 0.0,
    "joint_seed_collision_frames": 0,
    "joint_seed_rolled_back": False,
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


class JointSeedMathTests(unittest.TestCase):
    def test_allocate_segment_steps_preserves_total_and_balance(self):
        allocation = allocate_segment_steps(total_steps=10, segment_count=3)

        self.assertEqual(allocation, (4, 3, 3))
        self.assertEqual(sum(allocation), 10)
        self.assertEqual(max(allocation) - min(allocation), 1)

    def test_allocate_segment_steps_rejects_invalid_counts(self):
        invalid = ((0, 1), (4, 0), (2, 3), (2.5, 2), (4, True))
        for total_steps, segment_count in invalid:
            with self.subTest(total_steps=total_steps, segment_count=segment_count):
                with self.assertRaises(ValueError):
                    allocate_segment_steps(
                        total_steps=total_steps,
                        segment_count=segment_count,
                    )

    def test_interior_joint_bounds_move_both_limits_inward(self):
        lower, upper = interior_joint_bounds(
            [-1.0, -2.0],
            [1.0, 3.0],
            margin_rad=0.05,
        )

        np.testing.assert_allclose(lower, [-0.95, -1.95])
        np.testing.assert_allclose(upper, [0.95, 2.95])

    def test_interior_joint_bounds_reject_invalid_ranges(self):
        invalid = (
            ([-1.0], [1.0, 2.0], 0.01),
            ([float("nan")], [1.0], 0.01),
            ([-1.0], [float("inf")], 0.01),
            ([-1.0], [1.0], -0.01),
            ([-0.01], [0.01], 0.01),
        )
        for lower, upper, margin in invalid:
            with self.subTest(lower=lower, upper=upper, margin=margin):
                with self.assertRaises(ValueError):
                    interior_joint_bounds(lower, upper, margin_rad=margin)

    def test_directed_axis_interpolation_has_exact_normalized_endpoints(self):
        source = np.array([0.0, 2.0, 0.0])
        target = np.array([1.0, 1.0, 0.0])

        np.testing.assert_allclose(
            interpolate_directed_axis(source, target, fraction=0.0),
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            interpolate_directed_axis(source, target, fraction=1.0),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        )

    def test_directed_axis_interpolation_is_unit_and_monotonic(self):
        source = np.array([0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])
        values = [
            interpolate_directed_axis(source, target, fraction=fraction)
            for fraction in np.linspace(0.0, 1.0, 11)
        ]

        np.testing.assert_allclose(
            [np.linalg.norm(value) for value in values],
            np.ones(11),
        )
        errors = [closure_axis_error_degrees(value, target) for value in values]
        self.assertTrue(all(a >= b for a, b in zip(errors, errors[1:])))

    def test_directed_axis_interpolation_rejects_invalid_inputs(self):
        invalid = (
            ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], 0.5),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], -0.01),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1.01),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], float("nan")),
        )
        for source, target, fraction in invalid:
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    interpolate_directed_axis(source, target, fraction=fraction)

    @staticmethod
    def seed_inputs():
        return {
            "current_positions": {
                "right": np.array([1.0, 2.0, 3.0]),
                "left": np.array([4.0, 5.0, 6.0]),
            },
            "target_positions": {
                "right": np.array([1.0, 2.0, 3.0]),
                "left": np.array([4.0, 5.0, 6.0]),
            },
            "current_axes": {
                "right": np.array([0.0, 1.0, 0.0]),
                "left": np.array([0.0, -1.0, 0.0]),
            },
            "target_axes": {
                "right": np.array([0.0, 1.0, 0.0]),
                "left": np.array([0.0, -1.0, 0.0]),
            },
            "joints": np.zeros(12),
            "start_joints": np.zeros(12),
            "joint_ranges": np.full(12, 2.0),
            "position_scale_m": 0.01,
            "axis_scale": np.sin(np.deg2rad(5.0)),
            "regularization": 0.02,
        }

    def test_joint_seed_objective_is_zero_at_the_target_start(self):
        residual = joint_seed_objective_residual(**self.seed_inputs())

        self.assertEqual(residual.shape, (24,))
        np.testing.assert_allclose(residual, np.zeros(24), atol=1e-12)

    def test_joint_seed_objective_scales_position_axis_and_joint_terms(self):
        values = self.seed_inputs()
        values["current_positions"]["right"][0] += 0.01
        values["current_axes"]["left"] = np.array([0.0, 0.0, -1.0])
        values["joints"][0] = 1.0

        residual = joint_seed_objective_residual(**values)

        self.assertAlmostEqual(residual[0], 1.0)
        np.testing.assert_allclose(
            residual[9:12],
            np.array([0.0, 1.0, -1.0]) / np.sin(np.deg2rad(5.0)),
        )
        self.assertAlmostEqual(residual[12], 0.01)

    def test_nearest_directed_axis_target_fixes_the_closer_sign(self):
        np.testing.assert_allclose(
            nearest_directed_axis_target([0.0, -1.0, 0.0], [0.0, 2.0, 0.0]),
            [0.0, -1.0, 0.0],
        )

    def test_joint_seed_objective_rejects_invalid_inputs(self):
        overrides = (
            {"current_positions": {"right": np.zeros(3)}},
            {"current_axes": {"right": np.zeros(3), "left": np.ones(3)}},
            {"joints": np.zeros(11)},
            {"joint_ranges": np.r_[np.ones(11), 0.0]},
            {"position_scale_m": 0.0},
            {"axis_scale": float("nan")},
            {"regularization": -0.01},
        )
        common = self.seed_inputs()
        for override in overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    joint_seed_objective_residual(**{**common, **override})


class JointSeedParserTests(unittest.TestCase):
    def test_joint_seed_parser_defaults_are_strict_and_opt_in(self):
        args = parse_args(
            [
                "--candidate-root",
                "/tmp/candidate",
                "--expected-official-commit",
                "official-commit",
                "--output",
                "/tmp/result.json",
                "--trajectory",
                "/tmp/trajectory.json",
            ]
        )

        self.assertFalse(args.orientation_joint_seed)
        self.assertAlmostEqual(args.orientation_joint_seed_margin_rad, 0.03)
        self.assertEqual(args.orientation_joint_seed_max_nfev, 800)
        self.assertEqual(args.orientation_joint_seed_steps, 240)
        self.assertAlmostEqual(args.orientation_joint_seed_position_scale_m, 0.01)
        self.assertAlmostEqual(
            args.orientation_joint_seed_axis_scale,
            np.sin(np.deg2rad(5.0)),
        )
        self.assertAlmostEqual(args.orientation_joint_seed_regularization, 0.02)
        self.assertAlmostEqual(args.orientation_joint_seed_max_error_deg, 10.0)
        self.assertAlmostEqual(
            args.orientation_joint_seed_max_endpoint_position_error_m,
            0.015,
        )
        self.assertEqual(args.orientation_joint_seed_continuation_nodes, 1)


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

    def test_osc_orientation_command_preserves_axis_when_limited(self):
        axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        angle = np.pi / 2.0
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = np.eye(3) + skew * np.sin(angle) + (skew @ skew) * (
            1.0 - np.cos(angle)
        )

        command = normalized_osc_orientation_command(
            world_rotation_delta=rotation,
            controller_origin_rotation=np.eye(3),
            output_min=np.array([-0.05] * 3 + [-0.5] * 3),
            output_max=np.array([0.05] * 3 + [0.5] * 3),
            max_action=0.30,
        )

        self.assertAlmostEqual(float(np.linalg.norm(command)), 0.30)
        np.testing.assert_allclose(command / np.linalg.norm(command), axis, atol=1e-8)

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

    def test_orientation_action_schedule_slows_each_arm_near_target(self):
        self.assertAlmostEqual(
            scheduled_orientation_action_limit(
                error_deg=15.01,
                coarse_action=0.02,
                fine_action=0.005,
                fine_threshold_deg=15.0,
            ),
            0.02,
        )
        self.assertAlmostEqual(
            scheduled_orientation_action_limit(
                error_deg=15.0,
                coarse_action=0.02,
                fine_action=0.005,
                fine_threshold_deg=15.0,
            ),
            0.005,
        )

    def test_orientation_action_schedule_rejects_invalid_limits(self):
        invalid = (
            {"error_deg": float("nan")},
            {"coarse_action": 0.0},
            {"fine_action": 0.03},
            {"fine_threshold_deg": -1.0},
        )
        common = {
            "error_deg": 10.0,
            "coarse_action": 0.02,
            "fine_action": 0.005,
            "fine_threshold_deg": 15.0,
        }
        for override in invalid:
            with self.subTest(override=override):
                values = {**common, **override}
                with self.assertRaises(ValueError):
                    scheduled_orientation_action_limit(**values)


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


class JointSeedGateTests(unittest.TestCase):
    def test_joint_seed_node_uses_the_endpoint_orientation_gate(self):
        self.assertIsNone(
            joint_seed_node_failure(
                solver_success=True,
                right_error_deg=2.71,
                left_error_deg=5.94,
                position_error_m=0.008,
                min_bound_margin_rad=0.0,
                collision=False,
            )
        )

    def test_joint_seed_node_reports_the_first_failed_condition(self):
        common = {
            "solver_success": True,
            "right_error_deg": 2.0,
            "left_error_deg": 3.0,
            "position_error_m": 0.008,
            "min_bound_margin_rad": 0.0,
            "collision": False,
        }
        failures = (
            ({"solver_success": False}, "solver"),
            ({"collision": True}, "collision"),
            ({"left_error_deg": 10.01}, "orientation"),
            ({"position_error_m": 0.0151}, "position"),
            ({"min_bound_margin_rad": -0.0001}, "bounds"),
        )
        for override, expected in failures:
            with self.subTest(expected=expected):
                self.assertEqual(
                    joint_seed_node_failure(**{**common, **override}),
                    expected,
                )

    def test_joint_seed_gate_accepts_complete_bounded_evidence(self):
        self.assertEqual(joint_seed_failures(VALID_JOINT_SEED_RECORD), [])

    def test_joint_seed_gate_rejects_each_failed_condition(self):
        invalid_values = {
            "joint_seed_success": False,
            "joint_seed_right_error_deg": 10.01,
            "joint_seed_left_error_deg": 10.01,
            "joint_seed_max_endpoint_position_error_m": 0.0151,
            "joint_seed_max_path_position_drift_m": 0.0301,
            "joint_seed_min_bound_margin_rad": -0.0001,
            "joint_seed_collision_frames": 1,
            "joint_seed_rolled_back": True,
            "infrastructure_error": "simulator failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_JOINT_SEED_RECORD)
                record[key] = value
                self.assertIn(key, joint_seed_failures(record))

    def test_joint_seed_gate_rejects_missing_and_non_finite_evidence(self):
        numeric_fields = (
            "joint_seed_right_error_deg",
            "joint_seed_left_error_deg",
            "joint_seed_max_endpoint_position_error_m",
            "joint_seed_max_path_position_drift_m",
            "joint_seed_min_bound_margin_rad",
            "joint_seed_collision_frames",
        )
        for key in VALID_JOINT_SEED_RECORD:
            with self.subTest(missing=key):
                record = dict(VALID_JOINT_SEED_RECORD)
                del record[key]
                self.assertIn(key, joint_seed_failures(record))
        for key in numeric_fields:
            with self.subTest(non_finite=key):
                record = dict(VALID_JOINT_SEED_RECORD)
                record[key] = float("nan")
                self.assertIn(key, joint_seed_failures(record))


class JointSeedPathStateTests(unittest.TestCase):
    def test_path_state_accumulates_the_largest_two_arm_drift(self):
        state = next_joint_seed_path_state(
            {},
            waypoint_index=1,
            right_drift_m=0.01,
            left_drift_m=0.02,
            collision_pairs=(),
        )
        self.assertAlmostEqual(state["max_position_drift_m"], 0.02)
        self.assertFalse(state["terminate"])

        state = next_joint_seed_path_state(
            state,
            waypoint_index=2,
            right_drift_m=0.025,
            left_drift_m=0.01,
            collision_pairs=(),
        )
        self.assertAlmostEqual(state["max_position_drift_m"], 0.025)
        self.assertEqual(state["waypoint_count"], 2)
        self.assertFalse(state["terminate"])

    def test_path_state_stops_on_the_first_collision_and_keeps_pairs(self):
        state = next_joint_seed_path_state(
            {},
            waypoint_index=7,
            right_drift_m=0.01,
            left_drift_m=0.02,
            collision_pairs=(("robot_link", "table"),),
        )

        self.assertTrue(state["terminate"])
        self.assertEqual(state["failure"], "collision")
        self.assertEqual(state["failed_waypoint"], 7)
        self.assertEqual(state["collision_frames"], 1)
        self.assertEqual(state["collision_pairs"], [["robot_link", "table"]])

    def test_path_state_stops_on_excessive_position_drift(self):
        state = next_joint_seed_path_state(
            {},
            waypoint_index=3,
            right_drift_m=0.0301,
            left_drift_m=0.0,
            collision_pairs=(),
        )

        self.assertTrue(state["terminate"])
        self.assertEqual(state["failure"], "position_drift")
        self.assertEqual(state["failed_waypoint"], 3)

    def test_path_state_rejects_invalid_measurements(self):
        invalid = (
            {"waypoint_index": 0},
            {"right_drift_m": float("nan")},
            {"left_drift_m": -0.01},
            {"collision_pairs": (("only-one-name",),)},
            {"max_position_drift_m": 0.0},
        )
        common = {
            "waypoint_index": 1,
            "right_drift_m": 0.0,
            "left_drift_m": 0.0,
            "collision_pairs": (),
            "max_position_drift_m": 0.03,
        }
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    next_joint_seed_path_state({}, **{**common, **override})


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
