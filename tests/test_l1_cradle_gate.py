import inspect
import unittest
from types import SimpleNamespace

import numpy as np

import scripts.run_l1_cradle_gate as gate_module
from scripts.run_l1_cradle_gate import (
    _center_regrasp_probe,
    allocate_segment_steps,
    arm_transport_stroke_targets,
    bounded_base_advance_world_velocity,
    closure_axis_error_degrees,
    compensated_base_reset_step,
    cradle_gate_accepted,
    cradle_gate_failures,
    eef_site_pose,
    geometry_snapshot,
    has_bilateral_object_contact,
    interior_joint_bounds,
    interpolate_directed_axis,
    is_allowed_open_fork_support_geom,
    joint_seed_failures,
    joint_seed_joint_names,
    joint_seed_node_failure,
    joint_seed_objective_residual,
    minimum_undirected_axis_rotation,
    nearest_directed_axis_target,
    next_joint_seed_path_state,
    next_orientation_alignment_state,
    normalized_osc_orientation_command,
    open_fork_alignment_sufficient,
    open_fork_under_bottom_support_ready,
    open_fork_target_orientation,
    opposed_wall_clearance_targets,
    opposed_wall_squeeze_targets,
    orientation_alignment_failures,
    parse_args,
    push_gate_accepted,
    push_gate_failures,
    projected_planar_motion,
    rotation_error_degrees,
    scheduled_orientation_action_limit,
    table_edge_undercut_targets,
    trailing_corner_seat_targets,
    undercut_gate_accepted,
    undercut_gate_failures,
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

VALID_UNDERCUT_RECORD = {
    "open_gripper": True,
    "support_contact_steps": 5,
    "object_lift_m": 0.02,
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

VALID_CENTER_GRASP_TRANSPORT_RECORD = {
    "physical_grasp": True,
    "lift_m": 0.131,
    "hold_grasp_steps": 20,
    "transport_success": True,
    "object_translation_m": 1.001,
    "attachment_calls": 0,
    "object_pose_writes": 0,
    "collision_frames": 0,
    "dropped": False,
    "infrastructure_error": None,
}


class FingerpadBracketTests(unittest.TestCase):
    def test_reads_official_fingerpad_geom_positions(self):
        self.assertTrue(hasattr(gate_module, "fingerpad_world_positions"))
        names = {
            "left_left_pad": 0,
            "left_right_pad": 1,
            "right_left_pad": 2,
            "right_right_pad": 3,
        }
        model = SimpleNamespace(geom_name2id=names.__getitem__)
        data = SimpleNamespace(
            geom_xpos=np.array(
                [
                    [0.0, -0.25, 1.0],
                    [0.0, -0.15, 1.0],
                    [0.0, 0.15, 1.0],
                    [0.0, 0.25, 1.0],
                ]
            )
        )
        raw_env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
        robot = SimpleNamespace(
            gripper={
                "left": SimpleNamespace(
                    important_geoms={
                        "left_fingerpad": ["left_left_pad"],
                        "right_fingerpad": ["left_right_pad"],
                    }
                ),
                "right": SimpleNamespace(
                    important_geoms={
                        "left_fingerpad": ["right_left_pad"],
                        "right_fingerpad": ["right_right_pad"],
                    }
                ),
            }
        )

        positions = gate_module.fingerpad_world_positions(raw_env, robot)

        np.testing.assert_allclose(
            positions["left"],
            [[0.0, -0.25, 1.0], [0.0, -0.15, 1.0]],
        )
        np.testing.assert_allclose(
            positions["right"],
            [[0.0, 0.15, 1.0], [0.0, 0.25, 1.0]],
        )

    def test_reads_extreme_object_wall_centers_along_axis(self):
        self.assertTrue(hasattr(gate_module, "opposed_object_wall_centers"))
        geom_names = ["bottom", "front", "back", "robot"]
        model = SimpleNamespace(
            nbody=2,
            body_parentid=np.array([0, 0]),
            ngeom=4,
            geom_bodyid=np.array([1, 1, 1, 0]),
            geom_id2name=geom_names.__getitem__,
        )
        data = SimpleNamespace(
            geom_xpos=np.array(
                [
                    [0.0, 0.0, 0.9],
                    [0.0, -0.2, 1.0],
                    [0.0, 0.2, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )
        raw_env = SimpleNamespace(
            sim=SimpleNamespace(model=model, data=data),
            obj_body_id={"box": 1},
        )

        centers = gate_module.opposed_object_wall_centers(
            raw_env,
            "box",
            separation_axis=np.array([0.0, 1.0, 0.0]),
        )

        np.testing.assert_allclose(
            centers,
            [[0.0, -0.2, 1.0], [0.0, 0.2, 1.0]],
        )

    def test_distinct_opposed_walls_between_fingerpads_are_ready(self):
        self.assertTrue(hasattr(gate_module, "fingerpad_bracket_evidence"))

        evidence = gate_module.fingerpad_bracket_evidence(
            fingerpads={
                "left": np.array([[0.0, -0.25, 0.0], [0.0, -0.15, 0.0]]),
                "right": np.array([[0.0, 0.15, 0.0], [0.0, 0.25, 0.0]]),
            },
            wall_centers=np.array(
                [[0.0, -0.20, 0.0], [0.0, 0.20, 0.0]]
            ),
            separation_axis=np.array([0.0, 1.0, 0.0]),
        )

        self.assertTrue(evidence["ready"])
        self.assertEqual(evidence["arms"]["left"]["wall_index"], 0)
        self.assertEqual(evidence["arms"]["right"]["wall_index"], 1)

    def test_wall_outside_one_fingerpad_pair_is_not_ready(self):
        evidence = gate_module.fingerpad_bracket_evidence(
            fingerpads={
                "left": np.array([[0.0, -0.30, 0.0], [0.0, -0.25, 0.0]]),
                "right": np.array([[0.0, 0.15, 0.0], [0.0, 0.25, 0.0]]),
            },
            wall_centers=np.array(
                [[0.0, -0.20, 0.0], [0.0, 0.20, 0.0]]
            ),
            separation_axis=np.array([0.0, 1.0, 0.0]),
        )

        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["arms"]["left"]["bracketed"])

    def test_same_wall_assignment_is_not_ready(self):
        evidence = gate_module.fingerpad_bracket_evidence(
            fingerpads={
                "left": np.array([[0.0, -0.25, 0.0], [0.0, -0.15, 0.0]]),
                "right": np.array([[0.0, -0.24, 0.0], [0.0, -0.16, 0.0]]),
            },
            wall_centers=np.array(
                [[0.0, -0.20, 0.0], [0.0, 0.20, 0.0]]
            ),
            separation_axis=np.array([0.0, 1.0, 0.0]),
        )

        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["distinct_walls"])

    def test_invalid_axis_and_nonfinite_coordinates_are_rejected(self):
        valid_fingerpads = {
            "left": np.array([[0.0, -0.25, 0.0], [0.0, -0.15, 0.0]]),
            "right": np.array([[0.0, 0.15, 0.0], [0.0, 0.25, 0.0]]),
        }
        walls = np.array([[0.0, -0.20, 0.0], [0.0, 0.20, 0.0]])

        with self.assertRaisesRegex(ValueError, "separation_axis"):
            gate_module.fingerpad_bracket_evidence(
                fingerpads=valid_fingerpads,
                wall_centers=walls,
                separation_axis=np.zeros(3),
            )
        invalid_fingerpads = dict(valid_fingerpads)
        invalid_fingerpads["left"] = np.array(
            [[0.0, np.nan, 0.0], [0.0, -0.15, 0.0]]
        )
        with self.assertRaisesRegex(ValueError, "fingerpads"):
            gate_module.fingerpad_bracket_evidence(
                fingerpads=invalid_fingerpads,
                wall_centers=walls,
                separation_axis=np.array([0.0, 1.0, 0.0]),
            )


class CenterGraspTransportTests(unittest.TestCase):
    def test_complete_center_grasp_transport_record_is_accepted(self):
        self.assertTrue(
            hasattr(gate_module, "center_grasp_transport_failures")
        )
        self.assertTrue(
            hasattr(gate_module, "center_grasp_transport_accepted")
        )

        self.assertEqual(
            gate_module.center_grasp_transport_failures(
                VALID_CENTER_GRASP_TRANSPORT_RECORD
            ),
            [],
        )
        self.assertTrue(
            gate_module.center_grasp_transport_accepted(
                VALID_CENTER_GRASP_TRANSPORT_RECORD
            )
        )

    def test_each_center_grasp_transport_hard_condition_rejects_record(self):
        invalid_values = {
            "physical_grasp": False,
            "lift_m": 0.129,
            "hold_grasp_steps": 19,
            "transport_success": False,
            "object_translation_m": 1.0,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "dropped": True,
            "infrastructure_error": "RuntimeError: failed",
        }

        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_CENTER_GRASP_TRANSPORT_RECORD)
                record[key] = value
                self.assertIn(
                    key,
                    gate_module.center_grasp_transport_failures(record),
                )
                self.assertFalse(
                    gate_module.center_grasp_transport_accepted(record)
                )

    def test_forward_carry_target_moves_toward_object_by_requested_distance(self):
        self.assertTrue(hasattr(gate_module, "forward_carry_target"))

        target = gate_module.forward_carry_target(
            base_xy=np.array([8.0, 4.6]),
            object_xy=np.array([7.0, 4.6]),
            distance_m=0.2,
        )

        np.testing.assert_allclose(target, [7.8, 4.6])

    def test_forward_carry_target_can_retreat_from_object_for_physical_pull(self):
        target = gate_module.forward_carry_target(
            base_xy=np.array([8.0, 4.6]),
            object_xy=np.array([7.0, 4.6]),
            distance_m=0.2,
            toward_object=False,
        )

        np.testing.assert_allclose(target, [8.2, 4.6])

    def test_explicit_inchworm_world_direction_overrides_grasp_axis(self):
        direction = gate_module.resolve_inchworm_direction(
            base_xy=np.array([8.0, 4.6]),
            object_xy=np.array([7.0, 4.6]),
            toward_base=False,
            world_direction=np.array([0.0, -2.0]),
        )

        np.testing.assert_allclose(direction, [0.0, -1.0])

    def test_zero_carry_distance_keeps_current_base_position(self):
        target = gate_module.forward_carry_target(
            base_xy=np.array([8.0, 4.6]),
            object_xy=np.array([8.0, 4.6]),
            distance_m=0.0,
        )

        np.testing.assert_allclose(target, [8.0, 4.6])

    def test_invalid_carry_distance_and_direction_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distance_m"):
            gate_module.forward_carry_target(
                base_xy=np.array([8.0, 4.6]),
                object_xy=np.array([7.0, 4.6]),
                distance_m=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "distance_m"):
            gate_module.forward_carry_target(
                base_xy=np.array([8.0, 4.6]),
                object_xy=np.array([7.0, 4.6]),
                distance_m=np.nan,
            )
        with self.assertRaisesRegex(ValueError, "must differ"):
            gate_module.forward_carry_target(
                base_xy=np.array([8.0, 4.6]),
                object_xy=np.array([8.0, 4.6]),
                distance_m=0.1,
            )

    def test_trailing_corner_seat_moves_both_grippers_away_from_travel(self):
        targets = trailing_corner_seat_targets(
            {
                "right": np.array([7.25, 4.82, 1.36]),
                "left": np.array([7.25, 4.42, 1.36]),
            },
            travel_direction=np.array([-1.0, 0.0]),
            distance_m=0.08,
        )

        np.testing.assert_allclose(targets["right"], [7.33, 4.82, 1.36])
        np.testing.assert_allclose(targets["left"], [7.33, 4.42, 1.36])

    def test_corner_seat_rejects_invalid_direction_and_distance(self):
        current = {"right": np.zeros(3), "left": np.zeros(3)}
        for direction, distance in (
            (np.zeros(2), 0.08),
            (np.array([1.0, 0.0]), -0.01),
            (np.array([1.0, np.nan]), 0.08),
        ):
            with self.subTest(direction=direction, distance=distance):
                with self.assertRaises(ValueError):
                    trailing_corner_seat_targets(
                        current,
                        travel_direction=direction,
                        distance_m=distance,
                    )

    def test_arm_transport_stroke_moves_toward_goal_and_compensates_height(self):
        targets = arm_transport_stroke_targets(
            {
                "right": np.array([7.25, 4.82, 1.36]),
                "left": np.array([7.25, 4.42, 1.36]),
            },
            travel_direction=np.array([-1.0, 0.0]),
            stroke_m=0.08,
            lift_m=0.05,
        )

        np.testing.assert_allclose(targets["right"], [7.17, 4.82, 1.41])
        np.testing.assert_allclose(targets["left"], [7.17, 4.42, 1.41])

    def test_base_reset_step_advances_base_and_holds_grippers_in_world(self):
        base_command, arm_deltas = compensated_base_reset_step(
            travel_direction=np.array([-1.0, 0.0]),
            base_yaw=np.pi,
            remaining_m=0.08,
            max_speed_m_s=0.04,
            control_dt_s=0.05,
            gripper_world_errors={
                "right": np.array([0.01, 0.0, 0.002]),
                "left": np.array([0.008, 0.0, -0.001]),
            },
        )

        np.testing.assert_allclose(base_command, [0.04, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(arm_deltas["right"], [0.012, 0.0, 0.002])
        np.testing.assert_allclose(arm_deltas["left"], [0.010, 0.0, -0.001])

    def test_macro_progress_accounts_for_reset_backslip(self):
        stroke_progress, stroke_lateral = projected_planar_motion(
            np.array([-0.086, 0.001]),
            direction=np.array([-1.0, 0.0]),
        )
        reset_progress, reset_lateral = projected_planar_motion(
            np.array([0.031, 0.004]),
            direction=np.array([-1.0, 0.0]),
        )

        self.assertAlmostEqual(stroke_progress, 0.086)
        self.assertAlmostEqual(reset_progress, -0.031)
        self.assertAlmostEqual(stroke_progress + reset_progress, 0.055)
        self.assertAlmostEqual(stroke_lateral, 0.001)
        self.assertAlmostEqual(reset_lateral, 0.004)


class L1CradleGateTests(unittest.TestCase):
    def test_geometry_snapshot_includes_all_gripper_geometries(self):
        source = inspect.getsource(geometry_snapshot)

        self.assertIn('lowered.startswith("gripper0_")', source)

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


class CenterRegraspSequenceTests(unittest.TestCase):
    def test_physical_transport_runs_only_after_closed_hold(self):
        source = inspect.getsource(_center_regrasp_probe)

        hold_index = source.index('"hold_center_grasp"')
        transport_index = source.index("run_physical_transport(")
        self.assertLess(hold_index, transport_index)

    def test_optional_corner_seat_runs_after_hold_and_before_transport(self):
        source = inspect.getsource(_center_regrasp_probe)

        hold_index = source.index('"hold_center_grasp"')
        seat_index = source.index('"seat_trailing_corners"')
        transport_index = source.index("run_physical_transport(")
        self.assertLess(hold_index, seat_index)
        self.assertLess(seat_index, transport_index)

    def test_corner_seat_remains_available_before_repeating_inchworm(self):
        source = inspect.getsource(_center_regrasp_probe)
        seat_body = source.index("seat_base_xy")
        condition_start = source.rfind("if (", 0, seat_body)
        condition_end = source.index("):", condition_start) + 2
        condition = source[condition_start:condition_end]

        self.assertNotIn("center_carry_inchworm_distance_m", condition)
        seat_direction = source.index("seat_direction =", seat_body)
        reverse_direction = source.index(
            "if center_carry_inchworm_toward_base:", seat_direction
        )
        seat_targets = source.index("trailing_corner_seat_targets(", seat_direction)
        self.assertLess(reverse_direction, seat_targets)

    def test_optional_arm_stroke_runs_after_hold_and_before_transport(self):
        source = inspect.getsource(_center_regrasp_probe)

        hold_index = source.index('"hold_center_grasp"')
        stroke_index = source.index('"arm_transport_stroke"')
        transport_index = source.index("run_physical_transport(")
        self.assertLess(hold_index, stroke_index)
        self.assertLess(stroke_index, transport_index)

    def test_optional_base_reset_runs_after_arm_stroke_and_before_transport(self):
        source = inspect.getsource(_center_regrasp_probe)

        stroke_index = source.index('"arm_transport_stroke"')
        reset_index = source.index('"inchworm_base_reset"')
        transport_index = source.index("run_physical_transport(")
        self.assertLess(stroke_index, reset_index)
        self.assertLess(reset_index, transport_index)

    def test_repeating_inchworm_transport_runs_only_after_closed_hold(self):
        source = inspect.getsource(_center_regrasp_probe)

        hold_index = source.index('"hold_center_grasp"')
        inchworm_index = source.index("run_inchworm_transport(")
        self.assertLess(hold_index, inchworm_index)

    def test_single_arm_support_transition_runs_after_hold_before_transport(self):
        source = inspect.getsource(_center_regrasp_probe)

        hold_index = source.index('"hold_center_grasp"')
        clearance_index = source.index('"raise_for_under_support"')
        lower_index = source.index('f"lower_{center_support_moving_arm}_for_support"')
        inset_index = source.index('f"inset_{center_support_moving_arm}_under_object"')
        transport_index = source.index("run_physical_transport(")

        self.assertLess(hold_index, clearance_index)
        self.assertLess(clearance_index, lower_index)
        self.assertLess(lower_index, inset_index)
        self.assertLess(inset_index, transport_index)

    def test_single_arm_transition_stops_on_height_or_stationary_contact_loss(self):
        source = inspect.getsource(_center_regrasp_probe)

        self.assertEqual(source.count("required_contact_arm=stationary_arm"), 3)
        self.assertGreaterEqual(source.count("minimum_object_z="), 2)
        self.assertIn("require_bilateral_grasp=True", source)
        self.assertIn('safety_failure = "bilateral_grasp_loss"', source)
        self.assertIn('safety_failure = "height_loss"', source)
        self.assertIn('safety_failure = "required_contact_loss"', source)
        self.assertIn('"stationary_arm_contact": stationary_contact', source)

    def test_support_transition_can_keep_the_moving_gripper_closed(self):
        source = inspect.getsource(_center_regrasp_probe)

        self.assertIn("center_support_keep_moving_gripper_closed", source)
        self.assertIn("moving_gripper_value = (", source)
        self.assertIn("else -1.0", source)

    def test_support_transition_can_combine_descent_and_inset(self):
        source = inspect.getsource(_center_regrasp_probe)

        self.assertIn("center_support_combined_motion", source)
        self.assertIn(
            'f"lower_inset_{center_support_moving_arm}_under_object"',
            source,
        )

    def test_inchworm_extraction_can_reverse_toward_the_robot_base(self):
        direction = gate_module.resolve_inchworm_direction(
            base_xy=np.array([8.0, 4.6]),
            object_xy=np.array([7.0, 4.6]),
            toward_base=True,
        )

        np.testing.assert_allclose(direction, [1.0, 0.0])

    def test_contact_constrained_close_is_guarded_after_failed_approach(self):
        source = inspect.getsource(_center_regrasp_probe)

        approach_index = source.index('"approach_center_walls"')
        collision_guard_index = source.index(
            'if not bool(approach_stage["collision"]):',
            approach_index,
        )
        bracket_index = source.index(
            "fingerpad_bracket_evidence(",
            collision_guard_index,
        )
        close_index = source.index('"close_center_grasp"')
        self.assertLess(approach_index, collision_guard_index)
        self.assertLess(collision_guard_index, bracket_index)
        self.assertLess(bracket_index, close_index)

    def test_high_precenter_precedes_wall_approach_and_close(self):
        source = inspect.getsource(_center_regrasp_probe)

        squeeze_index = source.index('"squeeze_center_walls"')
        approach_index = source.index('"approach_center_walls"')
        close_index = source.index('"close_center_grasp"')
        self.assertLess(squeeze_index, approach_index)
        self.assertLess(approach_index, close_index)

    def test_complete_wall_approach_does_not_stop_on_first_contact(self):
        source = inspect.getsource(_center_regrasp_probe)

        stage_index = source.index('"approach_center_walls"')
        call_index = source.rfind("if not execute_stage(", 0, stage_index)
        approach_call = source[
            call_index:
            source.index('failure_stage = "approach_center_walls"')
        ]
        self.assertNotIn("stop_bilateral_contact_steps", approach_call)

    def test_complete_wall_squeeze_does_not_stop_on_first_contact(self):
        source = inspect.getsource(_center_regrasp_probe)

        stage_index = source.index('"squeeze_center_walls"')
        call_index = source.rfind("if not execute_stage(", 0, stage_index)
        squeeze_call = source[
            call_index:
            source.index('failure_stage = "squeeze_center_walls"')
        ]
        self.assertNotIn("stop_bilateral_contact_steps", squeeze_call)

    def test_center_regrasp_imports_its_base_advance_dependencies(self):
        source = inspect.getsource(_center_regrasp_probe)

        import_block = source[
            source.index("from robot_agent.skills.competition_transport import") :
            source.index("helpers = OfficialScriptedGraspDriver._helpers()")
        ]
        self.assertIn("OfficialPhysicalCarryDriver", import_block)
        self.assertIn("world_velocity_to_base_frame", import_block)

    def test_center_regrasp_closes_before_lift_and_stays_closed(self):
        source = inspect.getsource(_center_regrasp_probe)

        close_start = source.index('"close_center_grasp"')
        lift_start = source.index('"lift_center_grasp"')
        hold_start = source.index('"hold_center_grasp"')
        self.assertLess(close_start, lift_start)
        self.assertLess(lift_start, hold_start)

        close_block = source[close_start:lift_start]
        lift_block = source[lift_start:hold_start]
        hold_block = source[hold_start:]
        self.assertIn("close_schedule=True", close_block)
        self.assertIn("stop_grasp_contact_steps=3", close_block)
        self.assertIn("gripper_value=1.0", lift_block)
        self.assertIn("gripper_value=1.0", hold_block)

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


class L1TableEdgeUndercutTests(unittest.TestCase):
    @staticmethod
    def box_geometry(name, center, half_extents, *, is_object=False):
        return {
            "name": name,
            "is_object": is_object,
            "world_position": list(center),
            "world_rotation": np.eye(3).tolist(),
            "size": list(half_extents),
            "type": 6,
        }

    def test_open_fork_requires_real_planar_overlap_below_bottom(self):
        bottom = self.box_geometry(
            "container_col_bottom",
            [7.059, 4.619, 1.009],
            [0.300, 0.200, 0.009],
            is_object=True,
        )
        outside_x = self.box_geometry(
            "gripper0_right_left_fingertip_collision",
            [7.380, 4.805, 0.975],
            [0.015, 0.035, 0.004],
        )
        overlapping = self.box_geometry(
            "gripper0_right_left_fingertip_collision",
            [7.345, 4.805, 0.975],
            [0.015, 0.035, 0.004],
        )

        self.assertFalse(
            open_fork_under_bottom_support_ready(
                {"geometries": [bottom, outside_x]},
                minimum_planar_overlap_m=0.001,
            )
        )
        self.assertTrue(
            open_fork_under_bottom_support_ready(
                {"geometries": [bottom, overlapping]},
                minimum_planar_overlap_m=0.001,
            )
        )

    def test_open_fork_rejects_finger_above_the_bottom_surface(self):
        bottom = self.box_geometry(
            "container_col_bottom",
            [7.059, 4.619, 1.009],
            [0.300, 0.200, 0.009],
            is_object=True,
        )
        too_high = self.box_geometry(
            "gripper0_right_left_fingertip_collision",
            [7.345, 4.805, 1.010],
            [0.015, 0.035, 0.004],
        )

        self.assertFalse(
            open_fork_under_bottom_support_ready(
                {"geometries": [bottom, too_high]},
                minimum_planar_overlap_m=0.001,
            )
        )

    def test_open_fork_alignment_accepts_measured_safe_partial_rotation(self):
        measured = np.array(
            [
                [0.86882879, 0.12411114, -0.47930467],
                [-0.39157324, -0.42018957, -0.81860315],
                [-0.30299659, 0.89890886, -0.31647420],
            ]
        )

        self.assertTrue(
            open_fork_alignment_sufficient(
                measured,
                inward_axis=np.array([0.0, -1.0, 0.0]),
                min_inward_projection=0.80,
                max_closure_vertical=0.35,
            )
        )

    def test_open_fork_alignment_rejects_shallow_or_vertical_fork(self):
        shallow = np.eye(3)
        vertical_closure = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
        )

        self.assertFalse(
            open_fork_alignment_sufficient(
                shallow,
                inward_axis=np.array([0.0, -1.0, 0.0]),
                min_inward_projection=0.80,
                max_closure_vertical=0.35,
            )
        )
        self.assertFalse(
            open_fork_alignment_sufficient(
                vertical_closure,
                inward_axis=np.array([0.0, -1.0, 0.0]),
                min_inward_projection=0.80,
                max_closure_vertical=0.35,
            )
        )

    def test_open_fork_support_accepts_open_finger_links_for_matching_arm(self):
        self.assertTrue(
            is_allowed_open_fork_support_geom(
                "gripper0_right_right_fingerpad_collision",
                "right",
            )
        )
        self.assertTrue(
            is_allowed_open_fork_support_geom(
                "gripper0_right_hand_collision",
                "right",
            )
        )
        self.assertFalse(
            is_allowed_open_fork_support_geom(
                "gripper0_left_right_fingerpad_collision",
                "right",
            )
        )
        self.assertFalse(
            is_allowed_open_fork_support_geom(
                "input_5_table_collision",
                "right",
            )
        )

    def test_open_fork_orientation_points_tool_inward_and_opening_across_box(self):
        target = open_fork_target_orientation(
            inward_axis=np.array([0.0, -1.0, 0.0]),
            closure_axis=np.array([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(target[:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(target[:, 2], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(target.T @ target, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(target), 1.0)

    def test_open_fork_orientation_rejects_nonorthogonal_axes(self):
        with self.assertRaises(ValueError):
            open_fork_target_orientation(
                inward_axis=np.array([0.0, -1.0, 0.0]),
                closure_axis=np.array([0.0, -1.0, 0.0]),
            )

    def test_rotation_error_reports_full_frame_difference(self):
        target = open_fork_target_orientation(
            inward_axis=np.array([0.0, -1.0, 0.0]),
            closure_axis=np.array([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(rotation_error_degrees(target, target), 0.0)
        self.assertAlmostEqual(rotation_error_degrees(np.eye(3), target), 90.0)

    def test_targets_descend_outside_then_inset_above_the_table_edge(self):
        targets = table_edge_undercut_targets(
            object_center=np.array([7.035, 4.620, 1.125]),
            object_half_depth_m=0.20,
            object_half_height_m=0.125,
            table_edge_y=4.688,
            outside_clearance_m=0.08,
            edge_clearance_m=0.06,
            object_offset_x_m=0.20,
            above_clearance_m=0.15,
            below_bottom_clearance_m=0.05,
            raise_above_bottom_m=0.12,
        )

        np.testing.assert_allclose(targets["outside"], [7.235, 4.900, 1.400])
        np.testing.assert_allclose(targets["below"], [7.235, 4.900, 0.950])
        np.testing.assert_allclose(targets["undercut"], [7.235, 4.748, 0.950])
        np.testing.assert_allclose(targets["raise"], [7.235, 4.748, 1.120])

    def test_targets_reject_an_edge_without_exposed_bottom(self):
        with self.assertRaises(ValueError):
            table_edge_undercut_targets(
                object_center=np.array([7.035, 4.620, 1.125]),
                object_half_depth_m=0.20,
                object_half_height_m=0.125,
                table_edge_y=4.80,
                outside_clearance_m=0.08,
                edge_clearance_m=0.06,
                object_offset_x_m=0.20,
                above_clearance_m=0.15,
                below_bottom_clearance_m=0.05,
                raise_above_bottom_m=0.12,
            )

    def test_gate_accepts_complete_open_gripper_support_evidence(self):
        self.assertTrue(undercut_gate_accepted(VALID_UNDERCUT_RECORD))
        self.assertEqual(undercut_gate_failures(VALID_UNDERCUT_RECORD), [])

    def test_each_hard_condition_rejects_the_record(self):
        invalid_values = {
            "open_gripper": False,
            "support_contact_steps": 4,
            "object_lift_m": 0.019,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "infrastructure_error": "simulator failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_UNDERCUT_RECORD)
                record[key] = value
                self.assertFalse(undercut_gate_accepted(record))
                self.assertIn(key, undercut_gate_failures(record))

    def test_runner_skips_official_grasp_for_the_undercut_mode(self):
        source = inspect.getsource(gate_module.run_probe)

        self.assertIn("if not args.table_edge_undercut", source)
        self.assertIn("table_edge_undercut_no_grasp", source)
        self.assertIn("_table_edge_undercut_probe(", source)

    def test_clearance_target_is_captured_after_optional_base_advance(self):
        source = inspect.getsource(gate_module._table_edge_undercut_probe)

        advance_index = source.index("success = execute_base_advance()")
        capture_index = source.index("initial_eef = right_eef_position()")
        self.assertLess(advance_index, capture_index)

    def test_horizontal_fork_rotates_after_descent_and_skips_deep_inset(self):
        source = inspect.getsource(gate_module._table_edge_undercut_probe)

        descent_index = source.index('"descend_open_outside"')
        orientation_index = source.index('"orient_open_fork_inward"')
        inset_index = source.index('"inset_horizontal_fork_under_overhang"')
        raise_index = source.index('"raise_open_into_support"')
        self.assertLess(descent_index, orientation_index)
        self.assertLess(orientation_index, inset_index)
        self.assertLess(inset_index, raise_index)
        self.assertIn("if success and horizontal_fork", source)
        self.assertIn("open_fork_under_bottom_support_ready", source)
        self.assertIn("other_arm_world_target", source)
        self.assertIn('left_eef_position().copy()', source)
        self.assertIn('"raise_left_clearance_for_torso"', source)
        self.assertIn('"advance_base_for_fork_overlap"', source)
        self.assertIn("fork_raise_target = right_eef_position().copy()", source)


class JointSeedMathTests(unittest.TestCase):
    def test_joint_seed_joint_names_append_only_the_official_torso_joint(self):
        arms_only = joint_seed_joint_names(include_torso=False)
        with_torso = joint_seed_joint_names(include_torso=True)

        self.assertEqual(len(arms_only), 12)
        self.assertEqual(
            arms_only[:6],
            tuple(f"robot0_arm_right_{index}_joint" for index in range(1, 7)),
        )
        self.assertEqual(
            arms_only[6:],
            tuple(f"robot0_arm_left_{index}_joint" for index in range(1, 7)),
        )
        self.assertEqual(with_torso, (*arms_only, "robot0_torso_lift_joint"))

    def test_joint_seed_joint_names_require_a_boolean_option(self):
        with self.assertRaises(ValueError):
            joint_seed_joint_names(include_torso=1)


class BaseAdvanceMathTests(unittest.TestCase):
    def test_base_advance_velocity_points_to_object_and_respects_speed(self):
        velocity = bounded_base_advance_world_velocity(
            base_xy=[8.0, 4.6],
            object_xy=[7.0, 4.6],
            remaining_m=0.10,
            max_speed_m_s=0.04,
            control_dt_s=0.05,
        )

        np.testing.assert_allclose(velocity, [-0.04, 0.0], atol=1e-12)

    def test_base_advance_velocity_clips_the_final_step(self):
        velocity = bounded_base_advance_world_velocity(
            base_xy=[8.0, 4.6],
            object_xy=[7.0, 4.6],
            remaining_m=0.001,
            max_speed_m_s=0.04,
            control_dt_s=0.05,
        )

        np.testing.assert_allclose(velocity, [-0.02, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            bounded_base_advance_world_velocity(
                base_xy=[8.0, 4.6],
                object_xy=[7.0, 4.6],
                remaining_m=0.0,
                max_speed_m_s=0.04,
                control_dt_s=0.05,
            ),
            np.zeros(2),
        )

    def test_base_advance_velocity_rejects_invalid_inputs(self):
        common = {
            "base_xy": [8.0, 4.6],
            "object_xy": [7.0, 4.6],
            "remaining_m": 0.10,
            "max_speed_m_s": 0.04,
            "control_dt_s": 0.05,
        }
        invalid = (
            {"base_xy": [8.0]},
            {"object_xy": [float("nan"), 4.6]},
            {"remaining_m": -0.01},
            {"max_speed_m_s": 0.0},
            {"control_dt_s": float("inf")},
            {"object_xy": [8.0, 4.6]},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    bounded_base_advance_world_velocity(**{**common, **override})

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
        self.assertFalse(args.orientation_joint_seed_include_torso)
        self.assertAlmostEqual(args.orientation_joint_seed_torso_margin_m, 0.005)
        self.assertAlmostEqual(args.regrasp_base_advance_m, 0.0)
        self.assertAlmostEqual(args.center_carry_distance_m, 0.0)
        self.assertFalse(args.center_carry_away_from_object)
        self.assertAlmostEqual(args.center_carry_max_linear, 0.04)
        self.assertAlmostEqual(args.center_carry_corner_seat_m, 0.0)
        self.assertAlmostEqual(args.center_carry_arm_stroke_m, 0.0)
        self.assertAlmostEqual(args.center_carry_arm_stroke_lift_m, 0.0)
        self.assertAlmostEqual(args.center_carry_base_reset_m, 0.0)
        self.assertAlmostEqual(args.center_carry_inchworm_distance_m, 0.0)
        self.assertFalse(args.center_carry_inchworm_toward_base)
        self.assertAlmostEqual(args.center_carry_inchworm_stroke_m, 0.08)
        self.assertAlmostEqual(args.center_carry_inchworm_reset_m, 0.06)
        self.assertIsNone(args.center_carry_inchworm_world_direction_x)
        self.assertIsNone(args.center_carry_inchworm_world_direction_y)
        self.assertEqual(args.center_support_moving_arm, "none")
        self.assertAlmostEqual(args.center_support_clearance_lift_m, 0.08)
        self.assertAlmostEqual(args.center_support_descent_m, 0.12)
        self.assertAlmostEqual(args.center_support_inset_m, 0.04)
        self.assertFalse(args.center_support_keep_moving_gripper_closed)
        self.assertFalse(args.center_support_combined_motion)
        self.assertFalse(args.table_edge_undercut)
        self.assertAlmostEqual(args.undercut_table_edge_y, 4.688)
        self.assertAlmostEqual(args.undercut_outside_clearance_m, 0.08)
        self.assertAlmostEqual(args.undercut_edge_clearance_m, 0.06)
        self.assertAlmostEqual(args.undercut_above_clearance_m, 0.15)
        self.assertAlmostEqual(args.undercut_base_advance_m, 0.0)
        self.assertAlmostEqual(args.undercut_object_offset_x_m, 0.20)
        self.assertIsNone(args.undercut_torso_target_m)
        self.assertAlmostEqual(args.undercut_below_bottom_clearance_m, 0.05)
        self.assertAlmostEqual(args.undercut_raise_above_bottom_m, 0.12)
        self.assertFalse(args.undercut_horizontal_fork)
        self.assertAlmostEqual(args.undercut_orientation_max_action, 0.08)
        self.assertAlmostEqual(
            args.undercut_orientation_position_max_action,
            0.30,
        )
        self.assertAlmostEqual(args.undercut_orientation_tolerance_deg, 3.0)
        self.assertEqual(args.undercut_orientation_stable_steps, 5)
        self.assertEqual(args.undercut_orientation_max_steps, 240)
        self.assertAlmostEqual(
            args.undercut_orientation_min_inward_projection,
            0.80,
        )
        self.assertAlmostEqual(
            args.undercut_orientation_max_closure_vertical,
            0.35,
        )
        self.assertAlmostEqual(args.undercut_horizontal_inset_m, 0.06)
        self.assertAlmostEqual(args.undercut_left_clearance_lift_m, 0.25)
        self.assertAlmostEqual(args.undercut_post_inset_base_advance_m, 0.0)

    def test_center_carry_speed_can_be_overridden_for_single_variable_probe(self):
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
                "--center-carry-max-linear",
                "0.005",
                "--center-carry-away-from-object",
                "--center-carry-corner-seat-m",
                "0.08",
                "--center-carry-arm-stroke-m",
                "0.07",
                "--center-carry-arm-stroke-lift-m",
                "0.04",
                "--center-carry-base-reset-m",
                "0.07",
                "--center-carry-inchworm-distance-m",
                "0.06",
                "--center-carry-inchworm-toward-base",
                "--center-carry-inchworm-stroke-m",
                "0.06",
                "--center-carry-inchworm-reset-m",
                "0.04",
                "--center-carry-inchworm-world-direction-x",
                "0.0",
                "--center-carry-inchworm-world-direction-y",
                "-1.0",
                "--center-support-moving-arm",
                "right",
                "--center-support-clearance-lift-m",
                "0.09",
                "--center-support-descent-m",
                "0.13",
                "--center-support-inset-m",
                "0.05",
                "--center-support-keep-moving-gripper-closed",
                "--center-support-combined-motion",
                "--table-edge-undercut",
                "--undercut-table-edge-y",
                "4.70",
                "--undercut-outside-clearance-m",
                "0.09",
                "--undercut-edge-clearance-m",
                "0.05",
                "--undercut-above-clearance-m",
                "0.07",
                "--undercut-base-advance-m",
                "0.10",
                "--undercut-object-offset-x-m",
                "0.31",
                "--undercut-torso-target-m",
                "0.15",
                "--undercut-below-bottom-clearance-m",
                "0.03",
                "--undercut-raise-above-bottom-m",
                "0.14",
                "--undercut-horizontal-fork",
                "--undercut-orientation-max-action",
                "0.06",
                "--undercut-orientation-position-max-action",
                "0.25",
                "--undercut-orientation-tolerance-deg",
                "2.0",
                "--undercut-orientation-stable-steps",
                "7",
                "--undercut-orientation-max-steps",
                "300",
                "--undercut-orientation-min-inward-projection",
                "0.85",
                "--undercut-orientation-max-closure-vertical",
                "0.30",
                "--undercut-horizontal-inset-m",
                "0.07",
                "--undercut-left-clearance-lift-m",
                "0.20",
                "--undercut-post-inset-base-advance-m",
                "0.05",
            ]
        )

        self.assertAlmostEqual(args.center_carry_max_linear, 0.005)
        self.assertTrue(args.center_carry_away_from_object)
        self.assertAlmostEqual(args.center_carry_corner_seat_m, 0.08)
        self.assertAlmostEqual(args.center_carry_arm_stroke_m, 0.07)
        self.assertAlmostEqual(args.center_carry_arm_stroke_lift_m, 0.04)
        self.assertAlmostEqual(args.center_carry_base_reset_m, 0.07)
        self.assertAlmostEqual(args.center_carry_inchworm_distance_m, 0.06)
        self.assertTrue(args.center_carry_inchworm_toward_base)
        self.assertAlmostEqual(args.center_carry_inchworm_stroke_m, 0.06)
        self.assertAlmostEqual(args.center_carry_inchworm_reset_m, 0.04)
        self.assertAlmostEqual(args.center_carry_inchworm_world_direction_x, 0.0)
        self.assertAlmostEqual(args.center_carry_inchworm_world_direction_y, -1.0)
        self.assertEqual(args.center_support_moving_arm, "right")
        self.assertAlmostEqual(args.center_support_clearance_lift_m, 0.09)
        self.assertAlmostEqual(args.center_support_descent_m, 0.13)
        self.assertAlmostEqual(args.center_support_inset_m, 0.05)
        self.assertTrue(args.center_support_keep_moving_gripper_closed)
        self.assertTrue(args.center_support_combined_motion)
        self.assertTrue(args.table_edge_undercut)
        self.assertAlmostEqual(args.undercut_table_edge_y, 4.70)
        self.assertAlmostEqual(args.undercut_outside_clearance_m, 0.09)
        self.assertAlmostEqual(args.undercut_edge_clearance_m, 0.05)
        self.assertAlmostEqual(args.undercut_above_clearance_m, 0.07)
        self.assertAlmostEqual(args.undercut_base_advance_m, 0.10)
        self.assertAlmostEqual(args.undercut_object_offset_x_m, 0.31)
        self.assertAlmostEqual(args.undercut_torso_target_m, 0.15)
        self.assertAlmostEqual(args.undercut_below_bottom_clearance_m, 0.03)
        self.assertAlmostEqual(args.undercut_raise_above_bottom_m, 0.14)
        self.assertTrue(args.undercut_horizontal_fork)
        self.assertAlmostEqual(args.undercut_orientation_max_action, 0.06)
        self.assertAlmostEqual(
            args.undercut_orientation_position_max_action,
            0.25,
        )
        self.assertAlmostEqual(args.undercut_orientation_tolerance_deg, 2.0)
        self.assertEqual(args.undercut_orientation_stable_steps, 7)
        self.assertEqual(args.undercut_orientation_max_steps, 300)
        self.assertAlmostEqual(
            args.undercut_orientation_min_inward_projection,
            0.85,
        )
        self.assertAlmostEqual(
            args.undercut_orientation_max_closure_vertical,
            0.30,
        )
        self.assertAlmostEqual(args.undercut_horizontal_inset_m, 0.07)
        self.assertAlmostEqual(args.undercut_left_clearance_lift_m, 0.20)
        self.assertAlmostEqual(args.undercut_post_inset_base_advance_m, 0.05)


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
