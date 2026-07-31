import inspect
import unittest
from types import SimpleNamespace

import numpy as np

import scripts.run_l1_cradle_gate as gate_module
from scripts.run_l1_cradle_gate import (
    _center_regrasp_probe,
    _end_grasp_floor_push_probe,
    _end_grasp_regrasp_probe,
    _end_grasp_setdown_probe,
    allocate_segment_steps,
    arm_transport_stroke_targets,
    bounded_base_advance_world_velocity,
    closure_axis_error_degrees,
    compensated_base_reset_step,
    cradle_gate_accepted,
    cradle_gate_failures,
    eef_site_pose,
    floor_base_reposition_targets,
    floor_base_target_route,
    floor_base_tracking_velocity,
    floor_regrasp_safe_base_xy,
    floor_push_staging_targets,
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
    navigation_retract_targets,
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
    task_for_index,
    push_gate_accepted,
    push_gate_failures,
    hybrid_exit_gate_accepted,
    hybrid_exit_gate_failures,
    setdown_gate_accepted,
    setdown_gate_failures,
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

VALID_HYBRID_EXIT_RECORD = {
    "physical_grasp": True,
    "extraction_success": True,
    "floor_transition_detected": True,
    "navigation_retract_success": True,
    "floor_push_success": True,
    "physical_contact_steps": 20,
    "official_source_maximum_axis_displacement_m": 1.000001,
    "official_target_distance_m": 0.799999,
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

VALID_POSTURE_CARRY_RECORD = {
    "posture_carry_success": True,
    "projected_object_progress_m": 0.08,
    "lateral_object_drift_m": 0.03,
    "object_gripper_drift_m": 0.03,
    "final_object_lift_m": 0.10,
    "terminal_bilateral_contact": True,
    "collision_frames": 0,
    "attachment_activations": 0,
    "legacy_teleport_activations": 0,
    "object_pose_writes": 0,
    "infrastructure_error": None,
}

VALID_SETDOWN_RECORD = {
    "physical_grasp": True,
    "transport_success": True,
    "place_success": True,
    "support_detected": True,
    "released": True,
    "object_translation_m": 0.12,
    "net_projected_object_progress_m": 0.12,
    "net_lateral_object_drift_m": 0.05,
    "requested_macro_count": 1,
    "completed_macro_count": 1,
    "attachment_calls": 0,
    "object_pose_writes": 0,
    "collision_frames": 0,
    "infrastructure_error": None,
}


class EndGraspSetdownGateTests(unittest.TestCase):
    def test_navigation_retract_targets_are_base_relative_and_compact(self):
        targets = navigation_retract_targets(
            base_xy=[8.20, 4.60],
            base_yaw=np.pi,
            forward_m=0.20,
            lateral_m=0.15,
            target_z=1.45,
        )

        np.testing.assert_allclose(targets["right"], [8.00, 4.75, 1.45])
        np.testing.assert_allclose(targets["left"], [8.00, 4.45, 1.45])

    def test_floor_regrasp_safe_base_stays_on_current_outer_ray(self):
        object_xy = np.array([7.78, 4.48])
        current_base_xy = np.array([8.21, 4.61])

        safe_xy = floor_regrasp_safe_base_xy(
            object_xy=object_xy,
            current_base_xy=current_base_xy,
            clearance_m=1.20,
        )

        self.assertAlmostEqual(np.linalg.norm(safe_xy - object_xy), 1.20)
        self.assertGreater(safe_xy[0], current_base_xy[0])
        self.assertGreater(safe_xy[1], current_base_xy[1])

    def test_floor_push_staging_targets_preserve_safe_lane_offset(self):
        targets = floor_push_staging_targets(
            object_xy=[7.784, 4.482],
            current_base_xy=[8.210, 4.605],
            push_direction_xy=[0.0, -1.0],
            base_standoff_m=0.85,
            orientation_clearance_m=0.35,
            lateral_offset_m=None,
            maximum_lateral_offset_m=0.25,
            face_offset_m=0.24,
            hand_separation_m=0.28,
            hand_height_m=0.38,
            precontact_clearance_m=0.08,
        )

        np.testing.assert_allclose(targets["direction"], [0.0, -1.0])
        np.testing.assert_allclose(targets["stage_base_xy"], [8.034, 5.332])
        np.testing.assert_allclose(
            targets["orientation_base_xy"],
            [8.034, 5.682],
        )
        np.testing.assert_allclose(targets["escape_base_xy"], [8.210, 5.682])
        np.testing.assert_allclose(targets["contact"]["right"], [7.644, 4.722, 0.38])
        np.testing.assert_allclose(targets["contact"]["left"], [7.924, 4.722, 0.38])
        np.testing.assert_allclose(
            targets["precontact"]["right"],
            [7.644, 4.802, 0.46],
        )
        np.testing.assert_allclose(
            targets["precontact"]["left"],
            [7.924, 4.802, 0.46],
        )
        self.assertAlmostEqual(targets["target_yaw"], -np.pi / 2.0)
        self.assertAlmostEqual(targets["lateral_offset_m"], 0.25)

    def test_floor_push_staging_targets_accept_safe_signed_lane_offset(self):
        targets = floor_push_staging_targets(
            object_xy=[7.784, 4.482],
            current_base_xy=[8.210, 4.605],
            push_direction_xy=[0.0, -1.0],
            base_standoff_m=0.85,
            orientation_clearance_m=0.35,
            lateral_offset_m=-0.15,
            maximum_lateral_offset_m=0.25,
            face_offset_m=0.24,
            hand_separation_m=0.28,
            hand_height_m=0.38,
            precontact_clearance_m=0.08,
        )

        np.testing.assert_allclose(targets["stage_base_xy"], [7.634, 5.332])
        self.assertAlmostEqual(targets["lateral_offset_m"], -0.15)

    def test_floor_base_target_route_uses_safe_bottom_corridor(self):
        route = floor_base_target_route(
            start_object_xy=[7.979, 4.560],
            target_xy=[-0.166, -7.290],
            corridor_y=-8.40,
            arrival_radius_m=0.80,
            arrival_margin_m=0.05,
        )

        self.assertEqual(len(route["segments"]), 3)
        np.testing.assert_allclose(
            [segment["direction"] for segment in route["segments"]],
            [[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]],
        )
        np.testing.assert_allclose(
            [segment["distance_m"] for segment in route["segments"]],
            [12.960, 8.145, 0.360],
            atol=1e-9,
        )
        np.testing.assert_allclose(route["final_object_xy"], [-0.166, -8.040])
        self.assertAlmostEqual(route["final_target_distance_m"], 0.75)

    def test_floor_base_reposition_targets_go_around_object(self):
        targets = floor_base_reposition_targets(
            object_xy=[8.0, -8.4],
            current_base_xy=[8.0, -7.95],
            next_push_direction_xy=[-1.0, 0.0],
            retreat_clearance_m=0.90,
            base_standoff_m=0.65,
        )

        np.testing.assert_allclose(targets["retreat_base_xy"], [8.0, -7.50])
        np.testing.assert_allclose(targets["corner_base_xy"], [8.65, -7.50])
        np.testing.assert_allclose(targets["stage_base_xy"], [8.65, -8.40])
        self.assertAlmostEqual(targets["target_yaw"], np.pi)

    def test_floor_base_tracking_velocity_corrects_lateral_drift(self):
        velocity = floor_base_tracking_velocity(
            push_direction_xy=[0.0, -1.0],
            lateral_error_m=0.10,
            base_object_lateral_offset_m=0.0,
            forward_speed_m_s=0.04,
            lateral_gain=0.50,
            alignment_gain=0.50,
            lateral_deadband_m=0.05,
            maximum_base_object_offset_m=0.08,
            maximum_lateral_speed_m_s=0.02,
        )

        np.testing.assert_allclose(velocity, [0.0125, -0.04])

        recenter_velocity = floor_base_tracking_velocity(
            push_direction_xy=[0.0, -1.0],
            lateral_error_m=0.10,
            base_object_lateral_offset_m=-0.20,
            forward_speed_m_s=0.04,
            lateral_gain=0.50,
            alignment_gain=0.50,
            lateral_deadband_m=0.05,
            maximum_base_object_offset_m=0.08,
            maximum_lateral_speed_m_s=0.02,
        )
        np.testing.assert_allclose(recenter_velocity, [0.02, -0.04])

        deadband_velocity = floor_base_tracking_velocity(
            push_direction_xy=[0.0, -1.0],
            lateral_error_m=-0.04,
            base_object_lateral_offset_m=0.0,
            forward_speed_m_s=0.04,
            lateral_gain=0.50,
            alignment_gain=0.50,
            lateral_deadband_m=0.05,
            maximum_base_object_offset_m=0.08,
            maximum_lateral_speed_m_s=0.02,
        )
        np.testing.assert_allclose(deadband_velocity, [0.0, -0.04])

    def test_hybrid_exit_gate_requires_strict_official_exit_and_physics(self):
        self.assertEqual(hybrid_exit_gate_failures(VALID_HYBRID_EXIT_RECORD), [])
        self.assertTrue(hybrid_exit_gate_accepted(VALID_HYBRID_EXIT_RECORD))

        invalid_values = {
            "physical_grasp": False,
            "extraction_success": False,
            "floor_transition_detected": False,
            "navigation_retract_success": False,
            "floor_push_success": False,
            "physical_contact_steps": 19,
            "official_source_maximum_axis_displacement_m": 1.0,
            "official_target_distance_m": 0.8,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "infrastructure_error": "RuntimeError: failed",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_HYBRID_EXIT_RECORD)
                record[key] = value
                self.assertIn(key, hybrid_exit_gate_failures(record))
                self.assertFalse(hybrid_exit_gate_accepted(record))

    def test_setdown_gate_accepts_every_inclusive_boundary(self):
        self.assertEqual(setdown_gate_failures(VALID_SETDOWN_RECORD), [])
        self.assertTrue(setdown_gate_accepted(VALID_SETDOWN_RECORD))

    def test_each_setdown_hard_condition_rejects_the_record(self):
        invalid_values = {
            "physical_grasp": False,
            "transport_success": False,
            "place_success": False,
            "support_detected": False,
            "released": False,
            "object_translation_m": 0.119999,
            "net_projected_object_progress_m": 0.119999,
            "net_lateral_object_drift_m": 0.050001,
            "requested_macro_count": 0,
            "completed_macro_count": 0,
            "attachment_calls": 1,
            "object_pose_writes": 1,
            "collision_frames": 1,
            "infrastructure_error": "RuntimeError: failed",
        }

        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_SETDOWN_RECORD)
                record[key] = value
                self.assertIn(key, setdown_gate_failures(record))
                self.assertFalse(setdown_gate_accepted(record))

    def test_missing_and_nonfinite_setdown_evidence_is_rejected(self):
        for key in VALID_SETDOWN_RECORD:
            with self.subTest(missing=key):
                record = dict(VALID_SETDOWN_RECORD)
                del record[key]
                self.assertIn(key, setdown_gate_failures(record))

        for key in (
            "object_translation_m",
            "net_projected_object_progress_m",
            "net_lateral_object_drift_m",
            "requested_macro_count",
            "completed_macro_count",
            "attachment_calls",
            "object_pose_writes",
            "collision_frames",
        ):
            for value in (np.nan, np.inf, -np.inf, True, "invalid"):
                with self.subTest(key=key, value=value):
                    record = dict(VALID_SETDOWN_RECORD)
                    record[key] = value
                    self.assertIn(key, setdown_gate_failures(record))

    def test_setdown_gate_scales_net_progress_with_requested_macros(self):
        record = {
            **VALID_SETDOWN_RECORD,
            "requested_macro_count": 2,
            "completed_macro_count": 2,
            "net_projected_object_progress_m": 0.239999,
        }

        self.assertIn(
            "net_projected_object_progress_m",
            setdown_gate_failures(record),
        )
        record["net_projected_object_progress_m"] = 0.24
        self.assertEqual(setdown_gate_failures(record), [])

    def test_regrasp_probe_repeats_only_after_successful_physical_release(self):
        raw_env = SimpleNamespace(
            obj_body_id={"box": 0},
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos=np.array([[0.0, 0.0, 1.0]], dtype=float)
                )
            ),
        )
        backend = SimpleNamespace(env=raw_env)

        class Driver:
            def __init__(self):
                self.calls = []
                self._physical_hold = {"active": True}

            def move(self, source, *, carrying, object_name):
                self.calls.append(("move", source, carrying, object_name))
                return True

            def grasp(self, source, object_name):
                self.calls.append(("grasp", source, object_name))
                return {
                    "success": True,
                    "lift_success": True,
                    "contacts": {"right": True, "left": True},
                }

        driver = Driver()
        setdown_calls = []

        def setdown(_backend, object_name, **kwargs):
            setdown_calls.append((object_name, kwargs))
            raw_env.sim.data.body_xpos[0, 0] += 0.20
            return {
                "success": True,
                "failure_stage": None,
                "transport_success": True,
                "place_success": True,
                "support_detected": True,
                "released": True,
                "world_direction": [1.0, 0.0],
                "attachment_activations": 0,
                "object_pose_writes": 0,
            }

        result = _end_grasp_regrasp_probe(
            backend,
            driver,
            "input_5",
            "box",
            macro_count=2,
            distance_m=0.14,
            world_direction_x=1.0,
            world_direction_y=0.0,
            table_object_z=1.0,
            stroke_m=0.08,
            stroke_lift_m=0.0,
            height_gain=0.0,
            reset_m=0.06,
            minimum_lift_m=0.10,
            place_max_descent_m=0.35,
            _setdown_probe=setdown,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["requested_macro_count"], 2)
        self.assertEqual(result["completed_macro_count"], 2)
        self.assertEqual(len(result["macros"]), 2)
        self.assertEqual(len(setdown_calls), 2)
        self.assertEqual(
            driver.calls,
            [
                ("move", "input_5", False, "box"),
                ("grasp", "input_5", "box"),
            ],
        )
        self.assertIsNone(driver._physical_hold)
        self.assertAlmostEqual(result["measured_object_translation_m"], 0.40)

    def test_floor_regrasp_physically_retracts_before_navigation(self):
        raw_env = SimpleNamespace(
            obj_body_id={"box": 0},
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos=np.array([[0.0, 0.0, 0.20]], dtype=float)
                )
            ),
        )
        backend = SimpleNamespace(env=raw_env)
        calls = []

        class Driver:
            _physical_hold = {"active": True}

            def move(self, source, *, carrying, object_name):
                calls.append("move")
                return True

            def grasp(self, source, object_name):
                calls.append("grasp")
                return {
                    "success": True,
                    "lift_success": True,
                    "contacts": {"right": True, "left": True},
                }

        def setdown(_backend, object_name, **kwargs):
            calls.append("setdown")
            raw_env.sim.data.body_xpos[0, 0] += 0.20
            return {
                "success": True,
                "transport_success": True,
                "place_success": True,
                "support_detected": True,
                "released": True,
                "world_direction": [1.0, 0.0],
                "attachment_activations": 0,
                "object_pose_writes": 0,
            }

        def retract(_backend, **kwargs):
            calls.append("retract")
            return {"success": True, "collision": False, "targets": {}}

        def floor_move(_backend, _driver, source, object_name, **kwargs):
            calls.append("floor_move")
            return {"success": True, "collision": False}

        result = _end_grasp_regrasp_probe(
            backend,
            Driver(),
            "input_5",
            "box",
            macro_count=2,
            distance_m=0.14,
            world_direction_x=1.0,
            world_direction_y=0.0,
            table_object_z=1.0,
            stroke_m=0.08,
            stroke_lift_m=0.0,
            height_gain=0.0,
            reset_m=0.06,
            minimum_lift_m=0.10,
            place_max_descent_m=0.45,
            floor_retract_forward_m=0.20,
            floor_retract_lateral_m=0.15,
            floor_retract_target_z=1.45,
            floor_transition_margin_m=0.30,
            _setdown_probe=setdown,
            _navigation_retract=retract,
            _floor_regrasp_move=floor_move,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            calls,
            ["setdown", "retract", "floor_move", "grasp", "setdown"],
        )
        self.assertTrue(result["regrasps"][0]["navigation_retract"]["success"])

    def test_floor_push_runs_only_after_extraction_and_physical_retract(self):
        raw_env = SimpleNamespace(
            obj_body_id={"box": 0},
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos=np.array([[0.0, 0.0, 1.0]], dtype=float)
                )
            ),
        )
        backend = SimpleNamespace(env=raw_env)
        calls = []

        def extraction(_backend, _driver, source, object_name, **kwargs):
            calls.append("extraction")
            raw_env.sim.data.body_xpos[0] = [0.72, -0.14, 0.20]
            return {
                "success": True,
                "start_object_position": [0.0, 0.0, 1.0],
                "end_object_position": [0.72, -0.14, 0.20],
                "requested_macro_count": 2,
                "completed_macro_count": 2,
                "transport_success": True,
                "place_success": True,
                "support_detected": True,
                "released": True,
                "attachment_activations": 0,
                "object_pose_writes": 0,
            }

        def retract(_backend, **kwargs):
            calls.append("retract")
            return {"success": True, "collision": False}

        def floor_push(_backend, object_name, **kwargs):
            calls.append("floor_push")
            raw_env.sim.data.body_xpos[0] = [0.72, -1.01, 0.20]
            return {
                "success": True,
                "failure_stage": None,
                "physical_contact_steps": 30,
                "object_progress_m": 0.87,
            }

        result = _end_grasp_floor_push_probe(
            backend,
            SimpleNamespace(),
            "input_5",
            "box",
            macro_count=2,
            table_object_z=1.0,
            floor_transition_margin_m=0.30,
            push_direction_x=0.0,
            push_direction_y=-1.0,
            _extraction_probe=extraction,
            _navigation_retract=retract,
            _floor_push=floor_push,
        )

        self.assertTrue(result["success"])
        self.assertEqual(calls, ["extraction", "retract", "floor_push"])
        self.assertTrue(result["floor_transition_detected"])
        self.assertAlmostEqual(result["maximum_axis_displacement_m"], 1.01)
        self.assertEqual(result["physical_contact_steps"], 30)


class PostureCarryGateTests(unittest.TestCase):
    def test_posture_carry_gate_accepts_every_inclusive_boundary(self):
        self.assertTrue(hasattr(gate_module, "posture_carry_failures"))
        self.assertTrue(hasattr(gate_module, "posture_carry_accepted"))

        self.assertEqual(
            gate_module.posture_carry_failures(VALID_POSTURE_CARRY_RECORD),
            [],
        )
        self.assertTrue(
            gate_module.posture_carry_accepted(VALID_POSTURE_CARRY_RECORD)
        )

    def test_each_posture_carry_hard_condition_rejects_the_record(self):
        invalid_values = {
            "posture_carry_success": False,
            "projected_object_progress_m": 0.079999,
            "lateral_object_drift_m": 0.030001,
            "object_gripper_drift_m": 0.030001,
            "final_object_lift_m": 0.099999,
            "terminal_bilateral_contact": False,
            "collision_frames": 1,
            "attachment_activations": 1,
            "legacy_teleport_activations": 1,
            "object_pose_writes": 1,
            "infrastructure_error": "RuntimeError: failed",
        }

        for key, value in invalid_values.items():
            with self.subTest(key=key):
                record = dict(VALID_POSTURE_CARRY_RECORD)
                record[key] = value
                self.assertIn(
                    key,
                    gate_module.posture_carry_failures(record),
                )
                self.assertFalse(gate_module.posture_carry_accepted(record))

    def test_missing_and_nonfinite_posture_carry_evidence_is_rejected(self):
        for key in VALID_POSTURE_CARRY_RECORD:
            with self.subTest(missing=key):
                record = dict(VALID_POSTURE_CARRY_RECORD)
                del record[key]
                self.assertIn(key, gate_module.posture_carry_failures(record))

        numeric_fields = (
            "projected_object_progress_m",
            "lateral_object_drift_m",
            "object_gripper_drift_m",
            "final_object_lift_m",
            "collision_frames",
            "attachment_activations",
            "legacy_teleport_activations",
            "object_pose_writes",
        )
        for key in numeric_fields:
            for value in (np.nan, np.inf, -np.inf, True, "invalid"):
                with self.subTest(key=key, value=value):
                    record = dict(VALID_POSTURE_CARRY_RECORD)
                    record[key] = value
                    self.assertIn(
                        key,
                        gate_module.posture_carry_failures(record),
                    )


class TransportAttachmentAuditTests(unittest.TestCase):
    @staticmethod
    def fake_module():
        calls = []

        def capture(env, object_name):
            calls.append(("capture", env, object_name))
            return {"active": True, "object_name": object_name}

        def write(env, joint_name, qpos):
            calls.append(("write", env, joint_name, tuple(qpos)))
            return "written"

        module = SimpleNamespace(
            TRANSPORT_ATTACHMENT_ATTR="_transport_attachment",
            capture_transport_attachment=capture,
            set_object_qpos=write,
        )
        return module, calls

    def test_transport_attachment_audit_counts_without_suppressing_calls(self):
        self.assertTrue(hasattr(gate_module, "transport_attachment_audit"))
        module, calls = self.fake_module()
        original_capture = module.capture_transport_attachment
        original_write = module.set_object_qpos
        raw_env = SimpleNamespace(
            _transport_attachment={"active": False}
        )

        with gate_module.transport_attachment_audit(raw_env, module) as audit:
            attachment = module.capture_transport_attachment(raw_env, "box")
            result = module.set_object_qpos(raw_env, "box_free", [1.0, 2.0])

        self.assertEqual(attachment["object_name"], "box")
        self.assertEqual(result, "written")
        self.assertEqual([entry[0] for entry in calls], ["capture", "write"])
        self.assertEqual(audit["attachment_activations"], 1)
        self.assertEqual(audit["object_pose_writes"], 1)
        self.assertFalse(audit["active_before"])
        self.assertFalse(audit["active_after"])
        self.assertIs(module.capture_transport_attachment, original_capture)
        self.assertIs(module.set_object_qpos, original_write)

    def test_transport_attachment_audit_reports_active_state(self):
        module, _ = self.fake_module()
        raw_env = SimpleNamespace(
            _transport_attachment={"active": True}
        )

        with gate_module.transport_attachment_audit(raw_env, module) as audit:
            raw_env._transport_attachment["active"] = False

        self.assertTrue(audit["active_before"])
        self.assertFalse(audit["active_after"])

    def test_transport_attachment_audit_restores_functions_after_exception(self):
        module, _ = self.fake_module()
        original_capture = module.capture_transport_attachment
        original_write = module.set_object_qpos
        raw_env = SimpleNamespace()

        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with gate_module.transport_attachment_audit(raw_env, module):
                raise RuntimeError("probe failed")

        self.assertIs(module.capture_transport_attachment, original_capture)
        self.assertIs(module.set_object_qpos, original_write)


class DirectedPlanarProgressTests(unittest.TestCase):
    def test_normalizes_direction_and_reports_forward_progress(self):
        progress, lateral = gate_module.directed_planar_progress(
            start_xy=[1.0, 2.0],
            end_xy=[1.06, 2.08],
            direction_xy=[3.0, 4.0],
        )

        self.assertAlmostEqual(progress, 0.10)
        self.assertAlmostEqual(lateral, 0.0)

    def test_reports_signed_progress_and_unsigned_lateral_drift(self):
        progress, lateral = gate_module.directed_planar_progress(
            start_xy=[0.0, 0.0],
            end_xy=[0.06, 0.13],
            direction_xy=[3.0, 4.0],
        )
        reverse, reverse_lateral = gate_module.directed_planar_progress(
            start_xy=[0.06, 0.08],
            end_xy=[0.0, 0.0],
            direction_xy=[3.0, 4.0],
        )

        self.assertAlmostEqual(progress, 0.14)
        self.assertAlmostEqual(lateral, 0.03)
        self.assertAlmostEqual(reverse, -0.10)
        self.assertAlmostEqual(reverse_lateral, 0.0)

    def test_rejects_invalid_points_and_direction(self):
        invalid = (
            ([0.0], [0.0, 0.0], [1.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [float("nan"), 0.0], [1.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0], [float("inf"), 0.0]),
        )
        for start_xy, end_xy, direction_xy in invalid:
            with self.subTest(
                start_xy=start_xy,
                end_xy=end_xy,
                direction_xy=direction_xy,
            ):
                with self.assertRaises(ValueError):
                    gate_module.directed_planar_progress(
                        start_xy=start_xy,
                        end_xy=end_xy,
                        direction_xy=direction_xy,
                    )


class PostureLockedCarryProbeTests(unittest.TestCase):
    class FakeBackend:
        def __init__(self, *, legacy_held=False):
            self.base_xy = np.array([-1.0, 0.0], dtype=float)
            self.grippers = {
                "right": np.array([-0.10, -0.15, 1.25], dtype=float),
                "left": np.array([-0.10, 0.15, 1.25], dtype=float),
            }
            self.env = SimpleNamespace(
                obj_body_id={"box": 0},
                sim=SimpleNamespace(
                    data=SimpleNamespace(
                        body_xpos=np.array([[0.0, 0.0, 1.20]], dtype=float)
                    )
                ),
                robots=[SimpleNamespace()],
                _transport_attachment={"active": False},
                has_judge_collision=False,
            )
            self._held_crate_name = "box" if legacy_held else None
            self._held_crate_body_id = 0 if legacy_held else None
            self._max_linear = 0.70
            self.follow_calls = []
            self.follow_speeds = []
            self.record_calls = 0

        def get_base_pose(self):
            return self.base_xy.copy(), 0.0

        def _record_trajectory_frame(self, *args, **kwargs):
            del args, kwargs
            self.record_calls += 1

        def _update_held_crate_position(self):
            if self._held_crate_name is not None:
                self.env.sim.data.body_xpos[0] = np.array(
                    [self.base_xy[0], self.base_xy[1], 1.20], dtype=float
                )

        def follow_path(self, path, **kwargs):
            self.follow_calls.append((path, kwargs))
            self.follow_speeds.append(self._max_linear)
            self.base_xy = np.asarray(path[0], dtype=float).copy()
            displacement = np.array([0.09, 0.0, 0.0], dtype=float)
            self.env.sim.data.body_xpos[0] += displacement
            for arm in self.grippers:
                self.grippers[arm] += displacement
            self._record_trajectory_frame(_env=self.env)
            return True

    @staticmethod
    def fake_transport_module():
        return SimpleNamespace(
            TRANSPORT_ATTACHMENT_ATTR="_transport_attachment",
            capture_transport_attachment=lambda *args, **kwargs: None,
            set_object_qpos=lambda *args, **kwargs: None,
        )

    @staticmethod
    def contacts(*args, **kwargs):
        del args, kwargs
        return {"right": ("right_pad",), "left": ("left_pad",)}

    @staticmethod
    def gripper_position(backend):
        return lambda raw_env, robot, arm: backend.grippers[arm].copy()

    def test_probe_accepts_one_scene_relative_physical_waypoint(self):
        backend = self.FakeBackend()

        result = gate_module._posture_locked_carry_probe(
            backend,
            "box",
            distance_m=0.10,
            world_direction_x=None,
            world_direction_y=None,
            table_object_z=1.0,
            max_linear_m_s=0.04,
            _transport_module=self.fake_transport_module(),
            _gripper_position=self.gripper_position(backend),
            _contact_reader=self.contacts,
        )

        self.assertEqual(len(backend.follow_calls), 1)
        self.assertEqual(backend.follow_speeds, [0.04])
        self.assertAlmostEqual(backend._max_linear, 0.70)
        np.testing.assert_allclose(backend.follow_calls[0][0][0], [-0.9, 0.0])
        self.assertAlmostEqual(result["projected_object_progress_m"], 0.09)
        self.assertAlmostEqual(result["lateral_object_drift_m"], 0.0)
        self.assertAlmostEqual(result["object_gripper_drift_m"], 0.0)
        self.assertAlmostEqual(result["final_object_lift_m"], 0.20)
        self.assertTrue(result["terminal_bilateral_contact"])
        self.assertTrue(result["posture_carry_success"])
        self.assertTrue(gate_module.posture_carry_accepted(result))

    def test_probe_rejects_legacy_held_crate_teleport_state(self):
        backend = self.FakeBackend(legacy_held=True)

        result = gate_module._posture_locked_carry_probe(
            backend,
            "box",
            distance_m=0.10,
            world_direction_x=1.0,
            world_direction_y=0.0,
            table_object_z=1.0,
            max_linear_m_s=0.04,
            _transport_module=self.fake_transport_module(),
            _gripper_position=self.gripper_position(backend),
            _contact_reader=self.contacts,
        )

        self.assertEqual(backend.follow_calls, [])
        self.assertFalse(result["posture_carry_success"])
        self.assertGreater(result["legacy_teleport_activations"], 0)
        self.assertIn(
            "legacy_teleport_activations",
            gate_module.posture_carry_failures(result),
        )

    def test_probe_can_use_actuated_gripper_hold_without_follow_path(self):
        backend = self.FakeBackend()
        calls = []

        def config_factory(**kwargs):
            return SimpleNamespace(**kwargs)

        def actuated_transport(
            selected_backend,
            *,
            path,
            object_name,
            hold_yaw,
            minimum_object_z,
            config,
        ):
            calls.append(
                {
                    "backend": selected_backend,
                    "path": path,
                    "object_name": object_name,
                    "hold_yaw": hold_yaw,
                    "minimum_object_z": minimum_object_z,
                    "config": config,
                }
            )
            selected_backend.base_xy = np.asarray(path[0], dtype=float).copy()
            displacement = np.array([0.09, 0.0, 0.0], dtype=float)
            selected_backend.env.sim.data.body_xpos[0] += displacement
            for arm in selected_backend.grippers:
                selected_backend.grippers[arm] += displacement
            selected_backend._record_trajectory_frame(_env=selected_backend.env)
            return {"success": True, "failure_stage": None, "steps": 55}

        result = gate_module._posture_locked_carry_probe(
            backend,
            "box",
            distance_m=0.10,
            world_direction_x=None,
            world_direction_y=None,
            table_object_z=1.0,
            max_linear_m_s=0.04,
            actuated_gripper_hold=True,
            _transport_module=self.fake_transport_module(),
            _gripper_position=self.gripper_position(backend),
            _contact_reader=self.contacts,
            _actuated_transport=actuated_transport,
            _physical_carry_config_factory=config_factory,
        )

        self.assertEqual(backend.follow_calls, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["object_name"], "box")
        self.assertAlmostEqual(calls[0]["minimum_object_z"], 1.10)
        self.assertAlmostEqual(calls[0]["config"].max_linear, 0.04)
        self.assertAlmostEqual(calls[0]["config"].max_linear_delta, 0.005)
        self.assertEqual(result["control_mode"], "actuated_gripper_hold")
        self.assertTrue(result["posture_carry_success"])

    def test_actuated_driver_restores_only_non_gripper_posture(self):
        class Delegate:
            @staticmethod
            def capture_hold_targets(backend):
                del backend
                return {"torso": np.array([0.2])}

            @staticmethod
            def step(backend, **kwargs):
                del kwargs
                backend.env.sim.data.qpos[:] += 10.0
                backend.env.sim.data.qvel[:] += 20.0
                return {"collision": False}

        forward_calls = []
        model = SimpleNamespace(
            get_joint_qpos_addr={
                "arm": 0,
                "torso": 1,
                "head": 2,
                "gripper": 3,
            }.__getitem__,
            get_joint_qvel_addr={
                "arm": 0,
                "torso": 1,
                "head": 2,
                "gripper": 3,
            }.__getitem__,
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
            _record_trajectory_frame=lambda **kwargs: None,
        )
        driver = gate_module._PostureLockedActuatedCarryDriver(Delegate())

        driver.capture_hold_targets(backend)
        result = driver.step(backend)

        self.assertFalse(result["collision"])
        np.testing.assert_allclose(backend.env.sim.data.qpos, [1, 2, 3, 14])
        np.testing.assert_allclose(backend.env.sim.data.qvel, [5, 6, 7, 28])
        self.assertEqual(forward_calls, [True])

    def test_probe_passes_posture_locked_driver_to_actuated_transport(self):
        backend = self.FakeBackend()
        sentinel_driver = object()
        received = []

        def actuated_transport(selected_backend, **kwargs):
            received.append(kwargs["driver"])
            selected_backend.base_xy = np.asarray(
                kwargs["path"][0], dtype=float
            ).copy()
            displacement = np.array([0.09, 0.0, 0.0], dtype=float)
            selected_backend.env.sim.data.body_xpos[0] += displacement
            for arm in selected_backend.grippers:
                selected_backend.grippers[arm] += displacement
            return {"success": True}

        result = gate_module._posture_locked_carry_probe(
            backend,
            "box",
            distance_m=0.10,
            world_direction_x=None,
            world_direction_y=None,
            table_object_z=1.0,
            max_linear_m_s=0.04,
            actuated_gripper_hold=True,
            posture_lock_robot_joints=True,
            _transport_module=self.fake_transport_module(),
            _gripper_position=self.gripper_position(backend),
            _contact_reader=self.contacts,
            _actuated_transport=actuated_transport,
            _physical_carry_config_factory=lambda **kwargs: SimpleNamespace(
                **kwargs
            ),
            _actuated_driver=sentinel_driver,
        )

        self.assertEqual(received, [sentinel_driver])
        self.assertEqual(result["control_mode"], "actuated_posture_lock")
        self.assertTrue(result["posture_carry_success"])

    def test_runner_selects_probe_after_physical_grasp_and_uses_its_gate(self):
        source = inspect.getsource(gate_module.run_probe)

        posture_index = source.index(
            "if args.posture_locked_carry_distance_m > 0.0"
        )
        inchworm_index = source.index(
            "elif args.end_grasp_inchworm_distance_m > 0.0"
        )
        push_index = source.index("elif args.physical_push")
        self.assertLess(posture_index, inchworm_index)
        self.assertLess(inchworm_index, push_index)
        self.assertIn('record["mode"] = "posture_locked_physical_carry"', source)
        self.assertIn("_posture_locked_carry_probe(", source)
        self.assertIn('record["mode"] = "end_grasp_inchworm_transport"', source)
        self.assertIn("_end_grasp_inchworm_probe(", source)
        self.assertIn("if args.end_grasp_setdown_after_inchworm", source)
        self.assertIn('record["mode"] = "end_grasp_setdown_probe"', source)
        self.assertIn("_end_grasp_setdown_probe(", source)
        self.assertIn("_end_grasp_regrasp_probe(", source)
        self.assertIn(
            'record["gate_failures"] = posture_carry_failures(record)',
            source,
        )


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

    def test_open_fork_vertical_clearance_does_not_require_planar_overlap(self):
        bottom = self.box_geometry(
            "container_col_bottom",
            [7.059, 4.619, 1.009],
            [0.300, 0.200, 0.009],
            is_object=True,
        )
        outside_below = self.box_geometry(
            "gripper0_right_left_fingertip_collision",
            [7.345, 4.900, 0.995],
            [0.015, 0.035, 0.004],
        )
        outside_high = self.box_geometry(
            "gripper0_right_left_fingertip_collision",
            [7.345, 4.900, 1.005],
            [0.015, 0.035, 0.004],
        )

        self.assertTrue(
            gate_module.open_fork_below_bottom_ready(
                {"geometries": [bottom, outside_below]}
            )
        )
        self.assertFalse(
            gate_module.open_fork_below_bottom_ready(
                {"geometries": [bottom, outside_high]}
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

    def test_post_inset_uses_official_posture_locked_navigation(self):
        source = inspect.getsource(gate_module._table_edge_undercut_probe)
        insertion = source[source.index("def execute_post_inset_base_advance") :]
        insertion = insertion[: insertion.index("def execute_stage")]

        self.assertIn("segment_target_xy = segment_start_xy +", insertion)
        self.assertIn("backend.follow_path(", insertion)
        self.assertIn("int(np.ceil(requested_distance / 0.001)) + 5", insertion)
        self.assertIn("translation_reached", insertion)
        self.assertIn("requested_distance - 1e-4", insertion)
        self.assertNotIn("driver.step(", insertion)

    def test_torso_raise_uses_official_posture_lock_without_osc_compensation(self):
        source = inspect.getsource(gate_module._table_edge_undercut_probe)
        lift = source[source.index("def execute_torso_raise_into_support") :]
        lift = lift[: lift.index("def execute_post_inset_base_advance")]

        self.assertIn("_capture_upper_body_posture", lift)
        self.assertIn("_restore_upper_body_posture", lift)
        self.assertIn("env.step(idle_action)", lift)
        self.assertNotIn('arm_actions={"right": arm_action}', lift)

    def test_descent_waits_for_measured_fingertip_bottom_clearance(self):
        source = inspect.getsource(gate_module._table_edge_undercut_probe)

        self.assertIn('stage == "descend_open_outside"', source)
        self.assertIn("open_fork_below_bottom_ready(", source)
        self.assertIn("descent_max_steps = 480", source)
        self.assertIn("max_steps=descent_max_steps", source)

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
        self.assertIn('"raise_open_with_torso"', source)
        self.assertIn("start_eef = right_eef_position().copy()", source)
        self.assertIn("_capture_upper_body_posture", source)
        self.assertIn("_restore_upper_body_posture", source)
        self.assertIn("commanded_torso", source)
        self.assertIn("raw_env.step(idle_action)", source)
        self.assertIn('"orient_open_fork_at_clearance"', source)
        self.assertIn("orient_before_descent", source)
        self.assertIn('"safe_unrotated_inset_plateau"', source)
        self.assertIn('"safe_unrotated_descent_plateau"', source)
        self.assertIn('"safe_unrotated_base_assisted_inset"', source)


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
    def test_task_index_defaults_to_l1_and_can_select_another_level(self):
        common = [
            "--candidate-root",
            "/tmp/candidate",
            "--expected-official-commit",
            "official-commit",
            "--output",
            "/tmp/result.json",
            "--trajectory",
            "/tmp/trajectory.json",
        ]

        self.assertEqual(parse_args(common).task_index, 0)
        self.assertEqual(
            parse_args([*common, "--task-index", "3"]).task_index,
            3,
        )

    def test_task_for_index_returns_a_copy_and_rejects_invalid_indices(self):
        tasks = [{"level": "L1"}, {"level": "L2"}]

        selected = task_for_index(tasks, 1)

        self.assertEqual(selected, {"level": "L2"})
        self.assertIsNot(selected, tasks[1])
        for invalid in (-1, 2, True, 0.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError, IndexError)):
                    task_for_index(tasks, invalid)

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
        self.assertAlmostEqual(args.posture_locked_carry_distance_m, 0.0)
        self.assertAlmostEqual(args.posture_locked_carry_max_linear_m_s, 0.04)
        self.assertFalse(args.posture_locked_carry_actuated_gripper_hold)
        self.assertFalse(args.posture_locked_carry_posture_lock_robot_joints)
        self.assertIsNone(args.posture_locked_carry_world_direction_x)
        self.assertIsNone(args.posture_locked_carry_world_direction_y)
        self.assertAlmostEqual(args.end_grasp_inchworm_distance_m, 0.0)
        self.assertAlmostEqual(args.end_grasp_inchworm_stroke_m, 0.08)
        self.assertAlmostEqual(args.end_grasp_inchworm_stroke_lift_m, 0.015)
        self.assertAlmostEqual(args.end_grasp_inchworm_height_gain, 0.75)
        self.assertAlmostEqual(args.end_grasp_inchworm_reset_m, 0.06)
        self.assertIsNone(args.end_grasp_inchworm_world_direction_x)
        self.assertIsNone(args.end_grasp_inchworm_world_direction_y)
        self.assertIsNone(args.container_grasp_lift_height_m)
        self.assertAlmostEqual(args.end_grasp_minimum_lift_m, 0.10)
        self.assertFalse(args.end_grasp_setdown_after_inchworm)
        self.assertAlmostEqual(args.end_grasp_place_max_descent_m, 0.25)
        self.assertEqual(args.end_grasp_regrasp_macros, 1)
        self.assertAlmostEqual(args.floor_regrasp_retract_forward_m, 0.20)
        self.assertAlmostEqual(args.floor_regrasp_retract_lateral_m, 0.15)
        self.assertAlmostEqual(args.floor_regrasp_retract_target_z, 1.45)
        self.assertAlmostEqual(args.floor_transition_margin_m, 0.30)
        self.assertAlmostEqual(args.floor_regrasp_safe_clearance_m, 1.20)
        self.assertFalse(args.floor_corridor_push)
        self.assertIsNone(args.floor_push_world_direction_x)
        self.assertIsNone(args.floor_push_world_direction_y)
        self.assertAlmostEqual(args.floor_push_distance_m, 1.05)
        self.assertAlmostEqual(args.floor_push_base_standoff_m, 0.85)
        self.assertAlmostEqual(args.floor_push_orientation_clearance_m, 0.35)
        self.assertAlmostEqual(args.floor_push_oriented_retract_forward_m, 0.20)
        self.assertAlmostEqual(args.floor_push_oriented_retract_lateral_m, 0.08)
        self.assertIsNone(args.floor_push_lateral_offset_m)
        self.assertAlmostEqual(args.floor_push_torso_drop_m, 0.24)
        self.assertFalse(args.floor_push_base_pusher)
        self.assertAlmostEqual(args.floor_push_maximum_lateral_offset_m, 0.25)
        self.assertAlmostEqual(args.floor_push_face_offset_m, 0.24)
        self.assertAlmostEqual(args.floor_push_hand_separation_m, 0.28)
        self.assertAlmostEqual(args.floor_push_hand_height_m, 0.38)
        self.assertAlmostEqual(args.floor_push_precontact_clearance_m, 0.08)
        self.assertAlmostEqual(args.floor_push_base_speed_m_s, 0.025)
        self.assertEqual(args.floor_push_max_steps, 1200)
        self.assertFalse(args.floor_base_route_to_target)
        self.assertAlmostEqual(args.floor_base_route_corridor_y, -8.40)
        self.assertAlmostEqual(args.floor_base_route_arrival_margin_m, 0.05)
        self.assertAlmostEqual(args.floor_base_route_reposition_clearance_m, 0.90)
        self.assertAlmostEqual(args.floor_base_route_minimum_retract_z_m, 0.80)
        self.assertAlmostEqual(args.floor_base_tracking_gain, 0.50)
        self.assertAlmostEqual(args.floor_base_alignment_gain, 0.50)
        self.assertAlmostEqual(args.floor_base_tracking_deadband_m, 0.05)
        self.assertAlmostEqual(args.floor_base_maximum_contact_offset_m, 0.08)
        self.assertAlmostEqual(args.floor_base_max_lateral_speed_m_s, 0.02)
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
        self.assertAlmostEqual(args.undercut_torso_raise_m, 0.0)
        self.assertAlmostEqual(
            args.undercut_torso_raise_orientation_max_action,
            0.30,
        )
        self.assertAlmostEqual(
            args.undercut_torso_raise_base_correction_max_m,
            0.04,
        )
        self.assertFalse(args.undercut_orient_before_descent)
        self.assertIsNone(args.undercut_post_inset_world_direction_x)
        self.assertIsNone(args.undercut_post_inset_world_direction_y)

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
                "--posture-locked-carry-distance-m",
                "0.10",
                "--posture-locked-carry-max-linear-m-s",
                "0.02",
                "--posture-locked-carry-actuated-gripper-hold",
                "--posture-locked-carry-posture-lock-robot-joints",
                "--posture-locked-carry-world-direction-x",
                "-1.0",
                "--posture-locked-carry-world-direction-y",
                "0.25",
                "--end-grasp-inchworm-distance-m",
                "0.07",
                "--end-grasp-inchworm-stroke-m",
                "0.08",
                "--end-grasp-inchworm-stroke-lift-m",
                "0.005",
                "--end-grasp-inchworm-height-gain",
                "0.0",
                "--end-grasp-inchworm-reset-m",
                "0.05",
                "--end-grasp-inchworm-world-direction-x",
                "1.0",
                "--end-grasp-inchworm-world-direction-y",
                "0.0",
                "--container-grasp-lift-height-m",
                "0.04",
                "--end-grasp-minimum-lift-m",
                "0.015",
                "--end-grasp-setdown-after-inchworm",
                "--end-grasp-place-max-descent-m",
                "0.22",
                "--end-grasp-regrasp-macros",
                "2",
                "--floor-regrasp-retract-forward-m",
                "0.18",
                "--floor-regrasp-retract-lateral-m",
                "0.14",
                "--floor-regrasp-retract-target-z",
                "1.40",
                "--floor-transition-margin-m",
                "0.25",
                "--floor-regrasp-safe-clearance-m",
                "1.10",
                "--floor-corridor-push",
                "--floor-push-world-direction-x",
                "0.0",
                "--floor-push-world-direction-y",
                "-1.0",
                "--floor-push-distance-m",
                "0.95",
                "--floor-push-base-standoff-m",
                "0.90",
                "--floor-push-orientation-clearance-m",
                "0.40",
                "--floor-push-oriented-retract-forward-m",
                "0.18",
                "--floor-push-oriented-retract-lateral-m",
                "0.07",
                "--floor-push-lateral-offset-m",
                "-0.15",
                "--floor-push-torso-drop-m",
                "0.22",
                "--floor-push-base-pusher",
                "--floor-push-maximum-lateral-offset-m",
                "0.22",
                "--floor-push-face-offset-m",
                "0.21",
                "--floor-push-hand-separation-m",
                "0.26",
                "--floor-push-hand-height-m",
                "0.40",
                "--floor-push-precontact-clearance-m",
                "0.07",
                "--floor-push-base-speed-m-s",
                "0.02",
                "--floor-push-max-steps",
                "900",
                "--floor-base-route-to-target",
                "--floor-base-route-corridor-y",
                "-8.55",
                "--floor-base-route-arrival-margin-m",
                "0.06",
                "--floor-base-route-reposition-clearance-m",
                "0.95",
                "--floor-base-route-minimum-retract-z-m",
                "0.85",
                "--floor-base-tracking-gain",
                "0.60",
                "--floor-base-alignment-gain",
                "0.70",
                "--floor-base-tracking-deadband-m",
                "0.07",
                "--floor-base-maximum-contact-offset-m",
                "0.09",
                "--floor-base-max-lateral-speed-m-s",
                "0.03",
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
                "--undercut-torso-raise-m",
                "0.12",
                "--undercut-torso-raise-orientation-max-action",
                "0.25",
                "--undercut-torso-raise-base-correction-max-m",
                "0.06",
                "--undercut-orient-before-descent",
                "--undercut-post-inset-world-direction-x",
                "-1.0",
                "--undercut-post-inset-world-direction-y",
                "-0.25",
            ]
        )

        self.assertAlmostEqual(args.center_carry_max_linear, 0.005)
        self.assertAlmostEqual(args.posture_locked_carry_distance_m, 0.10)
        self.assertAlmostEqual(args.posture_locked_carry_max_linear_m_s, 0.02)
        self.assertTrue(args.posture_locked_carry_actuated_gripper_hold)
        self.assertTrue(args.posture_locked_carry_posture_lock_robot_joints)
        self.assertAlmostEqual(
            args.posture_locked_carry_world_direction_x,
            -1.0,
        )
        self.assertAlmostEqual(
            args.posture_locked_carry_world_direction_y,
            0.25,
        )
        self.assertAlmostEqual(args.end_grasp_inchworm_distance_m, 0.07)
        self.assertAlmostEqual(args.end_grasp_inchworm_stroke_m, 0.08)
        self.assertAlmostEqual(args.end_grasp_inchworm_stroke_lift_m, 0.005)
        self.assertAlmostEqual(args.end_grasp_inchworm_height_gain, 0.0)
        self.assertAlmostEqual(args.end_grasp_inchworm_reset_m, 0.05)
        self.assertAlmostEqual(args.end_grasp_inchworm_world_direction_x, 1.0)
        self.assertAlmostEqual(args.end_grasp_inchworm_world_direction_y, 0.0)
        self.assertAlmostEqual(args.container_grasp_lift_height_m, 0.04)
        self.assertAlmostEqual(args.end_grasp_minimum_lift_m, 0.015)
        self.assertTrue(args.end_grasp_setdown_after_inchworm)
        self.assertAlmostEqual(args.end_grasp_place_max_descent_m, 0.22)
        self.assertEqual(args.end_grasp_regrasp_macros, 2)
        self.assertAlmostEqual(args.floor_regrasp_retract_forward_m, 0.18)
        self.assertAlmostEqual(args.floor_regrasp_retract_lateral_m, 0.14)
        self.assertAlmostEqual(args.floor_regrasp_retract_target_z, 1.40)
        self.assertAlmostEqual(args.floor_transition_margin_m, 0.25)
        self.assertAlmostEqual(args.floor_regrasp_safe_clearance_m, 1.10)
        self.assertTrue(args.floor_corridor_push)
        self.assertAlmostEqual(args.floor_push_world_direction_x, 0.0)
        self.assertAlmostEqual(args.floor_push_world_direction_y, -1.0)
        self.assertAlmostEqual(args.floor_push_distance_m, 0.95)
        self.assertAlmostEqual(args.floor_push_base_standoff_m, 0.90)
        self.assertAlmostEqual(args.floor_push_orientation_clearance_m, 0.40)
        self.assertAlmostEqual(args.floor_push_oriented_retract_forward_m, 0.18)
        self.assertAlmostEqual(args.floor_push_oriented_retract_lateral_m, 0.07)
        self.assertAlmostEqual(args.floor_push_lateral_offset_m, -0.15)
        self.assertAlmostEqual(args.floor_push_torso_drop_m, 0.22)
        self.assertTrue(args.floor_push_base_pusher)
        self.assertAlmostEqual(args.floor_push_maximum_lateral_offset_m, 0.22)
        self.assertAlmostEqual(args.floor_push_face_offset_m, 0.21)
        self.assertAlmostEqual(args.floor_push_hand_separation_m, 0.26)
        self.assertAlmostEqual(args.floor_push_hand_height_m, 0.40)
        self.assertAlmostEqual(args.floor_push_precontact_clearance_m, 0.07)
        self.assertAlmostEqual(args.floor_push_base_speed_m_s, 0.02)
        self.assertEqual(args.floor_push_max_steps, 900)
        self.assertTrue(args.floor_base_route_to_target)
        self.assertAlmostEqual(args.floor_base_route_corridor_y, -8.55)
        self.assertAlmostEqual(args.floor_base_route_arrival_margin_m, 0.06)
        self.assertAlmostEqual(args.floor_base_route_reposition_clearance_m, 0.95)
        self.assertAlmostEqual(args.floor_base_route_minimum_retract_z_m, 0.85)
        self.assertAlmostEqual(args.floor_base_tracking_gain, 0.60)
        self.assertAlmostEqual(args.floor_base_alignment_gain, 0.70)
        self.assertAlmostEqual(args.floor_base_tracking_deadband_m, 0.07)
        self.assertAlmostEqual(args.floor_base_maximum_contact_offset_m, 0.09)
        self.assertAlmostEqual(args.floor_base_max_lateral_speed_m_s, 0.03)
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
        self.assertAlmostEqual(args.undercut_torso_raise_m, 0.12)
        self.assertAlmostEqual(
            args.undercut_torso_raise_orientation_max_action,
            0.25,
        )
        self.assertAlmostEqual(
            args.undercut_torso_raise_base_correction_max_m,
            0.06,
        )
        self.assertTrue(args.undercut_orient_before_descent)
        self.assertAlmostEqual(
            args.undercut_post_inset_world_direction_x,
            -1.0,
        )
        self.assertAlmostEqual(
            args.undercut_post_inset_world_direction_y,
            -0.25,
        )


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
