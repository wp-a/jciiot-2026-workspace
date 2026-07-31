import importlib.util
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
    / "competition_grasp.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("competition_grasp", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingBackend:
    def __init__(self):
        self.events = []

    def _mark_trajectory_event(self, name, **payload):
        self.events.append({"name": name, **payload})


class ScriptedDriver:
    def __init__(
        self,
        *,
        contacts=None,
        polished_contacts=None,
        lift_success=True,
        clearance_success=True,
    ):
        self.contacts = contacts or {"right": True, "left": True}
        self.polished_contacts = polished_contacts or self.contacts
        self.lift_success = lift_success
        self.clearance_success = clearance_success
        self.calls = []

    def raise_to_clearance(self, backend, object_name, config):
        self.calls.append("raise_clearance")
        return self.clearance_success

    def move_above_grasp_sites(self, backend, object_name, config):
        self.calls.append("move_above")
        return True

    def move_to_pregrasp(self, backend, object_name, config):
        self.calls.append("pregrasp")
        return True

    def approach_grasp_sites(self, backend, object_name, config):
        self.calls.append("approach")
        return True

    def adjust_wrist_for_reach(self, backend, object_name, config):
        self.calls.append("wrist_adjust")
        return True

    def close_and_check_contacts(self, backend, object_name, config):
        self.calls.append("close")
        return self.contacts

    def polish_contacts(self, backend, object_name, config, contacts):
        self.calls.append("polish")
        return self.polished_contacts

    def lift_and_verify(self, backend, object_name, config):
        self.calls.append("lift")
        return self.lift_success

    def physical_hold_metadata(self, backend, object_name):
        self.calls.append("hold_metadata")
        return {
            "base_yaw": 0.25,
            "object_pos": [1.0, 2.0, 1.15],
            "object_z": 1.15,
        }


class CompetitionGraspTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_position_delta_is_scaled_and_clipped(self):
        result = self.module.normalized_position_action(
            delta=np.array([0.20, -0.40, 0.025]),
            position_scale=np.array([0.10, 0.20, 0.10]),
            max_action=0.75,
        )

        np.testing.assert_allclose(result, [0.75, -0.75, 0.25, 0.0, 0.0, 0.0])

    def test_torso_target_drops_by_bounded_reach_offset(self):
        target = self.module.lowered_torso_target(
            np.array([0.35]),
            drop=0.04,
            minimum=0.10,
        )

        np.testing.assert_allclose(target, [0.31])

    def test_contact_micro_adjustments_are_incremental_and_bounded(self):
        targets = self.module.contact_micro_adjustment_targets(
            0.115,
            step=0.004,
            max_drop=0.020,
            minimum=0.100,
        )

        np.testing.assert_allclose(targets, [0.111, 0.107, 0.103, 0.100])

    def test_lift_defaults_use_official_short_verification_motion(self):
        config = self.module.ScriptedGraspConfig()

        self.assertEqual(config.lift_height, 0.04)
        self.assertEqual(config.lift_hold_steps, 0)
        self.assertIsNone(config.container_lift_height_override)

    def test_independent_gripper_action_keeps_stationary_arm_closed(self):
        robot = SimpleNamespace(
            gripper={
                "right": SimpleNamespace(dof=1),
                "left": SimpleNamespace(dof=1),
            },
            composite_controller=SimpleNamespace(
                _action_split_indexes={
                    "right": (0, 6),
                    "right_gripper": (6, 7),
                    "left": (7, 13),
                    "left_gripper": (13, 14),
                }
            ),
        )
        calls = []

        def build_action(robot_arg, arm_actions, gripper_value, hold_targets):
            calls.append(
                {
                    "robot": robot_arg,
                    "arm_actions": arm_actions,
                    "gripper_value": gripper_value,
                    "hold_targets": hold_targets,
                }
            )
            return np.zeros(14, dtype=float)

        arm_actions = {
            "right": np.ones(6, dtype=float),
            "left": np.zeros(6, dtype=float),
        }
        action = self.module.build_independent_gripper_action(
            robot,
            arm_actions=arm_actions,
            gripper_values={"right": 1.0, "left": -1.0},
            hold_targets={"torso": np.array([0.3])},
            build_action_fn=build_action,
        )

        self.assertEqual(action[6], 1.0)
        self.assertEqual(action[13], -1.0)
        self.assertEqual(calls[0]["gripper_value"], 0.0)
        self.assertIs(calls[0]["arm_actions"], arm_actions)

    def test_independent_gripper_action_requires_finite_commands_for_both_arms(self):
        robot = SimpleNamespace(
            gripper={
                "right": SimpleNamespace(dof=1),
                "left": SimpleNamespace(dof=1),
            },
            composite_controller=SimpleNamespace(
                _action_split_indexes={
                    "right_gripper": (0, 1),
                    "left_gripper": (1, 2),
                }
            ),
        )

        def build_action(*_args, **_kwargs):
            return np.zeros(2, dtype=float)

        with self.assertRaisesRegex(ValueError, "both arms"):
            self.module.build_independent_gripper_action(
                robot,
                arm_actions={},
                gripper_values={"right": 1.0},
                hold_targets={},
                build_action_fn=build_action,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            self.module.build_independent_gripper_action(
                robot,
                arm_actions={},
                gripper_values={"right": 1.0, "left": np.nan},
                hold_targets={},
                build_action_fn=build_action,
            )

    def test_mirrored_fingerpad_targets_reflect_and_lower_right_template(self):
        right_fingerpads = np.array(
            [
                [12.0306, 2.9367, 1.3996],
                [12.0535, 3.0389, 1.4735],
            ]
        )

        targets = self.module.world_x_mirrored_fingerpad_targets(
            right_fingerpads,
            object_x=11.8675,
            height_offset=0.13,
        )

        np.testing.assert_allclose(
            targets,
            [
                [11.7044, 2.9367, 1.2696],
                [11.6815, 3.0389, 1.3435],
            ],
        )

    def test_mirrored_fingerpad_targets_follow_rotated_arm_axis(self):
        right_fingerpads = np.array(
            [
                [0.6792, 8.5927, 1.5148],
                [0.6290, 8.6871, 1.5856],
            ]
        )

        targets = self.module.mirrored_fingerpad_targets(
            right_fingerpads,
            object_xy=np.array([0.4418, 8.4731]),
            mirror_normal_xy=np.array([0.0, 1.0]),
            height_offset=0.13,
        )

        np.testing.assert_allclose(
            targets,
            [
                [0.6792, 8.3535, 1.3848],
                [0.6290, 8.2591, 1.4556],
            ],
        )

    def test_transport_stow_path_excludes_start_and_reaches_target(self):
        path = self.module.joint_interpolation_path(
            np.array([0.8, -0.2, 2.3]),
            np.array([0.7, -0.1, 1.6]),
            steps=4,
        )

        self.assertEqual(path.shape, (4, 3))
        self.assertFalse(np.allclose(path[0], [0.8, -0.2, 2.3]))
        np.testing.assert_allclose(path[-1], [0.7, -0.1, 1.6])

    def test_contact_confirmation_requires_depth_beyond_first_touch(self):
        self.assertFalse(
            self.module.contact_margin_reached(
                first_contact=0.3095,
                current=0.3050,
                required_drop=0.007,
            )
        )
        self.assertTrue(
            self.module.contact_margin_reached(
                first_contact=0.3095,
                current=0.3025,
                required_drop=0.007,
            )
        )

    def test_follower_lift_offset_tracks_object_with_bounded_lead(self):
        self.assertAlmostEqual(
            self.module.follower_lift_offset(
                object_lift=0.0,
                lead=0.003,
                lift_height=0.05,
            ),
            0.003,
        )
        self.assertAlmostEqual(
            self.module.follower_lift_offset(
                object_lift=0.018,
                lead=0.003,
                lift_height=0.05,
            ),
            0.021,
        )
        self.assertAlmostEqual(
            self.module.follower_lift_offset(
                object_lift=0.060,
                lead=0.003,
                lift_height=0.05,
            ),
            0.05,
        )

    def test_lift_goal_counts_motion_that_happened_during_close(self):
        self.assertTrue(
            self.module.lift_goal_reached(
                reference_z=1.200,
                current_z=1.231,
                lift_height=0.040,
                tolerance=0.010,
            )
        )
        self.assertFalse(
            self.module.lift_goal_reached(
                reference_z=1.200,
                current_z=1.229,
                lift_height=0.040,
                tolerance=0.010,
            )
        )

    def test_controller_goals_are_refreshed_after_kinematic_adjustment(self):
        calls = []

        class Controller:
            def __init__(self, name):
                self.name = name

            def update(self, *, force):
                calls.append((self.name, "update", force))

            def reset_goal(self):
                calls.append((self.name, "reset"))

        class Composite:
            def __init__(self):
                self.part_controllers = {
                    name: Controller(name)
                    for name in ("right", "left", "torso")
                }

            def update_state(self):
                calls.append(("composite", "update_state"))

        robot = type("Robot", (), {"composite_controller": Composite()})()

        self.module.synchronize_controller_goals(robot)

        self.assertEqual(calls[0], ("composite", "update_state"))
        for part in ("right", "left", "torso"):
            self.assertIn((part, "update", True), calls)
            self.assertIn((part, "reset"), calls)

    def test_contact_stability_resets_when_either_arm_loses_contact(self):
        stable_steps = 0
        stable_steps = self.module.next_contact_stability(
            {"right": True, "left": True},
            stable_steps,
        )
        stable_steps = self.module.next_contact_stability(
            {"right": True, "left": True},
            stable_steps,
        )
        self.assertEqual(stable_steps, 2)

        stable_steps = self.module.next_contact_stability(
            {"right": True, "left": False},
            stable_steps,
        )
        self.assertEqual(stable_steps, 0)

    def test_wrist_adjustment_only_triggers_for_large_left_height_residual(self):
        self.assertTrue(
            self.module.wrist_adjustment_required(
                current_z=1.443,
                target_z=1.385,
                threshold=0.04,
            )
        )
        self.assertFalse(
            self.module.wrist_adjustment_required(
                current_z=1.402,
                target_z=1.385,
                threshold=0.04,
            )
        )

    def test_mirrored_open_grasp_is_limited_to_tote_geometry(self):
        self.assertTrue(
            self.module.uses_mirrored_open_grasp("green_tote_b01_lower")
        )
        self.assertFalse(
            self.module.uses_mirrored_open_grasp(
                "line_5_container_h01_near"
            )
        )
        self.assertFalse(
            self.module.uses_axis_aware_fingerpad_mirror(
                "green_tote_b01_lower"
            )
        )

    def test_ik_seed_is_clipped_inside_joint_bounds(self):
        seed = self.module.interior_joint_seed(
            np.array([-1.2, 0.5, 1.3]),
            lower=np.array([-1.0, 0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
        )

        self.assertGreater(seed[0], -1.0)
        self.assertAlmostEqual(seed[1], 0.5)
        self.assertLess(seed[2], 1.0)
        self.assertTrue(
            self.module.uses_axis_aware_fingerpad_mirror(
                "blue_tote_b01_near_right"
            )
        )
        self.assertTrue(
            self.module.uses_axis_aware_fingerpad_mirror(
                "white_tote_b01_left_front"
            )
        )

    def test_wall_side_tote_grasp_is_limited_to_white_left_objects(self):
        self.assertTrue(
            self.module.uses_station_side_tote_grasp(
                "white_tote_b01_left_front"
            )
        )
        self.assertFalse(
            self.module.uses_station_side_tote_grasp(
                "green_tote_b01_lower"
            )
        )
        self.assertFalse(
            self.module.should_swap_arm_targets(
                "white_tote_b01_left_front",
                requested=True,
            )
        )
        self.assertTrue(
            self.module.should_swap_arm_targets(
                "green_tote_b01_lower",
                requested=True,
            )
        )

    def test_rotated_blue_tote_keeps_near_left_far_right_assignment(self):
        self.assertFalse(
            self.module.should_swap_arm_targets(
                "blue_tote_b01_far_right",
                requested=True,
            )
        )

    def test_wall_side_tote_targets_reflect_near_site_across_heading(self):
        targets = self.module.station_side_tote_grasp_targets(
            {
                "right": np.array([-14.509088, 4.199868, 1.473978]),
                "left": np.array([-14.839088, 4.199868, 1.473978]),
            },
            object_xy=np.array([-14.674088, 4.414868]),
            base_xy=np.array([-13.65, 4.20]),
        )

        np.testing.assert_allclose(
            targets["right"],
            [-14.509088, 4.629868, 1.473978],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            targets["left"],
            [-14.509088, 4.199868, 1.473978],
            atol=1e-4,
        )

    def test_white_tote_profile_uses_measured_mirrored_ik_height(self):
        config = self.module.ScriptedGraspConfig()

        profiled = self.module.apply_object_grasp_profile(
            config,
            "white_tote_b01_left_front",
        )

        self.assertAlmostEqual(profiled.mirrored_ik_height_offset, 0.06)
        self.assertAlmostEqual(profiled.station_side_reach_offset, 0.04)
        self.assertEqual(profiled.clearance_translate_steps, 360)

    def test_green_tote_profile_inserts_grippers_inside_front_rim(self):
        config = self.module.ScriptedGraspConfig()

        profiled = self.module.apply_object_grasp_profile(
            config,
            "green_tote_b01_lower",
        )

        self.assertAlmostEqual(profiled.approach_tolerance, 0.08)
        self.assertAlmostEqual(profiled.face_insertion, 0.03)
        self.assertAlmostEqual(profiled.close_follow_max_distance, 0.10)

        upper = self.module.apply_object_grasp_profile(
            self.module.ScriptedGraspConfig(),
            "green_tote_b01_upper",
        )
        self.assertFalse(upper.hold_close_pose)
        self.assertAlmostEqual(upper.site_below_offset, 0.03)

    def test_blue_tote_profile_tracks_object_motion_during_close(self):
        config = self.module.apply_object_grasp_profile(
            self.module.ScriptedGraspConfig(),
            "blue_tote_b01_far_right",
        )

        self.assertAlmostEqual(config.close_follow_max_distance, 0.10)
        self.assertEqual(config.close_follow_arms, ("left",))

    def test_l5_followup_totes_use_a_full_upper_body_clearance_seed(self):
        self.assertIsNone(
            self.module.station_side_clearance_joint_seed(
                "white_tote_b01_left_front"
            )
        )

        center = self.module.station_side_clearance_joint_seed(
            "white_tote_b01_left_center"
        )
        back = self.module.station_side_clearance_joint_seed(
            "white_tote_b01_left_back"
        )

        self.assertEqual(center.shape, (13,))
        np.testing.assert_allclose(center, back)
        center[0] = -1.0
        self.assertGreater(
            self.module.station_side_clearance_joint_seed(
                "white_tote_b01_left_center"
            )[0],
            0.0,
        )

    def test_grasp_skill_has_no_kinematic_transport_path(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("transport_attachment", source)
        self.assertNotIn("attach_for_transport", source)
        self.assertNotIn("_transport_stow", source)

    def test_hold_metadata_preserves_pre_lift_support_reference(self):
        driver = self.module.OfficialScriptedGraspDriver()
        driver._close_lift_reference = ("box", 1.00)
        driver._helpers = lambda: {
            "object_center": lambda _env, _name: np.array([4.0, 5.0, 1.35])
        }
        backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([3.0, 5.0]), 0.25),
            env=SimpleNamespace(
                obj_body_id={"box": 0},
                sim=SimpleNamespace(
                    data=SimpleNamespace(
                        body_xpos=np.array([[4.0, 5.0, 1.20]])
                    )
                ),
            ),
        )

        hold = driver.physical_hold_metadata(backend, "box")

        self.assertAlmostEqual(hold["support_reference_object_z"], 1.15)
        self.assertAlmostEqual(hold["minimum_transport_object_z"], 1.25)

    def test_container_profile_restores_scored_l1_grasp_parameters(self):
        config = self.module.ScriptedGraspConfig()

        profiled = self.module.apply_object_grasp_profile(
            config,
            "line_5_container_h01_near",
        )

        self.assertIs(profiled, config)
        self.assertEqual(config.site_below_offset, 0.035)
        self.assertEqual(config.approach_tolerance, 0.012)
        self.assertEqual(config.close_steps, 80)
        self.assertEqual(config.close_increment_interval, 1)
        self.assertEqual(config.contact_settle_steps, 81)
        self.assertFalse(config.hold_close_pose)
        self.assertEqual(config.face_insertion, 0.0)
        self.assertEqual(config.lift_height, 0.15)
        self.assertEqual(config.lift_hold_steps, 20)
        self.assertEqual(config.lift_tolerance, 0.02)

    def test_container_profile_respects_explicit_research_lift_override(self):
        config = self.module.ScriptedGraspConfig(
            container_lift_height_override=0.04
        )

        profiled = self.module.apply_object_grasp_profile(
            config,
            "line_5_container_h01_near",
        )

        self.assertIs(profiled, config)
        self.assertEqual(profiled.lift_height, 0.04)

    def test_container_profile_rejects_invalid_research_lift_override(self):
        for value in (0.0, -0.01, np.nan, np.inf):
            with self.subTest(value=value):
                config = self.module.ScriptedGraspConfig(
                    container_lift_height_override=value
                )
                with self.assertRaises(ValueError):
                    self.module.apply_object_grasp_profile(
                        config,
                        "line_5_container_h01_near",
                    )

    def test_close_pose_can_hold_post_adjustment_gripper_positions(self):
        current = {
            "right": np.array([12.03, 2.98, 1.38]),
            "left": np.array([11.73, 2.99, 1.40]),
        }
        requested = {
            "right": np.array([12.03, 2.98, 1.36]),
            "left": np.array([11.70, 2.98, 1.36]),
        }

        held = self.module.close_pose_targets(
            current,
            requested,
            hold_current=True,
        )

        np.testing.assert_allclose(held["right"], requested["right"])
        np.testing.assert_allclose(held["left"], current["left"])

    def test_grasp_targets_insert_from_face_toward_object_center(self):
        targets = {
            "right": np.array([12.03, 2.98, 1.38]),
            "left": np.array([11.70, 2.98, 1.38]),
        }

        inserted = self.module.inward_face_targets(
            targets,
            object_xy=np.array([11.865, 3.195]),
            insertion=0.03,
        )

        np.testing.assert_allclose(inserted["right"], [12.03, 3.01, 1.38])
        np.testing.assert_allclose(inserted["left"], [11.70, 3.01, 1.38])

    def test_face_insertion_is_applied_only_to_active_close_targets(self):
        current = {
            "right": np.array([12.05, 2.99, 1.44]),
            "left": np.array([11.69, 2.99, 1.32]),
        }
        requested = {
            "right": np.array([12.03, 2.98, 1.38]),
            "left": np.array([11.70, 2.98, 1.38]),
        }

        targets = self.module.close_grasp_targets(
            current,
            requested,
            object_xy=np.array([11.865, 3.195]),
            insertion=0.03,
            hold_current=True,
        )

        np.testing.assert_allclose(targets["right"], [12.03, 3.01, 1.38])
        np.testing.assert_allclose(targets["left"], current["left"])

    def test_close_follow_offset_is_bounded_without_changing_direction(self):
        offset = self.module.bounded_planar_follow_offset(
            np.array([0.03, 0.04]),
            max_distance=0.03,
        )

        np.testing.assert_allclose(offset, [0.018, 0.024])

    def test_close_follow_moves_only_an_arm_without_contact(self):
        offset = np.array([-0.07, 0.06])

        np.testing.assert_allclose(
            self.module.contact_aware_follow_offset(
                offset,
                has_contact=False,
            ),
            offset,
        )
        np.testing.assert_allclose(
            self.module.contact_aware_follow_offset(
                offset,
                has_contact=True,
            ),
            [0.0, 0.0],
        )

    def test_configured_close_follow_keeps_unlisted_arm_fixed(self):
        offset = np.array([-0.07, 0.06])

        np.testing.assert_allclose(
            self.module.configured_close_follow_offset(
                offset,
                arm="right",
                follow_arms=("left",),
                has_contact=False,
            ),
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            self.module.configured_close_follow_offset(
                offset,
                arm="left",
                follow_arms=("left",),
                has_contact=False,
            ),
            offset,
        )

    def test_gripper_close_uses_spaced_positive_pulses(self):
        commands = [
            self.module.gripper_close_command(step, interval=20)
            for step in range(41)
        ]

        self.assertEqual([i for i, value in enumerate(commands) if value > 0], [0, 20, 40])

    def test_stage_requires_every_arm_within_tolerance(self):
        targets = {
            "right": np.array([1.0, 2.0, 3.0]),
            "left": np.array([4.0, 5.0, 6.0]),
        }
        almost = {
            "right": np.array([1.0, 2.0, 3.005]),
            "left": np.array([4.0, 5.0, 6.03]),
        }
        reached = {
            "right": np.array([1.0, 2.0, 3.005]),
            "left": np.array([4.0, 5.0, 6.009]),
        }

        self.assertFalse(self.module.targets_reached(almost, targets, tolerance=0.01))
        self.assertTrue(self.module.targets_reached(reached, targets, tolerance=0.01))

    def test_approach_tolerance_hands_small_residual_to_close_stage(self):
        config = self.module.ScriptedGraspConfig()
        targets = {
            "right": np.array([12.03, 2.98, 1.365]),
            "left": np.array([11.70, 2.98, 1.365]),
        }
        current = {
            "right": np.array([12.032, 2.981, 1.365]),
            "left": np.array([11.696, 2.982, 1.415]),
        }

        self.assertFalse(
            self.module.targets_reached(
                current,
                targets,
                tolerance=config.position_tolerance,
            )
        )
        self.assertTrue(
            self.module.targets_reached(
                current,
                targets,
                tolerance=config.approach_tolerance,
            )
        )
        self.assertEqual(config.approach_tolerance, 0.08)

    def test_high_clearance_translation_uses_approach_tolerance(self):
        driver = self.module.OfficialScriptedGraspDriver()
        config = self.module.ScriptedGraspConfig(
            position_tolerance=0.012,
            approach_tolerance=0.025,
        )
        targets = {
            "right": np.array([12.03, 2.98, 1.68]),
            "left": np.array([11.70, 2.98, 1.68]),
        }
        captured = {}
        driver._seed_station_side_clearance = lambda *_args: True
        driver._grasp_targets = lambda *_args, **_kwargs: targets

        def move_to_targets(_backend, actual_targets, _config, **kwargs):
            captured["targets"] = actual_targets
            captured.update(kwargs)
            return True

        driver._move_to_targets = move_to_targets

        reached = driver.move_above_grasp_sites(
            object(),
            "green_tote_b01_lower",
            config,
        )

        self.assertTrue(reached)
        self.assertIs(captured["targets"], targets)
        self.assertEqual(captured["tolerance"], config.approach_tolerance)
        self.assertEqual(captured["max_steps"], config.clearance_translate_steps)

    def test_vertical_clearance_targets_only_raise_grippers(self):
        current = {
            "right": np.array([12.48, 4.87, 1.16]),
            "left": np.array([12.47, 4.42, 1.14]),
        }
        grasp_targets = {
            "right": np.array([12.03, 4.41, 1.36]),
            "left": np.array([11.70, 4.41, 1.36]),
        }

        targets = self.module.vertical_clearance_targets(
            current,
            grasp_targets,
            clearance_height=0.30,
        )

        np.testing.assert_allclose(targets["right"], [12.48, 4.87, 1.66])
        np.testing.assert_allclose(targets["left"], [12.47, 4.42, 1.66])

    def test_default_clearance_matches_reachable_safe_height(self):
        config = self.module.ScriptedGraspConfig()

        self.assertEqual(config.clearance_height, 0.30)

    def test_default_grasp_depth_interpolates_reach_and_side_contact(self):
        config = self.module.ScriptedGraspConfig()

        self.assertEqual(config.site_below_offset, 0.015)
        self.assertEqual(config.left_wrist_adjustment, 0.10)

    def test_grasp_targets_can_be_swapped_after_base_rotation(self):
        raw_targets = {
            "right": np.array([12.03, 4.41, 1.36]),
            "left": np.array([11.70, 4.41, 1.36]),
        }

        targets = self.module.assigned_grasp_targets(raw_targets, swap=True)

        np.testing.assert_allclose(targets["right"], raw_targets["left"])
        np.testing.assert_allclose(targets["left"], raw_targets["right"])

    def test_success_requires_both_contacts_and_lift(self):
        self.assertTrue(
            self.module.verified_grasp(
                {"right": True, "left": True}, lift_success=True
            )
        )
        self.assertFalse(
            self.module.verified_grasp(
                {"right": True, "left": False}, lift_success=True
            )
        )
        self.assertFalse(
            self.module.verified_grasp(
                {"right": True, "left": True}, lift_success=False
            )
        )

    def test_grasp_end_event_uses_verified_combined_result(self):
        backend = RecordingBackend()

        success = self.module.mark_verified_grasp_end(
            backend,
            source="input_5",
            object_name="line_5_container_h01_near",
            contacts={"right": True, "left": True},
            lift_success=False,
        )

        self.assertFalse(success)
        self.assertEqual(
            backend.events,
            [
                {
                    "name": "grasp_end",
                    "source": "input_5",
                    "object_name": "line_5_container_h01_near",
                    "success": False,
                    "contact_right": True,
                    "contact_left": True,
                    "lift_success": False,
                }
            ],
        )

    def test_scripted_grasp_runs_verified_stage_sequence(self):
        backend = RecordingBackend()
        driver = ScriptedDriver()

        result = self.module.run_scripted_grasp(
            backend,
            source="input_5",
            object_name="line_5_container_h01_near",
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            driver.calls,
            [
                "raise_clearance",
                "move_above",
                "pregrasp",
                "approach",
                "wrist_adjust",
                "close",
                "lift",
                "hold_metadata",
            ],
        )
        self.assertEqual(
            result["hold"],
            {
                "base_yaw": 0.25,
                "object_pos": [1.0, 2.0, 1.15],
                "object_z": 1.15,
            },
        )
        self.assertEqual(backend.events[0]["name"], "grasp_start")
        self.assertEqual(backend.events[-1]["name"], "grasp_end")
        self.assertTrue(backend.events[-1]["success"])

    def test_white_tote_refreshes_controller_before_grasp_stages(self):
        calls = []

        class Composite:
            part_controllers = {}

            def update_state(self):
                calls.append("update_state")

        backend = RecordingBackend()
        backend.env = SimpleNamespace(
            robots=[SimpleNamespace(composite_controller=Composite())]
        )

        result = self.module.run_scripted_grasp(
            backend,
            source="input_1",
            object_name="white_tote_b01_left_center",
            driver=ScriptedDriver(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(calls, ["update_state"])

    def test_scripted_grasp_stops_before_lift_when_contact_is_incomplete(self):
        backend = RecordingBackend()
        driver = ScriptedDriver(contacts={"right": True, "left": False})

        result = self.module.run_scripted_grasp(
            backend,
            source="input_5",
            object_name="line_5_container_h01_near",
            driver=driver,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "contact")
        self.assertEqual(
            driver.calls,
            [
                "raise_clearance",
                "move_above",
                "pregrasp",
                "approach",
                "wrist_adjust",
                "close",
                "polish",
            ],
        )
        self.assertFalse(backend.events[-1]["success"])

    def test_contact_polish_can_recover_one_missing_arm(self):
        backend = RecordingBackend()
        driver = ScriptedDriver(
            contacts={"right": True, "left": False},
            polished_contacts={"right": True, "left": True},
        )

        result = self.module.run_scripted_grasp(
            backend,
            source="input_6",
            object_name="green_tote_b01_lower",
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            driver.calls,
            [
                "raise_clearance",
                "move_above",
                "pregrasp",
                "approach",
                "wrist_adjust",
                "close",
                "polish",
                "lift",
                "hold_metadata",
            ],
        )

    def test_scripted_grasp_stops_when_vertical_clearance_fails(self):
        backend = RecordingBackend()
        driver = ScriptedDriver(clearance_success=False)

        result = self.module.run_scripted_grasp(
            backend,
            source="input_6",
            object_name="green_tote_b01_upper",
            driver=driver,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "raise_clearance")
        self.assertEqual(driver.calls, ["raise_clearance"])

    def test_scripted_grasp_skips_redundant_raise_when_clearance_is_prepared(self):
        backend = RecordingBackend()
        driver = ScriptedDriver()
        config = self.module.ScriptedGraspConfig(clearance_prepared=True)

        result = self.module.run_scripted_grasp(
            backend,
            source="input_6",
            object_name="green_tote_b01_lower",
            config=config,
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            driver.calls,
            [
                "move_above",
                "pregrasp",
                "approach",
                "wrist_adjust",
                "close",
                "lift",
                "hold_metadata",
            ],
        )

    def test_upper_green_tote_preconditions_wrist_before_descent(self):
        backend = RecordingBackend()
        driver = ScriptedDriver()

        result = self.module.run_scripted_grasp(
            backend,
            source="input_6",
            object_name="green_tote_b01_upper",
            driver=driver,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            driver.calls,
            [
                "raise_clearance",
                "move_above",
                "wrist_adjust",
                "pregrasp",
                "approach",
                "close",
                "lift",
                "hold_metadata",
            ],
        )


if __name__ == "__main__":
    unittest.main()
