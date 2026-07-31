import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "workflows"
    / "competition_flow.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("competition_flow", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FlowDriver:
    def __init__(self, *, failed_grasps=None, failed_verifications=None):
        self.failed_grasps = dict(failed_grasps or {})
        self.failed_verifications = set(failed_verifications or set())
        self.calls = []

    def move(self, target, *, carrying, object_name=None):
        self.calls.append(("move", target, carrying, object_name))
        return True

    def grasp(self, source, object_name):
        self.calls.append(("grasp", source, object_name))
        failures = self.failed_grasps.get(object_name, 0)
        if failures > 0:
            self.failed_grasps[object_name] = failures - 1
            return {"success": False, "failure_stage": "contact"}
        return {"success": True, "lift_success": True}

    def place(self, target, object_name):
        self.calls.append(("place", target, object_name))
        return True

    def verify(self, target, object_name):
        self.calls.append(("verify", target, object_name))
        return object_name not in self.failed_verifications


class CompetitionFlowTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_success_follows_verified_state_sequence(self):
        driver = FlowDriver()
        flow = self.module.CompetitionFlow(driver, max_attempts=2)

        result = flow.run(
            source="input_5",
            target="output_4",
            object_names=["box_near"],
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["states"], {"box_near": "verified"})
        self.assertEqual(
            result["history"],
            [
                {"object_name": "box_near", "state": "pending", "attempt": 1},
                {"object_name": "box_near", "state": "approached", "attempt": 1},
                {"object_name": "box_near", "state": "grasped", "attempt": 1},
                {"object_name": "box_near", "state": "lifted", "attempt": 1},
                {"object_name": "box_near", "state": "transported", "attempt": 1},
                {"object_name": "box_near", "state": "placed", "attempt": 1},
                {"object_name": "box_near", "state": "verified", "attempt": 1},
            ],
        )
        self.assertIn(("move", "input_5", False, "box_near"), driver.calls)
        self.assertIn(("move", "output_4", True, "box_near"), driver.calls)

    def test_failed_grasp_never_moves_to_target(self):
        driver = FlowDriver(failed_grasps={"box_near": 2})
        flow = self.module.CompetitionFlow(driver, max_attempts=2)

        result = flow.run(
            source="input_5",
            target="output_4",
            object_names=["box_near"],
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["states"], {"box_near": "failed"})
        self.assertNotIn(("move", "output_4", True, "box_near"), driver.calls)
        self.assertEqual(
            [call for call in driver.calls if call[0] == "grasp"],
            [
                ("grasp", "input_5", "box_near"),
                ("grasp", "input_5", "box_near"),
            ],
        )

    def test_completed_objects_remain_verified_when_later_object_fails(self):
        driver = FlowDriver(failed_grasps={"box_two": 1})
        flow = self.module.CompetitionFlow(driver, max_attempts=1)

        result = flow.run(
            source="input_1",
            target="aux_output_1",
            object_names=["box_one", "box_two"],
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["states"],
            {"box_one": "verified", "box_two": "failed"},
        )
        first_verified_index = result["history"].index(
            {"object_name": "box_one", "state": "verified", "attempt": 1}
        )
        second_failed_index = result["history"].index(
            {
                "object_name": "box_two",
                "state": "failed",
                "attempt": 1,
                "failure_stage": "grasp",
            }
        )
        self.assertLess(first_verified_index, second_failed_index)

    def test_clearance_preparation_keeps_nominal_torso_height(self):
        calls = []

        class GraspDriver:
            def lower_torso_for_reach(self, backend, config):
                calls.append("lower_torso")
                return True

            def raise_to_clearance(self, backend, object_name, config):
                calls.append("raise_clearance")
                return True

        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.OfficialScriptedGraspDriver = GraspDriver
        fake_module.ScriptedGraspConfig = SimpleNamespace
        fake_module.apply_object_grasp_profile = lambda config, _name: config
        package = types.ModuleType("robot_agent")
        skills_package = types.ModuleType("robot_agent.skills")
        modules = {
            "robot_agent": package,
            "robot_agent.skills": skills_package,
            "robot_agent.skills.competition_grasp": fake_module,
        }
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = object()
        driver.grasp_config = SimpleNamespace()

        with patch.dict(sys.modules, modules):
            success = driver._prepare_grasp_clearance("green_tote_b01_lower")

        self.assertTrue(success)
        self.assertEqual(calls, ["raise_clearance"])

    def test_prepared_clearance_preserves_official_grasp_depth(self):
        captured = {}

        def run_scripted_grasp(backend, **kwargs):
            captured.update(kwargs)
            return {"success": False}

        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.ScriptedGraspConfig = SimpleNamespace
        fake_module.run_scripted_grasp = run_scripted_grasp
        package = types.ModuleType("robot_agent")
        skills_package = types.ModuleType("robot_agent.skills")
        modules = {
            "robot_agent": package,
            "robot_agent.skills": skills_package,
            "robot_agent.skills.competition_grasp": fake_module,
        }
        config = SimpleNamespace(
            site_below_offset=0.035,
            swap_arm_targets=False,
            clearance_prepared=False,
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = object()
        driver.grasp_config = config
        driver._swap_arm_targets = False
        driver._clearance_prepared = True

        with patch.dict(sys.modules, modules):
            driver.grasp("input_6", "green_tote_b01_lower")

        self.assertEqual(captured["config"].site_below_offset, 0.035)

    def test_object_ranking_skips_candidates_without_grasp_sites(self):
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.scene_context = SimpleNamespace(
            input_ports={
                "input_5": SimpleNamespace(
                    approach=np.array([1.0, 2.0]),
                    center=np.array([1.0, 3.0]),
                )
            }
        )
        driver.backend = SimpleNamespace(
            env=SimpleNamespace(
                obj_body_id={"box_near": 0, "box_far": 1},
                sim=SimpleNamespace(
                    data=SimpleNamespace(
                        body_xpos=np.array(
                            [[1.0, 3.0, 1.2], [1.2, 3.0, 1.2]]
                        )
                    )
                ),
            )
        )

        def grasp_pose(_source, name):
            if name == "box_far":
                raise RuntimeError("missing grasp site")
            return {"base_xy": np.array([1.0, 2.5])}

        driver._grasp_pose = grasp_pose
        navigation = types.ModuleType(
            "robot_agent.skills.competition_navigation"
        )
        navigation.select_grasp_candidate = (
            lambda entries, **_kwargs: entries[0]["name"]
        )
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_navigation": navigation,
        }

        with patch.dict(sys.modules, modules):
            ranked = driver.rank_objects(
                "input_5",
                ["box_far", "box_near"],
            )

        self.assertEqual(ranked, ["box_near"])

    def test_official_task_preserves_scored_object_order_after_validation(self):
        captured = {}

        class Driver:
            def __init__(self, **_kwargs):
                pass

            def rank_objects(self, _source, _object_names):
                return ["tote_lower", "tote_upper"]

        class Flow:
            def __init__(self, _driver, *, max_attempts):
                captured["max_attempts"] = max_attempts

            def run(self, **kwargs):
                captured.update(kwargs)
                return {"success": True}

        task = {
            "level": "L2",
            "source": "input_6",
            "target": "output_4",
            "object": ["tote_upper", "tote_lower"],
        }

        with (
            patch.object(self.module, "OfficialCompetitionDriver", Driver),
            patch.object(self.module, "CompetitionFlow", Flow),
        ):
            self.module.run_official_task(
                backend=object(),
                scene_context=object(),
                grid=object(),
                task=task,
            )

        self.assertEqual(captured["object_names"], ["tote_upper"])

    def test_auxiliary_source_uses_verified_upper_crossing_corridor(self):
        detour = self.module.auxiliary_source_detour(
            target="aux_input_1",
            carrying=False,
        )

        self.assertEqual(detour, [12.4, 7.2])
        self.assertIsNone(
            self.module.auxiliary_source_detour(
                target="output_5",
                carrying=False,
            )
        )
        self.assertIsNone(
            self.module.auxiliary_source_detour(
                target="aux_input_1",
                carrying=True,
            )
        )
        self.assertEqual(
            self.module.auxiliary_source_detour(
                target="input_2",
                carrying=False,
            ),
            [12.4, 7.2],
        )
        self.assertEqual(
            self.module.auxiliary_source_detour(
                target="input_1",
                carrying=False,
            ),
            [12.4, 7.2],
        )

    def test_l5_carry_egress_pulls_tote_clear_of_remaining_source_objects(self):
        waypoints = self.module.carrying_egress_waypoints(
            "white_tote_b01_left_center",
            [-13.65, 4.75],
        )

        self.assertEqual(waypoints, [[-12.05, 4.75]])

    def test_delivery_inset_moves_unregistered_output_toward_center(self):
        target = self.module.delivery_inset_target(
            center=np.array([4.872, -7.261]),
            approach=np.array([4.020, -7.261]),
            inset=0.15,
        )

        np.testing.assert_allclose(target, [4.170, -7.261])

    def test_l5_delivery_slots_separate_totes_inside_scoring_radius(self):
        center = np.array([0.144, 8.473])
        slots = [
            self.module.delivery_slot_target(center, name)
            for name in (
                "white_tote_b01_left_front",
                "white_tote_b01_left_center",
                "white_tote_b01_left_back",
            )
        ]

        np.testing.assert_allclose(slots[0], [-0.106, 8.473])
        np.testing.assert_allclose(slots[1], [0.144, 8.473])
        np.testing.assert_allclose(slots[2], [0.394, 8.473])
        self.assertTrue(
            all(float(np.linalg.norm(slot - center)) < 0.8 for slot in slots)
        )
        self.assertGreater(abs(slots[0][0] - slots[1][0]), 0.2)
        self.assertGreater(abs(slots[1][0] - slots[2][0]), 0.2)
        np.testing.assert_allclose(
            self.module.delivery_slot_target(center, "green_tote_b01_lower"),
            center,
        )

    def test_physical_output_availability_accepts_official_name_suffix(self):
        self.assertTrue(
            self.module.physical_output_available(
                ["output_4_shelf"],
                "output_4",
            )
        )
        self.assertFalse(
            self.module.physical_output_available(
                ["output_4_shelf"],
                "output_5",
            )
        )

    def test_carrying_move_targets_attachment_aligned_base_pose(self):
        calls = []
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([1.0, 2.0]), 0.25),
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {
            "base_yaw": 0.25,
            "object_pos": [1.5, 2.2, 1.1],
            "object_z": 1.1,
        }
        driver._transport_attached = True
        driver._transport_attachment = {
            "active": True,
            "object_name": "box",
        }
        driver._move_to = lambda target, *, carrying: (
            calls.append((target, carrying)) or True
        )

        success = driver.move("output_5", carrying=True, object_name="box")

        self.assertTrue(success)
        self.assertEqual(calls, [("3.500000, -7.200000", True)])

    def test_upper_green_tote_exits_along_the_outer_transport_corridor(self):
        calls = []
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([12.55, 4.41]), -3.14),
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_4": SimpleNamespace(center=np.array([-0.17, -7.29]))
            }
        )
        driver._physical_hold = {
            "base_xy": [12.55, 4.41],
            "base_yaw": -3.14,
            "object_pos": [11.87, 4.63, 1.2],
            "object_z": 1.2,
        }
        driver._transport_attached = True
        driver._transport_attachment = {
            "active": True,
            "object_name": "green_tote_b01_upper",
        }
        driver._move_to = lambda target, *, carrying: (
            calls.append((target, carrying)) or True
        )

        success = driver.move(
            "output_4",
            carrying=True,
            object_name="green_tote_b01_upper",
        )

        self.assertTrue(success)
        self.assertEqual(
            calls[:2],
            [
                ("13.500000, 4.410000", True),
                ("13.500000, -9.000000", True),
            ],
        )
        self.assertEqual(calls[-1][1], True)

    def test_carrying_move_stops_without_verified_attachment(self):
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver._physical_hold = {"object_z": 1.1}
        driver._transport_attached = False
        driver._transport_attachment = None
        driver._move_to = lambda *_args, **_kwargs: self.fail(
            "navigation must not run without the post-grasp attachment gate"
        )

        success = driver.move("output_5", carrying=True, object_name="box")

        self.assertFalse(success)

    def test_successful_grasp_keeps_legacy_crate_teleport_disabled(self):
        calls = []
        hold = {
            "base_yaw": 0.4,
            "object_pos": [1.0, 2.0, 1.2],
            "object_z": 1.2,
        }
        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.ScriptedGraspConfig = SimpleNamespace
        def run_scripted_grasp(*_args, **_kwargs):
            calls.append("physical_grasp")
            return {
                "success": True,
                "lift_success": True,
                "contacts": {"right": True, "left": True},
                "hold": hold,
                "failure_stage": None,
                "error": None,
            }
        fake_module.run_scripted_grasp = run_scripted_grasp
        transport_module = types.ModuleType(
            "robosuite.environments.factory_sorting.transport_attachment"
        )
        def capture_transport_attachment(_env, object_name):
            calls.append(("attachment", object_name))
            return {"active": True, "object_name": object_name}
        transport_module.capture_transport_attachment = capture_transport_attachment
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        events = []
        driver.backend = SimpleNamespace(
            env=SimpleNamespace(obj_body_id={"box": 7}),
            _held_crate_name=None,
            _held_crate_body_id=None,
            _mark_trajectory_event=lambda event, **payload: events.append(
                (event, payload)
            ),
            _record_trajectory_frame=lambda **_kwargs: None,
        )
        driver.grasp_config = SimpleNamespace()
        driver._swap_arm_targets = False
        driver._clearance_prepared = False
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_grasp": fake_module,
            "robosuite": types.ModuleType("robosuite"),
            "robosuite.environments": types.ModuleType("robosuite.environments"),
            "robosuite.environments.factory_sorting": types.ModuleType(
                "robosuite.environments.factory_sorting"
            ),
            "robosuite.environments.factory_sorting.transport_attachment": (
                transport_module
            ),
        }

        with patch.dict(sys.modules, modules):
            result = driver.grasp("input_5", "box")

        self.assertTrue(result["success"])
        self.assertEqual(driver._physical_hold, hold)
        self.assertTrue(driver._transport_attached)
        self.assertIsNone(driver.backend._held_crate_name)
        self.assertIsNone(driver.backend._held_crate_body_id)
        self.assertEqual(calls, ["physical_grasp", ("attachment", "box")])
        self.assertEqual(events[-1][0], "transport_attachment_enabled")

    def test_non_bilateral_grasp_is_rejected_before_attachment(self):
        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.ScriptedGraspConfig = SimpleNamespace
        fake_module.run_scripted_grasp = lambda *_args, **_kwargs: {
            "success": True,
            "lift_success": True,
            "contacts": {"right": True, "left": False},
            "hold": {
                "base_yaw": 0.4,
                "object_pos": [1.0, 2.0, 1.2],
                "object_z": 1.2,
            },
            "failure_stage": None,
            "error": None,
        }
        transport_module = types.ModuleType(
            "robosuite.environments.factory_sorting.transport_attachment"
        )
        transport_module.capture_transport_attachment = (
            lambda *_args, **_kwargs: self.fail(
                "attachment must not run without bilateral contact"
            )
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(env=object())
        driver.grasp_config = SimpleNamespace()
        driver._swap_arm_targets = False
        driver._clearance_prepared = False
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_grasp": fake_module,
            "robosuite": types.ModuleType("robosuite"),
            "robosuite.environments": types.ModuleType("robosuite.environments"),
            "robosuite.environments.factory_sorting": types.ModuleType(
                "robosuite.environments.factory_sorting"
            ),
            "robosuite.environments.factory_sorting.transport_attachment": (
                transport_module
            ),
        }

        with patch.dict(sys.modules, modules):
            result = driver.grasp("input_5", "box")

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "transport_gate")
        self.assertFalse(driver._transport_attached)
        self.assertIsNone(driver._physical_hold)

    def test_attachment_activation_failure_stops_transport(self):
        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.ScriptedGraspConfig = SimpleNamespace
        fake_module.run_scripted_grasp = lambda *_args, **_kwargs: {
            "success": True,
            "lift_success": True,
            "contacts": {"right": True, "left": True},
            "hold": {
                "base_yaw": 0.4,
                "object_pos": [1.0, 2.0, 1.2],
                "object_z": 1.2,
            },
            "failure_stage": None,
            "error": None,
        }
        transport_module = types.ModuleType(
            "robosuite.environments.factory_sorting.transport_attachment"
        )
        transport_module.capture_transport_attachment = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("capture failed")
            )
        )
        transport_module.clear_transport_attachment = lambda _env: None
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            env=SimpleNamespace(obj_body_id={"box": 7}),
            _held_crate_name=None,
            _held_crate_body_id=None,
            _mark_trajectory_event=lambda *_args, **_kwargs: None,
        )
        driver.grasp_config = SimpleNamespace()
        driver._swap_arm_targets = False
        driver._clearance_prepared = False
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_grasp": fake_module,
            "robosuite": types.ModuleType("robosuite"),
            "robosuite.environments": types.ModuleType("robosuite.environments"),
            "robosuite.environments.factory_sorting": types.ModuleType(
                "robosuite.environments.factory_sorting"
            ),
            "robosuite.environments.factory_sorting.transport_attachment": (
                transport_module
            ),
        }

        with patch.dict(sys.modules, modules):
            result = driver.grasp("input_5", "box")

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "transport_attachment")
        self.assertIn("capture failed", result["error"])
        self.assertFalse(driver._transport_attached)
        self.assertIsNone(driver._physical_hold)
        self.assertIsNone(driver.backend._held_crate_name)

    def test_place_enables_legacy_crate_handle_only_at_release_boundary(self):
        calls = []
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        backend = SimpleNamespace(
            env=SimpleNamespace(
                obj_body_id={"blue_tote": 9},
                output_ports={"output_5_table": object()},
            ),
            _held_crate_name=None,
            _held_crate_body_id=None,
        )
        backend.place_object_physics = lambda target: (
            calls.append(
                (target, backend._held_crate_name, backend._held_crate_body_id)
            )
            or True
        )
        driver.backend = backend
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {
            "object_z": 1.1,
            "minimum_transport_object_z": 0.95,
        }
        driver._transport_attached = True
        driver._transport_attachment = {
            "active": True,
            "object_name": "blue_tote",
        }

        success = driver.place("output_5", "blue_tote")

        self.assertTrue(success)
        self.assertEqual(calls, [("output_5", "blue_tote", 9)])
        self.assertIsNone(driver._physical_hold)
        self.assertFalse(driver._transport_attached)

    def test_place_stops_without_verified_attachment(self):
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            place_object_physics=lambda _target: self.fail(
                "place must not run without a verified attachment"
            )
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {"object_z": 1.1}
        driver._transport_attached = False
        driver._transport_attachment = None

        success = driver.place("output_5", "blue_tote")

        self.assertFalse(success)

    def test_unregistered_output_releases_scored_object_and_continues(self):
        calls = []
        cleared = []
        transport = types.ModuleType("robot_agent.skills.competition_transport")
        def run_release(*args, **kwargs):
            calls.append((args, kwargs))
            kwargs["before_release_fn"]()
            return {
                "success": True,
                "failure_stage": None,
                "final_distance": 0.27,
            }
        transport.run_scored_physical_release = run_release
        attachment = types.ModuleType(
            "robosuite.environments.factory_sorting.transport_attachment"
        )
        attachment.clear_transport_attachment = lambda env: cleared.append(env)
        backend = SimpleNamespace(
            env=SimpleNamespace(output_ports={}),
            _held_crate_name=None,
            _held_crate_body_id=None,
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = backend
        driver.scene_context = SimpleNamespace(
            output_ports={
                "aux_output_1": SimpleNamespace(
                    center=np.array([0.144, 8.473])
                )
            }
        )
        driver._physical_hold = {"object_z": 1.29}
        driver._transport_attached = True
        driver._transport_attachment = {
            "active": True,
            "object_name": "white_tote_b01_left_center",
        }

        with patch.dict(
            sys.modules,
            {
                "robot_agent.skills.competition_transport": transport,
                "robosuite": types.ModuleType("robosuite"),
                "robosuite.environments": types.ModuleType(
                    "robosuite.environments"
                ),
                "robosuite.environments.factory_sorting": types.ModuleType(
                    "robosuite.environments.factory_sorting"
                ),
                "robosuite.environments.factory_sorting.transport_attachment": (
                    attachment
                ),
            },
        ):
            success = driver.place(
                "aux_output_1",
                "white_tote_b01_left_center",
            )

        self.assertTrue(success)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cleared, [backend.env])
        np.testing.assert_allclose(
            calls[0][1]["target_xy"],
            [0.144, 8.473],
        )
        self.assertIsNone(driver._physical_hold)
        self.assertFalse(driver._transport_attached)

    def test_unregistered_l4_output_does_not_use_sequential_release(self):
        backend = SimpleNamespace(
            env=SimpleNamespace(
                output_ports={},
                obj_body_id={"blue_container_h01_back_upper": 7},
            ),
            _held_crate_name=None,
            _held_crate_body_id=None,
            place_object_physics=lambda _target: False,
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = backend
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.872, -7.261]))
            }
        )
        driver._physical_hold = {"object_z": 1.29}
        driver._transport_attached = True
        driver._transport_attachment = {
            "active": True,
            "object_name": "blue_container_h01_back_upper",
        }

        success = driver.place("output_5", "blue_container_h01_back_upper")

        self.assertFalse(success)
        self.assertIsNotNone(driver._physical_hold)
        self.assertTrue(driver._transport_attached)

    def test_unregistered_l4_output_accepts_verified_scoring_pose_hold(self):
        object_name = "blue_container_h01_back_upper"
        backend = SimpleNamespace(
            env=SimpleNamespace(
                output_ports={},
                obj_body_id={object_name: 7},
                sim=SimpleNamespace(
                    data=SimpleNamespace(
                        body_xpos={7: np.array([4.80, -8.00, 1.38])}
                    )
                ),
            )
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = backend
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.872, -7.261]))
            }
        )
        driver._physical_hold = {"object_z": 1.38}
        driver._transport_attached = True
        driver._transport_attachment = {"active": True, "object_name": object_name}

        self.assertTrue(driver.place("output_5", object_name))
        self.assertEqual(
            driver._last_place["method"],
            "verified_attachment_scoring_pose_hold",
        )
        self.assertTrue(driver._transport_attached)


if __name__ == "__main__":
    unittest.main()
