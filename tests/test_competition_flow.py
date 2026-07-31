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

        np.testing.assert_allclose(slots[0], [-0.456, 8.473])
        np.testing.assert_allclose(slots[1], [0.144, 8.473])
        np.testing.assert_allclose(slots[2], [0.744, 8.473])
        self.assertTrue(
            all(float(np.linalg.norm(slot - center)) < 0.8 for slot in slots)
        )
        self.assertGreater(abs(slots[0][0] - slots[1][0]), 0.4)
        self.assertGreater(abs(slots[1][0] - slots[2][0]), 0.4)
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

    def test_carrying_move_routes_through_physical_transport(self):
        captured = {}
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([1.0, 2.0]), 0.25),
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(
                    center=np.array([4.872, -7.261]),
                )
            }
        )
        driver._physical_hold = {
            "base_yaw": 0.25,
            "object_pos": [1.5, 2.2, 1.1],
            "object_z": 1.1,
        }
        driver.move_skill = SimpleNamespace(
            _plan=lambda start, goal: (
                captured.update(start=np.asarray(start), goal=np.asarray(goal))
                or [np.asarray(goal)]
            )
        )
        driver._move_to = lambda *_args, **_kwargs: self.fail(
            "carrying must not call direct navigation"
        )

        transport_module = types.ModuleType(
            "robot_agent.skills.competition_transport"
        )
        transport_module.PhysicalCarryConfig = lambda **kwargs: SimpleNamespace(
            object_drop_tolerance=0.025,
            **kwargs,
        )
        transport_module.PostureLockedPhysicalCarryDriver = (
            lambda: "posture-locked-driver"
        )
        transport_module.physical_carry_step_budget = lambda *_args, **_kwargs: 4321
        transport_module.transport_base_goal = (
            lambda **kwargs: (
                np.asarray(kwargs["object_target_xy"])
                - (
                    np.asarray(kwargs["object_xy"])
                    - np.asarray(kwargs["base_xy"])
                )
            )
        )

        def run_physical_transport(_backend, **kwargs):
            captured.update(transport=kwargs)
            return {"success": True, "failure_stage": None}

        transport_module.run_physical_transport = run_physical_transport
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_transport": transport_module,
        }

        with patch.dict(sys.modules, modules):
            success = driver.move("output_5", carrying=True, object_name="box")

        self.assertTrue(success)
        np.testing.assert_allclose(captured["start"], [1.0, 2.0])
        np.testing.assert_allclose(captured["goal"], [4.372, -7.461])
        np.testing.assert_allclose(
            captured["transport"]["path"],
            [[4.372, -7.461]],
        )
        self.assertEqual(captured["transport"]["object_name"], "box")
        self.assertAlmostEqual(captured["transport"]["hold_yaw"], 0.25)
        self.assertAlmostEqual(
            captured["transport"]["minimum_object_z"],
            1.075,
        )
        self.assertEqual(captured["transport"]["driver"], "posture-locked-driver")
        self.assertEqual(captured["transport"]["config"].max_steps, 4321)
        self.assertAlmostEqual(captured["transport"]["config"].max_linear, 0.04)
        self.assertAlmostEqual(
            captured["transport"]["config"].max_linear_delta,
            0.005,
        )
        self.assertAlmostEqual(
            captured["transport"]["config"].max_planar_grasp_drift,
            0.12,
        )
        self.assertAlmostEqual(
            captured["transport"]["config"].height_recovery_trigger,
            0.004,
        )
        self.assertAlmostEqual(
            captured["transport"]["config"].planar_recovery_trigger,
            0.015,
        )
        self.assertEqual(
            captured["transport"]["config"].planar_recovery_steps,
            4,
        )
        self.assertAlmostEqual(
            captured["transport"]["config"].planar_recovery_inward_delta,
            0.002,
        )
        self.assertFalse(captured["transport"]["config"].align_heading_to_path)

    def test_carrying_move_propagates_physical_contact_failure(self):
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            get_base_pose=lambda: (np.array([1.0, 2.0]), 0.0),
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {
            "base_yaw": 0.0,
            "object_pos": [1.5, 2.0, 1.1],
            "object_z": 1.1,
        }
        driver.move_skill = SimpleNamespace(
            _plan=lambda _start, goal: [np.asarray(goal)]
        )
        transport_module = types.ModuleType(
            "robot_agent.skills.competition_transport"
        )
        transport_module.PhysicalCarryConfig = lambda **kwargs: SimpleNamespace(
            object_drop_tolerance=0.025,
            **kwargs,
        )
        transport_module.PostureLockedPhysicalCarryDriver = (
            lambda: "posture-locked-driver"
        )
        transport_module.physical_carry_step_budget = lambda *_args, **_kwargs: 4321
        transport_module.transport_base_goal = lambda **kwargs: np.asarray(
            kwargs["object_target_xy"]
        )
        transport_module.run_physical_transport = lambda *_args, **_kwargs: {
            "success": False,
            "failure_stage": "contact",
        }
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_transport": transport_module,
        }

        with patch.dict(sys.modules, modules):
            success = driver.move("output_5", carrying=True, object_name="box")

        self.assertFalse(success)
        self.assertEqual(driver._last_transport["failure_stage"], "contact")

    def test_successful_grasp_persists_read_only_hold_metadata(self):
        hold = {
            "base_yaw": 0.4,
            "object_pos": [1.0, 2.0, 1.2],
            "object_z": 1.2,
        }
        fake_module = types.ModuleType("robot_agent.skills.competition_grasp")
        fake_module.ScriptedGraspConfig = SimpleNamespace
        fake_module.run_scripted_grasp = lambda *_args, **_kwargs: {
            "success": True,
            "lift_success": True,
            "hold": hold,
        }
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = object()
        driver.grasp_config = SimpleNamespace()
        driver._swap_arm_targets = False
        driver._clearance_prepared = False
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_grasp": fake_module,
        }

        with patch.dict(sys.modules, modules):
            result = driver.grasp("input_5", "box")

        self.assertTrue(result["success"])
        self.assertEqual(driver._physical_hold, hold)

    def test_place_routes_through_physical_descent_and_clears_hold_on_success(self):
        captured = {}
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = object()
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {
            "object_z": 1.1,
            "minimum_transport_object_z": 0.95,
        }
        transport_module = types.ModuleType(
            "robot_agent.skills.competition_transport"
        )

        def run_physical_target_alignment(_backend, **kwargs):
            captured.update(alignment=kwargs)
            return {"success": True, "failure_stage": None}

        def run_physical_place(_backend, **kwargs):
            captured.update(kwargs)
            return {"success": True, "failure_stage": None}

        transport_module.run_physical_place = run_physical_place
        transport_module.run_physical_target_alignment = (
            run_physical_target_alignment
        )
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_transport": transport_module,
        }

        with patch.dict(sys.modules, modules):
            success = driver.place("output_5", "blue_tote")

        self.assertTrue(success)
        self.assertEqual(captured["object_name"], "blue_tote")
        np.testing.assert_allclose(captured["target_xy"], [4.0, -7.0])
        self.assertEqual(captured["alignment"]["object_name"], "blue_tote")
        np.testing.assert_allclose(
            captured["alignment"]["target_xy"],
            [4.0, -7.0],
        )
        self.assertAlmostEqual(
            captured["alignment"]["minimum_object_z"],
            0.95,
        )
        self.assertIsNone(driver._physical_hold)

    def test_place_stops_when_physical_target_alignment_fails(self):
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = object()
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(center=np.array([4.0, -7.0]))
            }
        )
        driver._physical_hold = {"object_z": 1.1}
        transport_module = types.ModuleType(
            "robot_agent.skills.competition_transport"
        )
        transport_module.run_physical_target_alignment = (
            lambda *_args, **_kwargs: {
                "success": False,
                "failure_stage": "collision",
            }
        )
        transport_module.run_physical_place = lambda *_args, **_kwargs: self.fail(
            "placement must not run after alignment failure"
        )
        modules = {
            "robot_agent": types.ModuleType("robot_agent"),
            "robot_agent.skills": types.ModuleType("robot_agent.skills"),
            "robot_agent.skills.competition_transport": transport_module,
        }

        with patch.dict(sys.modules, modules):
            success = driver.place("output_5", "blue_tote")

        self.assertFalse(success)
        self.assertEqual(driver._last_alignment["failure_stage"], "collision")

    def test_workflow_source_has_no_transport_attachment(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("transport_attachment", source)
        self.assertNotIn("sync_transport_attachment", source)
        self.assertNotIn("relative_xy", source)


if __name__ == "__main__":
    unittest.main()
