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

    def test_delivery_inset_moves_unregistered_output_toward_center(self):
        target = self.module.delivery_inset_target(
            center=np.array([4.872, -7.261]),
            approach=np.array([4.020, -7.261]),
            inset=0.15,
        )

        np.testing.assert_allclose(target, [4.170, -7.261])

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

    def test_carrying_move_insets_output_without_physical_registration(self):
        calls = []
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            env=SimpleNamespace(output_ports={"output_4_shelf": {}})
        )
        driver.scene_context = SimpleNamespace(
            output_ports={
                "output_5": SimpleNamespace(
                    center=np.array([4.872, -7.261]),
                    approach=np.array([4.020, -7.261]),
                )
            }
        )
        driver._move_to = lambda target, *, carrying: (
            calls.append((target, carrying)) or True
        )

        success = driver.move("output_5", carrying=True, object_name="box")

        self.assertTrue(success)
        self.assertEqual(calls, [("4.170000, -7.261000", True)])

    def test_place_uses_physical_release_for_unregistered_output(self):
        calls = []
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = SimpleNamespace(
            env=SimpleNamespace(output_ports={"output_4_shelf": {}})
        )
        driver._release_at_current_pose = lambda target, object_name: (
            calls.append((target, object_name)) or True
        )

        success = driver.place("output_5", "blue_tote")

        self.assertTrue(success)
        self.assertEqual(calls, [("output_5", "blue_tote")])

    def test_unregistered_release_preserves_current_transport_position(self):
        calls = []
        place_module = types.ModuleType(
            "robosuite.environments.factory_sorting.place_on_table"
        )
        place_module.gripper_release_action = lambda _env: "release"
        transport_module = types.ModuleType(
            "robosuite.environments.factory_sorting.transport_attachment"
        )
        transport_module.clear_transport_attachment = (
            lambda _env: calls.append("clear")
        )

        def stale_sync(_env):
            raise AssertionError("stale attachment must not rewrite object pose")

        transport_module.sync_transport_attachment = stale_sync
        raw_env = SimpleNamespace(
            output_ports={},
            step=lambda action: (
                calls.append(("step", action))
                or (None, None, None, {"has_judge_collision": False})
            ),
        )
        backend = SimpleNamespace(
            env=raw_env,
            _held_crate_name="blue_tote",
            _held_crate_body_id=7,
            _rp={"place": {"release_steps": 2}},
        )
        driver = object.__new__(self.module.OfficialCompetitionDriver)
        driver.backend = backend
        modules = {
            "robosuite": types.ModuleType("robosuite"),
            "robosuite.environments": types.ModuleType(
                "robosuite.environments"
            ),
            "robosuite.environments.factory_sorting": types.ModuleType(
                "robosuite.environments.factory_sorting"
            ),
            "robosuite.environments.factory_sorting.place_on_table": place_module,
            "robosuite.environments.factory_sorting.transport_attachment": transport_module,
        }

        with patch.dict(sys.modules, modules):
            success = driver._release_at_current_pose(
                "output_5",
                "blue_tote",
            )

        self.assertTrue(success)
        self.assertEqual(calls, ["clear", ("step", "release"), ("step", "release")])
        self.assertIsNone(backend._held_crate_name)


if __name__ == "__main__":
    unittest.main()
