import importlib.util
import unittest
from pathlib import Path

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
    def __init__(self, *, contacts=None, lift_success=True, clearance_success=True):
        self.contacts = contacts or {"right": True, "left": True}
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

    def close_and_check_contacts(self, backend, object_name, config):
        self.calls.append("close")
        return self.contacts

    def lift_and_verify(self, backend, object_name, config):
        self.calls.append("lift")
        return self.lift_success

    def attach_for_transport(self, backend, object_name):
        self.calls.append("attach")


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
                "close",
                "lift",
                "attach",
            ],
        )
        self.assertEqual(backend.events[0]["name"], "grasp_start")
        self.assertEqual(backend.events[-1]["name"], "grasp_end")
        self.assertTrue(backend.events[-1]["success"])

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
            ["raise_clearance", "move_above", "pregrasp", "approach", "close"],
        )
        self.assertFalse(backend.events[-1]["success"])

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
            ["move_above", "pregrasp", "approach", "close", "lift", "attach"],
        )


if __name__ == "__main__":
    unittest.main()
