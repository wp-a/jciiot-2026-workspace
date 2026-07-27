import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
