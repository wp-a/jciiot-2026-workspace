import copy
import unittest

from scripts import audit_physical_transport_dataset as module


def valid_record():
    return {
        "accepted": False,
        "transport_success": False,
        "physical_grasp": True,
        "continuous_bilateral_contact": True,
        "dropped": False,
        "minimum_object_lift_m": 0.16,
        "max_object_gripper_drift_m": 0.02,
        "collision_frames": 0,
        "attachment_calls": 0,
        "attachment_activations": 0,
        "object_pose_writes": 0,
        "robot_state_writes": 0,
        "legacy_teleport_activations": 0,
        "infrastructure_error": None,
        "pre_grasp_object_position": [8.0, 4.6, 1.1],
        "full_physical_probe": {
            "start_object_position": [7.05, 4.62, 1.33],
            "final_object_position": [6.54, 4.62, 1.34],
        },
    }


class AuditRecordTests(unittest.TestCase):
    def test_accepts_only_recomputed_physical_object_transport(self):
        result = module.audit_record(valid_record())

        self.assertTrue(result["eligible"])
        self.assertEqual(result["classification"], "transport_success")
        self.assertEqual(result["failures"], [])
        self.assertAlmostEqual(result["metrics"]["object_translation_m"], 0.51)

    def test_ignores_old_acceptance_flag_and_base_motion(self):
        record = valid_record()
        record["accepted"] = True
        record["transport_success"] = True
        record["base_translation_m"] = 0.75
        record["full_physical_probe"]["final_object_position"] = [6.78, 4.62, 1.34]

        result = module.audit_record(record)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["classification"], "recovery")
        self.assertIn("object_translation_m", result["failures"])
        self.assertAlmostEqual(result["metrics"]["object_translation_m"], 0.27)

    def test_uses_transport_start_instead_of_pregrasp_position(self):
        record = valid_record()
        record["pre_grasp_object_position"] = [20.0, 20.0, 1.0]
        record["full_physical_probe"]["final_object_position"] = [6.80, 4.62, 1.34]

        result = module.audit_record(record)

        self.assertEqual(result["classification"], "recovery")
        self.assertAlmostEqual(result["metrics"]["object_translation_m"], 0.25)

    def test_missing_integrity_field_fails_closed(self):
        record = valid_record()
        del record["attachment_calls"]

        result = module.audit_record(record)

        self.assertEqual(result["classification"], "rejected")
        self.assertIn("missing:attachment_calls", result["failures"])

    def test_integrity_and_safety_violations_are_rejected(self):
        violations = {
            "attachment_calls": 1,
            "attachment_activations": 1,
            "object_pose_writes": 1,
            "robot_state_writes": 1,
            "legacy_teleport_activations": 1,
            "collision_frames": 1,
            "infrastructure_error": "EGL failed",
        }
        for field, value in violations.items():
            with self.subTest(field=field):
                record = valid_record()
                record[field] = value

                result = module.audit_record(record)

                self.assertEqual(result["classification"], "rejected")
                self.assertIn(field, result["failures"])

    def test_physical_failures_are_recovery_data(self):
        variants = {
            "physical_grasp": (False, "physical_grasp"),
            "continuous_bilateral_contact": (
                False,
                "continuous_bilateral_contact",
            ),
            "dropped": (True, "dropped"),
            "minimum_object_lift_m": (0.12, "minimum_object_lift_m"),
            "max_object_gripper_drift_m": (
                0.051,
                "max_object_gripper_drift_m",
            ),
        }
        for field, (value, expected_failure) in variants.items():
            with self.subTest(field=field):
                record = copy.deepcopy(valid_record())
                record[field] = value

                result = module.audit_record(record)

                self.assertEqual(result["classification"], "recovery")
                self.assertIn(expected_failure, result["failures"])

    def test_nonfinite_position_is_rejected(self):
        record = valid_record()
        record["full_physical_probe"]["final_object_position"][0] = float("nan")

        result = module.audit_record(record)

        self.assertEqual(result["classification"], "rejected")
        self.assertIn("object_positions", result["failures"])


if __name__ == "__main__":
    unittest.main()
