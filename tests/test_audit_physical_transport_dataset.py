import copy
import json
import tempfile
import unittest
from pathlib import Path

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


class AuditBatchTests(unittest.TestCase):
    def test_builds_sorted_ledger_and_rejects_duplicate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = valid_record()
            recovery = valid_record()
            recovery["full_physical_probe"]["final_object_position"] = [
                6.80,
                4.62,
                1.34,
            ]
            duplicate_text = json.dumps(success, sort_keys=True)
            (root / "c-success.json").write_text(duplicate_text, encoding="utf-8")
            (root / "a-duplicate.json").write_text(
                duplicate_text,
                encoding="utf-8",
            )
            (root / "b-recovery.json").write_text(
                json.dumps(recovery, sort_keys=True),
                encoding="utf-8",
            )

            ledger = module.audit_files(
                [
                    root / "c-success.json",
                    root / "b-recovery.json",
                    root / "a-duplicate.json",
                ]
            )

            self.assertEqual(
                [Path(row["source_path"]).name for row in ledger["records"]],
                ["a-duplicate.json", "b-recovery.json", "c-success.json"],
            )
            self.assertEqual(
                ledger["classification_counts"],
                {"recovery": 1, "rejected": 1, "transport_success": 1},
            )
            duplicate = ledger["records"][2]
            self.assertEqual(duplicate["classification"], "rejected")
            self.assertIn("duplicate_content", duplicate["failures"])
            self.assertEqual(
                Path(duplicate["duplicate_of"]).name,
                "a-duplicate.json",
            )
            self.assertEqual(len(duplicate["source_sha256"]), 64)

    def test_cli_writes_json_and_tsv_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "results"
            source_dir.mkdir()
            (source_dir / "accepted.json").write_text(
                json.dumps(valid_record()),
                encoding="utf-8",
            )
            (source_dir / "accepted-trajectory.json").write_text(
                json.dumps({"frames": []}),
                encoding="utf-8",
            )
            json_output = root / "ledger.json"
            tsv_output = root / "results.tsv"

            exit_code = module.main(
                [
                    str(source_dir),
                    "--json-output",
                    str(json_output),
                    "--tsv-output",
                    str(tsv_output),
                ]
            )

            self.assertEqual(exit_code, 0)
            ledger = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(ledger["record_count"], 1)
            self.assertEqual(ledger["eligible_count"], 1)
            rows = tsv_output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("object_translation_m", rows[0])
            self.assertFalse((root / "ledger.json.tmp").exists())
            self.assertFalse((root / "results.tsv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
