import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_scored_path


HARD_RULES = getattr(audit_scored_path, "HARD_RULES", set())
main = audit_scored_path.main
scan_file = audit_scored_path.scan_file
scan_submission = audit_scored_path.scan_submission


class ScoredPathAuditTests(unittest.TestCase):
    def _scan_source(self, source: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidate.py"
            path.write_text(source, encoding="utf-8")
            return scan_file(path)

    def test_reports_direct_qpos_assignment(self):
        violations = self._scan_source(
            "def move(raw_env, addr, value):\n"
            "    raw_env.sim.data.qpos[addr] = value\n"
        )

        self.assertEqual([item.rule for item in violations], ["direct_qpos_write"])
        self.assertEqual(violations[0].line, 2)
        self.assertEqual(getattr(violations[0], "severity", None), "warning")

    def test_reports_object_freejoint_assignment_as_hard_violation(self):
        violations = self._scan_source(
            "def move(raw_env, object_qpos_addr, value):\n"
            "    raw_env.sim.data.qpos[object_qpos_addr] = value\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["object_qpos_write"],
        )
        self.assertEqual(getattr(violations[0], "severity", None), "error")
        self.assertIn("object_qpos_write", HARD_RULES)

    def test_reports_attachment_relative_state_assignment(self):
        violations = self._scan_source(
            "def move(attachment, value):\n"
            "    attachment['relative_xy'] = value\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["attachment_relative_write"],
        )
        self.assertEqual(getattr(violations[0], "severity", None), "error")

    def test_reports_transport_sync_import_and_call_once_each(self):
        violations = self._scan_source(
            "from robosuite.environments.factory_sorting.transport_attachment "
            "import sync_transport_attachment\n\n"
            "def move(env):\n"
            "    sync_transport_attachment(env)\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            [
                "transport_attachment_import",
                "transport_sync_helper",
            ],
        )
        self.assertTrue(all(item.severity == "error" for item in violations))

    def test_reports_any_transport_attachment_import_as_hard_violation(self):
        violations = self._scan_source(
            "from robosuite.environments.factory_sorting.transport_attachment "
            "import capture_transport_attachment\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["transport_attachment_import"],
        )
        self.assertEqual(getattr(violations[0], "severity", None), "error")

    def test_reports_private_backend_call_and_assignment(self):
        violations = self._scan_source(
            "def move(backend):\n"
            "    backend._teleport('target')\n"
            "    backend._held_crate_name = 'box'\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["private_backend_member", "private_backend_member"],
        )

    def test_reports_private_backend_call_on_assignment_rhs(self):
        violations = self._scan_source(
            "def read(backend):\n"
            "    address = backend._get_object_joint_addr('box')\n"
            "    return address\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["private_backend_member"],
        )

    def test_reports_private_backend_helper_import(self):
        violations = self._scan_source(
            "from robot_agent.environments.robosuite_backend import "
            "_navigation_collisions\n"
        )

        self.assertEqual(
            [item.rule for item in violations],
            ["private_backend_import"],
        )

    def test_allows_public_step_observation_and_recording_hooks(self):
        violations = self._scan_source(
            "def run(backend, action):\n"
            "    observation = backend.observe()\n"
            "    backend._record_trajectory_frame()\n"
            "    backend._mark_trajectory_event('stage')\n"
            "    return backend.step(action), observation\n"
        )

        self.assertEqual(violations, [])

    def test_submission_scan_is_limited_to_skills_and_workflows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skills = root / "JCIIOT/src/robot_agent/skills"
            workflows = root / "JCIIOT/src/robot_agent/workflows"
            protected = root / "JCIIOT/src/robot_agent/environments"
            skills.mkdir(parents=True)
            workflows.mkdir(parents=True)
            protected.mkdir(parents=True)
            (skills / "bad.py").write_text(
                "def run(env):\n    env.sim.data.qpos[0] = 1\n",
                encoding="utf-8",
            )
            (workflows / "good.py").write_text(
                "def run(backend, action):\n    return backend.step(action)\n",
                encoding="utf-8",
            )
            (protected / "ignored.py").write_text(
                "def run(env):\n    env.sim.data.qpos[0] = 1\n",
                encoding="utf-8",
            )

            violations = scan_submission(root)

        self.assertEqual(len(violations), 1)
        self.assertTrue(violations[0].path.endswith("skills/bad.py"))

    def test_cli_writes_warning_report_but_exits_zero_without_hard_violation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skills = root / "JCIIOT/src/robot_agent/skills"
            (root / "JCIIOT/src/robot_agent/workflows").mkdir(parents=True)
            skills.mkdir(parents=True)
            (skills / "bad.py").write_text(
                "def run(env):\n    env.sim.data.qpos[0] = 1\n",
                encoding="utf-8",
            )
            output = root / "audit.json"

            exit_code = main(["--root", str(root), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["violation_count"], 1)
        self.assertEqual(report["hard_violation_count"], 0)
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(report["violations"][0]["rule"], "direct_qpos_write")

    def test_cli_exits_nonzero_for_object_pose_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skills = root / "JCIIOT/src/robot_agent/skills"
            (root / "JCIIOT/src/robot_agent/workflows").mkdir(parents=True)
            skills.mkdir(parents=True)
            (skills / "bad.py").write_text(
                "def run(env, object_qpos_addr):\n"
                "    env.sim.data.qpos[object_qpos_addr] = 1\n",
                encoding="utf-8",
            )
            output = root / "audit.json"

            exit_code = main(["--root", str(root), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["hard_violation_count"], 1)
        self.assertEqual(report["warning_count"], 0)
        self.assertEqual(report["violations"][0]["rule"], "object_qpos_write")


if __name__ == "__main__":
    unittest.main()
