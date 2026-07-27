import json
import subprocess
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OVERLAY = WORKSPACE / "submission"
MATERIALIZER = WORKSPACE / "scripts" / "materialize_submission.sh"
ALLOWED_PREFIXES = (
    "JCIIOT/src/robot_agent/skills/",
    "JCIIOT/src/robot_agent/workflows/",
)
ALLOWED_FILES = {"JCIIOT/knowledge/robot_params.json", "README.md"}


class SubmissionBoundaryTests(unittest.TestCase):
    def test_overlay_contains_only_allowed_official_paths(self):
        self.assertTrue(OVERLAY.is_dir(), "submission overlay is missing")
        files = [path for path in OVERLAY.rglob("*") if path.is_file()]
        self.assertTrue(files, "submission overlay is empty")

        violations = []
        for path in files:
            relative = path.relative_to(OVERLAY).as_posix()
            if relative in ALLOWED_FILES:
                continue
            if any(relative.startswith(prefix) for prefix in ALLOWED_PREFIXES):
                continue
            violations.append(relative)

        self.assertEqual(violations, [])

    def test_materializer_rejects_wrong_official_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official = root / "official"
            output = root / "candidate"
            official.mkdir()
            subprocess.run(
                ["git", "init", "--quiet", "--initial-branch=main", str(official)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(official), "config", "user.name", "Submission Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(official), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            (official / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(official), "add", "README.md"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(official), "commit", "--quiet", "-m", "fixture"],
                check=True,
            )

            result = subprocess.run(
                [
                    "bash",
                    str(MATERIALIZER),
                    "--workspace",
                    str(WORKSPACE),
                    "--official-root",
                    str(official),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("official commit mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_materializer_rejects_non_source_files_in_allowed_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            official = root / "official"
            overlay = root / "overlay"
            output = root / "candidate"
            (workspace / "config").mkdir(parents=True)
            official.mkdir()
            subprocess.run(
                ["git", "init", "--quiet", "--initial-branch=main", str(official)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(official), "config", "user.name", "Submission Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(official), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            (official / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(official), "add", "README.md"], check=True)
            subprocess.run(
                ["git", "-C", str(official), "commit", "--quiet", "-m", "fixture"],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(official), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (workspace / "config" / "upstream-lock.json").write_text(
                json.dumps({"repository": {"commit": commit}}),
                encoding="utf-8",
            )
            cache_file = (
                overlay
                / "JCIIOT"
                / "src"
                / "robot_agent"
                / "skills"
                / "__pycache__"
                / "cached.pyc"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"not source")

            result = subprocess.run(
                [
                    "bash",
                    str(MATERIALIZER),
                    "--workspace",
                    str(workspace),
                    "--official-root",
                    str(official),
                    "--overlay",
                    str(overlay),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("forbidden overlay path", result.stderr)
            self.assertFalse(output.exists())

    def test_locked_commit_is_full_sha(self):
        lock = json.loads(
            (WORKSPACE / "config" / "upstream-lock.json").read_text(encoding="utf-8")
        )
        commit = lock["repository"]["commit"]
        self.assertRegex(commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
