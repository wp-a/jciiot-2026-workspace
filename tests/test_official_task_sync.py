import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_official_tasks import ValidationError, validate_workspace


class OfficialTaskSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.vendor = self.workspace / "vendor" / "JCIIOT2026"
        self.map_dir = (
            self.vendor
            / "JCIIOT"
            / "robosuite"
            / "robosuite"
            / "environments"
            / "factory_sorting"
            / "generated_maps"
        )
        (self.workspace / "config").mkdir(parents=True)
        (self.vendor / "JCIIOT" / "knowledge").mkdir(parents=True)
        self.map_dir.mkdir(parents=True)

        self.commit = "a" * 40
        self.official_task = {
            "level": "L3",
            "scene_prefix": "factory_sorting_5_example",
            "env_name": "FactorySorting5_EXAMPLE",
            "source": "aux_input_1",
            "target": "output_5",
            "object": ["blue_tote_far", "blue_tote_near"],
            "max_score": 20,
        }
        self.workspace_task = {
            "level": "L3",
            "scene": "FactorySorting5_EXAMPLE",
            "source": "aux_input_1",
            "source_center_xy": [0.144, 8.473],
            "target": "output_5",
            "target_center_xy": [4.872, -7.261],
            "objects": ["blue_tote_far", "blue_tote_near"],
            "max_score": 20,
        }
        self.semantic_map = {
            "input_ports": {
                "aux_input_1": {"center": [0.144, 8.473]},
            },
            "output_ports": {
                "output_5": {"center": [4.872, -7.261]},
            },
        }
        self._write_fixture()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def _write_fixture(self):
        self._write_json(
            self.workspace / "config" / "upstream-lock.json",
            {
                "repository": {
                    "commit": self.commit,
                    "local_path": "vendor/JCIIOT2026",
                }
            },
        )
        self._write_json(
            self.workspace / "config" / "tasks.json",
            {
                "official_commit": self.commit,
                "tasks": [self.workspace_task],
            },
        )
        self._write_json(
            self.vendor / "JCIIOT" / "knowledge" / "task_config.json",
            {"tasks": [self.official_task]},
        )
        map_name = f"{self.official_task['scene_prefix']}_scene_regenerated_semantic_map.json"
        self._write_json(self.map_dir / map_name, self.semantic_map)

    def test_matching_workspace_passes(self):
        self.assertEqual(validate_workspace(self.workspace), ["L3"])

    def test_stale_source_fails_with_field_name(self):
        self.workspace_task["source"] = "input_6"
        self._write_fixture()

        with self.assertRaisesRegex(ValidationError, r"L3 source"):
            validate_workspace(self.workspace)

    def test_stale_target_fails_with_field_name(self):
        self.workspace_task["target"] = "output_6"
        self._write_fixture()

        with self.assertRaisesRegex(ValidationError, r"L3 target"):
            validate_workspace(self.workspace)

    def test_stale_coordinates_fail_with_field_name(self):
        self.workspace_task["source_center_xy"] = [11.937, 3.932]
        self._write_fixture()

        with self.assertRaisesRegex(ValidationError, r"L3 source_center_xy"):
            validate_workspace(self.workspace)

    def test_workspace_and_lock_commit_must_match(self):
        self.commit = "b" * 40
        self._write_json(
            self.workspace / "config" / "upstream-lock.json",
            {
                "repository": {
                    "commit": self.commit,
                    "local_path": "vendor/JCIIOT2026",
                }
            },
        )

        with self.assertRaisesRegex(ValidationError, "official_commit"):
            validate_workspace(self.workspace)


if __name__ == "__main__":
    unittest.main()
