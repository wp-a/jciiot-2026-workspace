import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("h5py is required for HDF5 inspector tests") from exc

from scripts.inspect_robomimic_hdf5 import (
    inspect_hdf5,
    write_json_atomic,
)


class RobomimicHdf5InspectorTests(unittest.TestCase):
    def _write_fixture(self, path: Path, *, action_dims=(20, 20)) -> None:
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps(
                {
                    "env_name": "FactorySorting1_TEST",
                    "env_kwargs": {"robots": ["Tiago"]},
                }
            )
            data.attrs["env_info"] = json.dumps({"camera": "robot0_robotview"})
            for index, (samples, action_dim) in enumerate(
                zip((3, 4), action_dims),
                start=1,
            ):
                demo = data.create_group(f"demo_{index}")
                demo.create_dataset(
                    "actions",
                    data=np.zeros((samples, action_dim), dtype=np.float32),
                )
                demo.create_dataset(
                    "states",
                    data=np.zeros((samples, 42), dtype=np.float64),
                )
                obs = demo.create_group("obs")
                obs.create_dataset(
                    "robot0_left_eef_pos",
                    data=np.zeros((samples, 3), dtype=np.float32),
                )
                obs.create_dataset(
                    "robot0_robotview_image",
                    data=np.zeros((samples, 8, 8, 3), dtype=np.uint8),
                )

    def test_inspects_environment_demos_actions_and_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.hdf5"
            self._write_fixture(path)

            summary = inspect_hdf5(path)

        self.assertTrue(summary["materialized"])
        self.assertEqual(summary["env_args"]["env_name"], "FactorySorting1_TEST")
        self.assertEqual(summary["demo_count"], 2)
        self.assertEqual(summary["total_samples"], 7)
        self.assertEqual(summary["action_dim"], 20)
        self.assertEqual(
            summary["observation_keys"],
            ["robot0_left_eef_pos", "robot0_robotview_image"],
        )
        self.assertEqual(summary["compatibility"]["classification"], "partially-reusable")

    def test_rejects_git_lfs_pointer_without_opening_it_as_hdf5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pointer.hdf5"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "a" * 64 + "\n"
                "size 591069600\n",
                encoding="ascii",
            )

            summary = inspect_hdf5(path)

        self.assertFalse(summary["materialized"])
        self.assertEqual(summary["compatibility"]["classification"], "lfs-pointer")
        self.assertEqual(summary["lfs_size"], 591069600)

    def test_inconsistent_action_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.hdf5"
            self._write_fixture(path, action_dims=(20, 19))

            with self.assertRaisesRegex(ValueError, "inconsistent action dimensions"):
                inspect_hdf5(path)

    def test_atomic_json_output_is_deterministic_and_has_newline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            value = {"z": 1, "a": {"b": 2}}

            write_json_atomic(path, value)
            first = path.read_bytes()
            write_json_atomic(path, value)
            second = path.read_bytes()

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(json.loads(first), value)


if __name__ == "__main__":
    unittest.main()
