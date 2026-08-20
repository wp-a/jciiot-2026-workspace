import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("h5py is required for HDF5 audit tests") from exc

from scripts.audit_physical_carry_hdf5 import audit_hdf5


EVENTS = [
    "grasp_start",
    "grasp_end",
    "transport_start",
    "transport_end",
    "place_end",
]


class PhysicalCarryHdf5AuditTests(unittest.TestCase):
    def _write_demo(
        self,
        data,
        name,
        *,
        seed,
        action_dim=20,
        state_dim=87,
        attachment_calls=0,
        object_pose_writes=0,
        collision_frames=0,
        min_lift_m=0.16,
        translation_m=0.62,
        bilateral_contact=True,
    ):
        samples = 5
        demo = data.create_group(name)
        demo.attrs["seed"] = seed
        demo.attrs["object_name"] = "sorting_object_0"
        demo.attrs["task_level"] = 1
        demo.create_dataset(
            "actions",
            data=np.zeros((samples, action_dim), dtype=np.float32),
        )
        obs = demo.create_group("obs")
        obs.create_dataset(
            "state",
            data=np.zeros((samples, state_dim), dtype=np.float32),
        )
        demo.create_dataset(
            "timestamps",
            data=np.arange(samples, dtype=np.float64) * 0.05,
        )
        demo.create_dataset("events", data=json.dumps(EVENTS))
        integrity = demo.create_group("integrity")
        integrity.create_dataset("attachment_calls", data=attachment_calls)
        integrity.create_dataset("object_pose_writes", data=object_pose_writes)
        integrity.create_dataset("collision_frames", data=collision_frames)
        integrity.create_dataset("min_lift_m", data=min_lift_m)
        integrity.create_dataset("true_object_translation_m", data=translation_m)
        integrity.create_dataset(
            "continuous_bilateral_contact",
            data=bilateral_contact,
        )

    def _write_fixture(self, path, **demo_overrides):
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps(
                {
                    "env_name": "FactorySorting1_TEST",
                    "env_kwargs": {"robots": ["Tiago"]},
                }
            )
            for index in range(3):
                overrides = demo_overrides if index == 0 else {}
                self._write_demo(
                    data,
                    f"demo_{index}",
                    seed=100 + index,
                    **overrides,
                )
            mask = handle.create_group("mask")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            mask.create_dataset("train", data=["demo_0"], dtype=string_dtype)
            mask.create_dataset("validation", data=["demo_1"], dtype=string_dtype)
            mask.create_dataset("heldout", data=["demo_2"], dtype=string_dtype)

    def test_accepts_complete_strict_physical_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            self._write_fixture(path)

            manifest = audit_hdf5(path)

        self.assertTrue(manifest["eligible"])
        self.assertEqual(manifest["demo_count"], 3)
        self.assertEqual(manifest["eligible_demo_count"], 3)
        self.assertEqual(manifest["action_dim"], 20)
        self.assertEqual(manifest["state_dim"], 87)
        self.assertEqual(manifest["failures"], [])

    def test_rejects_wrong_dimensions_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.hdf5"
            self._write_fixture(path, action_dim=19, state_dim=86)
            with h5py.File(path, "r+") as handle:
                handle["data/demo_1/actions"][2, 4] = np.nan

            manifest = audit_hdf5(path)

        self.assertFalse(manifest["eligible"])
        first = manifest["demos"][0]
        second = manifest["demos"][1]
        self.assertIn("action_dim:19", first["failures"])
        self.assertIn("state_dim:86", first["failures"])
        self.assertIn("actions_nonfinite", second["failures"])

    def test_rejects_shortcuts_but_keeps_low_lift_as_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            shortcut_path = Path(directory) / "shortcut.hdf5"
            recovery_path = Path(directory) / "recovery.hdf5"
            self._write_fixture(shortcut_path, attachment_calls=1)
            self._write_fixture(recovery_path, min_lift_m=0.08)

            shortcut = audit_hdf5(shortcut_path)
            recovery = audit_hdf5(recovery_path)

        self.assertEqual(shortcut["demos"][0]["classification"], "rejected")
        self.assertIn("attachment_calls", shortcut["demos"][0]["failures"])
        self.assertEqual(recovery["demos"][0]["classification"], "recovery")
        self.assertIn("min_lift_m", recovery["demos"][0]["failures"])

    def test_rejects_duplicate_seed_and_incomplete_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leakage.hdf5"
            self._write_fixture(path)
            with h5py.File(path, "r+") as handle:
                handle["data/demo_1"].attrs["seed"] = 100
                del handle["mask/heldout"]

            manifest = audit_hdf5(path)

        self.assertFalse(manifest["eligible"])
        self.assertIn("duplicate_seed:100", manifest["failures"])
        self.assertIn("missing_split:heldout", manifest["failures"])
        self.assertIn("unassigned_demo:demo_2", manifest["failures"])


if __name__ == "__main__":
    unittest.main()
