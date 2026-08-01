import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from scripts import merge_native_grasp_demos as module


def write_source(root: Path, run_name: str, *, samples: int = 3, action_dim: int = 20):
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    path = run_dir / "grasp-demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["total"] = samples
        data.attrs["num_demos"] = 1
        data.attrs["env_args"] = json.dumps(
            {"env_name": "FactorySorting1_3FO3ERFHISEM", "type": 1, "env_kwargs": {}}
        )
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = samples
        actions = np.zeros((samples, action_dim), dtype=np.float32)
        actions[:, 0] = 0.25
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=np.ones((samples, 4), dtype=np.float64))
        demo.create_dataset("rewards", data=np.zeros(samples, dtype=np.float32))
        demo.create_dataset("dones", data=np.array([0] * (samples - 1) + [1], dtype=np.uint8))
        obs = demo.create_group("obs")
        obs.create_dataset("eef", data=np.ones((samples, 3), dtype=np.float32))
        obs.create_dataset(
            "robot0_robotview_image",
            data=np.ones((samples, 4, 4, 3), dtype=np.uint8),
        )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(h5py is None, "h5py is exercised in the pinned server runtime")
class MergeNativeGraspDemosTests(unittest.TestCase):
    def test_merges_registered_splits_and_excludes_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, train_sha = write_source(root, "train-run", samples=3)
            valid_path, valid_sha = write_source(root, "valid-run", samples=4)
            heldout_path, heldout_sha = write_source(root, "heldout-run", samples=5)
            excluded_path, _ = write_source(root, "duplicate-run", samples=3)
            split = {
                "splits": {
                    "train": ["train-run"],
                    "valid": ["valid-run"],
                    "heldout": ["heldout-run"],
                    "excluded_duplicate": ["duplicate-run"],
                }
            }
            output = root / "merged.hdf5"

            summary = module.merge_registered_demos(root, split, output)

            self.assertEqual(summary["demo_count"], 3)
            self.assertEqual(summary["total_samples"], 12)
            self.assertEqual(summary["excluded_runs"], ["duplicate-run"])
            with h5py.File(output, "r") as handle:
                self.assertEqual(sorted(handle["data"].keys()), ["demo_0", "demo_1", "demo_2"])
                self.assertEqual(handle["data"].attrs["total"], 12)
                self.assertEqual(handle["data"].attrs["num_demos"], 3)
                self.assertEqual(handle["mask/train"][:].tolist(), [b"demo_0"])
                self.assertEqual(handle["mask/valid"][:].tolist(), [b"demo_1"])
                self.assertEqual(handle["mask/heldout"][:].tolist(), [b"demo_2"])
                expected = [
                    ("train-run", "train", train_sha, train_path),
                    ("valid-run", "valid", valid_sha, valid_path),
                    ("heldout-run", "heldout", heldout_sha, heldout_path),
                ]
                for index, (run, split_name, digest, source) in enumerate(expected):
                    demo = handle[f"data/demo_{index}"]
                    self.assertEqual(demo.attrs["source_run"], run)
                    self.assertEqual(demo.attrs["split"], split_name)
                    self.assertEqual(demo.attrs["source_sha256"], digest)
                    self.assertEqual(demo.attrs["source_path"], str(source.resolve()))

    def test_rejects_inconsistent_action_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root, "train-run", action_dim=20)
            write_source(root, "valid-run", action_dim=10)
            split = {
                "splits": {
                    "train": ["train-run"],
                    "valid": ["valid-run"],
                    "heldout": [],
                    "excluded_duplicate": [],
                }
            }

            with self.assertRaisesRegex(ValueError, "action dimension"):
                module.merge_registered_demos(root, split, root / "merged.hdf5")


if __name__ == "__main__":
    unittest.main()
