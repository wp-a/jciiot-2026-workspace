import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from scripts import collect_native_grasp_demo as module


class FakeSim:
    def __init__(self, env):
        self.env = env

    def get_state(self):
        return SimpleNamespace(
            flatten=lambda: np.array(
                [self.env.step_index, self.env.step_index + 0.5],
                dtype=float,
            )
        )

    def render(self, *, camera_name, width, height, depth):
        del camera_name, depth
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = self.env.step_index
        image[0, 0, 1] = 255
        return image


class FakeEnv:
    def __init__(self):
        self.step_index = 0
        self.sim = FakeSim(self)

    def _get_observations(self, *, force_update):
        self.assert_force_update = force_update
        return {
            "eef": np.array([self.step_index], dtype=float),
            "object": np.array([2.0, 3.0], dtype=float),
        }

    def step(self, action):
        self.step_index += 1
        return {"step": self.step_index, "action": np.asarray(action).copy()}


def valid_demo(samples=3, action_dim=20):
    actions = np.zeros((samples, action_dim), dtype=float)
    actions[:, 0] = np.linspace(0.1, 0.3, samples)
    actions[-1, -1] = 1.0
    images = np.zeros((samples, 4, 4, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = np.arange(samples, dtype=np.uint8)
    images[:, 1, 1, 1] = 255
    return {
        "actions": actions,
        "states": np.arange(samples * 2, dtype=float).reshape(samples, 2),
        "obs": {
            "eef": np.arange(samples, dtype=float).reshape(samples, 1),
            "object": np.ones((samples, 2), dtype=float),
            "robot0_robotview_image": images,
        },
        "start_event": {"name": "grasp_start", "object_name": "box"},
        "end_event": {"name": "grasp_end", "success": True},
    }


class GraspWindowRecorderTests(unittest.TestCase):
    def test_records_pre_action_observations_only_inside_grasp_window(self):
        env = FakeEnv()
        recorder = module.GraspWindowRecorder(
            env,
            observation_keys=("eef", "object"),
            image_size=4,
        )
        wrapped_step = recorder.wrap_step(env.step)

        wrapped_step(np.full(20, -1.0))
        recorder.handle_event("grasp_start", object_name="box")
        wrapped_step(np.full(20, 0.25))
        wrapped_step(np.full(20, 0.50))
        recorder.handle_event("grasp_end", success=True, lift_success=True)
        wrapped_step(np.full(20, 1.0))

        demo = recorder.as_demo()

        self.assertEqual(demo["actions"].shape, (2, 20))
        np.testing.assert_allclose(demo["actions"][:, 0], [0.25, 0.50])
        np.testing.assert_allclose(demo["obs"]["eef"][:, 0], [1.0, 2.0])
        np.testing.assert_allclose(demo["states"][:, 0], [1.0, 2.0])
        self.assertEqual(
            demo["obs"]["robot0_robotview_image"].shape,
            (2, 4, 4, 3),
        )
        self.assertEqual(demo["start_event"]["object_name"], "box")
        self.assertTrue(demo["end_event"]["success"])

    def test_rejects_second_grasp_window_in_one_demo(self):
        recorder = module.GraspWindowRecorder(
            FakeEnv(),
            observation_keys=("eef", "object"),
            image_size=4,
        )
        recorder.handle_event("grasp_start", object_name="box")
        recorder.handle_event("grasp_end", success=False)

        with self.assertRaisesRegex(RuntimeError, "one grasp window"):
            recorder.handle_event("grasp_start", object_name="box")

    def test_recorder_restore_does_not_reopen_closed_backend(self):
        env = FakeEnv()

        class ClosingBackend:
            def __init__(self):
                self._env = env
                self._mark_trajectory_event = lambda name, **details: (
                    name,
                    details,
                )

            @property
            def env(self):
                if self._env is None:
                    raise RuntimeError("backend is closed")
                return self._env

        backend = ClosingBackend()
        original_step = env.step
        original_marker = backend._mark_trajectory_event
        recorder = module.GraspWindowRecorder(
            env,
            observation_keys=("eef", "object"),
            image_size=4,
        )
        restore = module._install_recorder(backend, recorder)

        backend._env = None
        restore()

        self.assertEqual(env.step, original_step)
        self.assertEqual(backend._mark_trajectory_event, original_marker)


class NativeDemoValidationTests(unittest.TestCase):
    def test_accepts_aligned_twenty_dimensional_demo(self):
        summary = module.validate_demo(
            valid_demo(),
            required_observation_keys=("eef", "object"),
            minimum_samples=3,
        )

        self.assertEqual(summary["action_dim"], 20)
        self.assertEqual(summary["samples"], 3)
        self.assertGreater(summary["image_std"], 0.0)
        self.assertTrue(summary["grasp_success"])

    def test_rejects_incompatible_action_width(self):
        with self.assertRaisesRegex(ValueError, "action dimension"):
            module.validate_demo(
                valid_demo(action_dim=10),
                required_observation_keys=("eef", "object"),
                minimum_samples=3,
            )

    def test_rejects_observation_action_length_mismatch(self):
        demo = valid_demo()
        demo["obs"]["eef"] = demo["obs"]["eef"][:-1]

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            module.validate_demo(
                demo,
                required_observation_keys=("eef", "object"),
                minimum_samples=3,
            )


@unittest.skipIf(h5py is None, "h5py is exercised in the pinned server runtime")
class NativeDemoHDF5Tests(unittest.TestCase):
    def test_writes_robomimic_layout_and_trace_metadata(self):
        demo = valid_demo()
        metadata = {
            "official_commit": "0dcdddf",
            "workspace_commit": "ee59165",
            "seed": 20260840,
            "perturbation": {"tier": "nominal"},
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.hdf5"
            module.write_robomimic_hdf5(output, demo, metadata=metadata)

            with h5py.File(output, "r") as handle:
                data = handle["data"]
                episode = data["demo_0"]
                self.assertEqual(episode["actions"].shape, (3, 20))
                self.assertEqual(episode["states"].shape, (3, 2))
                self.assertEqual(episode["obs/eef"].shape, (3, 1))
                self.assertEqual(
                    episode["obs/robot0_robotview_image"].shape,
                    (3, 4, 4, 3),
                )
                self.assertEqual(episode.attrs["num_samples"], 3)
                self.assertEqual(data.attrs["total"], 3)
                self.assertEqual(data.attrs["num_demos"], 1)
                self.assertEqual(
                    json.loads(data.attrs["collection_metadata"]),
                    metadata,
                )


if __name__ == "__main__":
    unittest.main()
