import unittest
import random

import numpy as np

from scripts import run_bc_rnn_closed_loop as module


class FakePolicy:
    def __init__(self, actions):
        self.actions = [np.asarray(action, dtype=float) for action in actions]
        self.index = 0
        self.started = 0

    def start_episode(self):
        self.index = 0
        self.started += 1

    def __call__(self, *, ob):
        del ob
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action.copy()


class PolicyWindowTests(unittest.TestCase):
    def test_stops_after_stable_bilateral_contact_and_lift(self):
        policy = FakePolicy([np.full(20, 0.5)] * 5)
        state = {"step": 0}
        recorded = []

        def step_fn(action):
            state["step"] += 1
            recorded.append(np.asarray(action).copy())
            return {}, 0.0, False, {"has_judge_collision": False}

        result = module.execute_policy_window(
            policy=policy,
            observation_fn=lambda: {"eef": np.array([state["step"]])},
            step_fn=step_fn,
            contact_fn=lambda: {
                "right": state["step"] >= 2,
                "left": state["step"] >= 2,
            },
            object_z_fn=lambda: 1.0 + 0.05 * state["step"],
            record_fn=lambda: None,
            max_steps=5,
            required_lift_m=0.10,
            stable_steps=2,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["stable_steps"], 2)
        self.assertAlmostEqual(result["lift_m"], 0.15)
        self.assertEqual(policy.started, 1)
        self.assertEqual(len(recorded), 3)

    def test_aborts_on_collision(self):
        policy = FakePolicy([np.zeros(20)])

        result = module.execute_policy_window(
            policy=policy,
            observation_fn=lambda: {"eef": np.zeros(1)},
            step_fn=lambda action: (
                {},
                0.0,
                False,
                {"has_judge_collision": True},
            ),
            contact_fn=lambda: {"right": False, "left": False},
            object_z_fn=lambda: 1.0,
            record_fn=lambda: None,
            max_steps=5,
            required_lift_m=0.10,
            stable_steps=2,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "collision")
        self.assertEqual(result["steps"], 1)

    def test_rejects_wrong_action_dimension_before_step(self):
        policy = FakePolicy([np.zeros(10)])
        calls = []

        result = module.execute_policy_window(
            policy=policy,
            observation_fn=lambda: {"eef": np.zeros(1)},
            step_fn=lambda action: calls.append(action),
            contact_fn=lambda: {"right": False, "left": False},
            object_z_fn=lambda: 1.0,
            record_fn=lambda: None,
            max_steps=5,
            required_lift_m=0.10,
            stable_steps=2,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "action_shape")
        self.assertEqual(calls, [])


class PolicyMetadataTests(unittest.TestCase):
    def test_result_key_identifies_supported_policy_method(self):
        self.assertEqual(module.policy_result_key("bc_rnn_lowdim"), "bc_rnn")
        self.assertEqual(
            module.policy_result_key("diffusion_policy_lowdim"),
            "diffusion_policy",
        )

    def test_sampling_seed_controls_python_numpy_and_torch(self):
        class FakeCuda:
            def __init__(self):
                self.seeds = []

            def is_available(self):
                return True

            def manual_seed_all(self, seed):
                self.seeds.append(seed)

        class FakeTorch:
            def __init__(self):
                self.seeds = []
                self.cuda = FakeCuda()

            def manual_seed(self, seed):
                self.seeds.append(seed)

        fake_torch = FakeTorch()
        module.seed_policy_sampling(20260880, fake_torch)
        first = (random.random(), float(np.random.random()))
        module.seed_policy_sampling(20260880, fake_torch)
        second = (random.random(), float(np.random.random()))

        self.assertEqual(first, second)
        self.assertEqual(fake_torch.seeds, [20260880, 20260880])
        self.assertEqual(fake_torch.cuda.seeds, [20260880, 20260880])


if __name__ == "__main__":
    unittest.main()
