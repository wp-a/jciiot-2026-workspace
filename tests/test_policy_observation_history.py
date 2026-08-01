import unittest

import numpy as np

from scripts.policy_observation_history import ObservationHistoryPolicy


class FakePolicy:
    def __init__(self):
        self.started = 0
        self.observations = []

    def start_episode(self):
        self.started += 1

    def __call__(self, *, ob):
        self.observations.append(ob)
        return np.zeros(20)


class ObservationHistoryPolicyTests(unittest.TestCase):
    def test_repeats_first_frame_then_maintains_sliding_history(self):
        base_policy = FakePolicy()
        policy = ObservationHistoryPolicy(base_policy, horizon=2)
        policy.start_episode()

        policy(ob={"state": np.array([1.0, 2.0])})
        policy(ob={"state": np.array([3.0, 4.0])})

        np.testing.assert_array_equal(
            base_policy.observations[0]["state"],
            np.array([[1.0, 2.0], [1.0, 2.0]]),
        )
        np.testing.assert_array_equal(
            base_policy.observations[1]["state"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        self.assertEqual(base_policy.started, 1)

    def test_start_episode_clears_previous_history(self):
        base_policy = FakePolicy()
        policy = ObservationHistoryPolicy(base_policy, horizon=2)
        policy.start_episode()
        policy(ob={"state": np.array([1.0])})
        policy.start_episode()
        policy(ob={"state": np.array([9.0])})

        np.testing.assert_array_equal(
            base_policy.observations[-1]["state"],
            np.array([[9.0], [9.0]]),
        )

    def test_rejects_nonpositive_horizon(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ObservationHistoryPolicy(FakePolicy(), horizon=0)


if __name__ == "__main__":
    unittest.main()
