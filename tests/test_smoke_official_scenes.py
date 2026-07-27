import unittest
from types import SimpleNamespace

from scripts.smoke_official_scenes import build_summary, run_scene


class FakeAction:
    shape = (3,)


class SuccessfulEnv:
    def __init__(self):
        self.action_spec = (FakeAction(), FakeAction())
        self.sim = SimpleNamespace(
            model=SimpleNamespace(nq=31, nv=30, ngeom=412, ncam=5)
        )
        self.closed = False
        self.step_calls = 0

    def reset(self):
        return {"robot": "ready"}

    def step(self, action):
        self.step_calls += 1
        return {}, 0.0, False, {}

    def close(self):
        self.closed = True


class ResetFailureEnv(SuccessfulEnv):
    def reset(self):
        raise ValueError("broken scene asset")


class SceneSmokeTests(unittest.TestCase):
    def test_success_record_contains_dimensions_and_timing(self):
        env = SuccessfulEnv()

        result = run_scene(
            "FactorySorting1_EXAMPLE",
            lambda **kwargs: env,
            lambda low: FakeAction(),
            steps=2,
            seed=20260727,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["stage"], "complete")
        self.assertEqual(result["action_shape"], [3])
        self.assertEqual(result["model"], {"nq": 31, "nv": 30, "ngeom": 412, "ncam": 5})
        self.assertEqual(result["steps_completed"], 2)
        self.assertGreaterEqual(result["duration_s"], 0.0)
        self.assertEqual(env.step_calls, 2)
        self.assertTrue(env.closed)

    def test_failure_record_preserves_stage_and_exception(self):
        env = ResetFailureEnv()

        result = run_scene(
            "FactorySorting5_EXAMPLE",
            lambda **kwargs: env,
            lambda low: FakeAction(),
            steps=1,
            seed=20260727,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "reset")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertIn("broken scene asset", result["error"])
        self.assertTrue(env.closed)

    def test_any_scene_failure_returns_nonzero_summary(self):
        summary = build_summary(
            [
                {"scene": "L1", "success": True},
                {"scene": "L3", "success": False},
            ]
        )

        self.assertFalse(summary["success"])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failed_scenes"], ["L3"])


if __name__ == "__main__":
    unittest.main()
