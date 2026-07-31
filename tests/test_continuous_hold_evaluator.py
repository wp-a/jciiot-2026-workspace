import importlib
import math
import unittest


OBJECT_NAME = "blue_tote"


def _frame(*, base_xy, yaw, relative_xy, held=True):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    world_offset = [
        cosine * relative_xy[0] - sine * relative_xy[1],
        sine * relative_xy[0] + cosine * relative_xy[1],
    ]
    frame = {
        "base_pose": {
            "position": [base_xy[0], base_xy[1], 0.0],
            "orientation_xyzw": [
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
            ],
        },
        "object_positions": {
            OBJECT_NAME: [
                base_xy[0] + world_offset[0],
                base_xy[1] + world_offset[1],
                1.3,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        },
    }
    if held:
        frame["held_object"] = OBJECT_NAME
    return frame


def _trajectory(relative_offsets):
    frames = [
        _frame(
            base_xy=[float(index), -0.25 * index],
            yaw=0.25 * index,
            relative_xy=offset,
        )
        for index, offset in enumerate(relative_offsets)
    ]
    return {
        "events": [
            {
                "name": "transport_attachment_enabled",
                "frame": 0,
                "object_name": OBJECT_NAME,
            }
        ],
        "frames": frames,
    }


class ContinuousHoldEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.module = importlib.import_module(
                "scripts.evaluate_continuous_hold"
            )
        except ModuleNotFoundError as exc:
            raise AssertionError(
                "continuous-hold evaluator module must exist"
            ) from exc

    def test_accepts_rigid_gripper_relative_transport(self):
        report = self.module.evaluate_continuous_hold(
            _trajectory([[0.95, -0.02]] * 4)
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["held_frames"], 4)
        self.assertEqual(report["frames_below_min_distance"], 0)
        self.assertGreater(report["min_object_base_distance_m"], 0.94)
        self.assertLess(report["max_relative_xy_range_m"], 1e-9)

    def test_rejects_one_frame_teleported_to_base_center(self):
        report = self.module.evaluate_continuous_hold(
            _trajectory([[0.95, -0.02], [0.0, 0.0], [0.95, -0.02]])
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["frames_below_min_distance"], 1)
        self.assertAlmostEqual(report["min_object_base_distance_m"], 0.0)

    def test_rejects_more_than_twenty_mm_relative_drift(self):
        report = self.module.evaluate_continuous_hold(
            _trajectory([[0.95, -0.02], [0.95, 0.001]])
        )

        self.assertFalse(report["passed"])
        self.assertGreater(report["max_relative_xy_range_m"], 0.02)


if __name__ == "__main__":
    unittest.main()
