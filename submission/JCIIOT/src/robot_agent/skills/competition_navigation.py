"""Object-relative base pose planning and physical yaw control."""

from __future__ import annotations

import math


REFERENCE_BASE_DISTANCE = 0.941001


def grasp_aligned_base_pose(
    *,
    object_xy,
    right_site_xy,
    left_site_xy,
    station_center,
    station_approach,
    base_distance: float = REFERENCE_BASE_DISTANCE,
) -> dict:
    """Choose a collision-aware side perpendicular to the grasp-site axis."""
    axis_x = float(right_site_xy[0]) - float(left_site_xy[0])
    axis_y = float(right_site_xy[1]) - float(left_site_xy[1])
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm < 1e-6:
        raise ValueError("grasp sites must have distinct planar positions")
    axis = (axis_x / axis_norm, axis_y / axis_norm)

    clockwise = (axis[1], -axis[0])
    guidance = (
        float(station_approach[0]) - float(station_center[0])
        + float(object_xy[0]) - float(station_center[0]),
        float(station_approach[1]) - float(station_center[1])
        + float(object_xy[1]) - float(station_center[1]),
    )
    if clockwise[0] * guidance[0] + clockwise[1] * guidance[1] >= 0.0:
        base_direction = clockwise
    else:
        base_direction = (-clockwise[0], -clockwise[1])

    base_xy = [
        float(object_xy[0]) + float(base_distance) * base_direction[0],
        float(object_xy[1]) + float(base_distance) * base_direction[1],
    ]
    yaw = math.atan2(-base_direction[1], -base_direction[0])
    expected_right_to_left_axis = (math.sin(yaw), -math.cos(yaw))
    swap_arm_targets = (
        expected_right_to_left_axis[0] * axis[0]
        + expected_right_to_left_axis[1] * axis[1]
    ) < 0.0
    return {
        "base_xy": base_xy,
        "yaw": yaw,
        "swap_arm_targets": swap_arm_targets,
    }


def orient_base(
    backend,
    target_yaw: float,
    *,
    tolerance: float = 0.02,
    max_steps: int = 180,
    gain: float = 1.5,
    max_angular: float = 0.35,
) -> bool:
    """Rotate through the official mobile-base action controller."""
    import numpy as np

    from robot_agent.environments.robosuite_backend import (
        _build_base_action,
        _capture_upper_body_posture,
        _restore_upper_body_posture,
        _shortest_angle,
    )

    raw_env = backend.env
    robot = raw_env.robots[0]
    posture = _capture_upper_body_posture(raw_env, robot)
    recorder = getattr(backend, "_record_trajectory_frame", None)
    reached = False

    for _ in range(int(max_steps)):
        _, yaw = backend.get_base_pose()
        error = _shortest_angle(float(target_yaw) - float(yaw))
        if abs(error) <= float(tolerance):
            reached = True
            break
        angular = float(np.clip(float(gain) * error, -max_angular, max_angular))
        action = _build_base_action(robot, 0.0, 0.0, angular)
        _, _, _, info = raw_env.step(action)
        _restore_upper_body_posture(raw_env, posture)
        if callable(recorder):
            recorder(_env=raw_env)
        if bool((info or {}).get("has_judge_collision", False)):
            return False

    stop_action = _build_base_action(robot, 0.0, 0.0, 0.0)
    for _ in range(5):
        raw_env.step(stop_action)
        _restore_upper_body_posture(raw_env, posture)
        if callable(recorder):
            recorder(_env=raw_env)
    return reached
