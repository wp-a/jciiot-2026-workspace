"""Object-relative base pose planning and physical yaw control."""

from __future__ import annotations

import math


REFERENCE_BASE_TO_GRASP_CENTER = 0.651001


def bounded_yaw_step(
    *,
    current_yaw: float,
    target_yaw: float,
    max_step: float,
) -> float:
    """Advance toward a target yaw through the shortest wrapped angle."""
    error = (float(target_yaw) - float(current_yaw) + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    delta = max(-float(max_step), min(float(max_step), error))
    return (float(current_yaw) + delta + math.pi) % (2.0 * math.pi) - math.pi


def grasp_orientation_from_base(
    *,
    base_xy,
    right_site_xy,
    left_site_xy,
) -> dict:
    """Face the grasp center and choose the closest arm-to-site assignment."""
    grasp_center = (
        (float(right_site_xy[0]) + float(left_site_xy[0])) / 2.0,
        (float(right_site_xy[1]) + float(left_site_xy[1])) / 2.0,
    )
    yaw = math.atan2(
        grasp_center[1] - float(base_xy[1]),
        grasp_center[0] - float(base_xy[0]),
    )
    axis_x = float(right_site_xy[0]) - float(left_site_xy[0])
    axis_y = float(right_site_xy[1]) - float(left_site_xy[1])
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm < 1e-6:
        raise ValueError("grasp sites must have distinct planar positions")
    axis = (axis_x / axis_norm, axis_y / axis_norm)
    expected_right_to_left_axis = (math.sin(yaw), -math.cos(yaw))
    swap_arm_targets = (
        expected_right_to_left_axis[0] * axis[0]
        + expected_right_to_left_axis[1] * axis[1]
    ) < 0.0
    return {
        "yaw": yaw,
        "swap_arm_targets": swap_arm_targets,
    }


def grasp_aligned_base_pose(
    *,
    object_xy,
    right_site_xy,
    left_site_xy,
    station_center,
    station_approach,
    base_standoff: float = REFERENCE_BASE_TO_GRASP_CENTER,
) -> dict:
    """Choose a collision-aware side perpendicular to the grasp-site axis."""
    axis_x = float(right_site_xy[0]) - float(left_site_xy[0])
    axis_y = float(right_site_xy[1]) - float(left_site_xy[1])
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm < 1e-6:
        raise ValueError("grasp sites must have distinct planar positions")
    axis = (axis_x / axis_norm, axis_y / axis_norm)

    grasp_center = (
        (float(right_site_xy[0]) + float(left_site_xy[0])) / 2.0,
        (float(right_site_xy[1]) + float(left_site_xy[1])) / 2.0,
    )
    face = (
        grasp_center[0] - float(object_xy[0]),
        grasp_center[1] - float(object_xy[1]),
    )
    face_norm = math.hypot(face[0], face[1])
    if face_norm >= 0.05:
        base_direction = (face[0] / face_norm, face[1] / face_norm)
    else:
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
        grasp_center[0] + float(base_standoff) * base_direction[0],
        grasp_center[1] + float(base_standoff) * base_direction[1],
    ]
    projection = (
        (float(station_approach[0]) - base_xy[0]) * axis[0]
        + (float(station_approach[1]) - base_xy[1]) * axis[1]
    )
    staging_xy = [
        base_xy[0] + projection * axis[0],
        base_xy[1] + projection * axis[1],
    ]
    orientation = grasp_orientation_from_base(
        base_xy=base_xy,
        right_site_xy=right_site_xy,
        left_site_xy=left_site_xy,
    )
    return {
        "base_xy": base_xy,
        "staging_xy": staging_xy,
        "grasp_center_xy": list(grasp_center),
        "right_site_xy": [float(right_site_xy[0]), float(right_site_xy[1])],
        "left_site_xy": [float(left_site_xy[0]), float(left_site_xy[1])],
        "yaw": orientation["yaw"],
        "swap_arm_targets": orientation["swap_arm_targets"],
    }


def select_grasp_candidate(candidates, *, station_approach) -> str:
    """Prefer an approach pose clear of other task objects, then the station path."""
    entries = list(candidates)
    if not entries:
        raise ValueError("candidates must not be empty")

    def score(entry):
        base_xy = entry["base_xy"]
        other_positions = [
            other["object_xy"]
            for other in entries
            if other["name"] != entry["name"]
        ]
        if other_positions:
            clearance = min(
                math.hypot(
                    float(base_xy[0]) - float(position[0]),
                    float(base_xy[1]) - float(position[1]),
                )
                for position in other_positions
            )
        else:
            clearance = math.inf
        approach_distance = math.hypot(
            float(base_xy[0]) - float(station_approach[0]),
            float(base_xy[1]) - float(station_approach[1]),
        )
        return clearance, -approach_distance

    return str(max(entries, key=score)["name"])


def orient_base(
    backend,
    target_yaw: float,
    *,
    tolerance: float = 0.02,
    max_steps: int = 180,
    max_yaw_step: float = 0.025,
) -> bool:
    """Rotate smoothly using the official direct-navigation yaw helper."""
    import numpy as np

    from robot_agent.environments.robosuite_backend import (
        _capture_upper_body_posture,
        _restore_upper_body_posture,
        _shortest_angle,
        _set_base_world_yaw_direct,
    )

    raw_env = backend.env
    robot = raw_env.robots[0]
    posture = _capture_upper_body_posture(raw_env, robot)
    recorder = getattr(backend, "_record_trajectory_frame", None)
    idle_action = np.zeros_like(raw_env.action_spec[0])
    reached = False

    for _ in range(int(max_steps)):
        _, yaw = backend.get_base_pose()
        error = _shortest_angle(float(target_yaw) - float(yaw))
        if abs(error) <= float(tolerance):
            reached = True
            break
        next_yaw = bounded_yaw_step(
            current_yaw=yaw,
            target_yaw=target_yaw,
            max_step=max_yaw_step,
        )
        _set_base_world_yaw_direct(raw_env, robot, next_yaw)
        _, _, _, info = raw_env.step(idle_action)
        _restore_upper_body_posture(raw_env, posture)
        if callable(recorder):
            recorder(_env=raw_env)
        if bool((info or {}).get("has_judge_collision", False)):
            return False

    for _ in range(5):
        raw_env.step(idle_action)
        _restore_upper_body_posture(raw_env, posture)
        if callable(recorder):
            recorder(_env=raw_env)
    return reached
