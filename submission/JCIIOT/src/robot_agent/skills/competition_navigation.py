"""Object-relative base pose planning and physical yaw control."""

from __future__ import annotations

import math

import numpy as np


REFERENCE_BASE_TO_GRASP_CENTER = 0.651001
BLUE_TOTE_STATION_AXIS_STANDOFF = 0.78
SAFE_GRASP_YAW_CORRECTION = 0.15
PRECISE_GRASP_BASE_TOLERANCE = 0.04
PRECISE_GRASP_BASE_MAX_STEPS = 120


def station_axis_standoff_for_object(object_name: str) -> float | None:
    """Return the measured collision-free station-axis standoff when needed."""
    name = str(object_name).lower()
    if name == "green_tote_b01_upper":
        return REFERENCE_BASE_TO_GRASP_CENTER
    if "blue_tote_b01" in name:
        return BLUE_TOTE_STATION_AXIS_STANDOFF
    return None


def dominant_cardinal_axis(axis, *, max_minor_ratio: float = 0.10) -> np.ndarray:
    """Snap a nearly cardinal semantic axis to remove small map skew."""
    direction = np.asarray(axis, dtype=float).reshape(2)
    if not np.all(np.isfinite(direction)):
        raise ValueError("axis must contain finite coordinates")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("axis must be nonzero")
    direction /= norm
    magnitudes = np.abs(direction)
    dominant = int(np.argmax(magnitudes))
    minor = 1 - dominant
    if magnitudes[minor] <= float(max_minor_ratio) * magnitudes[dominant]:
        snapped = np.zeros(2, dtype=float)
        snapped[dominant] = math.copysign(1.0, direction[dominant])
        return snapped
    return direction


def align_base_for_grasp(
    backend,
    target_xy,
    *,
    tolerance: float = PRECISE_GRASP_BASE_TOLERANCE,
    max_steps: int = PRECISE_GRASP_BASE_MAX_STEPS,
) -> bool:
    """Finish a coarse approach at a collision-checked grasp-side pose."""
    goal = np.asarray(target_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(goal)):
        raise ValueError("target_xy must contain finite coordinates")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be at least 1")

    current_xy, _ = backend.get_base_pose()
    if float(np.linalg.norm(np.asarray(current_xy, dtype=float) - goal)) <= tolerance:
        return True
    reached = bool(
        backend.follow_path(
            [goal.copy()],
            waypoint_tolerance=tolerance,
            max_steps=int(max_steps),
        )
    )
    final_xy, _ = backend.get_base_pose()
    return bool(
        reached
        and float(np.linalg.norm(np.asarray(final_xy, dtype=float) - goal))
        <= tolerance
    )


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


def station_side_grasp_pose(
    *,
    grasp_center_xy,
    right_site_xy,
    left_site_xy,
    station_center,
    station_approach,
    base_standoff: float = 1.0,
) -> dict:
    """Approach a wall-side object along the station's reachable axis."""
    center = (float(grasp_center_xy[0]), float(grasp_center_xy[1]))
    approach_axis = (
        float(station_approach[0]) - float(station_center[0]),
        float(station_approach[1]) - float(station_center[1]),
    )
    axis_norm = math.hypot(*approach_axis)
    if axis_norm < 1e-6:
        raise ValueError("station approach must differ from station center")
    direction = (
        approach_axis[0] / axis_norm,
        approach_axis[1] / axis_norm,
    )
    base_xy = [
        center[0] + float(base_standoff) * direction[0],
        center[1] + float(base_standoff) * direction[1],
    ]
    staging_distance = (
        (float(station_approach[0]) - base_xy[0]) * direction[0]
        + (float(station_approach[1]) - base_xy[1]) * direction[1]
    )
    staging_xy = [
        base_xy[0] + staging_distance * direction[0],
        base_xy[1] + staging_distance * direction[1],
    ]
    orientation = grasp_orientation_from_base(
        base_xy=base_xy,
        right_site_xy=right_site_xy,
        left_site_xy=left_site_xy,
    )
    return {
        "base_xy": base_xy,
        "staging_xy": staging_xy,
        "grasp_center_xy": list(center),
        "right_site_xy": [float(right_site_xy[0]), float(right_site_xy[1])],
        "left_site_xy": [float(left_site_xy[0]), float(left_site_xy[1])],
        "yaw": orientation["yaw"],
        "swap_arm_targets": orientation["swap_arm_targets"],
    }


def station_axis_grasp_pose(
    *,
    grasp_center_xy,
    right_site_xy,
    left_site_xy,
    station_center,
    station_approach,
    base_standoff: float = REFERENCE_BASE_TO_GRASP_CENTER,
    facing_xy=None,
) -> dict:
    """Approach a tote along the free station axis instead of its proxy."""
    center = np.asarray(grasp_center_xy, dtype=float).reshape(2)
    station_center = np.asarray(station_center, dtype=float).reshape(2)
    station_approach = np.asarray(station_approach, dtype=float).reshape(2)
    axis = station_approach - station_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        raise ValueError("station approach must differ from station center")
    axis = dominant_cardinal_axis(axis)

    base_xy = center + float(base_standoff) * axis
    staging_distance = max(
        float(base_standoff),
        float(np.dot(station_approach - center, axis)),
    )
    staging_xy = center + staging_distance * axis
    orientation = grasp_orientation_from_base(
        base_xy=base_xy,
        right_site_xy=right_site_xy,
        left_site_xy=left_site_xy,
    )
    result = {
        "base_xy": base_xy.tolist(),
        "staging_xy": staging_xy.tolist(),
        "grasp_center_xy": center.tolist(),
        "right_site_xy": np.asarray(right_site_xy, dtype=float).tolist(),
        "left_site_xy": np.asarray(left_site_xy, dtype=float).tolist(),
        "yaw": orientation["yaw"],
        "swap_arm_targets": orientation["swap_arm_targets"],
        "precise_alignment": True,
    }
    if facing_xy is not None:
        facing = np.asarray(facing_xy, dtype=float).reshape(2)
        direction = facing - base_xy
        if not np.all(np.isfinite(facing)) or float(np.linalg.norm(direction)) <= 1e-9:
            raise ValueError("facing_xy must be finite and differ from base_xy")
        result["yaw"] = math.atan2(direction[1], direction[0])
        result["orientation_target_xy"] = facing.tolist()
    return result


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
        _set_base_xy_direct,
        _set_base_world_yaw_direct,
    )

    raw_env = backend.env
    robot = raw_env.robots[0]
    anchor_xy, _ = backend.get_base_pose()
    anchor_xy = np.asarray(anchor_xy, dtype=float).copy()
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
        _set_base_xy_direct(raw_env, robot, anchor_xy)
        _, _, _, info = raw_env.step(idle_action)
        _restore_upper_body_posture(raw_env, posture)
        _set_base_xy_direct(raw_env, robot, anchor_xy)
        if callable(recorder):
            recorder(_env=raw_env)
        if bool((info or {}).get("has_judge_collision", False)):
            return False

    for _ in range(5):
        _set_base_xy_direct(raw_env, robot, anchor_xy)
        raw_env.step(idle_action)
        _restore_upper_body_posture(raw_env, posture)
        _set_base_xy_direct(raw_env, robot, anchor_xy)
        if callable(recorder):
            recorder(_env=raw_env)
    return reached
