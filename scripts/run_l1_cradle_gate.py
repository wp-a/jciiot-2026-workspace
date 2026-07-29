#!/usr/bin/env python3
"""Run and validate the L1 physical cradle-transfer research gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


CRADLE_GATE_THRESHOLDS = {
    "lift_m": 0.13,
    "support_contact_steps": 20,
    "base_translation_m": 0.50,
}

ORIENTATION_ALIGNMENT_THRESHOLDS = {
    "error_deg": 5.0,
    "stable_steps": 5,
    "max_position_drift_m": 0.03,
}

JOINT_SEED_THRESHOLDS = {
    "error_deg": 10.0,
    "max_endpoint_position_error_m": 0.015,
    "max_path_position_drift_m": 0.03,
}

_REQUIRED_FIELDS = (
    "physical_grasp",
    "lift_m",
    "support_contact_steps",
    "base_translation_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "dropped",
    "infrastructure_error",
)


def cradle_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the hard L1 physical-transfer gate."""
    failures = [key for key in _REQUIRED_FIELDS if key not in record]

    def reject_numeric(key: str, *, minimum: float | None = None) -> None:
        if key not in record:
            return
        value = record[key]
        if isinstance(value, bool):
            failures.append(key)
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failures.append(key)
            return
        if not np.isfinite(numeric):
            failures.append(key)
            return
        if minimum is not None and numeric < minimum:
            failures.append(key)

    if record.get("physical_grasp") is not True:
        failures.append("physical_grasp")
    reject_numeric("lift_m", minimum=CRADLE_GATE_THRESHOLDS["lift_m"])
    reject_numeric(
        "support_contact_steps",
        minimum=CRADLE_GATE_THRESHOLDS["support_contact_steps"],
    )
    reject_numeric(
        "base_translation_m",
        minimum=CRADLE_GATE_THRESHOLDS["base_translation_m"],
    )

    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        reject_numeric(key)
        if key in record:
            try:
                if float(record[key]) != 0.0:
                    failures.append(key)
            except (TypeError, ValueError):
                pass

    if record.get("dropped") is not False:
        failures.append("dropped")
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")

    return list(dict.fromkeys(failures))


def cradle_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only a complete record that passes every hard condition."""
    return not cradle_gate_failures(record)


_CENTER_GRASP_TRANSPORT_REQUIRED_FIELDS = (
    "physical_grasp",
    "lift_m",
    "hold_grasp_steps",
    "transport_success",
    "object_translation_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "dropped",
    "infrastructure_error",
)


def center_grasp_transport_failures(
    record: Mapping[str, object],
) -> list[str]:
    """Return failed evidence fields for a physical center-grasp transport."""
    failures = [
        key for key in _CENTER_GRASP_TRANSPORT_REQUIRED_FIELDS if key not in record
    ]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    if record.get("physical_grasp") is not True:
        failures.append("physical_grasp")
    lift = numeric("lift_m")
    if lift is None or lift < CRADLE_GATE_THRESHOLDS["lift_m"]:
        failures.append("lift_m")
    hold_steps = numeric("hold_grasp_steps")
    if hold_steps is None or hold_steps < 20.0:
        failures.append("hold_grasp_steps")
    if record.get("transport_success") is not True:
        failures.append("transport_success")
    object_translation = numeric("object_translation_m")
    if object_translation is None or object_translation <= 1.0:
        failures.append("object_translation_m")
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("dropped") is not False:
        failures.append("dropped")
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def center_grasp_transport_accepted(record: Mapping[str, object]) -> bool:
    """Accept only complete, collision-free center-grasp transport evidence."""
    return not center_grasp_transport_failures(record)


_PUSH_REQUIRED_FIELDS = (
    "physical_contact_steps",
    "object_translation_m",
    "base_translation_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "infrastructure_error",
)


def push_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the hard L1 physical-push gate."""
    failures = [key for key in _PUSH_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    minimums = {
        "physical_contact_steps": 20.0,
        "object_translation_m": 0.50,
        "base_translation_m": 0.30,
    }
    for key, minimum in minimums.items():
        value = numeric(key)
        if value is None or value < minimum:
            failures.append(key)
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def push_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only complete, collision-free physical-push evidence."""
    return not push_gate_failures(record)


_UNDERCUT_REQUIRED_FIELDS = (
    "open_gripper",
    "support_contact_steps",
    "object_lift_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "infrastructure_error",
)


def undercut_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the open-gripper undercut gate."""
    failures = [key for key in _UNDERCUT_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    if record.get("open_gripper") is not True:
        failures.append("open_gripper")
    support_steps = numeric("support_contact_steps")
    if support_steps is None or support_steps < 5.0:
        failures.append("support_contact_steps")
    object_lift = numeric("object_lift_m")
    if object_lift is None or object_lift < 0.02:
        failures.append("object_lift_m")
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def undercut_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only measured, collision-free open-gripper support evidence."""
    return not undercut_gate_failures(record)


def table_edge_undercut_targets(
    *,
    object_center: object,
    object_half_depth_m: float,
    object_half_height_m: float,
    table_edge_y: float,
    outside_clearance_m: float,
    edge_clearance_m: float,
    object_offset_x_m: float,
    above_clearance_m: float,
    below_bottom_clearance_m: float,
    raise_above_bottom_m: float,
) -> dict[str, np.ndarray]:
    """Build an outside-down-in-up path through a measured table overhang."""
    center = np.asarray(object_center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object_center must be a finite three-vector")
    values = {
        "object_half_depth_m": object_half_depth_m,
        "object_half_height_m": object_half_height_m,
        "table_edge_y": table_edge_y,
        "outside_clearance_m": outside_clearance_m,
        "edge_clearance_m": edge_clearance_m,
        "object_offset_x_m": object_offset_x_m,
        "above_clearance_m": above_clearance_m,
        "below_bottom_clearance_m": below_bottom_clearance_m,
        "raise_above_bottom_m": raise_above_bottom_m,
    }
    for name, value in values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    for name in (
        "object_half_depth_m",
        "object_half_height_m",
        "outside_clearance_m",
        "edge_clearance_m",
        "above_clearance_m",
        "below_bottom_clearance_m",
        "raise_above_bottom_m",
    ):
        if float(values[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(object_half_depth_m) == 0.0 or float(object_half_height_m) == 0.0:
        raise ValueError("object half extents must be positive")

    outer_y = float(center[1]) + float(object_half_depth_m)
    outside_y = outer_y + float(outside_clearance_m)
    undercut_y = float(table_edge_y) + float(edge_clearance_m)
    if undercut_y >= outer_y:
        raise ValueError("configured table edge leaves no exposed bottom")
    bottom_z = float(center[2]) - float(object_half_height_m)
    target_x = float(center[0]) + float(object_offset_x_m)
    high_z = float(center[2]) + float(object_half_height_m) + float(
        above_clearance_m
    )
    below_z = bottom_z - float(below_bottom_clearance_m)
    return {
        "outside": np.array([target_x, outside_y, high_z], dtype=float),
        "below": np.array([target_x, outside_y, below_z], dtype=float),
        "undercut": np.array([target_x, undercut_y, below_z], dtype=float),
        "raise": np.array(
            [target_x, undercut_y, bottom_z + float(raise_above_bottom_m)],
            dtype=float,
        ),
    }


def joint_seed_joint_names(*, include_torso: bool) -> tuple[str, ...]:
    """Return the ordered Tiago joints controlled by the wrist seed solver."""
    if not isinstance(include_torso, (bool, np.bool_)):
        raise ValueError("include_torso must be boolean")
    names = tuple(
        f"robot0_arm_{arm}_{index}_joint"
        for arm in ("right", "left")
        for index in range(1, 7)
    )
    if bool(include_torso):
        names = (*names, "robot0_torso_lift_joint")
    return names


def bounded_base_advance_world_velocity(
    *,
    base_xy: object,
    object_xy: object,
    remaining_m: float,
    max_speed_m_s: float,
    control_dt_s: float,
) -> np.ndarray:
    """Return a bounded planar velocity from the base toward the object."""
    base = np.asarray(base_xy, dtype=float)
    target = np.asarray(object_xy, dtype=float)
    if (
        base.shape != (2,)
        or target.shape != (2,)
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("base_xy and object_xy must be finite planar vectors")
    remaining = float(remaining_m)
    max_speed = float(max_speed_m_s)
    control_dt = float(control_dt_s)
    if not np.isfinite(remaining) or remaining < 0.0:
        raise ValueError("remaining_m must be finite and non-negative")
    if not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("max_speed_m_s must be finite and positive")
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    if remaining == 0.0:
        return np.zeros(2, dtype=float)
    direction = target - base
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-12:
        raise ValueError("base and object positions must differ")
    speed = min(max_speed, remaining / control_dt)
    return direction * (speed / distance)


def forward_carry_target(
    *,
    base_xy: object,
    object_xy: object,
    distance_m: float,
    toward_object: bool = True,
) -> np.ndarray:
    """Return a base waypoint along or opposite the base-to-object axis."""
    base = np.asarray(base_xy, dtype=float).reshape(2)
    target = np.asarray(object_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(target)):
        raise ValueError("base_xy and object_xy must be finite planar vectors")
    distance = float(distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    if distance == 0.0:
        return base.copy()
    direction = target - base
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("base and object positions must differ")
    if not toward_object:
        direction = -direction
    return base + direction * (distance / direction_norm)


def resolve_inchworm_direction(
    *,
    base_xy: object,
    object_xy: object,
    toward_base: bool,
    world_direction: object | None = None,
) -> np.ndarray:
    """Resolve and normalize the requested world-frame transport direction."""
    base = np.asarray(base_xy, dtype=float).reshape(2)
    target = np.asarray(object_xy, dtype=float).reshape(2)
    if world_direction is None:
        direction = target - base
        if toward_base:
            direction = -direction
    else:
        direction = np.asarray(world_direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or norm <= 1e-12:
        raise ValueError("inchworm direction must be finite and non-zero")
    return direction / norm


def trailing_corner_seat_targets(
    current: Mapping[str, object],
    *,
    travel_direction: object,
    distance_m: float,
) -> dict[str, np.ndarray]:
    """Slide both closed grippers toward the trailing wall corner."""
    direction = np.asarray(travel_direction, dtype=float).reshape(2)
    distance = float(distance_m)
    if not np.all(np.isfinite(direction)):
        raise ValueError("travel_direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("travel_direction must be non-zero")
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    offset = -direction * (distance / norm)
    targets = {}
    for arm in ("right", "left"):
        position = np.asarray(current[arm], dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(position)):
            raise ValueError(f"{arm} position must be finite")
        position[:2] += offset
        targets[arm] = position
    return targets


def arm_transport_stroke_targets(
    current: Mapping[str, object],
    *,
    travel_direction: object,
    stroke_m: float,
    lift_m: float,
) -> dict[str, np.ndarray]:
    """Move a bilateral grasp through one bounded transport stroke."""
    direction = np.asarray(travel_direction, dtype=float).reshape(2)
    if not np.all(np.isfinite(direction)):
        raise ValueError("travel_direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("travel_direction must be non-zero")
    stroke = float(stroke_m)
    lift = float(lift_m)
    if not np.isfinite(stroke) or stroke < 0.0:
        raise ValueError("stroke_m must be finite and non-negative")
    if not np.isfinite(lift) or lift < 0.0:
        raise ValueError("lift_m must be finite and non-negative")
    planar_offset = direction * (stroke / norm)
    targets = {}
    for arm in ("right", "left"):
        position = np.asarray(current[arm], dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(position)):
            raise ValueError(f"{arm} position must be finite")
        position[:2] += planar_offset
        position[2] += lift
        targets[arm] = position
    return targets


def compensated_base_reset_step(
    *,
    travel_direction: object,
    base_yaw: float,
    remaining_m: float,
    max_speed_m_s: float,
    control_dt_s: float,
    gripper_world_errors: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Advance the base while opposing its world step with both arms."""
    direction = np.asarray(travel_direction, dtype=float).reshape(2)
    if not np.all(np.isfinite(direction)):
        raise ValueError("travel_direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("travel_direction must be non-zero")
    remaining = float(remaining_m)
    speed_limit = float(max_speed_m_s)
    control_dt = float(control_dt_s)
    if not np.isfinite(remaining) or remaining < 0.0:
        raise ValueError("remaining_m must be finite and non-negative")
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        raise ValueError("max_speed_m_s must be finite and positive")
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    world_step = direction * (min(speed_limit * control_dt, remaining) / norm)
    world_velocity = world_step / control_dt
    cosine = np.cos(float(base_yaw))
    sine = np.sin(float(base_yaw))
    base_velocity = np.array(
        [
            cosine * world_velocity[0] + sine * world_velocity[1],
            -sine * world_velocity[0] + cosine * world_velocity[1],
        ],
        dtype=float,
    )
    compensation = np.array([-world_step[0], -world_step[1], 0.0])
    errors = gripper_world_errors or {}
    arm_deltas = {}
    for arm in ("right", "left"):
        error = np.asarray(errors.get(arm, np.zeros(3)), dtype=float).reshape(3)
        if not np.all(np.isfinite(error)):
            raise ValueError(f"{arm} gripper world error must be finite")
        arm_deltas[arm] = compensation + error
    return (
        np.array([base_velocity[0], base_velocity[1], 0.0], dtype=float),
        arm_deltas,
    )


def projected_planar_motion(
    delta: object,
    *,
    direction: object,
) -> tuple[float, float]:
    """Split planar motion into signed progress and orthogonal drift."""
    planar_delta = np.asarray(delta, dtype=float).reshape(2)
    axis = np.asarray(direction, dtype=float).reshape(2)
    if not np.all(np.isfinite(planar_delta)) or not np.all(np.isfinite(axis)):
        raise ValueError("delta and direction must be finite")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    axis = axis / norm
    progress = float(np.dot(planar_delta, axis))
    lateral = float(np.linalg.norm(planar_delta - progress * axis))
    return progress, lateral


def allocate_segment_steps(*, total_steps: int, segment_count: int) -> tuple[int, ...]:
    """Distribute a fixed positive waypoint budget across path segments."""
    if (
        isinstance(total_steps, bool)
        or int(total_steps) != total_steps
        or int(total_steps) < 1
    ):
        raise ValueError("total_steps must be a positive integer")
    if (
        isinstance(segment_count, bool)
        or int(segment_count) != segment_count
        or int(segment_count) < 1
    ):
        raise ValueError("segment_count must be a positive integer")
    total = int(total_steps)
    segments = int(segment_count)
    if total < segments:
        raise ValueError("total_steps must provide at least one step per segment")
    base, remainder = divmod(total, segments)
    return tuple(base + int(index < remainder) for index in range(segments))


def interior_joint_bounds(
    lower: object,
    upper: object,
    *,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move finite joint limits inward by a fixed angular margin."""
    minimum = np.asarray(lower, dtype=float)
    maximum = np.asarray(upper, dtype=float)
    margin = float(margin_rad)
    if (
        minimum.ndim != 1
        or maximum.shape != minimum.shape
        or minimum.size == 0
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
    ):
        raise ValueError("joint bounds must be finite matching vectors")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("joint margin must be finite and non-negative")
    interior_lower = minimum + margin
    interior_upper = maximum - margin
    if np.any(interior_lower >= interior_upper):
        raise ValueError("joint margin leaves an empty interior interval")
    return interior_lower, interior_upper


def _normalized_axis(value: object, *, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return axis / norm


def nearest_directed_axis_target(
    source_axis: object,
    target_axis: object,
) -> np.ndarray:
    """Fix the sign of an undirected target nearest to the source axis."""
    source = _normalized_axis(source_axis, name="source_axis")
    target = _normalized_axis(target_axis, name="target_axis")
    return target if float(np.dot(source, target)) >= 0.0 else -target


def interpolate_directed_axis(
    source_axis: object,
    target_axis: object,
    *,
    fraction: float,
) -> np.ndarray:
    """Interpolate normalized directed axes within the same hemisphere."""
    source = _normalized_axis(source_axis, name="source_axis")
    target = _normalized_axis(target_axis, name="target_axis")
    if float(np.dot(source, target)) < -1e-12:
        raise ValueError("directed axes must lie in the same hemisphere")
    amount = float(fraction)
    if not np.isfinite(amount) or amount < 0.0 or amount > 1.0:
        raise ValueError("fraction must be finite and in [0, 1]")
    return _normalized_axis(
        (1.0 - amount) * source + amount * target,
        name="interpolated_axis",
    )


def joint_seed_objective_residual(
    *,
    current_positions: Mapping[str, object],
    target_positions: Mapping[str, object],
    current_axes: Mapping[str, object],
    target_axes: Mapping[str, object],
    joints: object,
    start_joints: object,
    joint_ranges: object,
    position_scale_m: float,
    axis_scale: float,
    regularization: float,
) -> np.ndarray:
    """Build the normalized simultaneous two-arm wrist-seed residual."""
    arms = ("right", "left")

    def position(mapping: Mapping[str, object], arm: str, name: str) -> np.ndarray:
        if not isinstance(mapping, Mapping) or arm not in mapping:
            raise ValueError(f"{name} must contain both arms")
        value = np.asarray(mapping[arm], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name}[{arm}] must be a finite three-vector")
        return value

    current = np.asarray(joints, dtype=float)
    start = np.asarray(start_joints, dtype=float)
    ranges = np.asarray(joint_ranges, dtype=float)
    if (
        current.ndim != 1
        or current.size == 0
        or start.shape != current.shape
        or ranges.shape != current.shape
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(ranges))
        or np.any(ranges <= 0.0)
    ):
        raise ValueError("joint vectors must be finite, matching, and non-empty")
    position_scale = float(position_scale_m)
    orientation_scale = float(axis_scale)
    joint_weight = float(regularization)
    if not np.isfinite(position_scale) or position_scale <= 0.0:
        raise ValueError("position_scale_m must be finite and positive")
    if not np.isfinite(orientation_scale) or orientation_scale <= 0.0:
        raise ValueError("axis_scale must be finite and positive")
    if not np.isfinite(joint_weight) or joint_weight < 0.0:
        raise ValueError("regularization must be finite and non-negative")

    position_errors = []
    axis_errors = []
    for arm in arms:
        position_errors.append(
            (
                position(current_positions, arm, "current_positions")
                - position(target_positions, arm, "target_positions")
            )
            / position_scale
        )
        axis_errors.append(
            (
                _normalized_axis(current_axes.get(arm), name=f"current_axes[{arm}]")
                - _normalized_axis(target_axes.get(arm), name=f"target_axes[{arm}]")
            )
            / orientation_scale
        )
    joint_error = joint_weight * (current - start) / ranges
    return np.concatenate([*position_errors, *axis_errors, joint_error])


def minimum_undirected_axis_rotation(
    source_axis: object,
    target_axis: object,
) -> np.ndarray:
    """Return the minimum rotation aligning an undirected source and target."""
    source = _normalized_axis(source_axis, name="source_axis")
    target = _normalized_axis(target_axis, name="target_axis")
    if float(np.dot(source, target)) < 0.0:
        target = -target
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        return np.eye(3)
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def closure_axis_error_degrees(source_axis: object, target_axis: object) -> float:
    """Return the unsigned angular error between two closure axes."""
    source = _normalized_axis(source_axis, name="source_axis")
    target = _normalized_axis(target_axis, name="target_axis")
    cosine = float(np.clip(abs(np.dot(source, target)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _validated_rotation_matrix(value: object, *, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation")
    return rotation


def _rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1e-12:
        return np.zeros(3)
    sine = float(np.sin(angle))
    if abs(sine) <= 1e-8:
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis = _normalized_axis(axis, name="rotation axis")
    else:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * sine)
    return axis * angle


def normalized_osc_orientation_command(
    *,
    world_rotation_delta: object,
    controller_origin_rotation: object,
    output_min: object,
    output_max: object,
    max_action: float,
) -> np.ndarray:
    """Convert a world-frame rotation delta into a bounded OSC command."""
    world_delta = _validated_rotation_matrix(
        world_rotation_delta,
        name="world_rotation_delta",
    )
    origin = _validated_rotation_matrix(
        controller_origin_rotation,
        name="controller_origin_rotation",
    )
    minimum = np.asarray(output_min, dtype=float)
    maximum = np.asarray(output_max, dtype=float)
    if (
        minimum.ndim != 1
        or maximum.shape != minimum.shape
        or minimum.size < 6
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
    ):
        raise ValueError("controller output bounds must be finite matching vectors")
    orientation_scale = np.maximum(
        np.abs(minimum[3:6]),
        np.abs(maximum[3:6]),
    )
    if np.any(orientation_scale <= 0.0):
        raise ValueError("controller orientation output scale must be positive")
    limit = float(max_action)
    if not np.isfinite(limit) or limit <= 0.0 or limit > 1.0:
        raise ValueError("max_action must be finite and in (0, 1]")

    local_delta = origin.T @ world_delta @ origin
    rotation_vector = _rotation_matrix_to_axis_angle(local_delta)
    command = rotation_vector / orientation_scale
    command_norm = float(np.linalg.norm(command))
    if command_norm > limit:
        command *= limit / command_norm
    return command


def scheduled_orientation_action_limit(
    *,
    error_deg: float,
    coarse_action: float,
    fine_action: float,
    fine_threshold_deg: float,
) -> float:
    """Select a per-arm fine action limit near the orientation target."""
    error = float(error_deg)
    coarse = float(coarse_action)
    fine = float(fine_action)
    threshold = float(fine_threshold_deg)
    if not all(np.isfinite(value) for value in (error, coarse, fine, threshold)):
        raise ValueError("orientation schedule values must be finite")
    if error < 0.0 or threshold < 0.0:
        raise ValueError("orientation errors and thresholds must be non-negative")
    if coarse <= 0.0 or coarse > 1.0:
        raise ValueError("coarse_action must be in (0, 1]")
    if fine <= 0.0 or fine > coarse:
        raise ValueError("fine_action must be in (0, coarse_action]")
    return fine if error <= threshold else coarse


_ORIENTATION_REQUIRED_FIELDS = (
    "orientation_right_error_deg",
    "orientation_left_error_deg",
    "orientation_stable_steps",
    "orientation_max_position_drift_m",
    "orientation_collision_frames",
    "infrastructure_error",
)


def orientation_alignment_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the high-clearance alignment gate."""
    failures = [key for key in _ORIENTATION_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    limits = {
        "orientation_right_error_deg": (
            "maximum",
            ORIENTATION_ALIGNMENT_THRESHOLDS["error_deg"],
        ),
        "orientation_left_error_deg": (
            "maximum",
            ORIENTATION_ALIGNMENT_THRESHOLDS["error_deg"],
        ),
        "orientation_stable_steps": (
            "minimum",
            ORIENTATION_ALIGNMENT_THRESHOLDS["stable_steps"],
        ),
        "orientation_max_position_drift_m": (
            "maximum",
            ORIENTATION_ALIGNMENT_THRESHOLDS["max_position_drift_m"],
        ),
        "orientation_collision_frames": ("maximum", 0.0),
    }
    for key, (kind, threshold) in limits.items():
        value = numeric(key)
        if value is None:
            failures.append(key)
        elif kind == "minimum" and value < threshold:
            failures.append(key)
        elif kind == "maximum" and value > threshold:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


_JOINT_SEED_REQUIRED_FIELDS = (
    "joint_seed_success",
    "joint_seed_right_error_deg",
    "joint_seed_left_error_deg",
    "joint_seed_max_endpoint_position_error_m",
    "joint_seed_max_path_position_drift_m",
    "joint_seed_min_bound_margin_rad",
    "joint_seed_collision_frames",
    "joint_seed_rolled_back",
    "infrastructure_error",
)


def joint_seed_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the robot-only joint seed gate."""
    failures = [key for key in _JOINT_SEED_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value >= 0.0 else None

    maximums = {
        "joint_seed_right_error_deg": JOINT_SEED_THRESHOLDS["error_deg"],
        "joint_seed_left_error_deg": JOINT_SEED_THRESHOLDS["error_deg"],
        "joint_seed_max_endpoint_position_error_m": JOINT_SEED_THRESHOLDS[
            "max_endpoint_position_error_m"
        ],
        "joint_seed_max_path_position_drift_m": JOINT_SEED_THRESHOLDS[
            "max_path_position_drift_m"
        ],
        "joint_seed_collision_frames": 0.0,
    }
    for key, maximum in maximums.items():
        value = numeric(key)
        if value is None or value > maximum:
            failures.append(key)
    if numeric("joint_seed_min_bound_margin_rad") is None:
        failures.append("joint_seed_min_bound_margin_rad")
    if record.get("joint_seed_success") is not True:
        failures.append("joint_seed_success")
    if record.get("joint_seed_rolled_back") is not False:
        failures.append("joint_seed_rolled_back")
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def joint_seed_node_failure(
    *,
    solver_success: bool,
    right_error_deg: float,
    left_error_deg: float,
    position_error_m: float,
    min_bound_margin_rad: float,
    collision: bool,
) -> str | None:
    """Return the first hard-gate failure for a continuation IK node."""
    if not isinstance(solver_success, (bool, np.bool_)):
        raise ValueError("solver_success must be boolean")
    if not isinstance(collision, (bool, np.bool_)):
        raise ValueError("collision must be boolean")
    errors = (float(right_error_deg), float(left_error_deg))
    position_error = float(position_error_m)
    bound_margin = float(min_bound_margin_rad)
    if not all(np.isfinite(value) and value >= 0.0 for value in errors):
        raise ValueError("node orientation errors must be finite and non-negative")
    if not np.isfinite(position_error) or position_error < 0.0:
        raise ValueError("node position error must be finite and non-negative")
    if not np.isfinite(bound_margin):
        raise ValueError("node bound margin must be finite")
    if not bool(solver_success):
        return "solver"
    if bool(collision):
        return "collision"
    if max(errors) > JOINT_SEED_THRESHOLDS["error_deg"]:
        return "orientation"
    if position_error > JOINT_SEED_THRESHOLDS["max_endpoint_position_error_m"]:
        return "position"
    if bound_margin < 0.0:
        return "bounds"
    return None


def next_joint_seed_path_state(
    previous: Mapping[str, object],
    *,
    waypoint_index: int,
    right_drift_m: float,
    left_drift_m: float,
    collision_pairs: object,
    max_position_drift_m: float = JOINT_SEED_THRESHOLDS[
        "max_path_position_drift_m"
    ],
) -> dict[str, object]:
    """Advance collision and grip-site drift evidence for a joint path."""
    if (
        isinstance(waypoint_index, bool)
        or int(waypoint_index) != waypoint_index
        or int(waypoint_index) < 1
    ):
        raise ValueError("waypoint_index must be a positive integer")
    measurements = (float(right_drift_m), float(left_drift_m))
    drift_limit = float(max_position_drift_m)
    if not all(np.isfinite(value) and value >= 0.0 for value in measurements):
        raise ValueError("joint seed drift measurements must be finite and non-negative")
    if not np.isfinite(drift_limit) or drift_limit <= 0.0:
        raise ValueError("max_position_drift_m must be finite and positive")
    try:
        pairs = [list(pair) for pair in collision_pairs]
    except TypeError as exc:
        raise ValueError("collision_pairs must contain geom-name pairs") from exc
    if any(
        len(pair) != 2 or not all(isinstance(name, str) for name in pair)
        for pair in pairs
    ):
        raise ValueError("collision_pairs must contain geom-name pairs")

    previous_drift = float(previous.get("max_position_drift_m", 0.0))
    previous_frames = previous.get("collision_frames", 0)
    if not np.isfinite(previous_drift) or previous_drift < 0.0:
        raise ValueError("previous path drift is invalid")
    if (
        isinstance(previous_frames, bool)
        or int(previous_frames) != previous_frames
        or int(previous_frames) < 0
    ):
        raise ValueError("previous collision frame count is invalid")
    previous_pairs = previous.get("collision_pairs", [])
    if not isinstance(previous_pairs, list):
        raise ValueError("previous collision pairs are invalid")

    maximum_drift = max(previous_drift, *measurements)
    collision_frames = int(previous_frames) + int(bool(pairs))
    all_pairs = [list(pair) for pair in previous_pairs]
    for pair in pairs:
        if pair not in all_pairs:
            all_pairs.append(pair)
    failure = "collision" if pairs else None
    if failure is None and maximum_drift > drift_limit:
        failure = "position_drift"
    return {
        "waypoint_count": int(waypoint_index),
        "max_position_drift_m": maximum_drift,
        "collision_frames": collision_frames,
        "collision_pairs": all_pairs,
        "terminate": failure is not None,
        "failure": failure,
        "failed_waypoint": int(waypoint_index) if failure is not None else None,
    }


def next_orientation_alignment_state(
    previous: Mapping[str, object],
    *,
    right_error_deg: float,
    left_error_deg: float,
    position_drift_m: float,
    collision: bool,
    tolerance_deg: float = ORIENTATION_ALIGNMENT_THRESHOLDS["error_deg"],
    required_stable_steps: int = ORIENTATION_ALIGNMENT_THRESHOLDS["stable_steps"],
    max_position_drift_m: float = ORIENTATION_ALIGNMENT_THRESHOLDS[
        "max_position_drift_m"
    ],
) -> dict[str, object]:
    """Advance the collision-aware high-clearance orientation gate state."""
    measurements = {
        "right_error_deg": float(right_error_deg),
        "left_error_deg": float(left_error_deg),
        "position_drift_m": float(position_drift_m),
        "tolerance_deg": float(tolerance_deg),
        "max_position_drift_m": float(max_position_drift_m),
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in measurements.values()):
        raise ValueError("alignment measurements and limits must be finite and non-negative")
    if not isinstance(collision, (bool, np.bool_)):
        raise ValueError("collision must be boolean")
    stable_steps = previous.get("stable_steps", 0)
    previous_drift = previous.get("max_position_drift_m", 0.0)
    if isinstance(stable_steps, bool) or int(stable_steps) != stable_steps:
        raise ValueError("stable_steps must be a non-negative integer")
    stable_steps = int(stable_steps)
    previous_drift = float(previous_drift)
    if stable_steps < 0 or not np.isfinite(previous_drift) or previous_drift < 0.0:
        raise ValueError("previous alignment state is invalid")
    required = int(required_stable_steps)
    if isinstance(required_stable_steps, bool) or required != required_stable_steps or required < 1:
        raise ValueError("required_stable_steps must be a positive integer")

    maximum_drift = max(previous_drift, measurements["position_drift_m"])
    failure = None
    if bool(collision):
        failure = "collision"
    elif maximum_drift > measurements["max_position_drift_m"]:
        failure = "position_drift"
    within_tolerance = (
        measurements["right_error_deg"] <= measurements["tolerance_deg"]
        and measurements["left_error_deg"] <= measurements["tolerance_deg"]
    )
    stable_steps = stable_steps + 1 if within_tolerance and failure is None else 0
    return {
        "stable_steps": stable_steps,
        "max_position_drift_m": maximum_drift,
        "right_error_deg": measurements["right_error_deg"],
        "left_error_deg": measurements["left_error_deg"],
        "aligned": stable_steps >= required and failure is None,
        "terminate": failure is not None,
        "failure": failure,
    }


def eef_site_pose(raw_env, robot, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the world pose of the grip site controlled by the arm OSC."""
    try:
        site_id = robot.eef_site_id[arm]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"missing OSC grip site for {arm}") from exc
    position = np.asarray(raw_env.sim.data.site_xpos[site_id], dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"invalid OSC grip-site position for {arm}")
    orientation = _validated_rotation_matrix(
        np.asarray(raw_env.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3),
        name=f"{arm} OSC grip-site orientation",
    )
    return position.copy(), orientation.copy()


def has_bilateral_object_contact(
    contacts: Mapping[str, tuple[str, ...]],
) -> bool:
    """Return whether both arms have at least one physical object contact."""
    return all(bool(contacts.get(arm)) for arm in ("right", "left"))


def fingerpad_bracket_evidence(
    *,
    fingerpads: Mapping[str, np.ndarray],
    wall_centers: np.ndarray,
    separation_axis: np.ndarray,
) -> dict[str, object]:
    """Measure whether distinct opposed walls lie between each fingerpad pair."""
    axis = np.asarray(separation_axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.all(np.isfinite(axis)) or axis_norm <= 1e-9:
        raise ValueError("separation_axis must be finite and nonzero")
    axis /= axis_norm
    walls = np.asarray(wall_centers, dtype=float).reshape(2, 3)
    if not np.all(np.isfinite(walls)):
        raise ValueError("wall_centers must be finite")
    wall_projections = walls @ axis
    arms: dict[str, object] = {}
    assigned_walls = []
    for arm in ("left", "right"):
        pad_positions = np.asarray(fingerpads[arm], dtype=float).reshape(2, 3)
        if not np.all(np.isfinite(pad_positions)):
            raise ValueError("fingerpads must be finite")
        pad_projections = pad_positions @ axis
        wall_index = int(
            np.argmin(np.abs(wall_projections - float(np.mean(pad_projections))))
        )
        wall_projection = float(wall_projections[wall_index])
        lower = float(np.min(pad_projections))
        upper = float(np.max(pad_projections))
        bracketed = lower <= wall_projection <= upper
        assigned_walls.append(wall_index)
        arms[arm] = {
            "wall_index": wall_index,
            "wall_projection": wall_projection,
            "fingerpad_projections": pad_projections.tolist(),
            "bracketed": bool(bracketed),
        }
    distinct = len(set(assigned_walls)) == 2
    return {
        "ready": bool(distinct and all(arm["bracketed"] for arm in arms.values())),
        "distinct_walls": distinct,
        "wall_centers": walls.tolist(),
        "wall_projections": wall_projections.tolist(),
        "fingerpads": {
            arm: np.asarray(fingerpads[arm], dtype=float).reshape(2, 3).tolist()
            for arm in ("left", "right")
        },
        "arms": arms,
    }


def fingerpad_world_positions(raw_env, robot) -> dict[str, np.ndarray]:
    """Read the two official important fingerpad geom centers for each arm."""
    model = raw_env.sim.model
    data = raw_env.sim.data
    result = {}
    for arm in ("left", "right"):
        important = getattr(robot.gripper[arm], "important_geoms", {})
        names = []
        for group in ("left_fingerpad", "right_fingerpad"):
            group_names = important.get(group, ())
            if not group_names:
                raise ValueError(f"missing {arm} {group} geometry")
            names.append(group_names[0])
        positions = np.stack(
            [
                np.asarray(data.geom_xpos[model.geom_name2id(name)], dtype=float)
                for name in names
            ],
            axis=0,
        )
        if positions.shape != (2, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"invalid {arm} fingerpad positions")
        result[arm] = positions.copy()
    return result


def opposed_object_wall_centers(
    raw_env,
    object_name: str,
    *,
    separation_axis: np.ndarray,
) -> np.ndarray:
    """Read the two extreme object geom centers along the opposed-wall axis."""
    axis = np.asarray(separation_axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.all(np.isfinite(axis)) or axis_norm <= 1e-9:
        raise ValueError("separation_axis must be finite and nonzero")
    axis /= axis_norm
    model = raw_env.sim.model
    data = raw_env.sim.data
    object_bodies = _object_body_ids(raw_env, object_name)
    centers = np.stack(
        [
            np.asarray(data.geom_xpos[geom_id], dtype=float)
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in object_bodies
        ],
        axis=0,
    )
    if centers.shape[0] < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("object must provide finite opposed wall geometries")
    projections = centers @ axis
    indices = np.array([int(np.argmin(projections)), int(np.argmax(projections))])
    if float(projections[indices[1]] - projections[indices[0]]) <= 1e-9:
        raise ValueError("object wall projections must be distinct")
    return centers[indices].copy()


def opposed_wall_clearance_targets(
    current_positions: Mapping[str, np.ndarray],
    *,
    separation_axis: np.ndarray,
    clearance_m: float,
) -> dict[str, np.ndarray]:
    """Move both end effectors outward before descending around opposed walls."""
    distance = float(clearance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("clearance_m must be a finite non-negative value")
    axis = np.asarray(separation_axis, dtype=float).reshape(3).copy()
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("separation_axis must have a horizontal component")
    axis /= norm
    right = np.asarray(current_positions["right"], dtype=float).reshape(3)
    left = np.asarray(current_positions["left"], dtype=float).reshape(3)
    if float(np.dot(right - left, axis)) < 0.0:
        axis *= -1.0
    return {
        "right": right + axis * distance,
        "left": left - axis * distance,
    }


def opposed_wall_squeeze_targets(
    current_positions: Mapping[str, np.ndarray],
    *,
    separation_axis: np.ndarray,
    squeeze_m: float,
) -> dict[str, np.ndarray]:
    """Move both end effectors inward to apply opposed wall pressure."""
    outward = opposed_wall_clearance_targets(
        current_positions,
        separation_axis=separation_axis,
        clearance_m=squeeze_m,
    )
    return {
        arm: 2.0 * np.asarray(current_positions[arm], dtype=float) - outward[arm]
        for arm in ("right", "left")
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_candidate(candidate_root: Path) -> Path:
    app_dir = candidate_root / "JCIIOT"
    if not (app_dir / "app.py").is_file():
        raise FileNotFoundError(f"official app not found: {app_dir / 'app.py'}")
    paths = (
        app_dir / "src",
        app_dir,
        app_dir / "robomimic",
        app_dir / "robosuite",
        app_dir / "robosuite" / "robosuite",
    )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.chdir(app_dir)
    return app_dir


def _load_scene(app_dir: Path, task: Mapping[str, Any], seed: int):
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.environments import RobosuiteBackend

    map_dir = (
        app_dir
        / "robosuite"
        / "robosuite"
        / "environments"
        / "factory_sorting"
        / "generated_maps"
    )
    prefix = str(task["scene_prefix"])
    semantic_path = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid_path = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    scene_data, grid = load_map_files(semantic_path, grid_path)
    scene_context = SceneContext.from_semantic_map(scene_data)
    backend = RobosuiteBackend(
        env_name=str(task["env_name"]),
        camera="birdview",
        headless=True,
        drive_mode="direct",
        seed=int(seed),
    )
    backend._scene_context = scene_context
    backend.reset()
    return backend, scene_context, grid


def _object_body_ids(raw_env, object_name: str) -> set[int]:
    model = raw_env.sim.model
    root = int(raw_env.obj_body_id[object_name])
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if body_id in descendants:
                continue
            if int(model.body_parentid[body_id]) in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def object_robot_contacts(raw_env, object_name: str) -> dict[str, tuple[str, ...]]:
    """Return robot geoms that currently contact the named physical object."""
    model = raw_env.sim.model
    object_bodies = _object_body_ids(raw_env, object_name)
    result = {"right": set(), "left": set()}
    for contact in raw_env.sim.data.contact[: raw_env.sim.data.ncon]:
        geom_ids = (int(contact.geom1), int(contact.geom2))
        object_sides = [
            int(model.geom_bodyid[geom_id]) in object_bodies
            for geom_id in geom_ids
        ]
        if object_sides[0] == object_sides[1]:
            continue
        robot_geom_id = geom_ids[1] if object_sides[0] else geom_ids[0]
        name = model.geom_id2name(robot_geom_id)
        if not name:
            continue
        lowered = name.lower()
        if lowered.startswith("gripper0_left_") or "_left_collision" in lowered:
            result["left"].add(name)
        elif lowered.startswith("gripper0_right_") or lowered.startswith("robot0_arm_"):
            result["right"].add(name)
    return {arm: tuple(sorted(names)) for arm, names in result.items()}


def geometry_snapshot(raw_env, object_name: str) -> dict[str, Any]:
    """Capture world geometry needed to design a physical support transition."""
    model = raw_env.sim.model
    data = raw_env.sim.data
    object_bodies = _object_body_ids(raw_env, object_name)
    selected = []
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        if not name:
            continue
        lowered = name.lower()
        is_object = int(model.geom_bodyid[geom_id]) in object_bodies
        is_support_link = lowered.startswith("gripper0_") or any(
            token in lowered
            for token in (
                "arm_5_collision",
                "arm_6_collision",
                "arm_5_left_collision",
                "arm_6_left_collision",
            )
        )
        if not (is_object or is_support_link):
            continue
        selected.append(
            {
                "name": name,
                "is_object": is_object,
                "world_position": np.asarray(data.geom_xpos[geom_id], dtype=float).tolist(),
                "world_rotation": np.asarray(data.geom_xmat[geom_id], dtype=float)
                .reshape(3, 3)
                .tolist(),
                "size": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
                "type": int(model.geom_type[geom_id]),
            }
        )
    body_id = raw_env.obj_body_id[object_name]
    return {
        "object_position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "geometries": selected,
    }


def _hold_probe(backend, object_name: str, *, steps: int) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        _is_allowed_cradle_geom,
        world_velocity_to_base_frame,
    )

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    hold_targets = helpers["capture_hold_targets"](robot)
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0
    observations = []
    for step in range(int(steps)):
        robot.composite_controller.update_state()
        action = helpers["build_action"](
            robot,
            arm_actions={},
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        contacts = object_robot_contacts(raw_env, object_name)
        supported = all(
            any(_is_allowed_cradle_geom(name, arm) for name in contacts[arm])
            for arm in ("right", "left")
        )
        stable_support_steps = stable_support_steps + 1 if supported else 0
        maximum_support_steps = max(maximum_support_steps, stable_support_steps)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        body_id = raw_env.obj_body_id[object_name]
        observations.append(
            {
                "step": step + 1,
                "object_z": float(raw_env.sim.data.body_xpos[body_id][2]),
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "bilateral_support": supported,
                "judge_collision": collision,
            }
        )
    return {
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "observations": observations,
    }


def _inward_support_probe(
    backend,
    object_name: str,
    *,
    inward_m: float,
    move_steps: int,
    hold_steps: int,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import _is_allowed_cradle_geom

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
    start_geometry = geometry_snapshot(raw_env, object_name)
    starts = {
        arm: np.asarray(
            helpers["gripper_position"](raw_env, robot, arm),
            dtype=float,
        )
        for arm in ("right", "left")
    }
    targets = {}
    for arm, start in starts.items():
        direction = object_position - start
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise RuntimeError(f"cannot resolve inward direction for {arm} arm")
        targets[arm] = start + direction / norm * float(inward_m)

    hold_targets = helpers["capture_hold_targets"](robot)
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0
    observations = []
    total_steps = int(move_steps) + int(hold_steps)
    for step in range(total_steps):
        robot.composite_controller.update_state()
        arm_actions = {}
        if step < int(move_steps):
            for arm in ("right", "left"):
                current = helpers["gripper_position"](raw_env, robot, arm)
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    targets[arm] - current,
                )
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    0.30,
                )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        contacts = object_robot_contacts(raw_env, object_name)
        supported = all(
            any(_is_allowed_cradle_geom(name, arm) for name in contacts[arm])
            for arm in ("right", "left")
        )
        stable_support_steps = stable_support_steps + 1 if supported else 0
        maximum_support_steps = max(maximum_support_steps, stable_support_steps)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        current_positions = {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }
        observations.append(
            {
                "step": step + 1,
                "phase": "inward" if step < int(move_steps) else "hold",
                "object_z": float(raw_env.sim.data.body_xpos[body_id][2]),
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "bilateral_support": supported,
                "judge_collision": collision,
                "eef_positions": {
                    arm: current_positions[arm].tolist()
                    for arm in ("right", "left")
                },
            }
        )
        if collision:
            break
    return {
        "requested_inward_m": float(inward_m),
        "move_steps": int(move_steps),
        "hold_steps": int(hold_steps),
        "start_eef_positions": {arm: starts[arm].tolist() for arm in starts},
        "target_eef_positions": {arm: targets[arm].tolist() for arm in targets},
        "start_geometry": start_geometry,
        "end_geometry": geometry_snapshot(raw_env, object_name),
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "observations": observations,
    }


def _physical_push_probe(
    backend,
    object_name: str,
    *,
    table_object_z: float,
    push_distance_m: float,
    max_push_steps: int,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        world_velocity_to_base_frame,
    )

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    hold_targets = helpers["capture_hold_targets"](robot)
    observations = []
    collision_steps = 0
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    start_object_xy = np.asarray(
        raw_env.sim.data.body_xpos[body_id][:2],
        dtype=float,
    )
    direction = start_object_xy - start_base_xy
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        raise RuntimeError("base and object centers coincide before physical push")
    direction /= direction_norm

    def eef_positions() -> dict[str, np.ndarray]:
        return {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }

    shift_starts = eef_positions()
    shift_targets = {
        arm: position
        + np.array([direction[0] * 0.20, direction[1] * 0.20, 0.0])
        for arm, position in shift_starts.items()
    }
    for step in range(180):
        robot.composite_controller.update_state()
        current = eef_positions()
        arm_actions = {}
        for arm in ("right", "left"):
            controller_delta = helpers["world_delta"](
                robot,
                arm,
                shift_targets[arm] - current[arm],
            )
            arm_actions[arm] = helpers["arm_action"](
                robot,
                arm,
                controller_delta,
                0.30,
            )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        observations.append(
            {
                "phase": "arm_shift_outward",
                "step": step + 1,
                "base_xy": np.asarray(backend.get_base_pose()[0], dtype=float).tolist(),
                "object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id],
                    dtype=float,
                ).tolist(),
                "contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
                "judge_collision": collision,
            }
        )
        if collision:
            break
        if all(
            float(np.linalg.norm(shift_targets[arm] - current[arm])) <= 0.012
            for arm in ("right", "left")
        ):
            break
    if collision_steps:
        return {
            "success": False,
            "failure_stage": "arm_shift_outward",
            "physical_contact_steps": 0,
            "object_translation_m": float(
                np.linalg.norm(raw_env.sim.data.body_xpos[body_id][:2] - start_object_xy)
            ),
            "base_translation_m": 0.0,
            "collision_steps": collision_steps,
            "observations": observations,
        }

    lower_starts = eef_positions()
    lower_targets = {
        arm: position + np.array([0.0, 0.0, -0.25], dtype=float)
        for arm, position in lower_starts.items()
    }
    lower_success = False
    for step in range(180):
        robot.composite_controller.update_state()
        current = eef_positions()
        arm_actions = {}
        for arm in ("right", "left"):
            controller_delta = helpers["world_delta"](
                robot,
                arm,
                lower_targets[arm] - current[arm],
            )
            arm_actions[arm] = helpers["arm_action"](
                robot,
                arm,
                controller_delta,
                0.30,
            )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        body_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
        observations.append(
            {
                "phase": "lower_to_table",
                "step": step + 1,
                "base_xy": np.asarray(backend.get_base_pose()[0], dtype=float).tolist(),
                "object_position": body_position.tolist(),
                "contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
                "judge_collision": collision,
            }
        )
        if collision:
            break
        if body_position[2] <= float(table_object_z) + 0.06:
            lower_success = True
            break

    if not lower_success:
        return {
            "success": False,
            "failure_stage": "lower_to_table",
            "physical_contact_steps": 0,
            "object_translation_m": float(
                np.linalg.norm(raw_env.sim.data.body_xpos[body_id][:2] - start_object_xy)
            ),
            "base_translation_m": float(
                np.linalg.norm(np.asarray(backend.get_base_pose()[0]) - start_base_xy)
            ),
            "collision_steps": collision_steps,
            "observations": observations,
        }

    _, base_yaw = backend.get_base_pose()
    base_velocity_xy = world_velocity_to_base_frame(
        direction * 0.04,
        base_yaw,
    )
    base_command = np.array(
        [base_velocity_xy[0], base_velocity_xy[1], 0.0],
        dtype=float,
    )
    stable_contact_steps = 0
    maximum_contact_steps = 0
    object_translation = 0.0
    base_translation = 0.0
    failure_stage = "timeout"
    success = False
    carry_driver = OfficialPhysicalCarryDriver()
    arm_push_delta = np.array(
        [direction[0] * 0.002, direction[1] * 0.002, 0.0],
        dtype=float,
    )
    for step in range(int(max_push_steps)):
        step_info = carry_driver.step(
            backend,
            object_name=object_name,
            base_command=base_command,
            hold_targets=hold_targets,
            arm_world_deltas={
                "right": arm_push_delta,
                "left": arm_push_delta,
            },
            gripper_value=1.0,
            base_control_dt=0.05,
        )
        collision = bool(step_info.get("collision", False))
        collision_steps += int(collision)
        contacts = object_robot_contacts(raw_env, object_name)
        has_contact = any(contacts[arm] for arm in ("right", "left"))
        stable_contact_steps = stable_contact_steps + 1 if has_contact else 0
        maximum_contact_steps = max(maximum_contact_steps, stable_contact_steps)
        base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
        base_translation = float(np.linalg.norm(base_xy - start_base_xy))
        object_translation = float(
            np.linalg.norm(object_position[:2] - start_object_xy)
        )
        observations.append(
            {
                "phase": "physical_push",
                "step": step + 1,
                "base_xy": base_xy.tolist(),
                "object_position": object_position.tolist(),
                "base_translation_m": base_translation,
                "object_translation_m": object_translation,
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "judge_collision": collision,
            }
        )
        if collision:
            from robot_agent.environments.robosuite_backend import (
                _navigation_collisions,
            )

            observations[-1]["judge_collision_pairs"] = [
                list(pair)
                for pair in _navigation_collisions(
                    raw_env,
                    robot,
                    getattr(backend, "_ignore_collision_geom", ()),
                )
            ]
            failure_stage = "collision"
            break
        if (
            base_translation >= 0.30
            and object_translation >= float(push_distance_m)
            and maximum_contact_steps >= 20
        ):
            success = True
            failure_stage = None
            break
        if base_translation >= 0.30:
            failure_stage = "object_distance"
            break

    return {
        "success": success,
        "failure_stage": failure_stage,
        "physical_contact_steps": maximum_contact_steps,
        "object_translation_m": object_translation,
        "base_translation_m": base_translation,
        "collision_steps": collision_steps,
        "push_direction": direction.tolist(),
        "observations": observations,
        "final_geometry": geometry_snapshot(raw_env, object_name),
    }


def _table_edge_undercut_probe(
    backend,
    object_name: str,
    *,
    table_edge_y: float,
    outside_clearance_m: float,
    edge_clearance_m: float,
    above_clearance_m: float,
    base_advance_m: float,
    object_offset_x_m: float,
    torso_target_m: float | None,
    raise_above_bottom_m: float,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        _is_allowed_cradle_geom,
        world_velocity_to_base_frame,
    )

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    hold_targets = helpers["capture_hold_targets"](robot)
    start_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    if not np.isfinite(float(base_advance_m)) or float(base_advance_m) < 0.0:
        raise ValueError("base_advance_m must be finite and non-negative")
    torso_joint_id = next(
        (
            index
            for index in range(raw_env.sim.model.njnt)
            if (raw_env.sim.model.joint_id2name(index) or "").endswith(
                "torso_lift_joint"
            )
        ),
        None,
    )
    if torso_joint_id is None:
        raise RuntimeError("torso lift joint is unavailable")
    torso_joint_name = raw_env.sim.model.joint_id2name(torso_joint_id)
    torso_qpos_addr = raw_env.sim.model.get_joint_qpos_addr(torso_joint_name)
    if isinstance(torso_qpos_addr, tuple):
        raise RuntimeError("torso lift joint must have a scalar qpos address")
    if torso_target_m is not None:
        torso_target = float(torso_target_m)
        torso_range = np.asarray(
            raw_env.sim.model.jnt_range[torso_joint_id], dtype=float
        )
        if (
            not np.isfinite(torso_target)
            or torso_target < float(torso_range[0])
            or torso_target > float(torso_range[1])
        ):
            raise ValueError("torso_target_m is outside the model joint range")
        if "torso" not in hold_targets:
            raise RuntimeError("torso controller hold target is unavailable")
    targets = table_edge_undercut_targets(
        object_center=start_object,
        object_half_depth_m=0.20,
        object_half_height_m=0.125,
        table_edge_y=table_edge_y,
        outside_clearance_m=outside_clearance_m,
        edge_clearance_m=edge_clearance_m,
        object_offset_x_m=object_offset_x_m,
        above_clearance_m=above_clearance_m,
        below_bottom_clearance_m=0.05,
        raise_above_bottom_m=raise_above_bottom_m,
    )
    observations: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    collision_steps = 0
    maximum_support_steps = 0

    def right_eef_position() -> np.ndarray:
        return np.asarray(
            helpers["gripper_position"](raw_env, robot, "right"),
            dtype=float,
        )

    def torso_position() -> float:
        return float(raw_env.sim.data.qpos[torso_qpos_addr])

    def execute_base_advance() -> bool:
        nonlocal collision_steps
        requested_distance = float(base_advance_m)
        if requested_distance == 0.0:
            return True
        control_dt = 0.05
        max_speed = 0.04
        driver = OfficialPhysicalCarryDriver()
        safety_failure = None
        success = False
        max_steps = int(np.ceil(requested_distance / (max_speed * control_dt))) + 5
        for local_step in range(max_steps):
            base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            translation = float(np.linalg.norm(base_xy - start_base_xy))
            remaining = max(0.0, requested_distance - translation)
            if remaining <= 1e-6:
                success = True
                break
            object_xy = np.asarray(
                raw_env.sim.data.body_xpos[body_id][:2], dtype=float
            )
            world_velocity = bounded_base_advance_world_velocity(
                base_xy=base_xy,
                object_xy=object_xy,
                remaining_m=remaining,
                max_speed_m_s=max_speed,
                control_dt_s=control_dt,
            )
            _, base_yaw = backend.get_base_pose()
            base_velocity = world_velocity_to_base_frame(world_velocity, base_yaw)
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.array(
                    [base_velocity[0], base_velocity[1], 0.0], dtype=float
                ),
                hold_targets=hold_targets,
                arm_world_deltas=None,
                gripper_value=-1.0,
                base_control_dt=control_dt,
            )
            collision = bool(step_info.get("collision", False))
            collision_steps += int(collision)
            contacts = object_robot_contacts(raw_env, object_name)
            premature_contact = any(contacts.values())
            measured_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            observations.append(
                {
                    "stage": "advance_base_for_undercut",
                    "step": local_step + 1,
                    "base_xy": measured_base_xy.tolist(),
                    "base_translation_m": float(
                        np.linalg.norm(measured_base_xy - start_base_xy)
                    ),
                    "eef_position": right_eef_position().tolist(),
                    "object_position": np.asarray(
                        raw_env.sim.data.body_xpos[body_id], dtype=float
                    ).tolist(),
                    "contacts": {
                        arm: list(names) for arm, names in contacts.items()
                    },
                    "judge_collision": collision,
                }
            )
            if collision:
                safety_failure = "collision"
                break
            if premature_contact:
                safety_failure = "premature_object_contact"
                break
        final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        final_translation = float(np.linalg.norm(final_base_xy - start_base_xy))
        success = bool(
            success
            or (
                safety_failure is None
                and final_translation >= requested_distance - 1e-6
            )
        )
        if not success and safety_failure is None:
            safety_failure = "timeout"
        stages.append(
            {
                "stage": "advance_base_for_undercut",
                "success": success,
                "steps": sum(
                    1
                    for item in observations
                    if item["stage"] == "advance_base_for_undercut"
                ),
                "safety_failure": safety_failure,
                "requested_translation_m": requested_distance,
                "base_translation_m": final_translation,
                "final_eef_position": right_eef_position().tolist(),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
            }
        )
        return success

    def execute_stage(
        stage: str,
        target: np.ndarray,
        *,
        allow_object_contact: bool,
        require_support: bool = False,
        max_steps: int = 180,
    ) -> bool:
        nonlocal collision_steps, maximum_support_steps
        stable_support_steps = 0
        safety_failure = None
        success = False
        for local_step in range(int(max_steps)):
            robot.composite_controller.update_state()
            current = right_eef_position()
            controller_delta = helpers["world_delta"](
                robot,
                "right",
                np.asarray(target, dtype=float) - current,
            )
            arm_action = helpers["arm_action"](
                robot,
                "right",
                controller_delta,
                0.12,
            )
            action = helpers["build_action"](
                robot,
                arm_actions={"right": arm_action},
                gripper_value=-1.0,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)
            measured_eef = right_eef_position()
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            )
            contacts = object_robot_contacts(raw_env, object_name)
            right_support = any(
                _is_allowed_cradle_geom(geom, "right")
                for geom in contacts["right"]
            )
            object_lift_m = float(object_position[2] - start_object[2])
            stable_support_steps = (
                stable_support_steps + 1
                if right_support and object_lift_m >= 0.02
                else 0
            )
            maximum_support_steps = max(
                maximum_support_steps,
                stable_support_steps,
            )
            collision = bool((info or {}).get("has_judge_collision", False))
            collision_steps += int(collision)
            observation = {
                "stage": stage,
                "step": local_step + 1,
                "target_eef_position": np.asarray(target, dtype=float).tolist(),
                "eef_position": measured_eef.tolist(),
                "object_position": object_position.tolist(),
                "object_lift_m": object_lift_m,
                "torso_position": torso_position(),
                "contacts": {
                    arm: list(names) for arm, names in contacts.items()
                },
                "right_support": right_support,
                "stable_support_steps": stable_support_steps,
                "judge_collision": collision,
            }
            observations.append(observation)
            if collision:
                from robot_agent.environments.robosuite_backend import (
                    _navigation_collisions,
                )

                observation["judge_collision_pairs"] = [
                    list(pair)
                    for pair in _navigation_collisions(
                        raw_env,
                        robot,
                        getattr(backend, "_ignore_collision_geom", ()),
                    )
                ]
                safety_failure = "collision"
                break
            if not allow_object_contact and any(contacts.values()):
                safety_failure = "premature_object_contact"
                break
            if require_support and stable_support_steps >= 5:
                success = True
                break
            if float(np.linalg.norm(np.asarray(target) - measured_eef)) <= 0.012:
                success = not require_support
                if require_support:
                    safety_failure = "target_without_support"
                break
        if not success and safety_failure is None:
            safety_failure = "timeout"
        stages.append(
            {
                "stage": stage,
                "success": success,
                "steps": sum(1 for item in observations if item["stage"] == stage),
                "safety_failure": safety_failure,
                "final_eef_position": right_eef_position().tolist(),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
                "final_torso_position": torso_position(),
                "final_contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(
                        raw_env, object_name
                    ).items()
                },
            }
        )
        return success

    success = execute_base_advance()
    failure_stage = None if success else "advance_base_for_undercut"
    if success:
        initial_eef = right_eef_position()
        clearance_target = initial_eef.copy()
        clearance_target[2] = max(
            clearance_target[2], float(targets["outside"][2])
        )
        sequence = (
            (
                "raise_open_clearance",
                clearance_target,
                False,
                False,
            ),
            ("move_open_outside", targets["outside"], False, False),
            ("descend_open_outside", targets["below"], False, False),
            ("inset_open_under_overhang", targets["undercut"], False, False),
            ("raise_open_into_support", targets["raise"], True, True),
        )
        for stage, target, allow_contact, require_support in sequence:
            if stage == "descend_open_outside" and torso_target_m is not None:
                hold_targets["torso"] = np.array(
                    [float(torso_target_m)], dtype=float
                )
            if not execute_stage(
                stage,
                target,
                allow_object_contact=allow_contact,
                require_support=require_support,
            ):
                success = False
                failure_stage = stage
                break
    final_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    return {
        "success": success,
        "failure_stage": failure_stage,
        "open_gripper": True,
        "support_contact_steps": maximum_support_steps,
        "object_lift_m": float(final_object[2] - start_object[2]),
        "collision_steps": collision_steps,
        "base_translation_m": float(
            np.linalg.norm(np.asarray(backend.get_base_pose()[0]) - start_base_xy)
        ),
        "torso_target_m": torso_target_m,
        "final_torso_position": torso_position(),
        "start_object_position": start_object.tolist(),
        "final_object_position": final_object.tolist(),
        "targets": {name: target.tolist() for name, target in targets.items()},
        "stages": stages,
        "observations": observations,
        "final_geometry": geometry_snapshot(raw_env, object_name),
    }


def _center_regrasp_probe(
    backend,
    object_name: str,
    *,
    table_object_z: float,
    center_shift_m: float,
    wall_clearance_m: float,
    wall_squeeze_m: float,
    base_advance_m: float,
    hold_steps: int,
    align_closure_axes: bool,
    orientation_max_action: float,
    orientation_fine_max_action: float | None,
    orientation_fine_threshold_deg: float,
    orientation_tolerance_deg: float,
    orientation_stable_steps: int,
    orientation_max_steps: int,
    orientation_max_position_drift_m: float,
    orientation_joint_seed: bool = False,
    orientation_joint_seed_margin_rad: float = 0.03,
    orientation_joint_seed_max_nfev: int = 800,
    orientation_joint_seed_steps: int = 240,
    orientation_joint_seed_position_scale_m: float = 0.01,
    orientation_joint_seed_axis_scale: float = float(np.sin(np.deg2rad(5.0))),
    orientation_joint_seed_regularization: float = 0.02,
    orientation_joint_seed_max_error_deg: float = JOINT_SEED_THRESHOLDS["error_deg"],
    orientation_joint_seed_max_endpoint_position_error_m: float = (
        JOINT_SEED_THRESHOLDS["max_endpoint_position_error_m"]
    ),
    orientation_joint_seed_continuation_nodes: int = 1,
    orientation_joint_seed_include_torso: bool = False,
    orientation_joint_seed_torso_margin_m: float = 0.005,
    center_carry_distance_m: float = 0.0,
    center_carry_max_linear: float = 0.04,
    center_carry_away_from_object: bool = False,
    center_carry_corner_seat_m: float = 0.0,
    center_carry_arm_stroke_m: float = 0.0,
    center_carry_arm_stroke_lift_m: float = 0.0,
    center_carry_base_reset_m: float = 0.0,
    center_carry_inchworm_distance_m: float = 0.0,
    center_carry_inchworm_toward_base: bool = False,
    center_carry_inchworm_stroke_m: float = 0.08,
    center_carry_inchworm_reset_m: float = 0.06,
    center_carry_inchworm_world_direction_x: float | None = None,
    center_carry_inchworm_world_direction_y: float | None = None,
    center_support_moving_arm: str = "none",
    center_support_clearance_lift_m: float = 0.08,
    center_support_descent_m: float = 0.12,
    center_support_inset_m: float = 0.04,
    center_support_keep_moving_gripper_closed: bool = False,
    center_support_combined_motion: bool = False,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        build_independent_gripper_action,
        gripper_close_command,
        joint_interpolation_path,
        synchronize_controller_goals,
    )
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        InchwormCarryConfig,
        PhysicalCarryConfig,
        _is_allowed_cradle_geom,
        run_inchworm_transport,
        run_physical_transport,
        single_arm_under_support_targets,
        world_velocity_to_base_frame,
    )

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    hold_targets = helpers["capture_hold_targets"](robot)
    observations = []
    stage_results = []
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0
    orientation_alignment = None
    joint_seed = None
    physical_grasp = False
    hold_grasp_steps = 0
    transport_result = None
    support_transition = None
    transport_object_translation = 0.0
    transport_base_translation = 0.0
    stroke_projected_progress = 0.0
    if orientation_joint_seed and not align_closure_axes:
        raise ValueError("orientation_joint_seed requires align_closure_axes")
    if center_support_moving_arm not in ("none", "right", "left"):
        raise ValueError("center_support_moving_arm must be none, right, or left")
    for parameter_name, parameter_value in (
        ("center_support_clearance_lift_m", center_support_clearance_lift_m),
        ("center_support_descent_m", center_support_descent_m),
        ("center_support_inset_m", center_support_inset_m),
    ):
        if not np.isfinite(float(parameter_value)) or float(parameter_value) < 0.0:
            raise ValueError(f"{parameter_name} must be finite and non-negative")

    def eef_positions() -> dict[str, np.ndarray]:
        return {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }

    def alignment_eef_poses() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            arm: eef_site_pose(raw_env, robot, arm)
            for arm in ("right", "left")
        }

    def execute_base_advance(distance_m: float) -> bool:
        nonlocal collision_steps
        requested_distance = float(distance_m)
        if not np.isfinite(requested_distance) or requested_distance < 0.0:
            raise ValueError("base advance distance must be finite and non-negative")
        if requested_distance == 0.0:
            return True
        control_dt = 0.05
        max_speed = 0.04
        start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        driver = OfficialPhysicalCarryDriver()
        success = False
        collision = False
        premature_contact = False
        max_steps = int(np.ceil(requested_distance / (max_speed * control_dt))) + 5
        for local_step in range(max_steps):
            base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            translation = float(np.linalg.norm(base_xy - start_base_xy))
            remaining = max(0.0, requested_distance - translation)
            if remaining <= 1e-6:
                success = True
                break
            object_xy = np.asarray(
                raw_env.sim.data.body_xpos[body_id][:2],
                dtype=float,
            )
            world_velocity = bounded_base_advance_world_velocity(
                base_xy=base_xy,
                object_xy=object_xy,
                remaining_m=remaining,
                max_speed_m_s=max_speed,
                control_dt_s=control_dt,
            )
            _, base_yaw = backend.get_base_pose()
            base_velocity = world_velocity_to_base_frame(
                world_velocity,
                base_yaw,
            )
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.array(
                    [base_velocity[0], base_velocity[1], 0.0],
                    dtype=float,
                ),
                hold_targets=hold_targets,
                arm_world_deltas=None,
                gripper_value=-1.0,
                base_control_dt=control_dt,
            )
            collision = bool(step_info.get("collision", False))
            collision_steps += int(collision)
            contacts = object_robot_contacts(raw_env, object_name)
            premature_contact = any(bool(names) for names in contacts.values())
            measured_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            translation = float(
                np.linalg.norm(measured_base_xy - start_base_xy)
            )
            observations.append(
                {
                    "stage": "advance_base_for_regrasp",
                    "step": local_step + 1,
                    "base_xy": measured_base_xy.tolist(),
                    "base_translation_m": translation,
                    "object_position": np.asarray(
                        raw_env.sim.data.body_xpos[body_id],
                        dtype=float,
                    ).tolist(),
                    "eef_positions": {
                        arm: position.tolist()
                        for arm, position in eef_positions().items()
                    },
                    "contacts": {
                        arm: list(names) for arm, names in contacts.items()
                    },
                    "judge_collision": collision,
                }
            )
            if collision or premature_contact:
                break
        final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        final_translation = float(np.linalg.norm(final_base_xy - start_base_xy))
        if final_translation >= requested_distance - 1e-6:
            success = success or not collision and not premature_contact
        stage_results.append(
            {
                "stage": "advance_base_for_regrasp",
                "success": bool(success and not collision and not premature_contact),
                "collision": collision,
                "premature_object_contact": premature_contact,
                "steps": sum(
                    1
                    for item in observations
                    if item["stage"] == "advance_base_for_regrasp"
                ),
                "requested_translation_m": requested_distance,
                "base_translation_m": final_translation,
                "start_base_xy": start_base_xy.tolist(),
                "final_base_xy": final_base_xy.tolist(),
            }
        )
        return bool(success and not collision and not premature_contact)

    def execute_joint_orientation_seed(target_axis: np.ndarray) -> dict[str, object]:
        nonlocal collision_steps
        arms = ("right", "left")
        model = raw_env.sim.model
        data = raw_env.sim.data
        include_torso = bool(orientation_joint_seed_include_torso)
        joint_names = list(
            joint_seed_joint_names(include_torso=include_torso)
        )
        joint_ids = [model.joint_name2id(name) for name in joint_names]
        qpos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        if any(isinstance(address, tuple) for address in qpos_addrs):
            raise RuntimeError("arm joints must have scalar qpos addresses")
        official_lower = np.asarray(
            [model.jnt_range[joint_id][0] for joint_id in joint_ids],
            dtype=float,
        )
        official_upper = np.asarray(
            [model.jnt_range[joint_id][1] for joint_id in joint_ids],
            dtype=float,
        )
        arm_lower, arm_upper = interior_joint_bounds(
            official_lower[:12],
            official_upper[:12],
            margin_rad=orientation_joint_seed_margin_rad,
        )
        if include_torso:
            torso_lower, torso_upper = interior_joint_bounds(
                official_lower[12:],
                official_upper[12:],
                margin_rad=orientation_joint_seed_torso_margin_m,
            )
            lower = np.concatenate([arm_lower, torso_lower])
            upper = np.concatenate([arm_upper, torso_upper])
        else:
            lower, upper = arm_lower, arm_upper
        start = np.asarray(data.qpos[qpos_addrs], dtype=float).copy()
        start_poses = alignment_eef_poses()
        start_positions = {arm: start_poses[arm][0] for arm in arms}
        start_axes = {arm: start_poses[arm][1][:, 0] for arm in arms}
        target_axes = {
            arm: nearest_directed_axis_target(start_axes[arm], target_axis)
            for arm in arms
        }
        joint_ranges = official_upper - official_lower
        steps = int(orientation_joint_seed_steps)
        max_nfev = int(orientation_joint_seed_max_nfev)
        max_error = float(orientation_joint_seed_max_error_deg)
        max_endpoint_position_error = float(
            orientation_joint_seed_max_endpoint_position_error_m
        )
        continuation_nodes = int(orientation_joint_seed_continuation_nodes)
        if steps < 1 or max_nfev < 1:
            raise ValueError("joint seed steps and max_nfev must be positive")
        if (
            isinstance(orientation_joint_seed_continuation_nodes, bool)
            or continuation_nodes != orientation_joint_seed_continuation_nodes
            or continuation_nodes < 1
            or continuation_nodes > steps
        ):
            raise ValueError(
                "joint seed continuation nodes must be in [1, total steps]"
            )
        if (
            not np.isfinite(max_error)
            or max_error < 0.0
            or max_error > JOINT_SEED_THRESHOLDS["error_deg"]
        ):
            raise ValueError("joint seed endpoint error exceeds the hard gate")
        if (
            not np.isfinite(max_endpoint_position_error)
            or max_endpoint_position_error < 0.0
            or max_endpoint_position_error
            > JOINT_SEED_THRESHOLDS["max_endpoint_position_error_m"]
        ):
            raise ValueError("joint seed endpoint position error exceeds the hard gate")

        summary: dict[str, object] = {
            "success": False,
            "failure": None,
            "right_error_deg": closure_axis_error_degrees(
                start_axes["right"], target_axes["right"]
            ),
            "left_error_deg": closure_axis_error_degrees(
                start_axes["left"], target_axes["left"]
            ),
            "max_endpoint_position_error_m": 0.0,
            "max_path_position_drift_m": 0.0,
            "min_bound_margin_rad": 0.0,
            "collision_frames": 0,
            "collision_pairs": [],
            "rolled_back": False,
            "controller_synchronized": False,
            "joint_names": joint_names,
            "include_torso": include_torso,
            "arm_margin_rad": float(orientation_joint_seed_margin_rad),
            "torso_margin_m": (
                float(orientation_joint_seed_torso_margin_m)
                if include_torso
                else None
            ),
            "initial_torso_joint_m": float(start[-1]) if include_torso else None,
            "refreshed_torso_hold_target": None,
            "start_joints": start.tolist(),
            "target_joints": None,
            "official_lower_bounds": official_lower.tolist(),
            "official_upper_bounds": official_upper.tolist(),
            "interior_lower_bounds": lower.tolist(),
            "interior_upper_bounds": upper.tolist(),
            "initial_positions": {
                arm: start_positions[arm].tolist() for arm in arms
            },
            "initial_closure_axes": {
                arm: start_axes[arm].tolist() for arm in arms
            },
            "target_closure_axes": {
                arm: target_axes[arm].tolist() for arm in arms
            },
            "solver": None,
            "continuation_node_count": continuation_nodes,
            "continuation_nodes": [],
            "endpoint_residual": None,
            "waypoint_count": 0,
            "failed_waypoint": None,
        }

        def restore_start(*, record_frame: bool) -> None:
            data.qpos[qpos_addrs] = start
            raw_env.sim.forward()
            synchronize_controller_goals(robot)
            summary["controller_synchronized"] = True
            if record_frame:
                recorder = getattr(backend, "_record_trajectory_frame", None)
                if callable(recorder):
                    recorder(_env=raw_env)

        from scipy.optimize import least_squares
        from robot_agent.environments.robosuite_backend import (
            _navigation_collisions,
        )

        solution = None
        proposal = None
        proposals: list[np.ndarray] = []
        node_records: list[dict[str, object]] = []
        solver_reference = start.copy()
        try:
            for node_index in range(1, continuation_nodes + 1):
                fraction = node_index / continuation_nodes
                node_target_axes = {
                    arm: interpolate_directed_axis(
                        start_axes[arm],
                        target_axes[arm],
                        fraction=fraction,
                    )
                    for arm in arms
                }
                solver_start = np.clip(
                    solver_reference,
                    lower + 1e-9,
                    upper - 1e-9,
                )

                def residual(joints: np.ndarray) -> np.ndarray:
                    data.qpos[qpos_addrs] = joints
                    raw_env.sim.forward()
                    poses = alignment_eef_poses()
                    return joint_seed_objective_residual(
                        current_positions={arm: poses[arm][0] for arm in arms},
                        target_positions=start_positions,
                        current_axes={arm: poses[arm][1][:, 0] for arm in arms},
                        target_axes=node_target_axes,
                        joints=joints,
                        start_joints=solver_reference,
                        joint_ranges=joint_ranges,
                        position_scale_m=orientation_joint_seed_position_scale_m,
                        axis_scale=orientation_joint_seed_axis_scale,
                        regularization=orientation_joint_seed_regularization,
                    )

                solution = least_squares(
                    residual,
                    solver_start,
                    bounds=(lower, upper),
                    max_nfev=max_nfev,
                )
                proposal = np.asarray(solution.x, dtype=float).copy()
                data.qpos[qpos_addrs] = proposal
                raw_env.sim.forward()
                node_poses = alignment_eef_poses()
                node_errors = {
                    arm: closure_axis_error_degrees(
                        node_poses[arm][1][:, 0], node_target_axes[arm]
                    )
                    for arm in arms
                }
                node_position_errors = {
                    arm: float(
                        np.linalg.norm(
                            node_poses[arm][0] - start_positions[arm]
                        )
                    )
                    for arm in arms
                }
                node_collisions = list(
                    _navigation_collisions(
                        raw_env,
                        robot,
                        getattr(backend, "_ignore_collision_geom", ()),
                    )
                )
                node_bound_margin = float(
                    np.min(
                        np.concatenate([proposal - lower, upper - proposal])
                    )
                )
                solver_record = {
                    "success": bool(solution.success),
                    "status": int(solution.status),
                    "message": str(solution.message),
                    "nfev": int(solution.nfev),
                    "cost": float(solution.cost),
                    "optimality": float(solution.optimality),
                }
                node_record: dict[str, object] = {
                    "node": node_index,
                    "fraction": fraction,
                    "target_axes": {
                        arm: node_target_axes[arm].tolist() for arm in arms
                    },
                    "joints": proposal.tolist(),
                    "right_error_deg": node_errors["right"],
                    "left_error_deg": node_errors["left"],
                    "position_errors_m": node_position_errors,
                    "max_position_error_m": max(node_position_errors.values()),
                    "min_bound_margin_rad": node_bound_margin,
                    "collision_pairs": [
                        [str(name) for name in pair] for pair in node_collisions
                    ],
                    "solver": solver_record,
                    "failure": None,
                }
                node_record["failure"] = joint_seed_node_failure(
                    solver_success=bool(solution.success),
                    right_error_deg=node_errors["right"],
                    left_error_deg=node_errors["left"],
                    position_error_m=max(node_position_errors.values()),
                    min_bound_margin_rad=node_bound_margin,
                    collision=bool(node_collisions),
                )
                node_records.append(node_record)
                if node_record["failure"] is not None:
                    if node_collisions:
                        collision_steps += 1
                        summary["collision_frames"] = int(
                            summary["collision_frames"]
                        ) + 1
                        summary["collision_pairs"] = [
                            [str(name) for name in pair]
                            for pair in node_collisions
                        ]
                    summary["failure"] = (
                        f"continuation_node_{node_index}_{node_record['failure']}"
                    )
                    break
                proposals.append(proposal)
                solver_reference = proposal
        except Exception as exc:
            summary["failure"] = "solver_exception"
            summary["solver"] = {
                "success": False,
                "exception": f"{type(exc).__name__}: {exc}",
            }
        finally:
            data.qpos[qpos_addrs] = start
            raw_env.sim.forward()

        summary["continuation_nodes"] = node_records
        if node_records:
            summary["solver"] = node_records[-1]["solver"]
        if (
            solution is None
            or proposal is None
            or len(proposals) != continuation_nodes
        ):
            summary["rolled_back"] = True
            restore_start(record_frame=True)
            stage_results.append({"stage": "joint_space_wrist_seed", **summary})
            return summary

        summary["target_joints"] = proposal.tolist()

        endpoint_collisions: list[tuple[str, str]] = []
        try:
            data.qpos[qpos_addrs] = proposal
            raw_env.sim.forward()
            endpoint_poses = alignment_eef_poses()
            endpoint_positions = {arm: endpoint_poses[arm][0] for arm in arms}
            endpoint_axes = {arm: endpoint_poses[arm][1][:, 0] for arm in arms}
            endpoint_errors = {
                arm: closure_axis_error_degrees(
                    endpoint_axes[arm], target_axes[arm]
                )
                for arm in arms
            }
            endpoint_position_errors = {
                arm: float(
                    np.linalg.norm(endpoint_positions[arm] - start_positions[arm])
                )
                for arm in arms
            }
            endpoint_residual = joint_seed_objective_residual(
                current_positions=endpoint_positions,
                target_positions=start_positions,
                current_axes=endpoint_axes,
                target_axes=target_axes,
                joints=proposal,
                start_joints=start,
                joint_ranges=joint_ranges,
                position_scale_m=orientation_joint_seed_position_scale_m,
                axis_scale=orientation_joint_seed_axis_scale,
                regularization=orientation_joint_seed_regularization,
            )
            endpoint_collisions = list(
                _navigation_collisions(
                    raw_env,
                    robot,
                    getattr(backend, "_ignore_collision_geom", ()),
                )
            )
            summary.update(
                {
                    "right_error_deg": endpoint_errors["right"],
                    "left_error_deg": endpoint_errors["left"],
                    "max_endpoint_position_error_m": max(
                        endpoint_position_errors.values()
                    ),
                    "min_bound_margin_rad": float(
                        np.min(
                            np.concatenate(
                                [proposal - lower, upper - proposal]
                            )
                        )
                    ),
                    "endpoint_positions": {
                        arm: endpoint_positions[arm].tolist() for arm in arms
                    },
                    "endpoint_closure_axes": {
                        arm: endpoint_axes[arm].tolist() for arm in arms
                    },
                    "endpoint_position_errors_m": endpoint_position_errors,
                    "endpoint_residual": endpoint_residual.tolist(),
                    "collision_pairs": [
                        [str(name) for name in pair]
                        for pair in endpoint_collisions
                    ],
                    "collision_frames": int(bool(endpoint_collisions)),
                }
            )
        finally:
            data.qpos[qpos_addrs] = start
            raw_env.sim.forward()

        endpoint_failure = None
        if not bool(solution.success):
            endpoint_failure = "solver"
        elif endpoint_collisions:
            endpoint_failure = "endpoint_collision"
        elif float(summary["right_error_deg"]) > max_error or float(
            summary["left_error_deg"]
        ) > max_error:
            endpoint_failure = "endpoint_orientation"
        elif (
            float(summary["max_endpoint_position_error_m"])
            > max_endpoint_position_error
        ):
            endpoint_failure = "endpoint_position"
        elif float(summary["min_bound_margin_rad"]) < 0.0:
            endpoint_failure = "endpoint_bounds"
        if endpoint_failure is not None:
            collision_steps += int(bool(endpoint_collisions))
            summary["failure"] = endpoint_failure
            summary["rolled_back"] = True
            restore_start(record_frame=True)
            stage_results.append({"stage": "joint_space_wrist_seed", **summary})
            return summary

        path_state: dict[str, object] = {
            "max_position_drift_m": 0.0,
            "collision_frames": 0,
            "collision_pairs": [],
        }
        segment_steps = allocate_segment_steps(
            total_steps=steps,
            segment_count=len(proposals),
        )
        segment_start = start
        waypoint_index = 0
        for node_index, (segment_target, local_steps) in enumerate(
            zip(proposals, segment_steps),
            start=1,
        ):
            for local_step, values in enumerate(
                joint_interpolation_path(
                    segment_start,
                    segment_target,
                    steps=local_steps,
                ),
                start=1,
            ):
                waypoint_index += 1
                data.qpos[qpos_addrs] = values
                raw_env.sim.forward()
                poses = alignment_eef_poses()
                drift = {
                    arm: float(
                        np.linalg.norm(poses[arm][0] - start_positions[arm])
                    )
                    for arm in arms
                }
                collisions = list(
                    _navigation_collisions(
                        raw_env,
                        robot,
                        getattr(backend, "_ignore_collision_geom", ()),
                    )
                )
                path_state = next_joint_seed_path_state(
                    path_state,
                    waypoint_index=waypoint_index,
                    right_drift_m=drift["right"],
                    left_drift_m=drift["left"],
                    collision_pairs=collisions,
                    max_position_drift_m=JOINT_SEED_THRESHOLDS[
                        "max_path_position_drift_m"
                    ],
                )
                observations.append(
                    {
                        "stage": "joint_space_wrist_seed",
                        "step": waypoint_index,
                        "continuation_node": node_index,
                        "node_local_step": local_step,
                        "joints": np.asarray(values, dtype=float).tolist(),
                        "eef_positions": {
                            arm: poses[arm][0].tolist() for arm in arms
                        },
                        "closure_axes": {
                            arm: poses[arm][1][:, 0].tolist() for arm in arms
                        },
                        "position_drift_m": drift,
                        "judge_collision_pairs": [
                            [str(name) for name in pair]
                            for pair in collisions
                        ],
                    }
                )
                recorder = getattr(backend, "_record_trajectory_frame", None)
                if callable(recorder):
                    recorder(_env=raw_env)
                if bool(path_state["terminate"]):
                    break
            if bool(path_state["terminate"]):
                break
            segment_start = segment_target

        summary.update(
            {
                "max_path_position_drift_m": float(
                    path_state["max_position_drift_m"]
                ),
                "waypoint_count": int(path_state["waypoint_count"]),
                "failed_waypoint": path_state["failed_waypoint"],
                "collision_frames": int(summary["collision_frames"])
                + int(path_state["collision_frames"]),
                "collision_pairs": list(summary["collision_pairs"])
                + [
                    pair
                    for pair in path_state["collision_pairs"]
                    if pair not in summary["collision_pairs"]
                ],
            }
        )
        if bool(path_state["terminate"]):
            collision_steps += int(path_state["collision_frames"])
            summary["failure"] = f"path_{path_state['failure']}"
            summary["rolled_back"] = True
            restore_start(record_frame=True)
        else:
            synchronize_controller_goals(robot)
            summary["controller_synchronized"] = True
            if include_torso:
                refreshed_holds = helpers["capture_hold_targets"](robot)
                hold_targets["torso"] = np.asarray(
                    refreshed_holds["torso"],
                    dtype=float,
                ).copy()
                summary["refreshed_torso_hold_target"] = hold_targets[
                    "torso"
                ].tolist()
            summary["success"] = True
        stage_results.append({"stage": "joint_space_wrist_seed", **summary})
        return summary

    def controller_origin_rotation(arm: str) -> np.ndarray:
        controller = robot.part_controllers[arm]
        if controller.name != "OSC_POSE" or controller.input_type != "delta":
            raise RuntimeError(
                f"orientation alignment requires {arm} OSC_POSE delta control"
            )
        input_ref_frame = getattr(controller, "input_ref_frame", "world")
        if input_ref_frame == "world":
            return np.eye(3)
        if input_ref_frame != "base":
            raise RuntimeError(
                f"unsupported orientation reference frame for {arm}: "
                f"{input_ref_frame}"
            )
        origin = controller.origin_ori
        if origin is None:
            _, origin = robot.composite_controller.get_controller_base_pose(
                controller_name=arm
            )
        return _validated_rotation_matrix(
            origin,
            name=f"{arm} controller origin rotation",
        )

    def execute_orientation_alignment(target_axis: np.ndarray) -> dict[str, object]:
        nonlocal collision_steps
        if int(orientation_max_steps) < 1:
            raise ValueError("orientation_max_steps must be positive")
        hold_poses = alignment_eef_poses()
        hold_positions = {arm: hold_poses[arm][0] for arm in ("right", "left")}
        state: dict[str, object] = {
            "stable_steps": 0,
            "max_position_drift_m": 0.0,
        }
        alignment_collision_frames = 0
        for local_step in range(int(orientation_max_steps)):
            robot.composite_controller.update_state()
            current_poses = alignment_eef_poses()
            current_positions = {
                arm: current_poses[arm][0] for arm in ("right", "left")
            }
            current_orientations = {
                arm: current_poses[arm][1] for arm in ("right", "left")
            }
            arm_actions = {}
            orientation_actions = {}
            orientation_action_limits = {}
            for arm in ("right", "left"):
                controller = robot.part_controllers[arm]
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    hold_positions[arm] - current_positions[arm],
                )
                arm_action = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    0.30,
                )
                closure_axis = current_orientations[arm][:, 0]
                current_error = closure_axis_error_degrees(
                    closure_axis,
                    target_axis,
                )
                action_limit = float(orientation_max_action)
                if orientation_fine_max_action is not None:
                    action_limit = scheduled_orientation_action_limit(
                        error_deg=current_error,
                        coarse_action=orientation_max_action,
                        fine_action=orientation_fine_max_action,
                        fine_threshold_deg=orientation_fine_threshold_deg,
                    )
                world_rotation_delta = minimum_undirected_axis_rotation(
                    closure_axis,
                    target_axis,
                )
                orientation_action = normalized_osc_orientation_command(
                    world_rotation_delta=world_rotation_delta,
                    controller_origin_rotation=controller_origin_rotation(arm),
                    output_min=controller.output_min,
                    output_max=controller.output_max,
                    max_action=action_limit,
                )
                arm_action[3:6] = orientation_action
                arm_actions[arm] = arm_action
                orientation_actions[arm] = orientation_action
                orientation_action_limits[arm] = action_limit
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=-1.0,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)

            measured_poses = alignment_eef_poses()
            measured_positions = {
                arm: measured_poses[arm][0] for arm in ("right", "left")
            }
            measured_orientations = {
                arm: measured_poses[arm][1] for arm in ("right", "left")
            }
            position_drift = max(
                float(
                    np.linalg.norm(
                        measured_positions[arm] - hold_positions[arm]
                    )
                )
                for arm in ("right", "left")
            )
            errors = {
                arm: closure_axis_error_degrees(
                    measured_orientations[arm][:, 0],
                    target_axis,
                )
                for arm in ("right", "left")
            }
            collision = bool((info or {}).get("has_judge_collision", False))
            collision_steps += int(collision)
            alignment_collision_frames += int(collision)
            state = next_orientation_alignment_state(
                state,
                right_error_deg=errors["right"],
                left_error_deg=errors["left"],
                position_drift_m=position_drift,
                collision=collision,
                tolerance_deg=orientation_tolerance_deg,
                required_stable_steps=orientation_stable_steps,
                max_position_drift_m=orientation_max_position_drift_m,
            )
            observations.append(
                {
                    "stage": "align_closure_axes_high",
                    "step": local_step + 1,
                    "object_position": np.asarray(
                        raw_env.sim.data.body_xpos[body_id],
                        dtype=float,
                    ).tolist(),
                    "eef_positions": {
                        arm: measured_positions[arm].tolist()
                        for arm in ("right", "left")
                    },
                    "eef_orientations": {
                        arm: measured_orientations[arm].tolist()
                        for arm in ("right", "left")
                    },
                    "closure_axes": {
                        arm: measured_orientations[arm][:, 0].tolist()
                        for arm in ("right", "left")
                    },
                    "target_closure_axis": np.asarray(
                        target_axis,
                        dtype=float,
                    ).tolist(),
                    "orientation_actions": {
                        arm: orientation_actions[arm].tolist()
                        for arm in ("right", "left")
                    },
                    "orientation_action_limits": orientation_action_limits,
                    "orientation_errors_deg": errors,
                    "position_drift_m": position_drift,
                    "orientation_stable_steps": int(state["stable_steps"]),
                    "judge_collision": collision,
                }
            )
            if bool(state["aligned"]) or bool(state["terminate"]):
                break

        summary = {
            "success": bool(state.get("aligned", False)),
            "failure": state.get("failure")
            or (None if state.get("aligned") else "timeout"),
            "right_error_deg": float(state.get("right_error_deg", float("inf"))),
            "left_error_deg": float(state.get("left_error_deg", float("inf"))),
            "stable_steps": int(state.get("stable_steps", 0)),
            "max_position_drift_m": float(state["max_position_drift_m"]),
            "collision_frames": alignment_collision_frames,
            "steps": sum(
                1
                for item in observations
                if item["stage"] == "align_closure_axes_high"
            ),
        }
        stage_results.append(
            {
                "stage": "align_closure_axes_high",
                **summary,
                "collision": bool(alignment_collision_frames),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id],
                    dtype=float,
                ).tolist(),
            }
        )
        return summary

    def execute_stage(
        name: str,
        targets: Mapping[str, np.ndarray],
        *,
        max_steps: int,
        gripper_value: float | Mapping[str, float],
        close_schedule: bool = False,
        stop_object_z: float | None = None,
        stop_object_z_at_least: float | None = None,
        stop_bilateral_support_steps: int | None = None,
        stop_bilateral_contact_steps: int | None = None,
        stop_grasp_contact_steps: int | None = None,
        support_seek_down_step: float = 0.0,
        support_seek_down_limit: float = 0.0,
        minimum_object_z: float | None = None,
        required_contact_arm: str | None = None,
        require_bilateral_grasp: bool = False,
        require_target: bool = True,
    ) -> bool:
        nonlocal stable_support_steps
        nonlocal maximum_support_steps
        nonlocal collision_steps
        reached = False
        collision = False
        stage_support_steps = 0
        stage_contact_steps = 0
        support_stop_met = False
        contact_stop_met = False
        object_stop_met = False
        grasp_stop_met = False
        stage_grasp_steps = 0
        safety_failure = None
        if required_contact_arm not in (None, "right", "left"):
            raise ValueError("required_contact_arm must be right, left, or None")
        active_targets = {
            arm: np.asarray(targets[arm], dtype=float).copy()
            for arm in ("right", "left")
        }
        initial_target_z = {
            arm: float(active_targets[arm][2]) for arm in ("right", "left")
        }
        for local_step in range(int(max_steps)):
            robot.composite_controller.update_state()
            current = eef_positions()
            arm_actions = {}
            for arm in ("right", "left"):
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    active_targets[arm] - current[arm],
                )
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    0.30,
                )
            if close_schedule:
                if isinstance(gripper_value, Mapping):
                    raise ValueError("close_schedule requires a scalar gripper value")
                command: float | Mapping[str, float] = gripper_close_command(
                    local_step,
                    interval=1,
                )
            else:
                command = (
                    {arm: float(gripper_value[arm]) for arm in ("right", "left")}
                    if isinstance(gripper_value, Mapping)
                    else float(gripper_value)
                )
            if isinstance(command, Mapping):
                action = build_independent_gripper_action(
                    robot,
                    arm_actions=arm_actions,
                    gripper_values=command,
                    hold_targets=hold_targets,
                    build_action_fn=helpers["build_action"],
                )
            else:
                action = helpers["build_action"](
                    robot,
                    arm_actions=arm_actions,
                    gripper_value=command,
                    hold_targets=hold_targets,
                )
            _, _, _, info = raw_env.step(action)
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)
            contacts = object_robot_contacts(raw_env, object_name)
            bilateral_contact = has_bilateral_object_contact(contacts)
            arm_supported = {
                arm: any(
                    _is_allowed_cradle_geom(geom, arm)
                    for geom in contacts[arm]
                )
                for arm in ("right", "left")
            }
            supported = all(arm_supported.values())
            grasp_contacts = helpers["grasp_status"](
                raw_env,
                robot,
                object_name,
            )
            grasped = all(
                bool(grasp_contacts.get(arm)) for arm in ("right", "left")
            )
            stage_grasp_steps = stage_grasp_steps + 1 if grasped else 0
            stage_support_steps = stage_support_steps + 1 if supported else 0
            stage_contact_steps = stage_contact_steps + 1 if bilateral_contact else 0
            stable_support_steps = stable_support_steps + 1 if supported else 0
            maximum_support_steps = max(maximum_support_steps, stable_support_steps)
            collision = bool((info or {}).get("has_judge_collision", False))
            collision_steps += int(collision)
            object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
            observations.append(
                {
                    "stage": name,
                    "step": local_step + 1,
                    "object_position": object_position.tolist(),
                    "eef_positions": {
                        arm: current[arm].tolist() for arm in ("right", "left")
                    },
                    "contacts": {arm: list(contacts[arm]) for arm in contacts},
                    "arm_support": arm_supported,
                    "grasp_contacts": {
                        arm: bool(grasp_contacts.get(arm))
                        for arm in ("right", "left")
                    },
                    "bilateral_support": supported,
                    "bilateral_grasp": grasped,
                    "judge_collision": collision,
                }
            )
            if collision:
                break
            if require_bilateral_grasp and not grasped:
                safety_failure = "bilateral_grasp_loss"
                break
            if (
                minimum_object_z is not None
                and float(object_position[2]) < float(minimum_object_z)
            ):
                safety_failure = "height_loss"
                break
            if required_contact_arm is not None and not bool(
                contacts.get(required_contact_arm)
            ):
                safety_failure = "required_contact_loss"
                break
            if float(support_seek_down_step) > 0.0:
                for arm in ("right", "left"):
                    if arm_supported[arm]:
                        continue
                    active_targets[arm][2] = max(
                        initial_target_z[arm] - float(support_seek_down_limit),
                        active_targets[arm][2] - float(support_seek_down_step),
                    )
            if stop_object_z is not None and object_position[2] <= float(stop_object_z):
                reached = True
                object_stop_met = True
                break
            if (
                stop_object_z_at_least is not None
                and object_position[2] >= float(stop_object_z_at_least)
            ):
                reached = True
                object_stop_met = True
                break
            if (
                stop_bilateral_support_steps is not None
                and stage_support_steps >= int(stop_bilateral_support_steps)
            ):
                reached = True
                support_stop_met = True
                break
            if (
                stop_bilateral_contact_steps is not None
                and stage_contact_steps >= int(stop_bilateral_contact_steps)
            ):
                reached = True
                contact_stop_met = True
                break
            if (
                stop_grasp_contact_steps is not None
                and stage_grasp_steps >= int(stop_grasp_contact_steps)
            ):
                reached = True
                grasp_stop_met = True
                break
            reached = all(
                float(np.linalg.norm(active_targets[arm] - current[arm])) <= 0.012
                for arm in ("right", "left")
            )
            if (
                reached
                and require_target
                and not close_schedule
                and stop_bilateral_support_steps is None
                and stop_bilateral_contact_steps is None
                and stop_grasp_contact_steps is None
                and stop_object_z is None
                and stop_object_z_at_least is None
            ):
                break
        if stop_bilateral_support_steps is not None:
            reached = support_stop_met
        if stop_bilateral_contact_steps is not None:
            reached = contact_stop_met
        if stop_grasp_contact_steps is not None:
            reached = grasp_stop_met
        if stop_object_z is not None or stop_object_z_at_least is not None:
            reached = object_stop_met
        if not require_target and not collision and safety_failure is None:
            reached = True
        stage_results.append(
            {
                "stage": name,
                "success": bool(
                    reached and not collision and safety_failure is None
                ),
                "collision": collision,
                "safety_failure": safety_failure,
                "steps": sum(1 for item in observations if item["stage"] == name),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id],
                    dtype=float,
                ).tolist(),
                "final_contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
            }
        )
        return bool(reached and not collision and safety_failure is None)

    current = eef_positions()
    lower_targets = {
        arm: position + np.array([0.0, 0.0, -0.25], dtype=float)
        for arm, position in current.items()
    }
    lowered_to_table = execute_stage(
        "lower_to_table",
        lower_targets,
        max_steps=180,
        gripper_value=1.0,
        stop_object_z=float(table_object_z) + 0.006,
    )
    if (
        not lowered_to_table
        and not stage_results[-1]["collision"]
        and float(raw_env.sim.data.body_xpos[body_id][2])
        <= float(table_object_z) + 0.06
    ):
        lowered_to_table = True
    if not lowered_to_table and not stage_results[-1]["collision"]:
        torso_target = hold_targets.get("torso")
        if torso_target is not None:
            torso_target = np.asarray(torso_target, dtype=float).copy()
            torso_target[0] = max(0.05, float(torso_target[0]) - 0.06)
            hold_targets["torso"] = torso_target
            current = eef_positions()
            torso_lower_targets = {
                arm: position + np.array([0.0, 0.0, -0.10], dtype=float)
                for arm, position in current.items()
            }
            lowered_to_table = execute_stage(
                "lower_torso_assist",
                torso_lower_targets,
                max_steps=140,
                gripper_value=1.0,
                stop_object_z=float(table_object_z) + 0.006,
            )
    if not lowered_to_table:
        return {
            "success": False,
            "failure_stage": stage_results[-1]["stage"],
            "support_contact_steps": maximum_support_steps,
            "collision_steps": collision_steps,
            "stages": stage_results,
            "observations": observations,
        }

    table_hold = eef_positions()
    if not execute_stage(
        "open_on_table",
        table_hold,
        max_steps=40,
        gripper_value=-1.0,
        require_target=False,
    ):
        failure_stage = "open_on_table"
    else:
        raw_targets, _ = helpers["get_targets"](raw_env, object_name, 0.0)
        separation_axis = np.asarray(raw_targets["right"] - raw_targets["left"], dtype=float)
        separation_axis[2] = 0.0
        separation_axis /= float(np.linalg.norm(separation_axis))
        clearance_height = 0.18
        current = eef_positions()
        clearance_targets = {
            arm: position + np.array([0.0, 0.0, clearance_height])
            for arm, position in current.items()
        }
        if not execute_stage(
            "raise_open_clearance",
            clearance_targets,
            max_steps=180,
            gripper_value=-1.0,
        ):
            return {
                "success": False,
                "failure_stage": "raise_open_clearance",
                "support_contact_steps": maximum_support_steps,
                "collision_steps": collision_steps,
                "stages": stage_results,
                "observations": observations,
            }
        if orientation_joint_seed:
            joint_seed = execute_joint_orientation_seed(separation_axis)
            if not joint_seed["success"]:
                return {
                    "success": False,
                    "failure_stage": "joint_space_wrist_seed",
                    "support_contact_steps": maximum_support_steps,
                    "collision_steps": collision_steps,
                    "joint_seed": joint_seed,
                    "stages": stage_results,
                    "observations": observations,
                }
        if align_closure_axes:
            orientation_alignment = execute_orientation_alignment(separation_axis)
            if not orientation_alignment["success"]:
                return {
                    "success": False,
                    "failure_stage": "align_closure_axes_high",
                    "support_contact_steps": maximum_support_steps,
                    "collision_steps": collision_steps,
                    "joint_seed": joint_seed,
                    "orientation_alignment": orientation_alignment,
                    "stages": stage_results,
                    "observations": observations,
                }
        if not execute_base_advance(base_advance_m):
            return {
                "success": False,
                "failure_stage": "advance_base_for_regrasp",
                "support_contact_steps": maximum_support_steps,
                "collision_steps": collision_steps,
                "joint_seed": joint_seed,
                "orientation_alignment": orientation_alignment,
                "stages": stage_results,
                "observations": observations,
            }
        current = eef_positions()
        retreat_targets = opposed_wall_clearance_targets(
            current,
            separation_axis=separation_axis,
            clearance_m=wall_clearance_m,
        )
        if not execute_stage(
            "retreat_from_walls",
            retreat_targets,
            max_steps=100,
            gripper_value=-1.0,
        ):
            failure_stage = "retreat_from_walls"
        else:
            current = eef_positions()
            midpoint = (current["right"] + current["left"]) * 0.5
            object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
            center_delta = object_position - midpoint
            center_delta[2] = 0.0
            center_delta -= separation_axis * float(
                np.dot(center_delta, separation_axis)
            )
            norm = float(np.linalg.norm(center_delta))
            if norm > float(center_shift_m):
                center_delta *= float(center_shift_m) / norm
            center_targets = {
                arm: position + center_delta for arm, position in current.items()
            }
            if not execute_stage(
                "translate_to_center",
                center_targets,
                max_steps=140,
                gripper_value=-1.0,
            ):
                failure_stage = "translate_to_center"
            else:
                squeeze_targets = opposed_wall_squeeze_targets(
                    eef_positions(),
                    separation_axis=separation_axis,
                    squeeze_m=wall_squeeze_m,
                )
                if not execute_stage(
                    "squeeze_center_walls",
                    squeeze_targets,
                    max_steps=120,
                    gripper_value=-1.0,
                ):
                    failure_stage = "squeeze_center_walls"
                else:
                    current = eef_positions()
                    approach_targets = {
                        arm: np.array(
                            [
                                position[0],
                                position[1],
                                float(table_object_z) + 0.115,
                            ],
                            dtype=float,
                        )
                        for arm, position in current.items()
                    }
                    approach_reached = execute_stage(
                        "approach_center_walls",
                        approach_targets,
                        max_steps=220,
                        gripper_value=-1.0,
                    )
                    approach_stage = stage_results[-1]
                    approach_stage["pose_target_reached"] = approach_reached
                    approach_stage["contact_constrained_ready"] = False
                    approach_stage["completion_mode"] = (
                        "pose_target" if approach_reached else None
                    )
                    if not approach_reached:
                        if not bool(approach_stage["collision"]):
                            bracket_evidence = fingerpad_bracket_evidence(
                                fingerpads=fingerpad_world_positions(
                                    raw_env,
                                    robot,
                                ),
                                wall_centers=opposed_object_wall_centers(
                                    raw_env,
                                    object_name,
                                    separation_axis=separation_axis,
                                ),
                                separation_axis=separation_axis,
                            )
                            approach_stage["fingerpad_bracket"] = bracket_evidence
                            approach_reached = bool(bracket_evidence["ready"])
                            approach_stage["contact_constrained_ready"] = (
                                approach_reached
                            )
                            if approach_reached:
                                approach_stage["success"] = True
                                approach_stage["completion_mode"] = (
                                    "fingerpad_bracket"
                                )
                    if not approach_reached:
                        failure_stage = "approach_center_walls"
                    else:
                        if not execute_stage(
                            "close_center_grasp",
                            eef_positions(),
                            max_steps=80,
                            gripper_value=1.0,
                            close_schedule=True,
                            stop_grasp_contact_steps=3,
                        ):
                            failure_stage = "close_center_grasp"
                        else:
                            current = eef_positions()
                            lift_targets = {
                                arm: position + np.array([0.0, 0.0, 0.17])
                                for arm, position in current.items()
                            }
                            if not execute_stage(
                                "lift_center_grasp",
                                lift_targets,
                                max_steps=180,
                                gripper_value=1.0,
                                stop_object_z_at_least=(
                                    float(table_object_z) + 0.14
                                ),
                            ):
                                failure_stage = "lift_center_grasp"
                            else:
                                elevated_targets = eef_positions()
                                if not execute_stage(
                                    "hold_center_grasp",
                                    elevated_targets,
                                    max_steps=max(20, int(hold_steps)),
                                    gripper_value=1.0,
                                    require_target=False,
                                ):
                                    failure_stage = "hold_center_grasp"
                                else:
                                    consecutive_grasp_steps = 0
                                    for item in observations:
                                        if item.get("stage") != "hold_center_grasp":
                                            continue
                                        consecutive_grasp_steps = (
                                            consecutive_grasp_steps + 1
                                            if bool(item.get("bilateral_grasp", False))
                                            else 0
                                        )
                                        hold_grasp_steps = max(
                                            hold_grasp_steps,
                                            consecutive_grasp_steps,
                                        )
                                    final_contacts = object_robot_contacts(
                                        raw_env,
                                        object_name,
                                    )
                                    physical_grasp = bool(
                                        hold_grasp_steps >= 20
                                        and has_bilateral_object_contact(final_contacts)
                                    )
                                    if (
                                        physical_grasp
                                        and center_support_moving_arm != "none"
                                    ):
                                        stationary_arm = (
                                            "left"
                                            if center_support_moving_arm == "right"
                                            else "right"
                                        )
                                        transition_start_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        clearance_targets = {
                                            arm: position
                                            + np.array(
                                                [
                                                    0.0,
                                                    0.0,
                                                    float(
                                                        center_support_clearance_lift_m
                                                    ),
                                                ]
                                            )
                                            for arm, position in eef_positions().items()
                                        }
                                        clearance_reached = execute_stage(
                                            "raise_for_under_support",
                                            clearance_targets,
                                            max_steps=160,
                                            gripper_value=1.0,
                                            minimum_object_z=(
                                                float(table_object_z) + 0.10
                                            ),
                                            require_bilateral_grasp=True,
                                        )
                                        clearance_stage = stage_results[-1]
                                        clearance_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        clearance_grasp = helpers["grasp_status"](
                                            raw_env,
                                            robot,
                                            object_name,
                                        )
                                        clearance_safe = bool(
                                            clearance_reached
                                            and all(
                                                bool(clearance_grasp.get(arm))
                                                for arm in ("right", "left")
                                            )
                                            and float(clearance_object[2])
                                            >= float(table_object_z) + 0.10
                                        )
                                        clearance_stage.update(
                                            {
                                                "requested_lift_m": float(
                                                    center_support_clearance_lift_m
                                                ),
                                                "object_lift_m": float(
                                                    clearance_object[2]
                                                    - transition_start_object[2]
                                                ),
                                                "bilateral_grasp": bool(
                                                    all(
                                                        bool(
                                                            clearance_grasp.get(arm)
                                                        )
                                                        for arm in ("right", "left")
                                                    )
                                                ),
                                                "height_safe": bool(
                                                    float(clearance_object[2])
                                                    >= float(table_object_z) + 0.10
                                                ),
                                            }
                                        )
                                        lower_reached = False
                                        inset_reached = False
                                        if clearance_safe:
                                            moving_gripper_value = (
                                                1.0
                                                if center_support_keep_moving_gripper_closed
                                                else -1.0
                                            )
                                            gripper_commands = {
                                                center_support_moving_arm: moving_gripper_value,
                                                stationary_arm: 1.0,
                                            }
                                            if center_support_combined_motion:
                                                combined_targets = (
                                                    single_arm_under_support_targets(
                                                        eef_positions(),
                                                        moving_arm=(
                                                            center_support_moving_arm
                                                        ),
                                                        separation_axis=(
                                                            separation_axis
                                                        ),
                                                        descent_m=(
                                                            center_support_descent_m
                                                        ),
                                                        inset_m=(
                                                            center_support_inset_m
                                                        ),
                                                    )
                                                )
                                                lower_reached = execute_stage(
                                                    f"lower_inset_{center_support_moving_arm}_under_object",
                                                    combined_targets,
                                                    max_steps=180,
                                                    gripper_value=gripper_commands,
                                                    minimum_object_z=(
                                                        float(table_object_z) + 0.10
                                                    ),
                                                    required_contact_arm=stationary_arm,
                                                )
                                                inset_reached = lower_reached
                                            else:
                                                lower_targets = (
                                                    single_arm_under_support_targets(
                                                        eef_positions(),
                                                        moving_arm=(
                                                            center_support_moving_arm
                                                        ),
                                                        separation_axis=(
                                                            separation_axis
                                                        ),
                                                        descent_m=(
                                                            center_support_descent_m
                                                        ),
                                                        inset_m=0.0,
                                                    )
                                                )
                                                lower_reached = execute_stage(
                                                    f"lower_{center_support_moving_arm}_for_support",
                                                    lower_targets,
                                                    max_steps=180,
                                                    gripper_value=gripper_commands,
                                                    minimum_object_z=(
                                                        float(table_object_z) + 0.10
                                                    ),
                                                    required_contact_arm=stationary_arm,
                                                )
                                        if lower_reached and not center_support_combined_motion:
                                            inset_targets = (
                                                single_arm_under_support_targets(
                                                    eef_positions(),
                                                    moving_arm=(
                                                        center_support_moving_arm
                                                    ),
                                                    separation_axis=separation_axis,
                                                    descent_m=0.0,
                                                    inset_m=center_support_inset_m,
                                                )
                                            )
                                            inset_reached = execute_stage(
                                                f"inset_{center_support_moving_arm}_under_object",
                                                inset_targets,
                                                max_steps=180,
                                                gripper_value=gripper_commands,
                                                minimum_object_z=(
                                                    float(table_object_z) + 0.10
                                                ),
                                                required_contact_arm=stationary_arm,
                                            )
                                        transition_contacts = object_robot_contacts(
                                            raw_env,
                                            object_name,
                                        )
                                        transition_grasp = helpers["grasp_status"](
                                            raw_env,
                                            robot,
                                            object_name,
                                        )
                                        transition_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        moving_support = any(
                                            _is_allowed_cradle_geom(
                                                geom,
                                                center_support_moving_arm,
                                            )
                                            for geom in transition_contacts[
                                                center_support_moving_arm
                                            ]
                                        )
                                        stationary_grasp = bool(
                                            transition_grasp.get(stationary_arm)
                                        )
                                        stationary_contact = bool(
                                            transition_contacts.get(stationary_arm)
                                        )
                                        height_safe = bool(
                                            float(transition_object[2])
                                            >= float(table_object_z) + 0.10
                                        )
                                        transition_success = bool(
                                            clearance_safe
                                            and lower_reached
                                            and inset_reached
                                            and moving_support
                                            and stationary_contact
                                            and height_safe
                                        )
                                        support_transition = {
                                            "success": transition_success,
                                            "moving_arm": center_support_moving_arm,
                                            "stationary_arm": stationary_arm,
                                            "moving_gripper_closed": bool(
                                                center_support_keep_moving_gripper_closed
                                            ),
                                            "combined_motion": bool(
                                                center_support_combined_motion
                                            ),
                                            "clearance_success": clearance_safe,
                                            "lower_success": lower_reached,
                                            "inset_success": inset_reached,
                                            "moving_arm_support": moving_support,
                                            "stationary_arm_contact": stationary_contact,
                                            "stationary_arm_grasp": stationary_grasp,
                                            "height_safe": height_safe,
                                            "object_lift_m": float(
                                                transition_object[2]
                                                - transition_start_object[2]
                                            ),
                                            "contacts": {
                                                arm: list(names)
                                                for arm, names in (
                                                    transition_contacts.items()
                                                )
                                            },
                                        }
                                        failure_stage = (
                                            None
                                            if transition_success
                                            else "single_arm_under_support"
                                        )
                                    if (
                                        physical_grasp
                                        and center_support_moving_arm == "none"
                                        and float(center_carry_corner_seat_m) > 0.0
                                    ):
                                        seat_base_xy = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        seat_start_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        seat_direction = (
                                            seat_start_object[:2] - seat_base_xy
                                        )
                                        if center_carry_inchworm_toward_base:
                                            seat_direction = -seat_direction
                                        seat_targets = trailing_corner_seat_targets(
                                            eef_positions(),
                                            travel_direction=seat_direction,
                                            distance_m=center_carry_corner_seat_m,
                                        )
                                        seat_reached = execute_stage(
                                            "seat_trailing_corners",
                                            seat_targets,
                                            max_steps=100,
                                            gripper_value=1.0,
                                        )
                                        seat_stage = stage_results[-1]
                                        seat_end_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        seat_stage.update(
                                            {
                                                "requested_distance_m": float(
                                                    center_carry_corner_seat_m
                                                ),
                                                "object_translation_m": float(
                                                    np.linalg.norm(
                                                        seat_end_object[:2]
                                                        - seat_start_object[:2]
                                                    )
                                                ),
                                            }
                                        )
                                        final_contacts = object_robot_contacts(
                                            raw_env, object_name
                                        )
                                        physical_grasp = bool(
                                            seat_reached
                                            and has_bilateral_object_contact(
                                                final_contacts
                                            )
                                            and float(seat_end_object[2])
                                            >= float(table_object_z) + 0.10
                                        )
                                        if not physical_grasp:
                                            failure_stage = "seat_trailing_corners"
                                    if (
                                        physical_grasp
                                        and center_support_moving_arm == "none"
                                        and float(center_carry_arm_stroke_m) > 0.0
                                        and float(center_carry_inchworm_distance_m) <= 0.0
                                    ):
                                        stroke_base_xy = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        stroke_start_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        stroke_direction = (
                                            stroke_start_object[:2] - stroke_base_xy
                                        )
                                        stroke_direction /= np.linalg.norm(
                                            stroke_direction
                                        )
                                        stroke_targets = arm_transport_stroke_targets(
                                            eef_positions(),
                                            travel_direction=stroke_direction,
                                            stroke_m=center_carry_arm_stroke_m,
                                            lift_m=center_carry_arm_stroke_lift_m,
                                        )
                                        stroke_reached = execute_stage(
                                            "arm_transport_stroke",
                                            stroke_targets,
                                            max_steps=120,
                                            gripper_value=1.0,
                                        )
                                        stroke_stage = stage_results[-1]
                                        stroke_end_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        stroke_delta = (
                                            stroke_end_object[:2]
                                            - stroke_start_object[:2]
                                        )
                                        projected_progress, lateral_drift = (
                                            projected_planar_motion(
                                                stroke_delta,
                                                direction=stroke_direction,
                                            )
                                        )
                                        stroke_projected_progress = (
                                            projected_progress
                                        )
                                        stroke_stage.update(
                                            {
                                                "requested_stroke_m": float(
                                                    center_carry_arm_stroke_m
                                                ),
                                                "requested_lift_m": float(
                                                    center_carry_arm_stroke_lift_m
                                                ),
                                                "projected_object_progress_m": (
                                                    projected_progress
                                                ),
                                                "lateral_object_drift_m": lateral_drift,
                                            }
                                        )
                                        final_contacts = object_robot_contacts(
                                            raw_env, object_name
                                        )
                                        physical_grasp = bool(
                                            stroke_reached
                                            and projected_progress >= 0.02
                                            and lateral_drift <= 0.03
                                            and has_bilateral_object_contact(
                                                final_contacts
                                            )
                                            and float(stroke_end_object[2])
                                            >= float(table_object_z) + 0.10
                                        )
                                        if not physical_grasp:
                                            failure_stage = "arm_transport_stroke"
                                    if (
                                        physical_grasp
                                        and center_support_moving_arm == "none"
                                        and float(center_carry_base_reset_m) > 0.0
                                        and float(center_carry_inchworm_distance_m) <= 0.0
                                    ):
                                        reset_driver = OfficialPhysicalCarryDriver()
                                        reset_start_base = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        reset_start_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        ).copy()
                                        reset_start_grippers = eef_positions()
                                        reset_direction = (
                                            reset_start_object[:2] - reset_start_base
                                        )
                                        reset_direction /= np.linalg.norm(
                                            reset_direction
                                        )
                                        reset_max_speed = float(
                                            center_carry_max_linear
                                        )
                                        reset_collision = False
                                        reset_grasped = True
                                        reset_height_safe = True
                                        reset_steps = 0
                                        reset_translation = 0.0
                                        maximum_object_drift = 0.0
                                        maximum_gripper_drift = 0.0
                                        reset_budget = int(
                                            np.ceil(
                                                float(center_carry_base_reset_m)
                                                / (reset_max_speed * 0.05)
                                            )
                                        ) + 20
                                        for _ in range(reset_budget):
                                            base_xy, base_yaw = backend.get_base_pose()
                                            base_xy = np.asarray(base_xy, dtype=float)
                                            reset_translation = float(
                                                np.dot(
                                                    base_xy - reset_start_base,
                                                    reset_direction,
                                                )
                                            )
                                            remaining = max(
                                                0.0,
                                                float(center_carry_base_reset_m)
                                                - reset_translation,
                                            )
                                            if remaining <= 1e-4:
                                                break
                                            current_grippers = eef_positions()
                                            base_command, arm_deltas = (
                                                compensated_base_reset_step(
                                                    travel_direction=reset_direction,
                                                    base_yaw=float(base_yaw),
                                                    remaining_m=remaining,
                                                    max_speed_m_s=reset_max_speed,
                                                    control_dt_s=0.05,
                                                    gripper_world_errors={
                                                        arm: (
                                                            reset_start_grippers[arm]
                                                            - current_grippers[arm]
                                                        )
                                                        for arm in (
                                                            "right",
                                                            "left",
                                                        )
                                                    },
                                                )
                                            )
                                            step_info = reset_driver.step(
                                                backend,
                                                object_name=object_name,
                                                base_command=base_command,
                                                hold_targets=hold_targets,
                                                arm_world_deltas=arm_deltas,
                                                gripper_value=1.0,
                                                base_control_dt=0.05,
                                            )
                                            reset_steps += 1
                                            reset_collision = bool(
                                                step_info.get("collision", False)
                                            )
                                            collision_steps += int(reset_collision)
                                            object_position = np.asarray(
                                                raw_env.sim.data.body_xpos[body_id],
                                                dtype=float,
                                            )
                                            current_grippers = eef_positions()
                                            maximum_object_drift = max(
                                                maximum_object_drift,
                                                float(
                                                    np.linalg.norm(
                                                        object_position[:2]
                                                        - reset_start_object[:2]
                                                    )
                                                ),
                                            )
                                            maximum_gripper_drift = max(
                                                maximum_gripper_drift,
                                                max(
                                                    float(
                                                        np.linalg.norm(
                                                            current_grippers[arm][:2]
                                                            - reset_start_grippers[arm][
                                                                :2
                                                            ]
                                                        )
                                                    )
                                                    for arm in ("right", "left")
                                                ),
                                            )
                                            reset_grasped = all(
                                                bool(value)
                                                for value in helpers["grasp_status"](
                                                    raw_env, robot, object_name
                                                ).values()
                                            )
                                            reset_height_safe = bool(
                                                float(object_position[2])
                                                >= float(table_object_z) + 0.10
                                            )
                                            if (
                                                reset_collision
                                                or not reset_grasped
                                                or not reset_height_safe
                                                or maximum_gripper_drift > 0.03
                                            ):
                                                break
                                        reset_end_base = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        reset_end_object = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id],
                                            dtype=float,
                                        )
                                        reset_translation = float(
                                            np.dot(
                                                reset_end_base - reset_start_base,
                                                reset_direction,
                                            )
                                        )
                                        (
                                            reset_object_progress,
                                            reset_object_lateral_drift,
                                        ) = projected_planar_motion(
                                            reset_end_object[:2]
                                            - reset_start_object[:2],
                                            direction=reset_direction,
                                        )
                                        macro_object_progress = float(
                                            stroke_projected_progress
                                            + reset_object_progress
                                        )
                                        reset_success = bool(
                                            reset_translation
                                            >= float(center_carry_base_reset_m) - 1e-4
                                            and not reset_collision
                                            and reset_grasped
                                            and reset_height_safe
                                            and maximum_gripper_drift <= 0.03
                                            and reset_object_lateral_drift <= 0.03
                                            and macro_object_progress >= 0.02
                                        )
                                        stage_results.append(
                                            {
                                                "stage": "inchworm_base_reset",
                                                "success": reset_success,
                                                "requested_distance_m": float(
                                                    center_carry_base_reset_m
                                                ),
                                                "base_translation_m": (
                                                    reset_translation
                                                ),
                                                "maximum_object_drift_m": (
                                                    maximum_object_drift
                                                ),
                                                "reset_object_progress_m": (
                                                    reset_object_progress
                                                ),
                                                "reset_object_lateral_drift_m": (
                                                    reset_object_lateral_drift
                                                ),
                                                "macro_object_progress_m": (
                                                    macro_object_progress
                                                ),
                                                "maximum_gripper_drift_m": (
                                                    maximum_gripper_drift
                                                ),
                                                "bilateral_grasp": reset_grasped,
                                                "height_safe": reset_height_safe,
                                                "collision": reset_collision,
                                                "steps": reset_steps,
                                            }
                                        )
                                        physical_grasp = reset_success
                                        if not physical_grasp:
                                            failure_stage = "inchworm_base_reset"
                                    if not physical_grasp:
                                        failure_stage = failure_stage or "final_contact"
                                    elif center_support_moving_arm != "none":
                                        failure_stage = (
                                            None
                                            if bool(
                                                (support_transition or {}).get(
                                                    "success"
                                                )
                                            )
                                            else failure_stage
                                            or "single_arm_under_support"
                                        )
                                    elif float(center_carry_inchworm_distance_m) > 0.0:
                                        transport_start_base_xy = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        transport_start_object_xy = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id][:2],
                                            dtype=float,
                                        ).copy()
                                        inchworm_direction = (
                                            None
                                            if center_carry_inchworm_world_direction_x
                                            is None
                                            and center_carry_inchworm_world_direction_y
                                            is None
                                            else np.array(
                                                [
                                                    center_carry_inchworm_world_direction_x,
                                                    center_carry_inchworm_world_direction_y,
                                                ],
                                                dtype=float,
                                            )
                                        )
                                        inchworm_direction = resolve_inchworm_direction(
                                            base_xy=transport_start_base_xy,
                                            object_xy=transport_start_object_xy,
                                            toward_base=(
                                                center_carry_inchworm_toward_base
                                            ),
                                            world_direction=inchworm_direction,
                                        )
                                        inchworm_distance = float(
                                            center_carry_inchworm_distance_m
                                        )
                                        transport_result = run_inchworm_transport(
                                            backend,
                                            object_name=object_name,
                                            travel_direction=inchworm_direction,
                                            travel_distance=inchworm_distance,
                                            minimum_object_z=(
                                                float(table_object_z) + 0.10
                                            ),
                                            config=InchwormCarryConfig(
                                                stroke_distance=(
                                                    center_carry_inchworm_stroke_m
                                                ),
                                                stroke_vertical_feedforward=0.015,
                                                stroke_height_gain=0.75,
                                                reset_distance=(
                                                    center_carry_inchworm_reset_m
                                                ),
                                                reset_max_linear=0.04,
                                                max_cycles=max(
                                                    2,
                                                    int(
                                                        np.ceil(
                                                            inchworm_distance / 0.02
                                                        )
                                                    )
                                                    + 2,
                                                ),
                                            ),
                                        )
                                        transport_end_base_xy = np.asarray(
                                            backend.get_base_pose()[0], dtype=float
                                        )
                                        transport_end_object_xy = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id][:2],
                                            dtype=float,
                                        ).copy()
                                        transport_base_translation = float(
                                            np.linalg.norm(
                                                transport_end_base_xy
                                                - transport_start_base_xy
                                            )
                                        )
                                        transport_object_translation = float(
                                            np.linalg.norm(
                                                transport_end_object_xy
                                                - transport_start_object_xy
                                            )
                                        )
                                        stage_results.append(
                                            {
                                                "stage": "inchworm_transport",
                                                **transport_result,
                                                "requested_distance_m": (
                                                    inchworm_distance
                                                ),
                                                "object_translation_m": (
                                                    transport_object_translation
                                                ),
                                                "base_translation_m": (
                                                    transport_base_translation
                                                ),
                                            }
                                        )
                                        failure_stage = (
                                            None
                                            if bool(transport_result.get("success"))
                                            else "inchworm_transport"
                                        )
                                    elif float(center_carry_distance_m) > 0.0:
                                        transport_start_base_xy, hold_yaw = (
                                            backend.get_base_pose()
                                        )
                                        transport_start_base_xy = np.asarray(
                                            transport_start_base_xy,
                                            dtype=float,
                                        )
                                        transport_start_object_xy = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id][:2],
                                            dtype=float,
                                        ).copy()
                                        carry_target = forward_carry_target(
                                            base_xy=transport_start_base_xy,
                                            object_xy=transport_start_object_xy,
                                            distance_m=center_carry_distance_m,
                                            toward_object=(
                                                not center_carry_away_from_object
                                            ),
                                        )
                                        control_dt = 0.05
                                        max_linear = float(center_carry_max_linear)
                                        if not np.isfinite(max_linear) or max_linear <= 0.0:
                                            raise ValueError(
                                                "center_carry_max_linear must be positive and finite"
                                            )
                                        minimum_steps = int(
                                            np.ceil(
                                                float(center_carry_distance_m)
                                                / (max_linear * control_dt)
                                            )
                                        )
                                        transport_result = run_physical_transport(
                                            backend,
                                            path=[carry_target],
                                            object_name=object_name,
                                            hold_yaw=float(hold_yaw),
                                            minimum_object_z=(
                                                float(table_object_z) + 0.10
                                            ),
                                            config=PhysicalCarryConfig(
                                                waypoint_tolerance=0.01,
                                                max_steps=max(600, minimum_steps * 2),
                                                max_linear=max_linear,
                                                max_angular=0.04,
                                                max_linear_delta=0.01,
                                                max_angular_delta=0.01,
                                                base_control_dt=control_dt,
                                            ),
                                        )
                                        transport_end_base_xy = np.asarray(
                                            backend.get_base_pose()[0],
                                            dtype=float,
                                        )
                                        transport_end_object_xy = np.asarray(
                                            raw_env.sim.data.body_xpos[body_id][:2],
                                            dtype=float,
                                        ).copy()
                                        transport_base_translation = float(
                                            np.linalg.norm(
                                                transport_end_base_xy
                                                - transport_start_base_xy
                                            )
                                        )
                                        transport_object_translation = float(
                                            np.linalg.norm(
                                                transport_end_object_xy
                                                - transport_start_object_xy
                                            )
                                        )
                                        stage_results.append(
                                            {
                                                "stage": "transport_center_grasp",
                                                **transport_result,
                                                "requested_distance_m": float(
                                                    center_carry_distance_m
                                                ),
                                                "max_linear_m_s": max_linear,
                                                "target_base_xy": carry_target.tolist(),
                                                "base_translation_m": (
                                                    transport_base_translation
                                                ),
                                                "object_translation_m": (
                                                    transport_object_translation
                                                ),
                                            }
                                        )
                                        failure_stage = (
                                            None
                                            if bool(transport_result.get("success"))
                                            else "transport_center_grasp"
                                        )
                                    else:
                                        failure_stage = None

    final_object_z = float(raw_env.sim.data.body_xpos[body_id][2])
    return {
        "success": failure_stage is None,
        "failure_stage": failure_stage,
        "physical_grasp": physical_grasp,
        "lift_m": final_object_z - float(table_object_z),
        "hold_grasp_steps": hold_grasp_steps,
        "transport_success": (
            bool(transport_result.get("success"))
            if isinstance(transport_result, Mapping)
            else False
        ),
        "requested_carry_distance_m": float(
            max(center_carry_distance_m, center_carry_inchworm_distance_m)
        ),
        "object_translation_m": transport_object_translation,
        "transport_base_translation_m": transport_base_translation,
        "transport": transport_result,
        "support_transition": support_transition,
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "joint_seed": joint_seed,
        "orientation_alignment": orientation_alignment,
        "stages": stage_results,
        "observations": observations,
        "final_geometry": geometry_snapshot(raw_env, object_name),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    candidate_root = args.candidate_root.resolve()
    app_dir = _configure_candidate(candidate_root)
    official_commit = _git_commit(candidate_root)
    if official_commit != args.expected_official_commit:
        raise RuntimeError(
            f"official commit mismatch: expected {args.expected_official_commit}, "
            f"got {official_commit}"
        )
    tasks = json.loads(
        (app_dir / "knowledge" / "task_config.json").read_text(encoding="utf-8")
    )["tasks"]
    task = dict(tasks[0])
    backend = None
    started = time.perf_counter()
    record: dict[str, Any] = {
        "physical_grasp": False,
        "lift_m": 0.0,
        "support_contact_steps": 0,
        "base_translation_m": 0.0,
        "attachment_calls": 0,
        "object_pose_writes": 0,
        "collision_frames": 0,
        "dropped": True,
        "infrastructure_error": None,
        "seed": int(args.seed),
        "official_commit": official_commit,
        "mode": "post_lift_hold_probe",
    }
    try:
        backend, scene_context, grid = _load_scene(app_dir, task, args.seed)
        from robot_agent.workflows.competition_flow import OfficialCompetitionDriver

        driver = OfficialCompetitionDriver(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
        )
        candidates = driver.rank_objects(str(task["source"]), task["object"])
        if not candidates:
            raise RuntimeError("no graspable L1 candidate resolved")
        object_name = candidates[0]
        record["object_name"] = object_name
        body_id = backend.env.obj_body_id[object_name]
        backend.start_recording()
        backend._record_trajectory_frame()
        moved = driver.move(
            str(task["source"]),
            carrying=False,
            object_name=object_name,
        )
        record["move_success"] = bool(moved)
        if moved:
            pre_grasp_z = float(backend.env.sim.data.body_xpos[body_id][2])
            pre_grasp_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            grasp = (
                driver.grasp(str(task["source"]), object_name)
                if not args.table_edge_undercut
                else {
                    "source": "table_edge_undercut_no_grasp",
                    "success": False,
                    "lift_success": False,
                    "contacts": {},
                }
            )
            post_grasp_z = float(backend.env.sim.data.body_xpos[body_id][2])
            record["grasp_result"] = grasp
            record["physical_grasp"] = bool(
                grasp.get("success")
                and grasp.get("lift_success")
                and all(bool(value) for value in grasp.get("contacts", {}).values())
            )
            record["pre_grasp_object_z"] = pre_grasp_z
            record["post_grasp_object_z"] = post_grasp_z
            record["lift_m"] = post_grasp_z - pre_grasp_z
            record["post_grasp_geometry"] = geometry_snapshot(
                backend.env,
                object_name,
            )
            if args.table_edge_undercut:
                probe = _table_edge_undercut_probe(
                    backend,
                    object_name,
                    table_edge_y=args.undercut_table_edge_y,
                    outside_clearance_m=args.undercut_outside_clearance_m,
                    edge_clearance_m=args.undercut_edge_clearance_m,
                    above_clearance_m=args.undercut_above_clearance_m,
                    base_advance_m=args.undercut_base_advance_m,
                    object_offset_x_m=args.undercut_object_offset_x_m,
                    torso_target_m=args.undercut_torso_target_m,
                    raise_above_bottom_m=(
                        args.undercut_raise_above_bottom_m
                    ),
                )
                record["mode"] = "table_edge_undercut_probe"
                record["open_gripper"] = bool(probe.get("open_gripper", False))
                record["support_contact_steps"] = int(
                    probe.get("support_contact_steps", 0)
                )
                record["object_lift_m"] = float(
                    probe.get("object_lift_m", 0.0)
                )
                record["hold_probe"] = probe
            elif record["physical_grasp"]:
                if args.physical_push:
                    probe = _physical_push_probe(
                        backend,
                        object_name,
                        table_object_z=pre_grasp_z,
                        push_distance_m=args.push_distance_m,
                        max_push_steps=args.max_push_steps,
                    )
                    record["mode"] = "physical_push_probe"
                    record["physical_contact_steps"] = int(
                        probe.get("physical_contact_steps", 0)
                    )
                    record["object_translation_m"] = float(
                        probe.get("object_translation_m", 0.0)
                    )
                elif args.center_regrasp:
                    probe = _center_regrasp_probe(
                        backend,
                        object_name,
                        table_object_z=pre_grasp_z,
                        center_shift_m=args.regrasp_center_shift_m,
                        wall_clearance_m=args.regrasp_wall_clearance_m,
                        wall_squeeze_m=args.regrasp_wall_squeeze_m,
                        base_advance_m=args.regrasp_base_advance_m,
                        hold_steps=args.hold_steps,
                        align_closure_axes=args.align_closure_axes,
                        orientation_max_action=args.orientation_max_action,
                        orientation_fine_max_action=(
                            args.orientation_fine_max_action
                        ),
                        orientation_fine_threshold_deg=(
                            args.orientation_fine_threshold_deg
                        ),
                        orientation_tolerance_deg=args.orientation_tolerance_deg,
                        orientation_stable_steps=args.orientation_stable_steps,
                        orientation_max_steps=args.orientation_max_steps,
                        orientation_max_position_drift_m=(
                            args.orientation_max_position_drift_m
                        ),
                        orientation_joint_seed=args.orientation_joint_seed,
                        orientation_joint_seed_margin_rad=(
                            args.orientation_joint_seed_margin_rad
                        ),
                        orientation_joint_seed_max_nfev=(
                            args.orientation_joint_seed_max_nfev
                        ),
                        orientation_joint_seed_steps=(
                            args.orientation_joint_seed_steps
                        ),
                        orientation_joint_seed_position_scale_m=(
                            args.orientation_joint_seed_position_scale_m
                        ),
                        orientation_joint_seed_axis_scale=(
                            args.orientation_joint_seed_axis_scale
                        ),
                        orientation_joint_seed_regularization=(
                            args.orientation_joint_seed_regularization
                        ),
                        orientation_joint_seed_max_error_deg=(
                            args.orientation_joint_seed_max_error_deg
                        ),
                        orientation_joint_seed_max_endpoint_position_error_m=(
                            args.orientation_joint_seed_max_endpoint_position_error_m
                        ),
                        orientation_joint_seed_continuation_nodes=(
                            args.orientation_joint_seed_continuation_nodes
                        ),
                        orientation_joint_seed_include_torso=(
                            args.orientation_joint_seed_include_torso
                        ),
                        orientation_joint_seed_torso_margin_m=(
                            args.orientation_joint_seed_torso_margin_m
                        ),
                        center_carry_distance_m=args.center_carry_distance_m,
                        center_carry_max_linear=args.center_carry_max_linear,
                        center_carry_away_from_object=(
                            args.center_carry_away_from_object
                        ),
                        center_carry_corner_seat_m=args.center_carry_corner_seat_m,
                        center_carry_arm_stroke_m=args.center_carry_arm_stroke_m,
                        center_carry_arm_stroke_lift_m=(
                            args.center_carry_arm_stroke_lift_m
                        ),
                        center_carry_base_reset_m=args.center_carry_base_reset_m,
                        center_carry_inchworm_distance_m=(
                            args.center_carry_inchworm_distance_m
                        ),
                        center_carry_inchworm_toward_base=(
                            args.center_carry_inchworm_toward_base
                        ),
                        center_carry_inchworm_stroke_m=(
                            args.center_carry_inchworm_stroke_m
                        ),
                        center_carry_inchworm_reset_m=(
                            args.center_carry_inchworm_reset_m
                        ),
                        center_carry_inchworm_world_direction_x=(
                            args.center_carry_inchworm_world_direction_x
                        ),
                        center_carry_inchworm_world_direction_y=(
                            args.center_carry_inchworm_world_direction_y
                        ),
                        center_support_moving_arm=(
                            args.center_support_moving_arm
                        ),
                        center_support_clearance_lift_m=(
                            args.center_support_clearance_lift_m
                        ),
                        center_support_descent_m=(
                            args.center_support_descent_m
                        ),
                        center_support_inset_m=args.center_support_inset_m,
                        center_support_keep_moving_gripper_closed=(
                            args.center_support_keep_moving_gripper_closed
                        ),
                        center_support_combined_motion=(
                            args.center_support_combined_motion
                        ),
                    )
                    if args.center_support_moving_arm != "none":
                        record["mode"] = "single_arm_under_support_probe"
                    elif max(
                            args.center_carry_distance_m,
                            args.center_carry_inchworm_distance_m,
                    ) > 0.0:
                        record["mode"] = "center_grasp_physical_transport"
                    else:
                        record["mode"] = "table_assisted_center_regrasp"
                    record["physical_grasp"] = bool(
                        probe.get("physical_grasp", probe.get("success", False))
                    )
                    record["lift_m"] = float(probe.get("lift_m", 0.0))
                    record["hold_grasp_steps"] = int(
                        probe.get("hold_grasp_steps", 0)
                    )
                    record["transport_success"] = bool(
                        probe.get("transport_success", False)
                    )
                    record["requested_carry_distance_m"] = float(
                        probe.get("requested_carry_distance_m", 0.0)
                    )
                    record["object_translation_m"] = float(
                        probe.get("object_translation_m", 0.0)
                    )
                    record["transport_base_translation_m"] = float(
                        probe.get("transport_base_translation_m", 0.0)
                    )
                    record["support_transition"] = probe.get(
                        "support_transition"
                    )
                    alignment = probe.get("orientation_alignment")
                    if isinstance(alignment, Mapping):
                        record.update(
                            {
                                "orientation_right_error_deg": alignment.get(
                                    "right_error_deg"
                                ),
                                "orientation_left_error_deg": alignment.get(
                                    "left_error_deg"
                                ),
                                "orientation_stable_steps": alignment.get(
                                    "stable_steps"
                                ),
                                "orientation_max_position_drift_m": alignment.get(
                                    "max_position_drift_m"
                                ),
                                "orientation_collision_frames": alignment.get(
                                    "collision_frames"
                                ),
                            }
                        )
                    seed_evidence = probe.get("joint_seed")
                    if isinstance(seed_evidence, Mapping):
                        record.update(
                            {
                                "joint_seed_success": seed_evidence.get("success"),
                                "joint_seed_right_error_deg": seed_evidence.get(
                                    "right_error_deg"
                                ),
                                "joint_seed_left_error_deg": seed_evidence.get(
                                    "left_error_deg"
                                ),
                                "joint_seed_max_endpoint_position_error_m": (
                                    seed_evidence.get(
                                        "max_endpoint_position_error_m"
                                    )
                                ),
                                "joint_seed_max_path_position_drift_m": (
                                    seed_evidence.get("max_path_position_drift_m")
                                ),
                                "joint_seed_min_bound_margin_rad": seed_evidence.get(
                                    "min_bound_margin_rad"
                                ),
                                "joint_seed_collision_frames": seed_evidence.get(
                                    "collision_frames"
                                ),
                                "joint_seed_rolled_back": seed_evidence.get(
                                    "rolled_back"
                                ),
                                "joint_seed_failure": seed_evidence.get("failure"),
                            }
                        )
                elif args.inward_probe_m > 0.0:
                    probe = _inward_support_probe(
                        backend,
                        object_name,
                        inward_m=args.inward_probe_m,
                        move_steps=args.inward_steps,
                        hold_steps=args.hold_steps,
                    )
                    record["mode"] = "inward_support_probe"
                else:
                    probe = _hold_probe(backend, object_name, steps=args.hold_steps)
                record["hold_probe"] = probe
                record["support_contact_steps"] = int(
                    probe.get("support_contact_steps", 0)
                )
            final_z = float(backend.env.sim.data.body_xpos[body_id][2])
            record["final_geometry"] = geometry_snapshot(
                backend.env,
                object_name,
            )
            final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            record["final_object_z"] = final_z
            record["base_translation_m"] = float(
                np.linalg.norm(final_base_xy - pre_grasp_base_xy)
            )
            record["dropped"] = bool(
                final_z < pre_grasp_z + CRADLE_GATE_THRESHOLDS["lift_m"]
            )
        backend._record_trajectory_frame()
        trajectory_path = args.trajectory.resolve()
        backend.save_trajectory(trajectory_path)
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        record["trajectory"] = str(trajectory_path)
        record["collision_frames"] = sum(
            int(bool(frame.get("has_collision", False)))
            for frame in trajectory.get("frames", [])
            if isinstance(frame, dict)
        )
    except Exception as exc:
        record["infrastructure_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
    record["elapsed_s"] = round(time.perf_counter() - started, 6)
    if record.get("mode") == "table_edge_undercut_probe":
        record["gate_failures"] = undercut_gate_failures(record)
    elif record.get("mode") == "physical_push_probe":
        record["gate_failures"] = push_gate_failures(record)
    elif record.get("mode") == "center_grasp_physical_transport":
        record["gate_failures"] = center_grasp_transport_failures(record)
    else:
        record["gate_failures"] = cradle_gate_failures(record)
    record["accepted"] = not record["gate_failures"]
    if args.center_regrasp and args.align_closure_axes:
        record["orientation_gate_failures"] = orientation_alignment_failures(record)
        record["orientation_accepted"] = not record["orientation_gate_failures"]
    if args.center_regrasp and args.orientation_joint_seed:
        record["joint_seed_gate_failures"] = joint_seed_failures(record)
        record["joint_seed_accepted"] = not record["joint_seed_gate_failures"]
    _atomic_json(args.output.resolve(), record)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--inward-probe-m", type=float, default=0.0)
    parser.add_argument("--inward-steps", type=int, default=40)
    parser.add_argument("--center-regrasp", action="store_true")
    parser.add_argument("--regrasp-center-shift-m", type=float, default=0.24)
    parser.add_argument("--regrasp-wall-clearance-m", type=float, default=0.10)
    parser.add_argument("--regrasp-wall-squeeze-m", type=float, default=0.025)
    parser.add_argument("--regrasp-base-advance-m", type=float, default=0.0)
    parser.add_argument("--center-carry-distance-m", type=float, default=0.0)
    parser.add_argument("--center-carry-max-linear", type=float, default=0.04)
    parser.add_argument("--center-carry-away-from-object", action="store_true")
    parser.add_argument("--center-carry-corner-seat-m", type=float, default=0.0)
    parser.add_argument("--center-carry-arm-stroke-m", type=float, default=0.0)
    parser.add_argument(
        "--center-carry-arm-stroke-lift-m", type=float, default=0.0
    )
    parser.add_argument("--center-carry-base-reset-m", type=float, default=0.0)
    parser.add_argument(
        "--center-carry-inchworm-distance-m", type=float, default=0.0
    )
    parser.add_argument(
        "--center-carry-inchworm-toward-base", action="store_true"
    )
    parser.add_argument(
        "--center-carry-inchworm-stroke-m", type=float, default=0.08
    )
    parser.add_argument(
        "--center-carry-inchworm-reset-m", type=float, default=0.06
    )
    parser.add_argument("--center-carry-inchworm-world-direction-x", type=float)
    parser.add_argument("--center-carry-inchworm-world-direction-y", type=float)
    parser.add_argument(
        "--center-support-moving-arm",
        choices=("none", "right", "left"),
        default="none",
    )
    parser.add_argument(
        "--center-support-clearance-lift-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--center-support-descent-m",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--center-support-inset-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--center-support-keep-moving-gripper-closed",
        action="store_true",
    )
    parser.add_argument(
        "--center-support-combined-motion",
        action="store_true",
    )
    parser.add_argument("--table-edge-undercut", action="store_true")
    parser.add_argument(
        "--undercut-table-edge-y",
        type=float,
        default=4.688,
    )
    parser.add_argument(
        "--undercut-outside-clearance-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--undercut-edge-clearance-m",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--undercut-above-clearance-m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--undercut-base-advance-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--undercut-object-offset-x-m",
        type=float,
        default=0.20,
    )
    parser.add_argument("--undercut-torso-target-m", type=float)
    parser.add_argument(
        "--undercut-raise-above-bottom-m",
        type=float,
        default=0.12,
    )
    parser.add_argument("--align-closure-axes", action="store_true")
    parser.add_argument("--orientation-max-action", type=float, default=0.30)
    parser.add_argument("--orientation-fine-max-action", type=float)
    parser.add_argument(
        "--orientation-fine-threshold-deg",
        type=float,
        default=0.0,
    )
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--orientation-stable-steps", type=int, default=5)
    parser.add_argument("--orientation-max-steps", type=int, default=160)
    parser.add_argument(
        "--orientation-max-position-drift-m",
        type=float,
        default=0.03,
    )
    parser.add_argument("--orientation-joint-seed", action="store_true")
    parser.add_argument(
        "--orientation-joint-seed-margin-rad",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--orientation-joint-seed-max-nfev",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--orientation-joint-seed-steps",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--orientation-joint-seed-position-scale-m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--orientation-joint-seed-axis-scale",
        type=float,
        default=float(np.sin(np.deg2rad(5.0))),
    )
    parser.add_argument(
        "--orientation-joint-seed-regularization",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--orientation-joint-seed-max-error-deg",
        type=float,
        default=JOINT_SEED_THRESHOLDS["error_deg"],
    )
    parser.add_argument(
        "--orientation-joint-seed-max-endpoint-position-error-m",
        type=float,
        default=JOINT_SEED_THRESHOLDS["max_endpoint_position_error_m"],
    )
    parser.add_argument(
        "--orientation-joint-seed-continuation-nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--orientation-joint-seed-include-torso",
        action="store_true",
    )
    parser.add_argument(
        "--orientation-joint-seed-torso-margin-m",
        type=float,
        default=0.005,
    )
    parser.add_argument("--physical-push", action="store_true")
    parser.add_argument("--push-distance-m", type=float, default=0.50)
    parser.add_argument("--max-push-steps", type=int, default=400)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run_probe(parse_args())
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    raise SystemExit(0 if result["accepted"] else 1)
