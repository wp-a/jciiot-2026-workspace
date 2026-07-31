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
from contextlib import contextmanager
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

POSTURE_CARRY_THRESHOLDS = {
    "projected_object_progress_m": 0.08,
    "lateral_object_drift_m": 0.03,
    "object_gripper_drift_m": 0.03,
    "final_object_lift_m": 0.10,
}


def directed_planar_progress(
    *,
    start_xy: object,
    end_xy: object,
    direction_xy: object,
) -> tuple[float, float]:
    """Return signed progress and absolute lateral drift along a 2-D direction."""
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    if any(value.shape != (2,) for value in (start, end, direction)):
        raise ValueError("start, end, and direction must each be planar 2-vectors")
    if not all(np.all(np.isfinite(value)) for value in (start, end, direction)):
        raise ValueError("start, end, and direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError("direction must be nonzero")

    unit_direction = direction / direction_norm
    displacement = end - start
    progress = float(np.dot(displacement, unit_direction))
    lateral = abs(
        float(
            unit_direction[0] * displacement[1]
            - unit_direction[1] * displacement[0]
        )
    )
    return progress, lateral


def navigation_retract_targets(
    *,
    base_xy: object,
    base_yaw: float,
    forward_m: float,
    lateral_m: float,
    target_z: float,
) -> dict[str, np.ndarray]:
    """Return compact, base-relative end-effector targets for safe navigation."""
    base = np.asarray(base_xy, dtype=float)
    values = np.asarray(
        [base_yaw, forward_m, lateral_m, target_z], dtype=float
    )
    if base.shape != (2,) or not np.all(np.isfinite(base)):
        raise ValueError("base_xy must be a finite planar vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("navigation retract parameters must be finite")
    if float(forward_m) <= 0.0 or float(lateral_m) <= 0.0:
        raise ValueError("navigation retract offsets must be positive")
    cosine = float(np.cos(float(base_yaw)))
    sine = float(np.sin(float(base_yaw)))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    targets = {}
    for arm, local_lateral in (
        ("right", -float(lateral_m)),
        ("left", float(lateral_m)),
    ):
        planar = base + rotation @ np.array(
            [float(forward_m), local_lateral], dtype=float
        )
        targets[arm] = np.array(
            [planar[0], planar[1], float(target_z)], dtype=float
        )
    return targets


def floor_regrasp_safe_base_xy(
    *,
    object_xy: object,
    current_base_xy: object,
    clearance_m: float,
) -> np.ndarray:
    """Move outward on the current object-to-base ray before changing yaw."""
    object_position = np.asarray(object_xy, dtype=float)
    base_position = np.asarray(current_base_xy, dtype=float)
    clearance = float(clearance_m)
    if object_position.shape != (2,) or base_position.shape != (2,):
        raise ValueError("floor regrasp positions must be planar vectors")
    if not np.all(np.isfinite(object_position)) or not np.all(
        np.isfinite(base_position)
    ):
        raise ValueError("floor regrasp positions must be finite")
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("floor regrasp clearance must be finite and positive")
    outward = base_position - object_position
    distance = float(np.linalg.norm(outward))
    if distance <= 1e-12:
        raise ValueError("floor regrasp base must not coincide with the object")
    return object_position + outward / distance * max(distance, clearance)


def floor_push_staging_targets(
    *,
    object_xy: object,
    current_base_xy: object,
    push_direction_xy: object,
    base_standoff_m: float,
    orientation_clearance_m: float,
    lateral_offset_m: float | None,
    maximum_lateral_offset_m: float,
    face_offset_m: float,
    hand_separation_m: float,
    hand_height_m: float,
    precontact_clearance_m: float,
) -> dict[str, Any]:
    """Build a lane-preserving base pose and paired floor-push targets."""
    object_position = np.asarray(object_xy, dtype=float)
    base_position = np.asarray(current_base_xy, dtype=float)
    direction = np.asarray(push_direction_xy, dtype=float)
    if any(value.shape != (2,) for value in (object_position, base_position, direction)):
        raise ValueError("floor push positions and direction must be planar vectors")
    if not all(
        np.all(np.isfinite(value))
        for value in (object_position, base_position, direction)
    ):
        raise ValueError("floor push positions and direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("floor push direction must be non-zero")
    direction /= direction_norm

    parameters = {
        "base_standoff_m": float(base_standoff_m),
        "orientation_clearance_m": float(orientation_clearance_m),
        "maximum_lateral_offset_m": float(maximum_lateral_offset_m),
        "face_offset_m": float(face_offset_m),
        "hand_separation_m": float(hand_separation_m),
        "hand_height_m": float(hand_height_m),
        "precontact_clearance_m": float(precontact_clearance_m),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in parameters.values()):
        raise ValueError("floor push geometry parameters must be finite and positive")

    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    if lateral_offset_m is None:
        raw_lateral_offset = float(np.dot(base_position - object_position, left_axis))
        lateral_offset = float(
            np.clip(
                raw_lateral_offset,
                -parameters["maximum_lateral_offset_m"],
                parameters["maximum_lateral_offset_m"],
            )
        )
    else:
        lateral_offset = float(lateral_offset_m)
        if not np.isfinite(lateral_offset):
            raise ValueError("requested floor push lateral offset must be finite")
        if abs(lateral_offset) > parameters["maximum_lateral_offset_m"]:
            raise ValueError(
                "requested floor push lateral offset exceeds configured maximum"
            )
    stage_base_xy = (
        object_position
        - direction * parameters["base_standoff_m"]
        + left_axis * lateral_offset
    )
    contact_center = object_position - direction * parameters["face_offset_m"]
    half_separation = parameters["hand_separation_m"] / 2.0
    contact = {
        "right": np.r_[
            contact_center - left_axis * half_separation,
            parameters["hand_height_m"],
        ],
        "left": np.r_[
            contact_center + left_axis * half_separation,
            parameters["hand_height_m"],
        ],
    }
    precontact_offset = np.r_[
        -direction * parameters["precontact_clearance_m"],
        parameters["precontact_clearance_m"],
    ]
    orientation_base_xy = (
        stage_base_xy - direction * parameters["orientation_clearance_m"]
    )
    escape_base_xy = base_position + direction * float(
        np.dot(orientation_base_xy - base_position, direction)
    )
    return {
        "direction": direction,
        "left_axis": left_axis,
        "stage_base_xy": stage_base_xy,
        "escape_base_xy": escape_base_xy,
        "orientation_base_xy": orientation_base_xy,
        "target_yaw": float(np.arctan2(direction[1], direction[0])),
        "lateral_offset_m": lateral_offset,
        "contact": contact,
        "precontact": {
            arm: target + precontact_offset for arm, target in contact.items()
        },
    }


def floor_base_target_route(
    *,
    start_object_xy: object,
    target_xy: object,
    corridor_y: float,
    arrival_radius_m: float,
    arrival_margin_m: float,
) -> dict[str, Any]:
    """Plan three axis-aligned floor pushes through a lower cross aisle."""
    start = np.asarray(start_object_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    values = np.asarray(
        [corridor_y, arrival_radius_m, arrival_margin_m], dtype=float
    )
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("floor route positions must be planar vectors")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("floor route positions must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("floor route parameters must be finite")
    radius = float(arrival_radius_m)
    margin = float(arrival_margin_m)
    if radius <= 0.0 or margin <= 0.0 or margin >= radius:
        raise ValueError("arrival margin must be positive and smaller than radius")

    final_y = float(target[1]) - (radius - margin)
    aisle_y = float(corridor_y)
    if aisle_y >= min(float(start[1]), final_y):
        raise ValueError("floor route corridor must lie below start and arrival")

    waypoints = (
        start,
        np.array([start[0], aisle_y], dtype=float),
        np.array([target[0], aisle_y], dtype=float),
        np.array([target[0], final_y], dtype=float),
    )
    segments = []
    for segment_start, segment_end in zip(waypoints, waypoints[1:]):
        delta = segment_end - segment_start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            continue
        segments.append(
            {
                "start_object_xy": segment_start.tolist(),
                "end_object_xy": segment_end.tolist(),
                "direction": (delta / distance).tolist(),
                "distance_m": distance,
            }
        )
    final_object = waypoints[-1]
    return {
        "segments": segments,
        "corridor_y": aisle_y,
        "target_xy": target.tolist(),
        "final_object_xy": final_object.tolist(),
        "final_target_distance_m": float(np.linalg.norm(final_object - target)),
    }


def floor_base_reposition_targets(
    *,
    object_xy: object,
    current_base_xy: object,
    next_push_direction_xy: object,
    retreat_clearance_m: float,
    base_standoff_m: float,
) -> dict[str, Any]:
    """Build an orthogonal, object-avoiding base path for a push direction change."""
    object_position = np.asarray(object_xy, dtype=float)
    base_position = np.asarray(current_base_xy, dtype=float)
    direction = np.asarray(next_push_direction_xy, dtype=float)
    if any(value.shape != (2,) for value in (object_position, base_position, direction)):
        raise ValueError("floor reposition positions and direction must be planar vectors")
    if not all(
        np.all(np.isfinite(value))
        for value in (object_position, base_position, direction)
    ):
        raise ValueError("floor reposition positions and direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("next floor push direction must be non-zero")
    direction /= direction_norm
    retreat_clearance = float(retreat_clearance_m)
    standoff = float(base_standoff_m)
    if (
        not np.isfinite(retreat_clearance)
        or not np.isfinite(standoff)
        or retreat_clearance <= standoff
        or standoff <= 0.0
    ):
        raise ValueError("retreat clearance must be finite and exceed standoff")

    outward = base_position - object_position
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-12:
        raise ValueError("floor reposition base must not coincide with the object")
    retreat = object_position + outward / outward_norm * retreat_clearance
    stage = object_position - direction * standoff
    corner = retreat + (stage - object_position)
    return {
        "direction": direction,
        "retreat_base_xy": retreat,
        "corner_base_xy": corner,
        "stage_base_xy": stage,
        "target_yaw": float(np.arctan2(direction[1], direction[0])),
    }


@contextmanager
def transport_attachment_audit(raw_env: object, transport_module: object):
    """Count transport attachment and direct object-pose operations in a scope."""
    attachment_attr = str(transport_module.TRANSPORT_ATTACHMENT_ATTR)
    original_capture = transport_module.capture_transport_attachment
    original_set_object_qpos = transport_module.set_object_qpos

    def attachment_active() -> bool:
        state = getattr(raw_env, attachment_attr, None)
        return bool(isinstance(state, Mapping) and state.get("active", False))

    audit = {
        "attachment_activations": 0,
        "object_pose_writes": 0,
        "active_before": attachment_active(),
        "active_after": False,
    }

    def capture_wrapper(*args: object, **kwargs: object):
        audit["attachment_activations"] += 1
        return original_capture(*args, **kwargs)

    def set_object_qpos_wrapper(*args: object, **kwargs: object):
        audit["object_pose_writes"] += 1
        return original_set_object_qpos(*args, **kwargs)

    transport_module.capture_transport_attachment = capture_wrapper
    transport_module.set_object_qpos = set_object_qpos_wrapper
    try:
        yield audit
    finally:
        audit["active_after"] = attachment_active()
        transport_module.capture_transport_attachment = original_capture
        transport_module.set_object_qpos = original_set_object_qpos

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


_POSTURE_CARRY_REQUIRED_FIELDS = (
    "posture_carry_success",
    "projected_object_progress_m",
    "lateral_object_drift_m",
    "object_gripper_drift_m",
    "final_object_lift_m",
    "terminal_bilateral_contact",
    "collision_frames",
    "attachment_activations",
    "legacy_teleport_activations",
    "object_pose_writes",
    "infrastructure_error",
)


def posture_carry_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for short posture-locked physical carry."""
    failures = [
        key for key in _POSTURE_CARRY_REQUIRED_FIELDS if key not in record
    ]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    if record.get("posture_carry_success") is not True:
        failures.append("posture_carry_success")
    progress = numeric("projected_object_progress_m")
    if (
        progress is None
        or progress
        < POSTURE_CARRY_THRESHOLDS["projected_object_progress_m"]
    ):
        failures.append("projected_object_progress_m")
    for key in ("lateral_object_drift_m", "object_gripper_drift_m"):
        value = numeric(key)
        if value is None or value > POSTURE_CARRY_THRESHOLDS[key]:
            failures.append(key)
    lift = numeric("final_object_lift_m")
    if lift is None or lift < POSTURE_CARRY_THRESHOLDS["final_object_lift_m"]:
        failures.append("final_object_lift_m")
    if record.get("terminal_bilateral_contact") is not True:
        failures.append("terminal_bilateral_contact")
    for key in (
        "collision_frames",
        "attachment_activations",
        "legacy_teleport_activations",
        "object_pose_writes",
    ):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def posture_carry_accepted(record: Mapping[str, object]) -> bool:
    """Accept only complete, attachment-free short physical carry evidence."""
    return not posture_carry_failures(record)


_SETDOWN_REQUIRED_FIELDS = (
    "physical_grasp",
    "transport_success",
    "place_success",
    "support_detected",
    "released",
    "object_translation_m",
    "net_projected_object_progress_m",
    "net_lateral_object_drift_m",
    "requested_macro_count",
    "completed_macro_count",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "infrastructure_error",
)


def setdown_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for transport followed by physical setdown."""
    failures = [key for key in _SETDOWN_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    for key in (
        "physical_grasp",
        "transport_success",
        "place_success",
        "support_detected",
        "released",
    ):
        if record.get(key) is not True:
            failures.append(key)
    translation = numeric("object_translation_m")
    if translation is None or translation < 0.12:
        failures.append("object_translation_m")
    requested_macros = numeric("requested_macro_count")
    if (
        requested_macros is None
        or requested_macros < 1.0
        or not requested_macros.is_integer()
    ):
        failures.append("requested_macro_count")
        requested_macros = 1.0
    completed_macros = numeric("completed_macro_count")
    if (
        completed_macros is None
        or not completed_macros.is_integer()
        or completed_macros != requested_macros
    ):
        failures.append("completed_macro_count")
    net_progress = numeric("net_projected_object_progress_m")
    if net_progress is None or net_progress < 0.12 * requested_macros:
        failures.append("net_projected_object_progress_m")
    net_lateral = numeric("net_lateral_object_drift_m")
    if net_lateral is None or net_lateral > 0.05:
        failures.append("net_lateral_object_drift_m")
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def setdown_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only collision-free physical extraction, support, and release."""
    return not setdown_gate_failures(record)


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


_HYBRID_EXIT_REQUIRED_FIELDS = (
    "physical_grasp",
    "extraction_success",
    "floor_transition_detected",
    "navigation_retract_success",
    "floor_push_success",
    "physical_contact_steps",
    "official_source_maximum_axis_displacement_m",
    "official_target_distance_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "infrastructure_error",
)


def hybrid_exit_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Require physical evidence for the official strict one-metre exit rule."""
    failures = [key for key in _HYBRID_EXIT_REQUIRED_FIELDS if key not in record]
    for key in (
        "physical_grasp",
        "extraction_success",
        "floor_transition_detected",
        "navigation_retract_success",
        "floor_push_success",
    ):
        if record.get(key) is not True:
            failures.append(key)

    def numeric(key: str) -> float | None:
        value = record.get(key)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    contact_steps = numeric("physical_contact_steps")
    if contact_steps is None or contact_steps < 20.0:
        failures.append("physical_contact_steps")
    displacement = numeric("official_source_maximum_axis_displacement_m")
    if displacement is None or displacement <= 1.0:
        failures.append("official_source_maximum_axis_displacement_m")
    target_distance = numeric("official_target_distance_m")
    if target_distance is None or target_distance >= 0.80:
        failures.append("official_target_distance_m")
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def hybrid_exit_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only a collision-free, attachment-free physical scene exit."""
    return not hybrid_exit_gate_failures(record)


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


def open_fork_target_orientation(
    *,
    inward_axis: object,
    closure_axis: object,
) -> np.ndarray:
    """Build a tool frame for an open horizontal fork under the object."""
    tool_z = _normalized_axis(inward_axis, name="inward_axis")
    tool_x = _normalized_axis(closure_axis, name="closure_axis")
    if abs(float(np.dot(tool_x, tool_z))) > 1e-8:
        raise ValueError("closure_axis must be orthogonal to inward_axis")
    tool_y = _normalized_axis(
        np.cross(tool_z, tool_x),
        name="open fork vertical axis",
    )
    return _validated_rotation_matrix(
        np.column_stack((tool_x, tool_y, tool_z)),
        name="open fork target orientation",
    )


def open_fork_alignment_sufficient(
    rotation: object,
    *,
    inward_axis: object,
    min_inward_projection: float,
    max_closure_vertical: float,
) -> bool:
    """Check whether the open fork is horizontal and points far enough inward."""
    tool_rotation = _validated_rotation_matrix(
        rotation,
        name="open fork rotation",
    )
    inward = _normalized_axis(inward_axis, name="inward_axis")
    minimum = float(min_inward_projection)
    maximum_vertical = float(max_closure_vertical)
    if not np.isfinite(minimum) or minimum < 0.0 or minimum > 1.0:
        raise ValueError("min_inward_projection must be in [0, 1]")
    if (
        not np.isfinite(maximum_vertical)
        or maximum_vertical < 0.0
        or maximum_vertical > 1.0
    ):
        raise ValueError("max_closure_vertical must be in [0, 1]")
    inward_projection = float(np.dot(tool_rotation[:, 2], inward))
    closure_vertical = abs(float(tool_rotation[2, 0]))
    return bool(
        inward_projection >= minimum
        and closure_vertical <= maximum_vertical
    )


def is_allowed_open_fork_support_geom(geom_name: str, arm: str) -> bool:
    """Accept real load-bearing links of an open, non-clamping fork."""
    name = str(geom_name).lower()
    side = str(arm).lower()
    if side not in ("right", "left") or "collision" not in name:
        return False
    if name.startswith(f"gripper0_{side}_"):
        return "finger" in name or "hand_collision" in name
    if side == "left":
        return any(f"arm_{index}_left_collision" in name for index in (4, 5, 6))
    return "_left_" not in name and any(
        f"arm_{index}_collision" in name for index in (4, 5, 6)
    )


def _oriented_box_world_bounds(geometry: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(geometry.get("world_position"), dtype=float)
    half_extents = np.asarray(geometry.get("size"), dtype=float)
    if (
        center.shape != (3,)
        or half_extents.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(half_extents))
        or np.any(half_extents < 0.0)
    ):
        raise ValueError("box geometry must contain finite center and half extents")
    rotation = _validated_rotation_matrix(
        geometry.get("world_rotation"),
        name="box geometry rotation",
    )
    world_half_extents = np.abs(rotation) @ half_extents
    return center - world_half_extents, center + world_half_extents


def open_fork_below_bottom_ready(snapshot: Mapping[str, object]) -> bool:
    """Return whether a real open fingertip is vertically below the object."""
    geometries = snapshot.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("snapshot geometries must be a list")
    bottom = next(
        (
            geometry
            for geometry in geometries
            if isinstance(geometry, Mapping)
            and bool(geometry.get("is_object"))
            and "col_bottom" in str(geometry.get("name", "")).lower()
            and int(geometry.get("type", -1)) == 6
        ),
        None,
    )
    if bottom is None:
        return False
    bottom_minimum, _ = _oriented_box_world_bounds(bottom)
    for geometry in geometries:
        if not isinstance(geometry, Mapping) or int(geometry.get("type", -1)) != 6:
            continue
        name = str(geometry.get("name", ""))
        lowered = name.lower()
        if not is_allowed_open_fork_support_geom(name, "right"):
            continue
        if not any(
            token in lowered
            for token in ("fingertip_collision", "fingerpad_collision")
        ):
            continue
        _, support_maximum = _oriented_box_world_bounds(geometry)
        if float(support_maximum[2]) <= float(bottom_minimum[2]) + 1e-4:
            return True
    return False


def open_fork_under_bottom_support_ready(
    snapshot: Mapping[str, object],
    *,
    minimum_planar_overlap_m: float,
) -> bool:
    """Require a real open fingertip box beneath the object's bottom box."""
    minimum_overlap = float(minimum_planar_overlap_m)
    if not np.isfinite(minimum_overlap) or minimum_overlap <= 0.0:
        raise ValueError("minimum_planar_overlap_m must be finite and positive")
    geometries = snapshot.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("snapshot geometries must be a list")
    bottom = next(
        (
            geometry
            for geometry in geometries
            if isinstance(geometry, Mapping)
            and bool(geometry.get("is_object"))
            and "col_bottom" in str(geometry.get("name", "")).lower()
            and int(geometry.get("type", -1)) == 6
        ),
        None,
    )
    if bottom is None:
        return False
    bottom_minimum, bottom_maximum = _oriented_box_world_bounds(bottom)
    for geometry in geometries:
        if not isinstance(geometry, Mapping) or int(geometry.get("type", -1)) != 6:
            continue
        name = str(geometry.get("name", ""))
        lowered = name.lower()
        if not is_allowed_open_fork_support_geom(name, "right"):
            continue
        if not any(
            token in lowered
            for token in ("fingertip_collision", "fingerpad_collision")
        ):
            continue
        support_minimum, support_maximum = _oriented_box_world_bounds(geometry)
        planar_overlap = np.minimum(bottom_maximum[:2], support_maximum[:2]) - np.maximum(
            bottom_minimum[:2], support_minimum[:2]
        )
        if (
            np.all(planar_overlap >= minimum_overlap)
            and float(support_maximum[2]) <= float(bottom_minimum[2]) + 1e-4
        ):
            return True
    return False


def rotation_error_degrees(current: object, target: object) -> float:
    """Return the geodesic angle between two complete tool frames."""
    current_rotation = _validated_rotation_matrix(
        current,
        name="current rotation",
    )
    target_rotation = _validated_rotation_matrix(
        target,
        name="target rotation",
    )
    delta = target_rotation @ current_rotation.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


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


def object_all_robot_contacts(raw_env, object_name: str) -> tuple[str, ...]:
    """Return every robot geometry physically touching the named object."""
    model = raw_env.sim.model
    object_bodies = _object_body_ids(raw_env, object_name)
    result = set()
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
        if name and name.lower().startswith(("robot0_", "gripper0_")):
            result.add(name)
    return tuple(sorted(result))


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


class _PostureLockedActuatedCarryDriver:
    """Keep robot posture base-relative while delegating physical grip actions."""

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._posture = None

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    @staticmethod
    def _capture_robot_posture(backend) -> dict[str, object]:
        raw_env = backend.env
        robot = raw_env.robots[0]
        joint_names = list(getattr(robot, "robot_arm_joints", ()))
        joint_names.extend(getattr(robot.robot_model, "torso_joints", ()))
        joint_names.extend(getattr(robot.robot_model, "head_joints", ()))
        joint_names = list(dict.fromkeys(joint_names))
        qpos_indexes = [
            raw_env.sim.model.get_joint_qpos_addr(name) for name in joint_names
        ]
        qvel_indexes = [
            raw_env.sim.model.get_joint_qvel_addr(name) for name in joint_names
        ]
        return {
            "qpos_indexes": qpos_indexes,
            "qvel_indexes": qvel_indexes,
            "qpos": np.asarray(
                raw_env.sim.data.qpos[qpos_indexes], dtype=float
            ).copy(),
            "qvel": np.asarray(
                raw_env.sim.data.qvel[qvel_indexes], dtype=float
            ).copy(),
        }

    def capture_hold_targets(self, backend):
        targets = self._delegate.capture_hold_targets(backend)
        self._posture = self._capture_robot_posture(backend)
        return targets

    def _restore_robot_posture(self, backend) -> None:
        if self._posture is None:
            raise RuntimeError("robot posture must be captured before carry")
        raw_env = backend.env
        raw_env.sim.data.qpos[self._posture["qpos_indexes"]] = self._posture[
            "qpos"
        ]
        raw_env.sim.data.qvel[self._posture["qvel_indexes"]] = self._posture[
            "qvel"
        ]
        raw_env.sim.forward()
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)

    def step(self, backend, **kwargs):
        result = self._delegate.step(backend, **kwargs)
        self._restore_robot_posture(backend)
        return result

    def recover_height(self, backend, **kwargs):
        result = self._delegate.recover_height(backend, **kwargs)
        if result:
            self._posture = self._capture_robot_posture(backend)
        return result


def _posture_locked_carry_probe(
    backend,
    object_name: str,
    *,
    distance_m: float,
    world_direction_x: float | None,
    world_direction_y: float | None,
    table_object_z: float,
    max_linear_m_s: float,
    actuated_gripper_hold: bool = False,
    posture_lock_robot_joints: bool = False,
    _transport_module=None,
    _gripper_position=None,
    _contact_reader=object_robot_contacts,
    _actuated_transport=None,
    _physical_carry_config_factory=None,
    _actuated_driver=None,
) -> dict[str, Any]:
    """Measure one attachment-free physical carry under posture-locked navigation."""
    requested_distance = float(distance_m)
    table_z = float(table_object_z)
    requested_max_linear = float(max_linear_m_s)
    if (
        not np.isfinite(requested_distance)
        or requested_distance <= 0.0
        or not np.isfinite(table_z)
        or not np.isfinite(requested_max_linear)
        or requested_max_linear <= 0.0
    ):
        raise ValueError(
            "carry distance and max speed must be positive and table height finite"
        )
    if (world_direction_x is None) != (world_direction_y is None):
        raise ValueError("both world direction components must be provided together")
    if posture_lock_robot_joints and not actuated_gripper_hold:
        raise ValueError("robot posture lock requires actuated gripper hold")

    if _transport_module is None:
        from robosuite.environments.factory_sorting import (
            transport_attachment as _transport_module,
        )
    if _gripper_position is None:
        from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver

        _gripper_position = OfficialScriptedGraspDriver._helpers()[
            "gripper_position"
        ]
    if actuated_gripper_hold and _actuated_transport is None:
        from robot_agent.skills.competition_transport import (
            OfficialPhysicalCarryDriver,
            PhysicalCarryConfig,
            run_physical_transport,
        )

        _actuated_transport = run_physical_transport
        _physical_carry_config_factory = PhysicalCarryConfig
        if posture_lock_robot_joints:
            _actuated_driver = _PostureLockedActuatedCarryDriver(
                OfficialPhysicalCarryDriver()
            )
    if actuated_gripper_hold and _physical_carry_config_factory is None:
        raise ValueError("actuated carry requires a physical carry config factory")

    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]

    def object_position() -> np.ndarray:
        position = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=float,
        )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise RuntimeError("invalid object position during carry probe")
        return position.copy()

    def gripper_positions() -> dict[str, np.ndarray]:
        positions = {
            arm: np.asarray(
                _gripper_position(raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }
        if any(
            position.shape != (3,) or not np.all(np.isfinite(position))
            for position in positions.values()
        ):
            raise RuntimeError("invalid gripper position during carry probe")
        return {arm: position.copy() for arm, position in positions.items()}

    def legacy_transport_active() -> bool:
        return bool(
            getattr(backend, "_held_crate_name", None) is not None
            or getattr(backend, "_held_crate_body_id", None) is not None
        )

    start_base_pose = backend.get_base_pose()
    start_base_xy = np.asarray(start_base_pose[0], dtype=float)
    start_base_yaw = float(start_base_pose[1])
    if start_base_xy.shape != (2,) or not np.all(np.isfinite(start_base_xy)):
        raise RuntimeError("invalid base position during carry probe")
    if not np.isfinite(start_base_yaw):
        raise RuntimeError("invalid base yaw during carry probe")
    start_object = object_position()
    start_grippers = gripper_positions()
    if world_direction_x is None:
        direction = start_object[:2] - start_base_xy
    else:
        direction = np.array(
            [float(world_direction_x), float(world_direction_y)],
            dtype=float,
        )
    _, _ = directed_planar_progress(
        start_xy=start_base_xy,
        end_xy=start_base_xy,
        direction_xy=direction,
    )
    direction /= float(np.linalg.norm(direction))
    target_base_xy = start_base_xy + direction * requested_distance

    object_heights = [float(start_object[2])]
    original_recorder = getattr(backend, "_record_trajectory_frame")
    original_legacy_update = getattr(backend, "_update_held_crate_position")
    backend_dict = getattr(backend, "__dict__", {})
    recorder_was_local = "_record_trajectory_frame" in backend_dict
    legacy_update_was_local = "_update_held_crate_position" in backend_dict
    local_recorder = backend_dict.get("_record_trajectory_frame")
    local_legacy_update = backend_dict.get("_update_held_crate_position")
    original_max_linear = float(getattr(backend, "_max_linear"))
    legacy_calls = 0

    def record_and_sample(*args: object, **kwargs: object):
        result = original_recorder(*args, **kwargs)
        object_heights.append(float(object_position()[2]))
        return result

    def audited_legacy_update(*args: object, **kwargs: object):
        nonlocal legacy_calls
        if legacy_transport_active():
            legacy_calls += 1
        return original_legacy_update(*args, **kwargs)

    setattr(backend, "_record_trajectory_frame", record_and_sample)
    setattr(backend, "_update_held_crate_position", audited_legacy_update)
    setattr(backend, "_max_linear", requested_max_linear)
    navigation_reached = False
    control_result = None
    try:
        with transport_attachment_audit(raw_env, _transport_module) as audit:
            legacy_active_before = legacy_transport_active()
            if not audit["active_before"] and not legacy_active_before:
                if actuated_gripper_hold:
                    transport_kwargs = {
                        "path": [target_base_xy],
                        "object_name": object_name,
                        "hold_yaw": start_base_yaw,
                        "minimum_object_z": (
                            table_z
                            + POSTURE_CARRY_THRESHOLDS["final_object_lift_m"]
                        ),
                        "config": _physical_carry_config_factory(
                            waypoint_tolerance=1e-4,
                            max_steps=max(
                                80,
                                int(
                                    np.ceil(
                                        requested_distance
                                        / (requested_max_linear * 0.05)
                                    )
                                )
                                * 5,
                            ),
                            max_linear=requested_max_linear,
                            max_angular=0.04,
                            max_linear_delta=min(0.005, requested_max_linear),
                            max_angular_delta=0.01,
                            base_control_dt=0.05,
                            max_planar_grasp_drift=(
                                POSTURE_CARRY_THRESHOLDS[
                                    "object_gripper_drift_m"
                                ]
                            ),
                        ),
                    }
                    if _actuated_driver is not None:
                        transport_kwargs["driver"] = _actuated_driver
                    control_result = _actuated_transport(
                        backend,
                        **transport_kwargs,
                    )
                    navigation_reached = bool(
                        isinstance(control_result, Mapping)
                        and control_result.get("success", False)
                    )
                else:
                    navigation_reached = bool(
                        backend.follow_path(
                            [target_base_xy],
                            max_steps=max(
                                20,
                                int(np.ceil(requested_distance / 0.001)) + 5,
                            ),
                            waypoint_tolerance=1e-5,
                            stop_on_collision=True,
                            record_every=1,
                        )
                    )
            legacy_active_after = legacy_transport_active()
    finally:
        if recorder_was_local:
            setattr(backend, "_record_trajectory_frame", local_recorder)
        else:
            delattr(backend, "_record_trajectory_frame")
        if legacy_update_was_local:
            setattr(backend, "_update_held_crate_position", local_legacy_update)
        else:
            delattr(backend, "_update_held_crate_position")
        setattr(backend, "_max_linear", original_max_linear)

    end_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    end_object = object_position()
    end_grippers = gripper_positions()
    base_progress, _ = directed_planar_progress(
        start_xy=start_base_xy,
        end_xy=end_base_xy,
        direction_xy=direction,
    )
    object_progress, lateral_drift = directed_planar_progress(
        start_xy=start_object[:2],
        end_xy=end_object[:2],
        direction_xy=direction,
    )
    gripper_drift = max(
        float(
            np.linalg.norm(
                (end_grippers[arm][:2] - end_object[:2])
                - (start_grippers[arm][:2] - start_object[:2])
            )
        )
        for arm in ("right", "left")
    )
    contacts = _contact_reader(raw_env, object_name)
    bilateral_contact = has_bilateral_object_contact(contacts)
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    attachment_activations = int(audit["attachment_activations"]) + int(
        bool(audit["active_before"])
    ) + int(bool(audit["active_after"]))
    legacy_teleport_activations = (
        int(legacy_calls)
        + int(legacy_active_before)
        + int(legacy_active_after)
    )
    object_pose_writes = int(audit["object_pose_writes"]) + int(legacy_calls)
    final_lift = float(end_object[2] - table_z)
    translation_reached = bool(
        navigation_reached or base_progress >= requested_distance - 1e-4
    )
    success = bool(
        translation_reached
        and object_progress >= POSTURE_CARRY_THRESHOLDS[
            "projected_object_progress_m"
        ]
        and lateral_drift <= POSTURE_CARRY_THRESHOLDS["lateral_object_drift_m"]
        and gripper_drift <= POSTURE_CARRY_THRESHOLDS["object_gripper_drift_m"]
        and final_lift >= POSTURE_CARRY_THRESHOLDS["final_object_lift_m"]
        and bilateral_contact
        and not collision
        and attachment_activations == 0
        and legacy_teleport_activations == 0
        and object_pose_writes == 0
    )

    return {
        "posture_carry_success": success,
        "requested_distance_m": requested_distance,
        "max_linear_m_s": requested_max_linear,
        "control_mode": (
            "actuated_posture_lock"
            if posture_lock_robot_joints
            else (
                "actuated_gripper_hold"
                if actuated_gripper_hold
                else "official_follow_path"
            )
        ),
        "control_result": control_result,
        "world_direction": direction.tolist(),
        "target_base_xy": target_base_xy.tolist(),
        "navigation_reached": navigation_reached,
        "base_progress_m": base_progress,
        "projected_object_progress_m": object_progress,
        "lateral_object_drift_m": lateral_drift,
        "object_gripper_drift_m": gripper_drift,
        "final_object_lift_m": final_lift,
        "minimum_object_lift_m": float(min(object_heights) - table_z),
        "terminal_bilateral_contact": bilateral_contact,
        "terminal_contacts": {
            arm: list(names) for arm, names in contacts.items()
        },
        "collision_frames": int(collision),
        "attachment_activations": attachment_activations,
        "transport_attachment_active_before": bool(audit["active_before"]),
        "transport_attachment_active_after": bool(audit["active_after"]),
        "legacy_teleport_activations": legacy_teleport_activations,
        "legacy_transport_active_before": legacy_active_before,
        "legacy_transport_active_after": legacy_active_after,
        "object_pose_writes": object_pose_writes,
        "infrastructure_error": None,
        "start_base_xy": start_base_xy.tolist(),
        "end_base_xy": end_base_xy.tolist(),
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
    }


def _end_grasp_inchworm_probe(
    backend,
    object_name: str,
    *,
    distance_m: float,
    world_direction_x: float | None,
    world_direction_y: float | None,
    table_object_z: float,
    stroke_m: float,
    stroke_lift_m: float,
    height_gain: float,
    reset_m: float,
    minimum_lift_m: float,
) -> dict[str, Any]:
    """Probe arm-first extraction using the existing physical inchworm controller."""
    from robosuite.environments.factory_sorting import transport_attachment
    from robot_agent.skills.competition_transport import (
        InchwormCarryConfig,
        run_inchworm_transport,
    )

    if (world_direction_x is None) != (world_direction_y is None):
        raise ValueError("both inchworm direction components must be provided together")
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    start_base = np.asarray(backend.get_base_pose()[0], dtype=float).copy()
    if world_direction_x is None:
        direction = start_object[:2] - start_base
    else:
        direction = np.array(
            [float(world_direction_x), float(world_direction_y)],
            dtype=float,
        )
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0 or not np.all(np.isfinite(direction)):
        raise ValueError("inchworm direction must be finite and non-zero")
    direction /= direction_norm
    minimum_lift = float(minimum_lift_m)
    if not np.isfinite(minimum_lift) or minimum_lift <= 0.0:
        raise ValueError("minimum_lift_m must be finite and positive")

    with transport_attachment_audit(raw_env, transport_attachment) as audit:
        result = run_inchworm_transport(
            backend,
            object_name=object_name,
            travel_direction=direction,
            travel_distance=float(distance_m),
            minimum_object_z=float(table_object_z) + minimum_lift,
            config=InchwormCarryConfig(
                stroke_distance=float(stroke_m),
                stroke_vertical_feedforward=float(stroke_lift_m),
                stroke_height_gain=float(height_gain),
                reset_distance=float(reset_m),
                max_cycles=64,
            ),
        )
    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    return {
        **result,
        "world_direction": direction.tolist(),
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "measured_object_translation_m": float(
            np.linalg.norm(end_object[:2] - start_object[:2])
        ),
        "attachment_activations": int(audit["attachment_activations"]),
        "object_pose_writes": int(audit["object_pose_writes"]),
        "transport_attachment_active_before": bool(audit["active_before"]),
        "transport_attachment_active_after": bool(audit["active_after"]),
    }


def _end_grasp_setdown_probe(
    backend,
    object_name: str,
    *,
    distance_m: float,
    world_direction_x: float | None,
    world_direction_y: float | None,
    table_object_z: float,
    stroke_m: float,
    stroke_lift_m: float,
    height_gain: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
) -> dict[str, Any]:
    """Extract with the end grasp, then physically support and release the object."""
    from robosuite.environments.factory_sorting import transport_attachment
    from robot_agent.skills.competition_transport import (
        PhysicalCarryConfig,
        run_physical_place,
    )

    max_descent = float(place_max_descent_m)
    if not np.isfinite(max_descent) or max_descent <= 0.0:
        raise ValueError("place_max_descent_m must be finite and positive")

    transport = _end_grasp_inchworm_probe(
        backend,
        object_name,
        distance_m=distance_m,
        world_direction_x=world_direction_x,
        world_direction_y=world_direction_y,
        table_object_z=table_object_z,
        stroke_m=stroke_m,
        stroke_lift_m=stroke_lift_m,
        height_gain=height_gain,
        reset_m=reset_m,
        minimum_lift_m=minimum_lift_m,
    )
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    place = None
    place_audit = {
        "attachment_activations": 0,
        "object_pose_writes": 0,
        "active_before": False,
        "active_after": False,
    }
    if bool(transport.get("success", False)):
        setdown_xy = np.asarray(
            raw_env.sim.data.body_xpos[body_id][:2], dtype=float
        ).copy()
        with transport_attachment_audit(
            raw_env, transport_attachment
        ) as place_audit:
            place = run_physical_place(
                backend,
                object_name=object_name,
                target_xy=setdown_xy,
                config=PhysicalCarryConfig(max_descent=max_descent),
            )

    end_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    start_object = np.asarray(
        transport["start_object_position"], dtype=float
    )
    place_success = bool(isinstance(place, Mapping) and place.get("success"))
    support_detected = bool(
        isinstance(place, Mapping) and place.get("support_detected")
    )
    contacts = dict(place.get("contacts", {})) if isinstance(place, Mapping) else {}
    released = bool(
        place_success
        and contacts
        and not any(bool(contacts.get(arm, False)) for arm in ("right", "left"))
    )
    transport_success = bool(transport.get("success", False))
    success = bool(
        transport_success and place_success and support_detected and released
    )
    if not transport_success:
        failure_stage = f"transport:{transport.get('failure_stage') or 'unknown'}"
    elif not place_success:
        place_failure = (
            place.get("failure_stage") if isinstance(place, Mapping) else "missing"
        )
        failure_stage = f"place:{place_failure or 'unknown'}"
    elif not released:
        failure_stage = "place:release"
    else:
        failure_stage = None

    return {
        "success": success,
        "failure_stage": failure_stage,
        "requested_macro_count": 1,
        "completed_macro_count": int(success),
        "transport_success": transport_success,
        "place_success": place_success,
        "support_detected": support_detected,
        "released": released,
        "transport": transport,
        "place": place,
        "world_direction": list(transport["world_direction"]),
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "measured_object_translation_m": float(
            np.linalg.norm(end_object[:2] - start_object[:2])
        ),
        "attachment_activations": int(
            transport.get("attachment_activations", 0)
        )
        + int(place_audit["attachment_activations"]),
        "object_pose_writes": int(transport.get("object_pose_writes", 0))
        + int(place_audit["object_pose_writes"]),
        "transport_attachment_active_before": bool(
            transport.get("transport_attachment_active_before", False)
        )
        or bool(place_audit["active_before"]),
        "transport_attachment_active_after": bool(
            transport.get("transport_attachment_active_after", False)
        )
        or bool(place_audit["active_after"]),
    }


def _navigation_retract_probe(
    backend,
    *,
    forward_m: float,
    lateral_m: float,
    target_z: float,
) -> dict[str, Any]:
    """Physically retract both open grippers before floor-level navigation."""
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        ScriptedGraspConfig,
    )

    base_xy, base_yaw = backend.get_base_pose()
    targets = navigation_retract_targets(
        base_xy=base_xy,
        base_yaw=base_yaw,
        forward_m=forward_m,
        lateral_m=lateral_m,
        target_z=target_z,
    )
    retract_driver = OfficialScriptedGraspDriver()
    config = ScriptedGraspConfig(
        max_action=0.30,
        position_tolerance=0.02,
    )
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "navigation_retract_start",
            targets={arm: target.tolist() for arm, target in targets.items()},
        )
    reached = bool(
        retract_driver._move_to_targets(
            backend,
            targets,
            config,
            max_steps=240,
            gripper_value=-1.0,
            tolerance=config.position_tolerance,
        )
    )
    helpers = retract_driver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    final_positions = {
        arm: np.asarray(
            helpers["gripper_position"](raw_env, robot, arm), dtype=float
        ).copy()
        for arm in ("right", "left")
    }
    maximum_error = max(
        float(np.linalg.norm(final_positions[arm] - targets[arm]))
        for arm in ("right", "left")
    )
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    success = bool(reached and not collision)
    if callable(marker):
        marker(
            "navigation_retract_end",
            success=success,
            maximum_error_m=maximum_error,
            collision=collision,
        )
    return {
        "success": success,
        "collision": collision,
        "maximum_error_m": maximum_error,
        "targets": {arm: target.tolist() for arm, target in targets.items()},
        "final_positions": {
            arm: position.tolist() for arm, position in final_positions.items()
        },
    }


def _floor_regrasp_move_probe(
    backend,
    driver,
    source: str,
    object_name: str,
    *,
    safe_clearance_m: float,
) -> dict[str, Any]:
    """Translate outward, orient in clearance, then approach a floor object."""
    from robot_agent.skills.competition_navigation import orient_base

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    object_xy = np.asarray(
        raw_env.sim.data.body_xpos[body_id][:2], dtype=float
    ).copy()
    current_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    grasp_pose = driver._grasp_pose(source, object_name)
    safe_base_xy = floor_regrasp_safe_base_xy(
        object_xy=object_xy,
        current_base_xy=current_base_xy,
        clearance_m=safe_clearance_m,
    )
    target_base_xy = np.asarray(grasp_pose["base_xy"], dtype=float)
    target_yaw = float(grasp_pose["yaw"])
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_regrasp_move_start",
            safe_base_xy=safe_base_xy.tolist(),
            target_base_xy=target_base_xy.tolist(),
            target_yaw=target_yaw,
        )

    safe_reached = bool(
        backend.follow_path(
            [safe_base_xy],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    oriented = bool(safe_reached and orient_base(backend, target_yaw))
    clearance_prepared = bool(
        oriented and driver._prepare_grasp_clearance(object_name)
    )
    target_reached = bool(
        clearance_prepared
        and backend.follow_path(
            [target_base_xy],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    success = bool(
        safe_reached
        and oriented
        and clearance_prepared
        and target_reached
        and not collision
    )
    if success:
        driver._grasp_yaw = target_yaw
        driver._swap_arm_targets = bool(grasp_pose["swap_arm_targets"])
        driver._clearance_prepared = True
    if callable(marker):
        marker(
            "floor_regrasp_move_end",
            success=success,
            safe_reached=safe_reached,
            oriented=oriented,
            clearance_prepared=clearance_prepared,
            target_reached=target_reached,
            collision=collision,
        )
    final_base_xy, final_base_yaw = backend.get_base_pose()
    return {
        "success": success,
        "collision": collision,
        "safe_reached": safe_reached,
        "oriented": oriented,
        "clearance_prepared": clearance_prepared,
        "target_reached": target_reached,
        "safe_base_xy": safe_base_xy.tolist(),
        "target_base_xy": target_base_xy.tolist(),
        "target_yaw": target_yaw,
        "final_base_xy": np.asarray(final_base_xy, dtype=float).tolist(),
        "final_base_yaw": float(final_base_yaw),
    }


def _reposition_base_for_floor_push(
    backend,
    object_name: str,
    *,
    direction_xy: object,
    retreat_clearance_m: float,
    base_standoff_m: float,
    retract_forward_m: float,
    retract_lateral_m: float,
    retract_target_z: float,
) -> dict[str, Any]:
    """Move around a settled object and face the next physical push direction."""
    from robot_agent.skills.competition_navigation import orient_base

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    object_position = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    current_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    targets = floor_base_reposition_targets(
        object_xy=object_position[:2],
        current_base_xy=current_base_xy,
        next_push_direction_xy=direction_xy,
        retreat_clearance_m=retreat_clearance_m,
        base_standoff_m=base_standoff_m,
    )
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_base_reposition_start",
            object_name=object_name,
            direction=targets["direction"].tolist(),
            retreat_base_xy=targets["retreat_base_xy"].tolist(),
            corner_base_xy=targets["corner_base_xy"].tolist(),
            stage_base_xy=targets["stage_base_xy"].tolist(),
        )

    retreat_reached = bool(
        backend.follow_path(
            [targets["retreat_base_xy"], targets["corner_base_xy"]],
            max_steps=1800,
            waypoint_tolerance=0.03,
        )
    )
    oriented = bool(
        retreat_reached and orient_base(backend, targets["target_yaw"])
    )
    retract = (
        _navigation_retract_probe(
            backend,
            forward_m=retract_forward_m,
            lateral_m=retract_lateral_m,
            target_z=retract_target_z,
        )
        if oriented
        else None
    )
    refined_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    refined_targets = floor_base_reposition_targets(
        object_xy=refined_object[:2],
        current_base_xy=np.asarray(backend.get_base_pose()[0], dtype=float),
        next_push_direction_xy=direction_xy,
        retreat_clearance_m=retreat_clearance_m,
        base_standoff_m=base_standoff_m,
    )
    stage_reached = bool(
        isinstance(retract, Mapping)
        and retract.get("success", False)
        and backend.follow_path(
            [refined_targets["stage_base_xy"]],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    success = bool(stage_reached and not collision)
    if callable(marker):
        marker(
            "floor_base_reposition_end",
            object_name=object_name,
            success=success,
            collision=collision,
            final_stage_base_xy=refined_targets["stage_base_xy"].tolist(),
        )
    return {
        "success": success,
        "collision": collision,
        "retreat_reached": retreat_reached,
        "oriented": oriented,
        "retract": retract,
        "stage_reached": stage_reached,
        "targets": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in refined_targets.items()
        },
    }


def _physical_base_push_segment(
    backend,
    object_name: str,
    *,
    direction_xy: object,
    distance_m: float,
    base_speed_m_s: float,
    max_steps: int,
) -> dict[str, Any]:
    """Push one floor segment using only actuated base-object contact."""
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        world_velocity_to_base_frame,
    )

    direction = np.asarray(direction_xy, dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("base push direction must be a finite planar vector")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("base push direction must be non-zero")
    direction /= direction_norm
    requested_distance = float(distance_m)
    requested_speed = float(base_speed_m_s)
    requested_steps = int(max_steps)
    if not np.isfinite(requested_distance) or requested_distance <= 0.0:
        raise ValueError("base push distance must be finite and positive")
    if not np.isfinite(requested_speed) or requested_speed <= 0.0:
        raise ValueError("base push speed must be finite and positive")
    if isinstance(max_steps, bool) or requested_steps != max_steps or requested_steps < 1:
        raise ValueError("base push max_steps must be a positive integer")

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    carry_driver = OfficialPhysicalCarryDriver()
    hold_targets = carry_driver.capture_hold_targets(backend)
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_base_push_segment_start",
            object_name=object_name,
            direction=direction.tolist(),
            requested_distance_m=requested_distance,
        )

    observations = []
    stable_contact_steps = 0
    maximum_contact_steps = 0
    no_contact_steps = 0
    object_progress = 0.0
    lateral_drift = 0.0
    base_progress = 0.0
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    failure_stage = "collision" if collision else None
    steps = 0
    base_control_dt = 0.05
    if failure_stage is None:
        for step in range(requested_steps):
            _, base_yaw = backend.get_base_pose()
            base_velocity = world_velocity_to_base_frame(
                direction * requested_speed,
                base_yaw,
            )
            step_info = carry_driver.step(
                backend,
                object_name=object_name,
                base_command=np.array(
                    [base_velocity[0], base_velocity[1], 0.0], dtype=float
                ),
                hold_targets=hold_targets,
                arm_world_deltas=None,
                gripper_value=-1.0,
                base_control_dt=base_control_dt,
            )
            steps = step + 1
            collision = bool(step_info.get("collision", False))
            contacts = object_all_robot_contacts(raw_env, object_name)
            has_contact = bool(contacts)
            stable_contact_steps = stable_contact_steps + 1 if has_contact else 0
            maximum_contact_steps = max(maximum_contact_steps, stable_contact_steps)
            no_contact_steps = 0 if has_contact else no_contact_steps + 1
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            object_delta = object_position[:2] - start_object[:2]
            object_progress = float(np.dot(object_delta, direction))
            lateral_drift = abs(float(np.dot(object_delta, left_axis)))
            base_delta = (
                np.asarray(backend.get_base_pose()[0], dtype=float) - start_base_xy
            )
            base_progress = float(np.dot(base_delta, direction))
            if step % 25 == 0 or collision or object_progress >= requested_distance:
                observations.append(
                    {
                        "step": steps,
                        "object_position": object_position.tolist(),
                        "object_progress_m": object_progress,
                        "lateral_drift_m": lateral_drift,
                        "base_progress_m": base_progress,
                        "contacts": list(contacts),
                        "stable_contact_steps": stable_contact_steps,
                        "judge_collision": collision,
                    }
                )
            if collision:
                failure_stage = "collision"
                break
            if lateral_drift > 0.30:
                failure_stage = "lateral_drift"
                break
            if object_progress >= requested_distance and maximum_contact_steps >= 20:
                failure_stage = None
                break
            if maximum_contact_steps >= 20 and no_contact_steps >= 80:
                failure_stage = "contact_lost"
                break
        else:
            failure_stage = "timeout"

    success = bool(
        failure_stage is None
        and object_progress >= requested_distance
        and maximum_contact_steps >= 20
        and not collision
    )
    final_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    if callable(marker):
        marker(
            "floor_base_push_segment_end",
            object_name=object_name,
            success=success,
            failure_stage=failure_stage,
            object_progress_m=object_progress,
            physical_contact_steps=maximum_contact_steps,
        )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "collision": collision,
        "steps": steps,
        "physical_contact_steps": maximum_contact_steps,
        "object_progress_m": object_progress,
        "lateral_drift_m": lateral_drift,
        "base_progress_m": base_progress,
        "start_object_position": start_object.tolist(),
        "end_object_position": final_object.tolist(),
        "direction": direction.tolist(),
        "requested_distance_m": requested_distance,
        "observations": observations,
    }


def _floor_corridor_push_probe(
    backend,
    object_name: str,
    *,
    push_direction_x: float,
    push_direction_y: float,
    push_distance_m: float,
    base_standoff_m: float,
    orientation_clearance_m: float,
    lateral_offset_m: float | None,
    torso_drop_m: float,
    base_pusher: bool,
    oriented_retract_forward_m: float,
    oriented_retract_lateral_m: float,
    oriented_retract_target_z: float,
    maximum_lateral_offset_m: float,
    face_offset_m: float,
    hand_separation_m: float,
    hand_height_m: float,
    precontact_clearance_m: float,
    base_speed_m_s: float,
    max_steps: int,
    route_target_xy: object | None = None,
    route_corridor_y: float = -8.40,
    route_arrival_radius_m: float = 0.80,
    route_arrival_margin_m: float = 0.05,
    route_reposition_clearance_m: float = 0.90,
) -> dict[str, Any]:
    """Push a floor object through its current lane using two actuated arms."""
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        ScriptedGraspConfig,
    )
    from robot_agent.skills.competition_navigation import orient_base
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        world_velocity_to_base_frame,
    )

    requested_distance = float(push_distance_m)
    requested_speed = float(base_speed_m_s)
    requested_torso_drop = float(torso_drop_m)
    requested_steps = int(max_steps)
    if not np.isfinite(requested_distance) or requested_distance <= 0.0:
        raise ValueError("floor push distance must be finite and positive")
    if not np.isfinite(requested_speed) or requested_speed <= 0.0:
        raise ValueError("floor push speed must be finite and positive")
    if not np.isfinite(requested_torso_drop) or requested_torso_drop <= 0.0:
        raise ValueError("floor push torso drop must be finite and positive")
    if isinstance(max_steps, bool) or requested_steps != max_steps or requested_steps < 1:
        raise ValueError("floor push max_steps must be a positive integer")

    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    targets = floor_push_staging_targets(
        object_xy=start_object[:2],
        current_base_xy=start_base_xy,
        push_direction_xy=[push_direction_x, push_direction_y],
        base_standoff_m=base_standoff_m,
        orientation_clearance_m=orientation_clearance_m,
        lateral_offset_m=lateral_offset_m,
        maximum_lateral_offset_m=maximum_lateral_offset_m,
        face_offset_m=face_offset_m,
        hand_separation_m=hand_separation_m,
        hand_height_m=hand_height_m,
        precontact_clearance_m=precontact_clearance_m,
    )
    navigation_targets = targets
    direction = np.asarray(targets["direction"], dtype=float)
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_corridor_push_start",
            object_name=object_name,
            escape_base_xy=targets["escape_base_xy"].tolist(),
            orientation_base_xy=targets["orientation_base_xy"].tolist(),
            stage_base_xy=targets["stage_base_xy"].tolist(),
            push_direction=direction.tolist(),
            requested_distance_m=requested_distance,
        )

    escape_stage_reached = bool(
        backend.follow_path(
            [targets["escape_base_xy"]],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    orientation_stage_reached = bool(
        escape_stage_reached
        and backend.follow_path(
            [targets["orientation_base_xy"]],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    oriented = bool(
        orientation_stage_reached
        and orient_base(backend, targets["target_yaw"])
    )
    oriented_retract = (
        _navigation_retract_probe(
            backend,
            forward_m=oriented_retract_forward_m,
            lateral_m=oriented_retract_lateral_m,
            target_z=oriented_retract_target_z,
        )
        if oriented
        else None
    )
    interaction_start_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    route_plan = None
    if route_target_xy is not None:
        if not base_pusher:
            raise ValueError("floor target route requires the physical base pusher")
        route_plan = floor_base_target_route(
            start_object_xy=interaction_start_object[:2],
            target_xy=route_target_xy,
            corridor_y=route_corridor_y,
            arrival_radius_m=route_arrival_radius_m,
            arrival_margin_m=route_arrival_margin_m,
        )
        first_route_segment = route_plan["segments"][0]
        if not np.allclose(first_route_segment["direction"], direction, atol=1e-9):
            raise ValueError(
                "initial floor push direction must match the first target-route segment"
            )
        requested_distance = float(first_route_segment["distance_m"])
        if callable(marker):
            marker(
                "floor_base_target_route",
                target_xy=route_plan["target_xy"],
                corridor_y=route_plan["corridor_y"],
                final_object_xy=route_plan["final_object_xy"],
                segment_count=len(route_plan["segments"]),
            )
    if isinstance(oriented_retract, Mapping) and oriented_retract.get(
        "success", False
    ):
        refined_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        targets = floor_push_staging_targets(
            object_xy=interaction_start_object[:2],
            current_base_xy=refined_base_xy,
            push_direction_xy=[push_direction_x, push_direction_y],
            base_standoff_m=base_standoff_m,
            orientation_clearance_m=orientation_clearance_m,
            lateral_offset_m=lateral_offset_m,
            maximum_lateral_offset_m=maximum_lateral_offset_m,
            face_offset_m=face_offset_m,
            hand_separation_m=hand_separation_m,
            hand_height_m=hand_height_m,
            precontact_clearance_m=precontact_clearance_m,
        )
        if callable(marker):
            marker(
                "floor_corridor_push_refined",
                object_position=interaction_start_object.tolist(),
                stage_base_xy=targets["stage_base_xy"].tolist(),
            )
    stage_reached = bool(
        isinstance(oriented_retract, Mapping)
        and oriented_retract.get("success", False)
        and backend.follow_path(
            [targets["stage_base_xy"]],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    position_driver = OfficialScriptedGraspDriver()
    position_config = ScriptedGraspConfig(
        max_action=0.30,
        position_tolerance=0.03,
        torso_drop=requested_torso_drop,
        torso_minimum=0.10,
        torso_steps=160,
    )
    torso_lowered = False
    precontact_reached = False
    if not base_pusher:
        torso_lowered = bool(
            stage_reached
            and position_driver.lower_torso_for_reach(backend, position_config)
        )
        precontact_reached = bool(
            torso_lowered
            and position_driver._move_to_targets(
                backend,
                targets["precontact"],
                position_config,
                max_steps=480,
                gripper_value=-1.0,
                tolerance=position_config.position_tolerance,
            )
        )
    carry_driver = OfficialPhysicalCarryDriver()
    observations = []
    stable_contact_steps = 0
    maximum_contact_steps = 0
    no_contact_steps = 0
    object_progress = 0.0
    lateral_drift = 0.0
    base_progress = 0.0
    contact_acquire_steps = 0
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    contact_reached = False
    if precontact_reached and not collision and not base_pusher:
        acquire_hold_targets = carry_driver.capture_hold_targets(backend)
        acquire_delta = np.r_[direction * 0.003, -0.003]
        for step in range(240):
            step_info = carry_driver.step(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=acquire_hold_targets,
                arm_world_deltas={
                    "right": acquire_delta,
                    "left": acquire_delta,
                },
                gripper_value=1.0,
            )
            contact_acquire_steps = step + 1
            collision = bool(step_info.get("collision", False))
            contacts = object_robot_contacts(raw_env, object_name)
            has_contact = any(contacts[arm] for arm in ("right", "left"))
            stable_contact_steps = stable_contact_steps + 1 if has_contact else 0
            maximum_contact_steps = max(
                maximum_contact_steps,
                stable_contact_steps,
            )
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            object_delta = object_position[:2] - interaction_start_object[:2]
            object_progress = float(np.dot(object_delta, direction))
            lateral_drift = abs(
                float(np.dot(object_delta, np.asarray(targets["left_axis"])))
            )
            if step % 10 == 0 or collision or stable_contact_steps >= 5:
                observations.append(
                    {
                        "stage": "contact_acquire",
                        "step": contact_acquire_steps,
                        "object_position": object_position.tolist(),
                        "object_progress_m": object_progress,
                        "lateral_drift_m": lateral_drift,
                        "contacts": {
                            arm: list(contacts[arm]) for arm in contacts
                        },
                        "stable_contact_steps": stable_contact_steps,
                        "judge_collision": collision,
                    }
                )
            if collision:
                break
            if stable_contact_steps >= 5:
                contact_reached = True
                break
    collision_pairs = []
    if collision:
        from robot_agent.environments.robosuite_backend import _navigation_collisions

        collision_pairs = [
            list(pair)
            for pair in _navigation_collisions(
                raw_env,
                robot,
                getattr(backend, "_ignore_collision_geom", ()),
            )
        ]
    failure_stage = None
    if not escape_stage_reached:
        failure_stage = "escape_stage_base"
    elif not orientation_stage_reached:
        failure_stage = "orientation_stage_base"
    elif not oriented:
        failure_stage = "orient"
    elif not isinstance(oriented_retract, Mapping) or not oriented_retract.get(
        "success", False
    ):
        failure_stage = "oriented_retract"
    elif not stage_reached:
        failure_stage = "stage_base"
    elif not base_pusher and not torso_lowered:
        failure_stage = "lower_torso"
    elif not base_pusher and not precontact_reached:
        failure_stage = "precontact"
    elif not base_pusher and not contact_reached:
        failure_stage = "contact_acquire"
    elif collision:
        failure_stage = "collision"

    steps = 0
    if failure_stage is None:
        hold_targets = carry_driver.capture_hold_targets(backend)
        push_start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        base_control_dt = 0.05
        arm_step = direction * requested_speed * base_control_dt
        for step in range(requested_steps):
            _, base_yaw = backend.get_base_pose()
            base_velocity = world_velocity_to_base_frame(
                direction * requested_speed,
                base_yaw,
            )
            arm_world_deltas = None
            gripper_value = -1.0
            if not base_pusher:
                arm_world_deltas = {
                    "right": np.r_[arm_step, 0.0],
                    "left": np.r_[arm_step, 0.0],
                }
                gripper_value = 1.0
            step_info = carry_driver.step(
                backend,
                object_name=object_name,
                base_command=np.array(
                    [base_velocity[0], base_velocity[1], 0.0],
                    dtype=float,
                ),
                hold_targets=hold_targets,
                arm_world_deltas=arm_world_deltas,
                gripper_value=gripper_value,
                base_control_dt=base_control_dt,
            )
            steps = step + 1
            collision = bool(step_info.get("collision", False))
            if base_pusher:
                base_contacts = object_all_robot_contacts(raw_env, object_name)
                contacts = {"base": base_contacts}
                has_contact = bool(base_contacts)
            else:
                contacts = object_robot_contacts(raw_env, object_name)
                has_contact = any(contacts[arm] for arm in ("right", "left"))
            stable_contact_steps = stable_contact_steps + 1 if has_contact else 0
            maximum_contact_steps = max(maximum_contact_steps, stable_contact_steps)
            if stable_contact_steps >= 5:
                contact_reached = True
            no_contact_steps = 0 if has_contact else no_contact_steps + 1
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            object_delta = object_position[:2] - interaction_start_object[:2]
            object_progress = float(np.dot(object_delta, direction))
            lateral_drift = abs(
                float(np.dot(object_delta, np.asarray(targets["left_axis"])))
            )
            base_delta = (
                np.asarray(backend.get_base_pose()[0], dtype=float)
                - push_start_base_xy
            )
            base_progress = float(np.dot(base_delta, direction))
            if step % 10 == 0 or collision or object_progress >= requested_distance:
                observations.append(
                    {
                        "step": steps,
                        "object_position": object_position.tolist(),
                        "object_progress_m": object_progress,
                        "lateral_drift_m": lateral_drift,
                        "base_progress_m": base_progress,
                        "contacts": {arm: list(contacts[arm]) for arm in contacts},
                        "judge_collision": collision,
                    }
                )
            if collision:
                failure_stage = "collision"
                break
            if lateral_drift > 0.30:
                failure_stage = "lateral_drift"
                break
            if object_progress >= requested_distance and maximum_contact_steps >= 20:
                failure_stage = None
                break
            if maximum_contact_steps >= 20 and no_contact_steps >= 80:
                failure_stage = "contact_lost"
                break
        else:
            failure_stage = "timeout"

    success = bool(
        failure_stage is None
        and object_progress >= requested_distance
        and maximum_contact_steps >= 20
        and not collision
    )
    total_steps = steps
    total_physical_contact_steps = maximum_contact_steps
    maximum_route_lateral_drift = lateral_drift
    route_segments = [
        {
            "index": 1,
            "direction": direction.tolist(),
            "requested_distance_m": requested_distance,
            "success": success,
            "failure_stage": failure_stage,
            "steps": steps,
            "physical_contact_steps": maximum_contact_steps,
            "object_progress_m": object_progress,
            "lateral_drift_m": lateral_drift,
        }
    ]
    if success and route_plan is not None:
        for route_index, planned_segment in enumerate(
            route_plan["segments"][1:], start=2
        ):
            segment_direction = np.asarray(
                planned_segment["direction"], dtype=float
            )
            current_object = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            desired_end_xy = np.asarray(
                planned_segment["end_object_xy"], dtype=float
            )
            remaining_distance = float(
                np.dot(desired_end_xy - current_object[:2], segment_direction)
            )
            if remaining_distance <= 0.01:
                route_segments.append(
                    {
                        "index": route_index,
                        "direction": segment_direction.tolist(),
                        "requested_distance_m": remaining_distance,
                        "success": True,
                        "failure_stage": None,
                        "skipped": True,
                    }
                )
                continue

            reposition = _reposition_base_for_floor_push(
                backend,
                object_name,
                direction_xy=segment_direction,
                retreat_clearance_m=route_reposition_clearance_m,
                base_standoff_m=base_standoff_m,
                retract_forward_m=oriented_retract_forward_m,
                retract_lateral_m=oriented_retract_lateral_m,
                retract_target_z=oriented_retract_target_z,
            )
            if not reposition["success"]:
                success = False
                collision = bool(reposition["collision"])
                failure_stage = f"route_{route_index}:reposition"
                route_segments.append(
                    {
                        "index": route_index,
                        "direction": segment_direction.tolist(),
                        "requested_distance_m": remaining_distance,
                        "success": False,
                        "failure_stage": failure_stage,
                        "reposition": reposition,
                    }
                )
                break

            segment = _physical_base_push_segment(
                backend,
                object_name,
                direction_xy=segment_direction,
                distance_m=remaining_distance,
                base_speed_m_s=requested_speed,
                max_steps=requested_steps,
            )
            segment["index"] = route_index
            segment["reposition"] = reposition
            route_segments.append(segment)
            total_steps += int(segment["steps"])
            total_physical_contact_steps += int(
                segment["physical_contact_steps"]
            )
            maximum_route_lateral_drift = max(
                maximum_route_lateral_drift,
                float(segment["lateral_drift_m"]),
            )
            collision = bool(segment["collision"])
            if not segment["success"]:
                success = False
                failure_stage = (
                    f"route_{route_index}:"
                    f"{segment.get('failure_stage') or 'unknown'}"
                )
                break

    final_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    final_target_distance = None
    if route_plan is not None:
        final_target_distance = float(
            np.linalg.norm(
                final_object[:2] - np.asarray(route_plan["target_xy"], dtype=float)
            )
        )
        if success and final_target_distance >= float(route_arrival_radius_m):
            success = False
            failure_stage = "target_distance"
    if callable(marker):
        marker(
            "floor_corridor_push_end",
            object_name=object_name,
            success=success,
            failure_stage=failure_stage,
            object_progress_m=object_progress,
            physical_contact_steps=total_physical_contact_steps,
            final_target_distance_m=final_target_distance,
        )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "pusher": "base" if base_pusher else "dual_arm",
        "escape_stage_reached": escape_stage_reached,
        "orientation_stage_reached": orientation_stage_reached,
        "stage_reached": stage_reached,
        "torso_lowered": torso_lowered,
        "oriented": oriented,
        "oriented_retract": oriented_retract,
        "precontact_reached": precontact_reached,
        "contact_reached": contact_reached,
        "contact_acquire_steps": contact_acquire_steps,
        "collision": collision,
        "collision_pairs": collision_pairs,
        "steps": total_steps,
        "physical_contact_steps": total_physical_contact_steps,
        "object_progress_m": object_progress,
        "lateral_drift_m": maximum_route_lateral_drift,
        "base_progress_m": base_progress,
        "start_object_position": start_object.tolist(),
        "interaction_start_object_position": interaction_start_object.tolist(),
        "end_object_position": final_object.tolist(),
        "route_plan": route_plan,
        "route_segments": route_segments,
        "final_target_distance_m": final_target_distance,
        "targets": {
            "escape_base_xy": navigation_targets["escape_base_xy"].tolist(),
            "orientation_base_xy": navigation_targets[
                "orientation_base_xy"
            ].tolist(),
            "stage_base_xy": targets["stage_base_xy"].tolist(),
            "target_yaw": float(targets["target_yaw"]),
            "lateral_offset_m": float(targets["lateral_offset_m"]),
            "precontact": {
                arm: target.tolist()
                for arm, target in targets["precontact"].items()
            },
            "contact": {
                arm: target.tolist() for arm, target in targets["contact"].items()
            },
        },
        "observations": observations,
    }


def _end_grasp_regrasp_probe(
    backend,
    driver,
    source: str,
    object_name: str,
    *,
    macro_count: int,
    distance_m: float,
    world_direction_x: float | None,
    world_direction_y: float | None,
    table_object_z: float,
    stroke_m: float,
    stroke_lift_m: float,
    height_gain: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    floor_retract_forward_m: float = 0.20,
    floor_retract_lateral_m: float = 0.15,
    floor_retract_target_z: float = 1.45,
    floor_transition_margin_m: float = 0.30,
    floor_regrasp_safe_clearance_m: float = 1.20,
    _setdown_probe=None,
    _navigation_retract=None,
    _floor_regrasp_move=None,
) -> dict[str, Any]:
    """Repeat physical setdown, dynamic base repositioning, and regrasp."""
    if isinstance(macro_count, bool) or int(macro_count) != macro_count:
        raise ValueError("macro_count must be a positive integer")
    requested_macros = int(macro_count)
    if requested_macros < 1:
        raise ValueError("macro_count must be a positive integer")
    setdown_probe = _setdown_probe or _end_grasp_setdown_probe
    retract_probe = _navigation_retract or _navigation_retract_probe
    floor_move_probe = _floor_regrasp_move or _floor_regrasp_move_probe
    floor_margin = float(floor_transition_margin_m)
    if not np.isfinite(floor_margin) or floor_margin <= 0.0:
        raise ValueError("floor_transition_margin_m must be finite and positive")
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    setdown_kwargs = {
        "distance_m": distance_m,
        "world_direction_x": world_direction_x,
        "world_direction_y": world_direction_y,
        "table_object_z": table_object_z,
        "stroke_m": stroke_m,
        "stroke_lift_m": stroke_lift_m,
        "height_gain": height_gain,
        "reset_m": reset_m,
        "minimum_lift_m": minimum_lift_m,
        "place_max_descent_m": place_max_descent_m,
    }
    macros = []
    regrasps = []
    completed_macros = 0
    attachment_activations = 0
    object_pose_writes = 0
    attachment_active_before = False
    attachment_active_after = False
    failure_stage = None

    for macro_index in range(requested_macros):
        macro = setdown_probe(
            backend,
            object_name,
            **setdown_kwargs,
        )
        macros.append({"macro": macro_index + 1, **macro})
        attachment_activations += int(macro.get("attachment_activations", 0))
        object_pose_writes += int(macro.get("object_pose_writes", 0))
        attachment_active_before = bool(
            attachment_active_before
            or macro.get("transport_attachment_active_before", False)
        )
        attachment_active_after = bool(
            attachment_active_after
            or macro.get("transport_attachment_active_after", False)
        )
        if not bool(macro.get("success", False)):
            failure_stage = (
                f"macro_{macro_index + 1}:"
                f"{macro.get('failure_stage') or 'unknown'}"
            )
            break
        completed_macros += 1
        driver._physical_hold = None
        if completed_macros >= requested_macros:
            break

        regrasp_start = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        ).copy()
        navigation_retract = None
        floor_transition = bool(
            float(regrasp_start[2]) < float(table_object_z) - floor_margin
        )
        if floor_transition:
            navigation_retract = retract_probe(
                backend,
                forward_m=floor_retract_forward_m,
                lateral_m=floor_retract_lateral_m,
                target_z=floor_retract_target_z,
            )
            if not bool(navigation_retract.get("success", False)):
                regrasps.append(
                    {
                        "after_macro": macro_index + 1,
                        "navigation_retract": navigation_retract,
                        "move_success": False,
                        "physical_grasp": False,
                        "grasp": None,
                        "start_object_position": regrasp_start.tolist(),
                        "end_object_position": regrasp_start.tolist(),
                        "object_translation_m": 0.0,
                    }
                )
                failure_stage = f"regrasp_{macro_index + 1}:retract"
                break
        floor_regrasp_move = None
        if floor_transition:
            floor_regrasp_move = floor_move_probe(
                backend,
                driver,
                source,
                object_name,
                safe_clearance_m=floor_regrasp_safe_clearance_m,
            )
            moved = bool(floor_regrasp_move.get("success", False))
        else:
            moved = bool(
                driver.move(
                    source,
                    carrying=False,
                    object_name=object_name,
                )
            )
        grasp = driver.grasp(source, object_name) if moved else None
        regrasp_end = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        ).copy()
        physical_grasp = bool(
            isinstance(grasp, Mapping)
            and grasp.get("success")
            and grasp.get("lift_success")
            and all(bool(value) for value in grasp.get("contacts", {}).values())
        )
        regrasps.append(
            {
                "after_macro": macro_index + 1,
                "navigation_retract": navigation_retract,
                "floor_regrasp_move": floor_regrasp_move,
                "move_success": moved,
                "physical_grasp": physical_grasp,
                "grasp": grasp,
                "start_object_position": regrasp_start.tolist(),
                "end_object_position": regrasp_end.tolist(),
                "object_translation_m": float(
                    np.linalg.norm(regrasp_end[:2] - regrasp_start[:2])
                ),
            }
        )
        if not moved:
            failure_stage = f"regrasp_{macro_index + 1}:move"
            break
        if not physical_grasp:
            failure_stage = f"regrasp_{macro_index + 1}:grasp"
            break

    end_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    last_macro = macros[-1] if macros else {}
    success = bool(
        failure_stage is None and completed_macros == requested_macros
    )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "requested_macro_count": requested_macros,
        "completed_macro_count": completed_macros,
        "transport_success": bool(
            macros and all(bool(macro.get("transport_success")) for macro in macros)
        ),
        "place_success": bool(last_macro.get("place_success", False)),
        "support_detected": bool(last_macro.get("support_detected", False)),
        "released": bool(last_macro.get("released", False)),
        "macros": macros,
        "regrasps": regrasps,
        "world_direction": list(last_macro.get("world_direction", [])),
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "measured_object_translation_m": float(
            np.linalg.norm(end_object[:2] - start_object[:2])
        ),
        "attachment_activations": attachment_activations,
        "object_pose_writes": object_pose_writes,
        "transport_attachment_active_before": attachment_active_before,
        "transport_attachment_active_after": attachment_active_after,
    }


def _end_grasp_floor_push_probe(
    backend,
    driver,
    source: str,
    object_name: str,
    *,
    macro_count: int = 2,
    distance_m: float = 0.14,
    world_direction_x: float | None = 1.0,
    world_direction_y: float | None = 0.0,
    table_object_z: float,
    stroke_m: float = 0.08,
    stroke_lift_m: float = 0.0,
    height_gain: float = 0.0,
    reset_m: float = 0.06,
    minimum_lift_m: float = 0.10,
    place_max_descent_m: float = 0.45,
    floor_retract_forward_m: float = 0.20,
    floor_retract_lateral_m: float = 0.15,
    floor_retract_target_z: float = 1.45,
    floor_transition_margin_m: float = 0.30,
    push_direction_x: float,
    push_direction_y: float,
    push_distance_m: float = 1.05,
    push_base_standoff_m: float = 0.85,
    push_orientation_clearance_m: float = 0.35,
    push_oriented_retract_forward_m: float = 0.20,
    push_oriented_retract_lateral_m: float = 0.08,
    push_lateral_offset_m: float | None = None,
    push_torso_drop_m: float = 0.24,
    push_base_pusher: bool = False,
    push_maximum_lateral_offset_m: float = 0.25,
    push_face_offset_m: float = 0.24,
    push_hand_separation_m: float = 0.28,
    push_hand_height_m: float = 0.38,
    push_precontact_clearance_m: float = 0.08,
    push_base_speed_m_s: float = 0.025,
    push_max_steps: int = 1200,
    source_center_xy: object | None = None,
    target_center_xy: object | None = None,
    floor_base_route_to_target: bool = False,
    floor_base_route_corridor_y: float = -8.40,
    floor_base_route_arrival_margin_m: float = 0.05,
    floor_base_route_reposition_clearance_m: float = 0.90,
    _extraction_probe=None,
    _navigation_retract=None,
    _floor_push=None,
) -> dict[str, Any]:
    """Extract from the table, retract, then physically push along a floor lane."""
    extraction_probe = _extraction_probe or _end_grasp_regrasp_probe
    retract_probe = _navigation_retract or _navigation_retract_probe
    floor_push_probe = _floor_push or _floor_corridor_push_probe
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    extraction = extraction_probe(
        backend,
        driver,
        source,
        object_name,
        macro_count=macro_count,
        distance_m=distance_m,
        world_direction_x=world_direction_x,
        world_direction_y=world_direction_y,
        table_object_z=table_object_z,
        stroke_m=stroke_m,
        stroke_lift_m=stroke_lift_m,
        height_gain=height_gain,
        reset_m=reset_m,
        minimum_lift_m=minimum_lift_m,
        place_max_descent_m=place_max_descent_m,
        floor_retract_forward_m=floor_retract_forward_m,
        floor_retract_lateral_m=floor_retract_lateral_m,
        floor_retract_target_z=floor_retract_target_z,
        floor_transition_margin_m=floor_transition_margin_m,
    )
    after_extraction = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    floor_transition = bool(
        after_extraction[2]
        < float(table_object_z) - float(floor_transition_margin_m)
    )
    navigation_retract = None
    floor_push = None
    failure_stage = None
    if not bool(extraction.get("success", False)):
        failure_stage = "extraction"
    elif not floor_transition:
        failure_stage = "floor_transition"
    else:
        navigation_retract = retract_probe(
            backend,
            forward_m=floor_retract_forward_m,
            lateral_m=floor_retract_lateral_m,
            target_z=floor_retract_target_z,
        )
        if not bool(navigation_retract.get("success", False)):
            failure_stage = "navigation_retract"
        else:
            floor_push = floor_push_probe(
                backend,
                object_name,
                push_direction_x=push_direction_x,
                push_direction_y=push_direction_y,
                push_distance_m=push_distance_m,
                base_standoff_m=push_base_standoff_m,
                orientation_clearance_m=push_orientation_clearance_m,
                lateral_offset_m=push_lateral_offset_m,
                torso_drop_m=push_torso_drop_m,
                base_pusher=push_base_pusher,
                oriented_retract_forward_m=push_oriented_retract_forward_m,
                oriented_retract_lateral_m=push_oriented_retract_lateral_m,
                oriented_retract_target_z=floor_retract_target_z,
                maximum_lateral_offset_m=push_maximum_lateral_offset_m,
                face_offset_m=push_face_offset_m,
                hand_separation_m=push_hand_separation_m,
                hand_height_m=push_hand_height_m,
                precontact_clearance_m=push_precontact_clearance_m,
                base_speed_m_s=push_base_speed_m_s,
                max_steps=push_max_steps,
                route_target_xy=(
                    target_center_xy if floor_base_route_to_target else None
                ),
                route_corridor_y=floor_base_route_corridor_y,
                route_arrival_radius_m=0.80,
                route_arrival_margin_m=floor_base_route_arrival_margin_m,
                route_reposition_clearance_m=(
                    floor_base_route_reposition_clearance_m
                ),
            )
            if not bool(floor_push.get("success", False)):
                failure_stage = f"floor_push:{floor_push.get('failure_stage') or 'unknown'}"

    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    displacement = end_object[:2] - start_object[:2]
    official_source_center = np.asarray(
        start_object[:2] if source_center_xy is None else source_center_xy,
        dtype=float,
    )
    if official_source_center.shape != (2,) or not np.all(
        np.isfinite(official_source_center)
    ):
        raise ValueError("official source center must be a finite planar vector")
    official_source_displacement = end_object[:2] - official_source_center
    official_target_distance = float("inf")
    if target_center_xy is not None:
        official_target_center = np.asarray(target_center_xy, dtype=float)
        if official_target_center.shape != (2,) or not np.all(
            np.isfinite(official_target_center)
        ):
            raise ValueError("official target center must be a finite planar vector")
        official_target_distance = float(
            np.linalg.norm(end_object[:2] - official_target_center)
        )
    success = bool(failure_stage is None)
    return {
        "success": success,
        "failure_stage": failure_stage,
        "extraction_success": bool(extraction.get("success", False)),
        "floor_transition_detected": floor_transition,
        "navigation_retract_success": bool(
            isinstance(navigation_retract, Mapping)
            and navigation_retract.get("success", False)
        ),
        "floor_push_success": bool(
            isinstance(floor_push, Mapping) and floor_push.get("success", False)
        ),
        "physical_contact_steps": int(
            floor_push.get("physical_contact_steps", 0)
            if isinstance(floor_push, Mapping)
            else 0
        ),
        "maximum_axis_displacement_m": float(np.max(np.abs(displacement))),
        "official_source_maximum_axis_displacement_m": float(
            np.max(np.abs(official_source_displacement))
        ),
        "official_target_distance_m": official_target_distance,
        "start_object_position": start_object.tolist(),
        "after_extraction_object_position": after_extraction.tolist(),
        "end_object_position": end_object.tolist(),
        "measured_object_translation_m": float(np.linalg.norm(displacement)),
        "requested_macro_count": int(extraction.get("requested_macro_count", 0)),
        "completed_macro_count": int(extraction.get("completed_macro_count", 0)),
        "transport_success": bool(extraction.get("transport_success", False)),
        "place_success": bool(extraction.get("place_success", False)),
        "support_detected": bool(extraction.get("support_detected", False)),
        "released": bool(extraction.get("released", False)),
        "world_direction": [float(push_direction_x), float(push_direction_y)],
        "attachment_activations": int(extraction.get("attachment_activations", 0)),
        "object_pose_writes": int(extraction.get("object_pose_writes", 0)),
        "extraction": extraction,
        "navigation_retract": navigation_retract,
        "floor_push": floor_push,
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
    below_bottom_clearance_m: float,
    raise_above_bottom_m: float,
    horizontal_fork: bool,
    orientation_max_action: float,
    orientation_position_max_action: float,
    orientation_tolerance_deg: float,
    orientation_stable_steps: int,
    orientation_max_steps: int,
    orientation_min_inward_projection: float,
    orientation_max_closure_vertical: float,
    horizontal_inset_m: float,
    left_clearance_lift_m: float,
    post_inset_base_advance_m: float,
    torso_raise_m: float,
    torso_raise_orientation_max_action: float,
    torso_raise_base_correction_max_m: float,
    orient_before_descent: bool,
    post_inset_world_direction_x: float | None,
    post_inset_world_direction_y: float | None,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        _is_allowed_cradle_geom,
        direct_base_step_target,
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
    if not np.isfinite(float(torso_raise_m)) or float(torso_raise_m) < 0.0:
        raise ValueError("torso_raise_m must be finite and non-negative")
    if not np.isfinite(float(torso_raise_orientation_max_action)) or not (
        0.0 < float(torso_raise_orientation_max_action) <= 1.0
    ):
        raise ValueError(
            "torso_raise_orientation_max_action must be in (0, 1]"
        )
    if not np.isfinite(float(torso_raise_base_correction_max_m)) or float(
        torso_raise_base_correction_max_m
    ) <= 0.0:
        raise ValueError(
            "torso_raise_base_correction_max_m must be finite and positive"
        )
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
    torso_range = np.asarray(
        raw_env.sim.model.jnt_range[torso_joint_id], dtype=float
    )
    if torso_target_m is not None:
        torso_target = float(torso_target_m)
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
        below_bottom_clearance_m=below_bottom_clearance_m,
        raise_above_bottom_m=raise_above_bottom_m,
    )
    observations: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    collision_steps = 0
    maximum_support_steps = 0

    def is_right_support(geom_name: str) -> bool:
        return is_allowed_open_fork_support_geom(geom_name, "right")

    def right_eef_position() -> np.ndarray:
        return np.asarray(
            helpers["gripper_position"](raw_env, robot, "right"),
            dtype=float,
        )

    def left_eef_position() -> np.ndarray:
        return np.asarray(
            helpers["gripper_position"](raw_env, robot, "left"),
            dtype=float,
        )

    def right_eef_pose() -> tuple[np.ndarray, np.ndarray]:
        return eef_site_pose(raw_env, robot, "right")

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

    def execute_torso_raise_into_support() -> bool:
        """Lift an inserted open fork while preserving its robot joint posture."""
        nonlocal collision_steps, maximum_support_steps
        from robot_agent.environments.robosuite_backend import (
            _capture_upper_body_posture,
            _restore_upper_body_posture,
        )

        requested_raise = float(torso_raise_m)
        if not np.isfinite(requested_raise) or requested_raise <= 0.0:
            raise ValueError("torso_raise_m must be finite and positive")
        stage = "raise_open_with_torso"
        start_torso = torso_position()
        target_torso = min(start_torso + requested_raise, float(torso_range[1]))
        if target_torso <= start_torso + 1e-6:
            raise ValueError("torso_raise_m has no room inside the joint range")

        posture = _capture_upper_body_posture(raw_env, robot)
        try:
            torso_posture_index = posture["joint_names"].index(torso_joint_name)
        except ValueError as exc:
            raise RuntimeError("captured posture omits the torso lift joint") from exc
        posture["qvel"][:] = 0.0
        start_eef = right_eef_position().copy()
        idle_action = np.zeros_like(raw_env.action_spec[0])
        lift_step_m = 0.001
        lift_steps = int(np.ceil((target_torso - start_torso) / lift_step_m))
        settle_steps = 20
        safety_failure = None
        stable_support_steps = 0
        success = False

        for local_step in range(lift_steps + settle_steps):
            commanded_torso = min(
                target_torso,
                start_torso + lift_step_m * float(local_step + 1),
            )
            posture["qpos"][torso_posture_index] = commanded_torso
            _restore_upper_body_posture(raw_env, posture)
            _, _, _, info = raw_env.step(idle_action)
            _restore_upper_body_posture(raw_env, posture)
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)

            measured_eef = right_eef_position()
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            )
            object_lift_m = float(object_position[2] - start_object[2])
            contacts = object_robot_contacts(raw_env, object_name)
            right_support = any(
                is_right_support(name) for name in contacts["right"]
            )
            invalid_right_contact = any(
                not is_right_support(name) for name in contacts["right"]
            )
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
            observations.append(
                {
                    "stage": stage,
                    "step": local_step + 1,
                    "posture_locked": True,
                    "start_torso_position": start_torso,
                    "target_torso_position": target_torso,
                    "commanded_torso_position": commanded_torso,
                    "torso_position": torso_position(),
                    "start_eef_position": start_eef.tolist(),
                    "eef_position": measured_eef.tolist(),
                    "eef_lift_m": float(measured_eef[2] - start_eef[2]),
                    "object_position": object_position.tolist(),
                    "object_lift_m": object_lift_m,
                    "contacts": {
                        arm: list(names) for arm, names in contacts.items()
                    },
                    "right_support": right_support,
                    "stable_support_steps": stable_support_steps,
                    "judge_collision": collision,
                }
            )
            if collision:
                safety_failure = "collision"
                break
            if contacts["left"] or invalid_right_contact:
                safety_failure = "unsafe_object_contact"
                break
            if stable_support_steps >= 5:
                success = True
                break
        if not success and safety_failure is None:
            safety_failure = "target_without_support"
        stages.append(
            {
                "stage": stage,
                "success": success,
                "steps": sum(
                    1 for item in observations if item.get("stage") == stage
                ),
                "safety_failure": safety_failure,
                "posture_locked": True,
                "requested_raise_m": requested_raise,
                "start_torso_position": start_torso,
                "target_torso_position": target_torso,
                "final_torso_position": torso_position(),
                "start_eef_position": start_eef.tolist(),
                "final_eef_position": right_eef_position().tolist(),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
                "final_contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(
                        raw_env, object_name
                    ).items()
                },
                "stable_support_steps": stable_support_steps,
            }
        )
        return success

    def execute_post_inset_base_advance() -> bool:
        """Shift the physically held open fork into measured bottom overlap."""
        nonlocal collision_steps
        requested_distance = float(post_inset_base_advance_m)
        if not np.isfinite(requested_distance) or requested_distance < 0.0:
            raise ValueError(
                "post_inset_base_advance_m must be finite and non-negative"
        )
        stage = "advance_base_for_fork_overlap"
        segment_start_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        custom_direction = None
        if (post_inset_world_direction_x is None) != (
            post_inset_world_direction_y is None
        ):
            raise ValueError(
                "post-inset world direction requires both x and y"
            )
        if post_inset_world_direction_x is not None:
            custom_direction = np.array(
                [
                    float(post_inset_world_direction_x),
                    float(post_inset_world_direction_y),
                ],
                dtype=float,
            )
            direction_norm = float(np.linalg.norm(custom_direction))
            if not np.all(np.isfinite(custom_direction)) or direction_norm <= 1e-9:
                raise ValueError(
                    "post-inset world direction must be finite and nonzero"
                )
            custom_direction /= direction_norm
        if custom_direction is None:
            object_xy = np.asarray(
                raw_env.sim.data.body_xpos[body_id][:2], dtype=float
            )
            custom_direction = object_xy - segment_start_xy
            direction_norm = float(np.linalg.norm(custom_direction))
            if direction_norm <= 1e-9:
                raise RuntimeError("cannot resolve post-inset base direction")
            custom_direction /= direction_norm
        segment_target_xy = segment_start_xy + (
            custom_direction * requested_distance
        )
        safety_failure = None
        navigation_reached = backend.follow_path(
            [segment_target_xy],
            max_steps=max(
                20,
                int(np.ceil(requested_distance / 0.001)) + 5,
            ),
            waypoint_tolerance=1e-5,
            stop_on_collision=True,
            record_every=1,
        )
        final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        final_translation = float(
            np.linalg.norm(final_base_xy - segment_start_xy)
        )
        translation_reached = bool(
            navigation_reached
            or final_translation >= requested_distance - 1e-4
        )
        final_geometry = geometry_snapshot(raw_env, object_name)
        geometry_ready = open_fork_under_bottom_support_ready(
            final_geometry,
            minimum_planar_overlap_m=0.001,
        )
        contacts = object_robot_contacts(raw_env, object_name)
        invalid_right_contact = any(
            not is_allowed_open_fork_support_geom(name, "right")
            for name in contacts["right"]
        )
        collision = bool(getattr(raw_env, "has_judge_collision", False))
        collision_steps += int(collision)
        if collision:
            safety_failure = "collision"
        elif contacts["left"] or invalid_right_contact:
            safety_failure = "unsafe_object_contact"
        elif not translation_reached:
            safety_failure = "base_navigation"
        success = bool(safety_failure is None and geometry_ready)
        if not success and safety_failure is None:
            safety_failure = "bottom_overlap_not_reached"
        observations.append(
            {
                "stage": stage,
                "step": 1,
                "base_xy": final_base_xy.tolist(),
                "base_translation_m": final_translation,
                "translation_reached": translation_reached,
                "eef_position": right_eef_position().tolist(),
                "object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
                "contacts": {
                    arm: list(names) for arm, names in contacts.items()
                },
                "geometry_ready": geometry_ready,
                "judge_collision": collision,
            }
        )
        stages.append(
            {
                "stage": stage,
                "success": success,
                "steps": sum(
                    1 for item in observations if item.get("stage") == stage
                ),
                "safety_failure": safety_failure,
                "requested_translation_m": requested_distance,
                "base_translation_m": final_translation,
                "translation_reached": translation_reached,
                "geometry_ready": geometry_ready,
                "world_direction": (
                    None if custom_direction is None else custom_direction.tolist()
                ),
                "final_eef_position": right_eef_position().tolist(),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
                "final_contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(
                        raw_env, object_name
                    ).items()
                },
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
        other_arm_world_target: np.ndarray | None = None,
        right_orientation_target: np.ndarray | None = None,
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
            measured_orientation = None
            orientation_error_deg = None
            orientation_aligned = True
            if right_orientation_target is not None:
                controller = robot.part_controllers["right"]
                if controller.name != "OSC_POSE" or controller.input_type != "delta":
                    raise RuntimeError(
                        "open fork pose hold requires OSC_POSE delta control"
                    )
                _, measured_orientation = right_eef_pose()
                input_ref_frame = getattr(controller, "input_ref_frame", "world")
                if input_ref_frame == "world":
                    origin_rotation = np.eye(3)
                elif input_ref_frame == "base":
                    origin_rotation = controller.origin_ori
                    if origin_rotation is None:
                        _, origin_rotation = (
                            robot.composite_controller.get_controller_base_pose(
                                controller_name="right"
                            )
                        )
                else:
                    raise RuntimeError(
                        "unsupported orientation reference frame for right: "
                        f"{input_ref_frame}"
                    )
                world_rotation_delta = (
                    np.asarray(right_orientation_target, dtype=float)
                    @ measured_orientation.T
                )
                orientation_action = normalized_osc_orientation_command(
                    world_rotation_delta=world_rotation_delta,
                    controller_origin_rotation=origin_rotation,
                    output_min=controller.output_min,
                    output_max=controller.output_max,
                    max_action=float(torso_raise_orientation_max_action),
                )
                arm_action[3:6] = orientation_action
            arm_actions = {"right": arm_action}
            measured_left = None
            if other_arm_world_target is not None:
                measured_left = left_eef_position()
                left_controller_delta = helpers["world_delta"](
                    robot,
                    "left",
                    np.asarray(other_arm_world_target, dtype=float) - measured_left,
                )
                arm_actions["left"] = helpers["arm_action"](
                    robot,
                    "left",
                    left_controller_delta,
                    0.30,
                )
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=-1.0,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            measured_left_after = (
                None
                if other_arm_world_target is None
                else left_eef_position()
            )
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)
            measured_eef = right_eef_position()
            if right_orientation_target is not None:
                _, measured_orientation = right_eef_pose()
                orientation_error_deg = rotation_error_degrees(
                    measured_orientation,
                    right_orientation_target,
                )
                orientation_aligned = bool(
                    orientation_error_deg <= float(orientation_tolerance_deg)
                    or open_fork_alignment_sufficient(
                        measured_orientation,
                        inward_axis=np.array([0.0, -1.0, 0.0]),
                        min_inward_projection=orientation_min_inward_projection,
                        max_closure_vertical=orientation_max_closure_vertical,
                    )
                )
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            )
            contacts = object_robot_contacts(raw_env, object_name)
            right_support = any(
                is_right_support(geom)
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
                "other_arm_world_target": (
                    None
                    if other_arm_world_target is None
                    else np.asarray(other_arm_world_target, dtype=float).tolist()
                ),
                "other_arm_eef_position": (
                    None
                    if measured_left_after is None
                    else measured_left_after.tolist()
                ),
                "object_position": object_position.tolist(),
                "object_lift_m": object_lift_m,
                "torso_position": torso_position(),
                "contacts": {
                    arm: list(names) for arm, names in contacts.items()
                },
                "right_support": right_support,
                "stable_support_steps": stable_support_steps,
                "judge_collision": collision,
                "orientation_error_deg": orientation_error_deg,
                "orientation_aligned": orientation_aligned,
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
            right_reached = bool(
                float(np.linalg.norm(np.asarray(target) - measured_eef)) <= 0.012
                and orientation_aligned
            )
            other_reached = bool(
                other_arm_world_target is None
                or (
                    measured_left_after is not None
                    and float(
                        np.linalg.norm(
                            np.asarray(other_arm_world_target, dtype=float)
                            - measured_left_after
                        )
                    )
                    <= 0.012
                )
            )
            if right_reached and other_reached:
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

    def execute_orientation_stage(
        stage: str,
        target_orientation: np.ndarray,
        *,
        max_position_drift_m: float = 0.04,
    ) -> bool:
        nonlocal collision_steps, maximum_support_steps
        if not np.isfinite(float(orientation_max_action)) or not (
            0.0 < float(orientation_max_action) <= 1.0
        ):
            raise ValueError("orientation_max_action must be in (0, 1]")
        if not np.isfinite(float(orientation_position_max_action)) or not (
            0.0 < float(orientation_position_max_action) <= 1.0
        ):
            raise ValueError(
                "orientation_position_max_action must be in (0, 1]"
            )
        if not np.isfinite(float(orientation_tolerance_deg)) or float(
            orientation_tolerance_deg
        ) < 0.0:
            raise ValueError("orientation_tolerance_deg must be non-negative")
        if int(orientation_stable_steps) < 1 or int(orientation_max_steps) < 1:
            raise ValueError("orientation step limits must be positive")
        target_rotation = _validated_rotation_matrix(
            target_orientation,
            name="open fork target orientation",
        )
        hold_position, start_orientation = right_eef_pose()
        controller = robot.part_controllers["right"]
        if controller.name != "OSC_POSE" or controller.input_type != "delta":
            raise RuntimeError("open fork orientation requires OSC_POSE delta control")
        input_ref_frame = getattr(controller, "input_ref_frame", "world")
        if input_ref_frame == "world":
            origin_rotation = np.eye(3)
        elif input_ref_frame == "base":
            origin_rotation = controller.origin_ori
            if origin_rotation is None:
                _, origin_rotation = robot.composite_controller.get_controller_base_pose(
                    controller_name="right"
                )
        else:
            raise RuntimeError(
                f"unsupported orientation reference frame for right: {input_ref_frame}"
            )
        origin_rotation = _validated_rotation_matrix(
            origin_rotation,
            name="right controller origin rotation",
        )
        stable_steps = 0
        max_drift = 0.0
        safety_failure = None
        success = False
        for local_step in range(int(orientation_max_steps)):
            robot.composite_controller.update_state()
            current_position, current_orientation = right_eef_pose()
            controller_delta = helpers["world_delta"](
                robot,
                "right",
                hold_position - current_position,
            )
            arm_action = helpers["arm_action"](
                robot,
                "right",
                controller_delta,
                float(orientation_position_max_action),
            )
            world_rotation_delta = target_rotation @ current_orientation.T
            orientation_action = normalized_osc_orientation_command(
                world_rotation_delta=world_rotation_delta,
                controller_origin_rotation=origin_rotation,
                output_min=controller.output_min,
                output_max=controller.output_max,
                max_action=float(orientation_max_action),
            )
            arm_action[3:6] = orientation_action
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
            measured_position, measured_orientation = right_eef_pose()
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            )
            contacts = object_robot_contacts(raw_env, object_name)
            right_support = any(
                is_right_support(geom)
                for geom in contacts["right"]
            )
            object_lift_m = float(object_position[2] - start_object[2])
            support_steps = (
                1 + int(observations[-1].get("stable_support_steps", 0))
                if observations
                and observations[-1].get("stage") == stage
                and right_support
                and object_lift_m >= 0.02
                else int(right_support and object_lift_m >= 0.02)
            )
            maximum_support_steps = max(maximum_support_steps, support_steps)
            error_deg = rotation_error_degrees(
                measured_orientation,
                target_rotation,
            )
            inward_axis = np.array([0.0, -1.0, 0.0])
            inward_projection = float(
                np.dot(measured_orientation[:, 2], inward_axis)
            )
            closure_vertical = abs(float(measured_orientation[2, 0]))
            geometry_aligned = open_fork_alignment_sufficient(
                measured_orientation,
                inward_axis=inward_axis,
                min_inward_projection=orientation_min_inward_projection,
                max_closure_vertical=orientation_max_closure_vertical,
            )
            drift_m = float(np.linalg.norm(measured_position - hold_position))
            max_drift = max(max_drift, drift_m)
            collision = bool((info or {}).get("has_judge_collision", False))
            collision_steps += int(collision)
            stable_steps = (
                stable_steps + 1
                if (
                    error_deg <= float(orientation_tolerance_deg)
                    or geometry_aligned
                )
                and drift_m <= float(max_position_drift_m)
                else 0
            )
            observations.append(
                {
                    "stage": stage,
                    "step": local_step + 1,
                    "target_eef_position": hold_position.tolist(),
                    "eef_position": measured_position.tolist(),
                    "start_eef_orientation": start_orientation.tolist(),
                    "target_eef_orientation": target_rotation.tolist(),
                    "eef_orientation": measured_orientation.tolist(),
                    "orientation_action": orientation_action.tolist(),
                    "orientation_error_deg": error_deg,
                    "inward_projection": inward_projection,
                    "closure_vertical": closure_vertical,
                    "geometry_aligned": geometry_aligned,
                    "orientation_stable_steps": stable_steps,
                    "position_drift_m": drift_m,
                    "object_position": object_position.tolist(),
                    "object_lift_m": object_lift_m,
                    "contacts": {
                        arm: list(names) for arm, names in contacts.items()
                    },
                    "right_support": right_support,
                    "stable_support_steps": support_steps,
                    "judge_collision": collision,
                }
            )
            if collision:
                safety_failure = "collision"
                break
            if drift_m > float(max_position_drift_m):
                safety_failure = "position_drift"
                break
            if stable_steps >= int(orientation_stable_steps):
                success = True
                break
        if not success and safety_failure is None:
            safety_failure = "timeout"
        final_position, final_orientation = right_eef_pose()
        stages.append(
            {
                "stage": stage,
                "success": success,
                "steps": sum(
                    1 for item in observations if item.get("stage") == stage
                ),
                "safety_failure": safety_failure,
                "orientation_error_deg": rotation_error_degrees(
                    final_orientation,
                    target_rotation,
                ),
                "orientation_stable_steps": stable_steps,
                "inward_projection": float(
                    np.dot(final_orientation[:, 2], [0.0, -1.0, 0.0])
                ),
                "closure_vertical": abs(float(final_orientation[2, 0])),
                "max_position_drift_m": max_drift,
                "final_eef_position": final_position.tolist(),
                "final_eef_orientation": final_orientation.tolist(),
                "target_eef_orientation": target_rotation.tolist(),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).tolist(),
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
    fork_orientation = (
        open_fork_target_orientation(
            inward_axis=np.array([0.0, -1.0, 0.0]),
            closure_axis=np.array([1.0, 0.0, 0.0]),
        )
        if horizontal_fork
        else None
    )
    fork_orientation_completed = False
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
        )
        for stage, target, allow_contact, require_support in sequence:
            other_arm_world_target = None
            if stage == "descend_open_outside" and torso_target_m is not None:
                clearance_lift = float(left_clearance_lift_m)
                if not np.isfinite(clearance_lift) or clearance_lift <= 0.0:
                    raise ValueError(
                        "left_clearance_lift_m must be finite and positive"
                    )
                left_clearance_target = left_eef_position().copy()
                left_clearance_target[2] += clearance_lift
                if not execute_stage(
                    "raise_left_clearance_for_torso",
                    right_eef_position().copy(),
                    allow_object_contact=False,
                    other_arm_world_target=left_clearance_target,
                ):
                    success = False
                    failure_stage = "raise_left_clearance_for_torso"
                    break
                other_arm_world_target = left_eef_position().copy()
                if horizontal_fork and orient_before_descent:
                    if not execute_orientation_stage(
                        "orient_open_fork_at_clearance",
                        fork_orientation,
                    ):
                        success = False
                        failure_stage = "orient_open_fork_at_clearance"
                        break
                    fork_orientation_completed = True
                hold_targets["torso"] = np.array(
                    [float(torso_target_m)], dtype=float
                )
            descent_max_steps = 480 if stage == "descend_open_outside" else 180
            stage_success = execute_stage(
                stage,
                target,
                allow_object_contact=allow_contact,
                require_support=require_support,
                max_steps=descent_max_steps,
                other_arm_world_target=other_arm_world_target,
                right_orientation_target=(
                    fork_orientation
                    if fork_orientation_completed and stage == "descend_open_outside"
                    else None
                ),
            )
            if (
                not stage_success
                and stage == "descend_open_outside"
                and not horizontal_fork
                and stages[-1].get("safety_failure") == "timeout"
                and not any(object_robot_contacts(raw_env, object_name).values())
                and open_fork_below_bottom_ready(
                    geometry_snapshot(raw_env, object_name)
                )
                and (
                    torso_target_m is None
                    or abs(torso_position() - float(torso_target_m)) <= 0.005
                )
            ):
                stage_success = True
                stages[-1]["success"] = True
                stages[-1]["safety_failure"] = None
                stages[-1]["success_source"] = (
                    "safe_unrotated_descent_plateau"
                )
            if not stage_success:
                success = False
                failure_stage = stage
                break
        if success and horizontal_fork:
            if not fork_orientation_completed and not execute_orientation_stage(
                "orient_open_fork_inward",
                fork_orientation,
            ):
                success = False
                failure_stage = "orient_open_fork_inward"
            else:
                fork_orientation_completed = True
                inset_distance = float(horizontal_inset_m)
                if not np.isfinite(inset_distance) or inset_distance <= 0.0:
                    raise ValueError("horizontal_inset_m must be finite and positive")
                fork_inset_target = np.asarray(targets["below"], dtype=float).copy()
                fork_inset_target[1] = float(targets["outside"][1]) - inset_distance
                inset_success = execute_stage(
                    "inset_horizontal_fork_under_overhang",
                    fork_inset_target,
                    allow_object_contact=True,
                )
                insertion_geometry = geometry_snapshot(raw_env, object_name)
                geometry_ready = open_fork_under_bottom_support_ready(
                    insertion_geometry,
                    minimum_planar_overlap_m=0.001,
                )
                stages[-1]["geometry_ready"] = geometry_ready
                if (
                    not inset_success
                    and geometry_ready
                    and stages[-1].get("safety_failure") == "timeout"
                ):
                    inset_success = True
                    stages[-1]["success"] = True
                    stages[-1]["safety_failure"] = None
                    stages[-1]["success_source"] = "measured_bottom_overlap"
                if not inset_success:
                    success = False
                    failure_stage = "inset_horizontal_fork_under_overhang"
                else:
                    if not geometry_ready and not execute_post_inset_base_advance():
                        success = False
                        failure_stage = "advance_base_for_fork_overlap"
                    elif float(torso_raise_m) > 0.0:
                        if not execute_torso_raise_into_support():
                            success = False
                            failure_stage = "raise_open_with_torso"
                    else:
                        fork_raise_target = right_eef_position().copy()
                        fork_raise_target[2] = float(targets["raise"][2])
                        if not execute_stage(
                            "raise_open_into_support",
                            fork_raise_target,
                            allow_object_contact=True,
                            require_support=True,
                        ):
                            success = False
                            failure_stage = "raise_open_into_support"
        elif success:
            inset_success = execute_stage(
                "inset_open_under_overhang",
                targets["undercut"],
                allow_object_contact=False,
            )
            inset_stage = stages[-1]
            inset_distance_to_target = float(
                np.linalg.norm(
                    np.asarray(targets["undercut"], dtype=float)
                    - right_eef_position()
                )
            )
            inset_progress_m = max(
                0.0,
                float(targets["outside"][1]) - float(right_eef_position()[1]),
            )
            inset_stage["inset_progress_m"] = inset_progress_m
            inset_stage["distance_to_target_m"] = inset_distance_to_target
            if (
                not inset_success
                and inset_stage.get("safety_failure") == "timeout"
                and not any(object_robot_contacts(raw_env, object_name).values())
                and (
                    inset_progress_m >= 0.04
                    or (
                        float(post_inset_base_advance_m) > 0.0
                        and post_inset_world_direction_x is not None
                        and post_inset_world_direction_y is not None
                        and open_fork_below_bottom_ready(
                            geometry_snapshot(raw_env, object_name)
                        )
                    )
                )
            ):
                inset_success = True
                inset_stage["success"] = True
                inset_stage["safety_failure"] = None
                inset_stage["success_source"] = (
                    "safe_unrotated_inset_plateau"
                    if inset_progress_m >= 0.04
                    else "safe_unrotated_base_assisted_inset"
                )
            insertion_geometry = geometry_snapshot(raw_env, object_name)
            geometry_ready = open_fork_under_bottom_support_ready(
                insertion_geometry,
                minimum_planar_overlap_m=0.001,
            )
            inset_stage["geometry_ready"] = geometry_ready
            if not inset_success:
                success = False
                failure_stage = "inset_open_under_overhang"
            elif not geometry_ready and not execute_post_inset_base_advance():
                success = False
                failure_stage = "advance_base_for_fork_overlap"
            elif float(torso_raise_m) > 0.0:
                if not execute_torso_raise_into_support():
                    success = False
                    failure_stage = "raise_open_with_torso"
            else:
                fork_raise_target = right_eef_position().copy()
                fork_raise_target[2] = float(targets["raise"][2])
                if not execute_stage(
                    "raise_open_into_support",
                    fork_raise_target,
                    allow_object_contact=True,
                    require_support=True,
                ):
                    success = False
                    failure_stage = "raise_open_into_support"
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
        "horizontal_fork": bool(horizontal_fork),
        "horizontal_inset_m": float(horizontal_inset_m),
        "post_inset_base_advance_m": float(post_inset_base_advance_m),
        "torso_raise_m": float(torso_raise_m),
        "torso_raise_orientation_max_action": float(
            torso_raise_orientation_max_action
        ),
        "torso_raise_base_correction_max_m": float(
            torso_raise_base_correction_max_m
        ),
        "orient_before_descent": bool(orient_before_descent),
        "post_inset_world_direction": [
            post_inset_world_direction_x,
            post_inset_world_direction_y,
        ],
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
        from robot_agent.skills.competition_grasp import ScriptedGraspConfig
        from robot_agent.workflows.competition_flow import OfficialCompetitionDriver

        grasp_config = None
        if args.container_grasp_lift_height_m is not None:
            grasp_config = ScriptedGraspConfig(
                container_lift_height_override=(
                    args.container_grasp_lift_height_m
                )
            )
        driver = OfficialCompetitionDriver(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            grasp_config=grasp_config,
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
            pre_grasp_object_position = np.asarray(
                backend.env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            pre_grasp_z = float(pre_grasp_object_position[2])
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
            record["pre_grasp_object_position"] = (
                pre_grasp_object_position.tolist()
            )
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
                    below_bottom_clearance_m=(
                        args.undercut_below_bottom_clearance_m
                    ),
                    raise_above_bottom_m=(
                        args.undercut_raise_above_bottom_m
                    ),
                    horizontal_fork=args.undercut_horizontal_fork,
                    orientation_max_action=(
                        args.undercut_orientation_max_action
                    ),
                    orientation_position_max_action=(
                        args.undercut_orientation_position_max_action
                    ),
                    orientation_tolerance_deg=(
                        args.undercut_orientation_tolerance_deg
                    ),
                    orientation_stable_steps=(
                        args.undercut_orientation_stable_steps
                    ),
                    orientation_max_steps=(
                        args.undercut_orientation_max_steps
                    ),
                    orientation_min_inward_projection=(
                        args.undercut_orientation_min_inward_projection
                    ),
                    orientation_max_closure_vertical=(
                        args.undercut_orientation_max_closure_vertical
                    ),
                    horizontal_inset_m=args.undercut_horizontal_inset_m,
                    left_clearance_lift_m=(
                        args.undercut_left_clearance_lift_m
                    ),
                    post_inset_base_advance_m=(
                        args.undercut_post_inset_base_advance_m
                    ),
                    torso_raise_m=args.undercut_torso_raise_m,
                    torso_raise_orientation_max_action=(
                        args.undercut_torso_raise_orientation_max_action
                    ),
                    torso_raise_base_correction_max_m=(
                        args.undercut_torso_raise_base_correction_max_m
                    ),
                    orient_before_descent=(
                        args.undercut_orient_before_descent
                    ),
                    post_inset_world_direction_x=(
                        args.undercut_post_inset_world_direction_x
                    ),
                    post_inset_world_direction_y=(
                        args.undercut_post_inset_world_direction_y
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
                if args.posture_locked_carry_distance_m > 0.0:
                    record["mode"] = "posture_locked_physical_carry"
                    probe = _posture_locked_carry_probe(
                        backend,
                        object_name,
                        distance_m=args.posture_locked_carry_distance_m,
                        world_direction_x=(
                            args.posture_locked_carry_world_direction_x
                        ),
                        world_direction_y=(
                            args.posture_locked_carry_world_direction_y
                        ),
                        table_object_z=pre_grasp_z,
                        max_linear_m_s=(
                            args.posture_locked_carry_max_linear_m_s
                        ),
                        actuated_gripper_hold=(
                            args.posture_locked_carry_actuated_gripper_hold
                        ),
                        posture_lock_robot_joints=(
                            args.posture_locked_carry_posture_lock_robot_joints
                        ),
                    )
                    for key in _POSTURE_CARRY_REQUIRED_FIELDS:
                        record[key] = probe[key]
                elif args.end_grasp_inchworm_distance_m > 0.0:
                    inchworm_kwargs = {
                        "distance_m": args.end_grasp_inchworm_distance_m,
                        "world_direction_x": (
                            args.end_grasp_inchworm_world_direction_x
                        ),
                        "world_direction_y": (
                            args.end_grasp_inchworm_world_direction_y
                        ),
                        "table_object_z": pre_grasp_z,
                        "stroke_m": args.end_grasp_inchworm_stroke_m,
                        "stroke_lift_m": args.end_grasp_inchworm_stroke_lift_m,
                        "height_gain": args.end_grasp_inchworm_height_gain,
                        "reset_m": args.end_grasp_inchworm_reset_m,
                        "minimum_lift_m": args.end_grasp_minimum_lift_m,
                    }
                    if args.end_grasp_setdown_after_inchworm:
                        record["mode"] = "end_grasp_setdown_probe"
                        if args.floor_corridor_push:
                            if (
                                args.floor_push_world_direction_x is None
                                or args.floor_push_world_direction_y is None
                            ):
                                raise ValueError(
                                    "floor corridor push requires both world direction components"
                                )
                            record["mode"] = "end_grasp_floor_push_probe"
                            probe = _end_grasp_floor_push_probe(
                                backend,
                                driver,
                                str(task["source"]),
                                object_name,
                                macro_count=args.end_grasp_regrasp_macros,
                                place_max_descent_m=(
                                    args.end_grasp_place_max_descent_m
                                ),
                                floor_retract_forward_m=(
                                    args.floor_regrasp_retract_forward_m
                                ),
                                floor_retract_lateral_m=(
                                    args.floor_regrasp_retract_lateral_m
                                ),
                                floor_retract_target_z=(
                                    args.floor_regrasp_retract_target_z
                                ),
                                floor_transition_margin_m=(
                                    args.floor_transition_margin_m
                                ),
                                push_direction_x=(
                                    args.floor_push_world_direction_x
                                ),
                                push_direction_y=(
                                    args.floor_push_world_direction_y
                                ),
                                push_distance_m=args.floor_push_distance_m,
                                push_base_standoff_m=(
                                    args.floor_push_base_standoff_m
                                ),
                                push_orientation_clearance_m=(
                                    args.floor_push_orientation_clearance_m
                                ),
                                push_oriented_retract_forward_m=(
                                    args.floor_push_oriented_retract_forward_m
                                ),
                                push_oriented_retract_lateral_m=(
                                    args.floor_push_oriented_retract_lateral_m
                                ),
                                push_lateral_offset_m=(
                                    args.floor_push_lateral_offset_m
                                ),
                                push_torso_drop_m=args.floor_push_torso_drop_m,
                                push_base_pusher=args.floor_push_base_pusher,
                                push_maximum_lateral_offset_m=(
                                    args.floor_push_maximum_lateral_offset_m
                                ),
                                push_face_offset_m=args.floor_push_face_offset_m,
                                push_hand_separation_m=(
                                    args.floor_push_hand_separation_m
                                ),
                                push_hand_height_m=args.floor_push_hand_height_m,
                                push_precontact_clearance_m=(
                                    args.floor_push_precontact_clearance_m
                                ),
                                push_base_speed_m_s=(
                                    args.floor_push_base_speed_m_s
                                ),
                                push_max_steps=args.floor_push_max_steps,
                                source_center_xy=scene_context.input_port(
                                    str(task["source"])
                                ).center,
                                target_center_xy=scene_context.output_port(
                                    str(task["target"])
                                ).center,
                                floor_base_route_to_target=(
                                    args.floor_base_route_to_target
                                ),
                                floor_base_route_corridor_y=(
                                    args.floor_base_route_corridor_y
                                ),
                                floor_base_route_arrival_margin_m=(
                                    args.floor_base_route_arrival_margin_m
                                ),
                                floor_base_route_reposition_clearance_m=(
                                    args.floor_base_route_reposition_clearance_m
                                ),
                                **inchworm_kwargs,
                            )
                            record["extraction_success"] = bool(
                                probe.get("extraction_success", False)
                            )
                            record["floor_transition_detected"] = bool(
                                probe.get("floor_transition_detected", False)
                            )
                            record["navigation_retract_success"] = bool(
                                probe.get("navigation_retract_success", False)
                            )
                            record["floor_push_success"] = bool(
                                probe.get("floor_push_success", False)
                            )
                            record["physical_contact_steps"] = int(
                                probe.get("physical_contact_steps", 0)
                            )
                            record["maximum_axis_displacement_m"] = float(
                                probe.get("maximum_axis_displacement_m", 0.0)
                            )
                            record[
                                "official_source_maximum_axis_displacement_m"
                            ] = float(
                                probe.get(
                                    "official_source_maximum_axis_displacement_m",
                                    0.0,
                                )
                            )
                            record["official_target_distance_m"] = float(
                                probe.get("official_target_distance_m", float("inf"))
                            )
                        elif args.end_grasp_regrasp_macros > 1:
                            probe = _end_grasp_regrasp_probe(
                                backend,
                                driver,
                                str(task["source"]),
                                object_name,
                                macro_count=args.end_grasp_regrasp_macros,
                                place_max_descent_m=(
                                    args.end_grasp_place_max_descent_m
                                ),
                                floor_retract_forward_m=(
                                    args.floor_regrasp_retract_forward_m
                                ),
                                floor_retract_lateral_m=(
                                    args.floor_regrasp_retract_lateral_m
                                ),
                                floor_retract_target_z=(
                                    args.floor_regrasp_retract_target_z
                                ),
                                floor_transition_margin_m=(
                                    args.floor_transition_margin_m
                                ),
                                floor_regrasp_safe_clearance_m=(
                                    args.floor_regrasp_safe_clearance_m
                                ),
                                **inchworm_kwargs,
                            )
                        else:
                            probe = _end_grasp_setdown_probe(
                                backend,
                                object_name,
                                place_max_descent_m=(
                                    args.end_grasp_place_max_descent_m
                                ),
                                **inchworm_kwargs,
                            )
                        record["transport_success"] = bool(
                            probe.get("transport_success")
                        )
                        record["place_success"] = bool(probe.get("place_success"))
                        record["support_detected"] = bool(
                            probe.get("support_detected")
                        )
                        record["released"] = bool(probe.get("released"))
                        record["requested_macro_count"] = int(
                            probe.get("requested_macro_count", 1)
                        )
                        record["completed_macro_count"] = int(
                            probe.get("completed_macro_count", 0)
                        )
                    else:
                        record["mode"] = "end_grasp_inchworm_transport"
                        probe = _end_grasp_inchworm_probe(
                            backend,
                            object_name,
                            **inchworm_kwargs,
                        )
                        record["transport_success"] = bool(probe.get("success"))
                    record["object_translation_m"] = float(
                        probe.get("measured_object_translation_m", 0.0)
                    )
                    record["transport_world_direction"] = list(
                        probe.get("world_direction", [])
                    )
                    record["attachment_calls"] = int(
                        probe.get("attachment_activations", 0)
                    )
                    record["object_pose_writes"] = int(
                        probe.get("object_pose_writes", 0)
                    )
                elif args.physical_push:
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
            final_object_position = np.asarray(
                backend.env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            final_z = float(final_object_position[2])
            record["final_geometry"] = geometry_snapshot(
                backend.env,
                object_name,
            )
            final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            record["final_object_z"] = final_z
            record["final_object_position"] = final_object_position.tolist()
            if record.get("mode") == "end_grasp_floor_push_probe":
                record["maximum_axis_displacement_m"] = float(
                    np.max(
                        np.abs(
                            final_object_position[:2]
                            - pre_grasp_object_position[:2]
                        )
                    )
                )
            direction = np.asarray(
                record.get("transport_world_direction", []), dtype=float
            )
            if direction.shape == (2,) and np.all(np.isfinite(direction)):
                net_progress, net_lateral = directed_planar_progress(
                    start_xy=pre_grasp_object_position[:2],
                    end_xy=final_object_position[:2],
                    direction_xy=direction,
                )
                record["net_projected_object_progress_m"] = net_progress
                record["net_lateral_object_drift_m"] = net_lateral
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
    elif record.get("mode") == "posture_locked_physical_carry":
        record["gate_failures"] = posture_carry_failures(record)
    elif record.get("mode") == "end_grasp_setdown_probe":
        record["gate_failures"] = setdown_gate_failures(record)
    elif record.get("mode") == "end_grasp_floor_push_probe":
        record["gate_failures"] = hybrid_exit_gate_failures(record)
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
    parser.add_argument(
        "--posture-locked-carry-distance-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--posture-locked-carry-max-linear-m-s",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--posture-locked-carry-actuated-gripper-hold",
        action="store_true",
    )
    parser.add_argument(
        "--posture-locked-carry-posture-lock-robot-joints",
        action="store_true",
    )
    parser.add_argument("--posture-locked-carry-world-direction-x", type=float)
    parser.add_argument("--posture-locked-carry-world-direction-y", type=float)
    parser.add_argument(
        "--end-grasp-inchworm-distance-m", type=float, default=0.0
    )
    parser.add_argument("--container-grasp-lift-height-m", type=float)
    parser.add_argument(
        "--end-grasp-minimum-lift-m", type=float, default=0.10
    )
    parser.add_argument(
        "--end-grasp-inchworm-stroke-m", type=float, default=0.08
    )
    parser.add_argument(
        "--end-grasp-inchworm-stroke-lift-m", type=float, default=0.015
    )
    parser.add_argument(
        "--end-grasp-inchworm-height-gain", type=float, default=0.75
    )
    parser.add_argument(
        "--end-grasp-inchworm-reset-m", type=float, default=0.06
    )
    parser.add_argument("--end-grasp-inchworm-world-direction-x", type=float)
    parser.add_argument("--end-grasp-inchworm-world-direction-y", type=float)
    parser.add_argument("--end-grasp-setdown-after-inchworm", action="store_true")
    parser.add_argument(
        "--end-grasp-place-max-descent-m", type=float, default=0.25
    )
    parser.add_argument("--end-grasp-regrasp-macros", type=int, default=1)
    parser.add_argument(
        "--floor-regrasp-retract-forward-m", type=float, default=0.20
    )
    parser.add_argument(
        "--floor-regrasp-retract-lateral-m", type=float, default=0.15
    )
    parser.add_argument(
        "--floor-regrasp-retract-target-z", type=float, default=1.45
    )
    parser.add_argument(
        "--floor-transition-margin-m", type=float, default=0.30
    )
    parser.add_argument(
        "--floor-regrasp-safe-clearance-m", type=float, default=1.20
    )
    parser.add_argument("--floor-corridor-push", action="store_true")
    parser.add_argument("--floor-push-world-direction-x", type=float)
    parser.add_argument("--floor-push-world-direction-y", type=float)
    parser.add_argument("--floor-push-distance-m", type=float, default=1.05)
    parser.add_argument(
        "--floor-push-base-standoff-m", type=float, default=0.85
    )
    parser.add_argument(
        "--floor-push-orientation-clearance-m", type=float, default=0.35
    )
    parser.add_argument(
        "--floor-push-oriented-retract-forward-m", type=float, default=0.20
    )
    parser.add_argument(
        "--floor-push-oriented-retract-lateral-m", type=float, default=0.08
    )
    parser.add_argument("--floor-push-lateral-offset-m", type=float)
    parser.add_argument("--floor-push-torso-drop-m", type=float, default=0.24)
    parser.add_argument("--floor-push-base-pusher", action="store_true")
    parser.add_argument(
        "--floor-push-maximum-lateral-offset-m", type=float, default=0.25
    )
    parser.add_argument("--floor-push-face-offset-m", type=float, default=0.24)
    parser.add_argument(
        "--floor-push-hand-separation-m", type=float, default=0.28
    )
    parser.add_argument("--floor-push-hand-height-m", type=float, default=0.38)
    parser.add_argument(
        "--floor-push-precontact-clearance-m", type=float, default=0.08
    )
    parser.add_argument(
        "--floor-push-base-speed-m-s", type=float, default=0.025
    )
    parser.add_argument("--floor-push-max-steps", type=int, default=1200)
    parser.add_argument("--floor-base-route-to-target", action="store_true")
    parser.add_argument(
        "--floor-base-route-corridor-y", type=float, default=-8.40
    )
    parser.add_argument(
        "--floor-base-route-arrival-margin-m", type=float, default=0.05
    )
    parser.add_argument(
        "--floor-base-route-reposition-clearance-m", type=float, default=0.90
    )
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
        "--undercut-below-bottom-clearance-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--undercut-raise-above-bottom-m",
        type=float,
        default=0.12,
    )
    parser.add_argument("--undercut-horizontal-fork", action="store_true")
    parser.add_argument(
        "--undercut-orientation-max-action",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--undercut-orientation-position-max-action",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--undercut-orientation-tolerance-deg",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--undercut-orientation-stable-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--undercut-orientation-max-steps",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--undercut-orientation-min-inward-projection",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--undercut-orientation-max-closure-vertical",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--undercut-horizontal-inset-m",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--undercut-left-clearance-lift-m",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--undercut-post-inset-base-advance-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--undercut-torso-raise-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--undercut-torso-raise-orientation-max-action",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--undercut-torso-raise-base-correction-max-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--undercut-orient-before-descent",
        action="store_true",
    )
    parser.add_argument(
        "--undercut-post-inset-world-direction-x",
        type=float,
    )
    parser.add_argument(
        "--undercut-post-inset-world-direction-y",
        type=float,
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
