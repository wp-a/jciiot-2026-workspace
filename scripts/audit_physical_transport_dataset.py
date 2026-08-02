#!/usr/bin/env python3
"""Audit attachment-free physical transport evidence for dataset admission."""

from __future__ import annotations

import math
from typing import Any, Mapping


INTEGRITY_COUNTERS = (
    "attachment_calls",
    "attachment_activations",
    "object_pose_writes",
    "robot_state_writes",
    "legacy_teleport_activations",
    "collision_frames",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _planar_translation(record: Mapping[str, Any]) -> float | None:
    probe = record.get("full_physical_probe")
    if not isinstance(probe, Mapping):
        return None
    start = probe.get("start_object_position")
    final = probe.get("final_object_position")
    if not isinstance(start, (list, tuple)) or len(start) < 2:
        return None
    if not isinstance(final, (list, tuple)) or len(final) < 2:
        return None
    start_xy = [_finite_number(value) for value in start[:2]]
    final_xy = [_finite_number(value) for value in final[:2]]
    if any(value is None for value in (*start_xy, *final_xy)):
        return None
    return math.hypot(
        float(final_xy[0]) - float(start_xy[0]),
        float(final_xy[1]) - float(start_xy[1]),
    )


def audit_record(
    record: Mapping[str, Any],
    *,
    minimum_object_translation_m: float = 0.50,
    minimum_object_lift_m: float = 0.13,
    maximum_object_gripper_drift_m: float = 0.05,
) -> dict[str, Any]:
    """Classify one record using measured object motion and integrity evidence."""
    failures: list[str] = []
    rejection_failures: list[str] = []

    required_fields = (
        "physical_grasp",
        "continuous_bilateral_contact",
        "dropped",
        "minimum_object_lift_m",
        "max_object_gripper_drift_m",
        *INTEGRITY_COUNTERS,
        "infrastructure_error",
    )
    for field in required_fields:
        if field not in record:
            failure = f"missing:{field}"
            failures.append(failure)
            rejection_failures.append(failure)

    object_translation_m = _planar_translation(record)
    if object_translation_m is None:
        failures.append("object_positions")
        rejection_failures.append("object_positions")

    for field in INTEGRITY_COUNTERS:
        if field not in record:
            continue
        count = _finite_number(record[field])
        if count is None or count != 0.0:
            failures.append(field)
            rejection_failures.append(field)

    if "infrastructure_error" in record and record.get("infrastructure_error"):
        failures.append("infrastructure_error")
        rejection_failures.append("infrastructure_error")

    if record.get("physical_grasp") is not True:
        failures.append("physical_grasp")
    if record.get("continuous_bilateral_contact") is not True:
        failures.append("continuous_bilateral_contact")
    if record.get("dropped") is not False:
        failures.append("dropped")

    lift_m = _finite_number(record.get("minimum_object_lift_m"))
    if lift_m is None or lift_m < float(minimum_object_lift_m):
        failures.append("minimum_object_lift_m")

    drift_m = _finite_number(record.get("max_object_gripper_drift_m"))
    if (
        drift_m is None
        or drift_m > float(maximum_object_gripper_drift_m)
    ):
        failures.append("max_object_gripper_drift_m")

    if (
        object_translation_m is None
        or object_translation_m < float(minimum_object_translation_m)
    ):
        failures.append("object_translation_m")

    failures = list(dict.fromkeys(failures))
    rejection_failures = list(dict.fromkeys(rejection_failures))
    eligible = not failures
    if eligible:
        classification = "transport_success"
    elif rejection_failures:
        classification = "rejected"
    else:
        classification = "recovery"
    return {
        "classification": classification,
        "eligible": eligible,
        "failures": failures,
        "metrics": {
            "object_translation_m": object_translation_m,
            "minimum_object_lift_m": lift_m,
            "max_object_gripper_drift_m": drift_m,
        },
        "thresholds": {
            "minimum_object_translation_m": float(
                minimum_object_translation_m
            ),
            "minimum_object_lift_m": float(minimum_object_lift_m),
            "maximum_object_gripper_drift_m": float(
                maximum_object_gripper_drift_m
            ),
        },
    }
