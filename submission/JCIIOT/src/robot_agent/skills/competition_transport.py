"""Physical carrying and placement controls for the competition workflow."""

from __future__ import annotations

import math

import numpy as np


class PhysicalCarryConfig:
    """Bounded control parameters for physical carrying and placement."""

    def __init__(
        self,
        *,
        waypoint_tolerance: float = 0.10,
        max_steps: int = 6000,
        k_linear: float = 0.8,
        k_angular: float = 1.0,
        max_linear: float = 0.12,
        max_angular: float = 0.08,
        max_linear_delta: float = 0.01,
        max_angular_delta: float = 0.01,
        yaw_tolerance: float = 0.04,
        object_drop_tolerance: float = 0.025,
        descent_step: float = 0.001,
        max_descent: float = 0.12,
        support_stability_steps: int = 8,
        support_motion_tolerance: float = 0.0002,
        release_steps: int = 40,
        settle_steps: int = 80,
        max_arm_action: float = 0.30,
    ) -> None:
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.max_steps = int(max_steps)
        self.k_linear = float(k_linear)
        self.k_angular = float(k_angular)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.max_linear_delta = float(max_linear_delta)
        self.max_angular_delta = float(max_angular_delta)
        self.yaw_tolerance = float(yaw_tolerance)
        self.object_drop_tolerance = float(object_drop_tolerance)
        self.descent_step = float(descent_step)
        self.max_descent = float(max_descent)
        self.support_stability_steps = int(support_stability_steps)
        self.support_motion_tolerance = float(support_motion_tolerance)
        self.release_steps = int(release_steps)
        self.settle_steps = int(settle_steps)
        self.max_arm_action = float(max_arm_action)


def world_velocity_to_base_frame(world_xy, yaw: float) -> np.ndarray:
    """Rotate a planar world-frame command into the current base frame."""
    world_xy = np.asarray(world_xy, dtype=float).reshape(2)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return np.array(
        [
            cosine * world_xy[0] + sine * world_xy[1],
            -sine * world_xy[0] + cosine * world_xy[1],
        ],
        dtype=float,
    )


def slew_limited_command(previous, requested, max_delta) -> np.ndarray:
    """Bound per-step command changes without changing the requested sign."""
    previous = np.asarray(previous, dtype=float)
    requested = np.asarray(requested, dtype=float)
    max_delta = np.asarray(max_delta, dtype=float)
    if previous.shape != requested.shape or previous.shape != max_delta.shape:
        raise ValueError("previous, requested, and max_delta must have one shape")
    if np.any(max_delta < 0.0):
        raise ValueError("max_delta must be non-negative")
    return previous + np.clip(requested - previous, -max_delta, max_delta)


def transport_base_goal(
    *,
    object_target_xy,
    base_xy,
    base_yaw: float,
    object_xy,
) -> np.ndarray:
    """Find the base goal that carries the measured object offset to target."""
    object_target_xy = np.asarray(object_target_xy, dtype=float).reshape(2)
    base_xy = np.asarray(base_xy, dtype=float).reshape(2)
    object_xy = np.asarray(object_xy, dtype=float).reshape(2)
    relative_base = world_velocity_to_base_frame(object_xy - base_xy, base_yaw)
    cosine = math.cos(float(base_yaw))
    sine = math.sin(float(base_yaw))
    relative_world = np.array(
        [
            cosine * relative_base[0] - sine * relative_base[1],
            sine * relative_base[0] + cosine * relative_base[1],
        ],
        dtype=float,
    )
    return object_target_xy - relative_world


def next_contact_stability(contacts, stable_steps: int) -> int:
    """Count consecutive bilateral-contact steps, resetting on either loss."""
    if bool(contacts.get("right", False)) and bool(contacts.get("left", False)):
        return int(stable_steps) + 1
    return 0
