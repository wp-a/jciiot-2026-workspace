"""Physical carrying and placement controls for the competition workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CradleObservation:
    """Read-only evidence used to validate bilateral robot-link support."""

    base_xy: tuple[float, float]
    object_z: float
    minimum_object_z: float
    gripper_contacts: Mapping[str, bool]
    support_contacts: Mapping[str, tuple[str, ...]]
    object_to_wrist_drift_m: Mapping[str, float]
    max_drift_m: float
    judge_collision: bool = False


def _is_allowed_cradle_geom(geom_name: str, arm: str) -> bool:
    name = str(geom_name).lower()
    if "finger" in name:
        return False
    if any(token in name for token in ("wrist", "palm", "hand_collision")):
        return arm in name
    if arm == "left":
        return any(f"arm_{index}_left_collision" in name for index in (4, 5, 6))
    if arm == "right":
        return "_left_" not in name and any(
            f"arm_{index}_collision" in name for index in (4, 5, 6)
        )
    return False


def is_cradle_supported(observation: CradleObservation) -> bool:
    """Require safe, bilateral contact with load-bearing robot links."""
    if observation.judge_collision:
        return False
    if float(observation.object_z) < float(observation.minimum_object_z):
        return False
    for arm in ("right", "left"):
        if float(observation.object_to_wrist_drift_m.get(arm, math.inf)) > float(
            observation.max_drift_m
        ):
            return False
        contacts = observation.support_contacts.get(arm, ())
        if not any(_is_allowed_cradle_geom(name, arm) for name in contacts):
            return False
    return True


def next_cradle_stability(
    observation: CradleObservation,
    stable_steps: int,
) -> int:
    """Count only consecutive bilateral, height-safe cradle observations."""
    return int(stable_steps) + 1 if is_cradle_supported(observation) else 0


def _bounded_vector(vector, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm == 0.0:
        return vector
    return vector * (float(max_norm) / norm)


def bounded_symmetric_cradle_deltas(
    *,
    center_delta,
    separation_axis,
    inward_delta: float,
    max_delta: float,
) -> dict[str, np.ndarray]:
    """Build mirrored arm deltas and bound each arm by Euclidean norm."""
    if float(max_delta) < 0.0:
        raise ValueError("max_delta must be non-negative")
    center = np.asarray(center_delta, dtype=float).reshape(3)
    axis = np.asarray(separation_axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("separation_axis must be non-zero")
    axis = axis / axis_norm
    separation = axis * float(inward_delta)
    return {
        "right": _bounded_vector(center - separation, max_delta),
        "left": _bounded_vector(center + separation, max_delta),
    }


def single_arm_under_support_targets(
    current_positions: Mapping[str, np.ndarray],
    *,
    moving_arm: str,
    separation_axis,
    descent_m: float,
    inset_m: float,
) -> dict[str, np.ndarray]:
    """Move one arm down and toward the measured midpoint of the two arms."""
    if moving_arm not in ("right", "left"):
        raise ValueError("moving_arm must be 'right' or 'left'")
    positions = {
        arm: np.asarray(current_positions[arm], dtype=float).reshape(3).copy()
        for arm in ("right", "left")
    }
    if any(not np.all(np.isfinite(position)) for position in positions.values()):
        raise ValueError("current_positions must be finite")
    axis = np.asarray(separation_axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.all(np.isfinite(axis)) or axis_norm == 0.0:
        raise ValueError("separation_axis must be finite and non-zero")
    axis = axis / axis_norm
    descent = float(descent_m)
    inset = float(inset_m)
    if not np.isfinite(descent) or descent < 0.0:
        raise ValueError("descent_m must be finite and non-negative")
    if not np.isfinite(inset) or inset < 0.0:
        raise ValueError("inset_m must be finite and non-negative")

    midpoint_projection = 0.5 * sum(
        float(np.dot(position, axis)) for position in positions.values()
    )
    moving_projection = float(np.dot(positions[moving_arm], axis))
    side = float(np.sign(moving_projection - midpoint_projection))
    if side == 0.0 and inset > 0.0:
        raise ValueError("moving arm must differ from the arm midpoint")
    positions[moving_arm] -= axis * side * inset
    positions[moving_arm][2] -= descent
    return positions


def _cradle_result(
    *,
    success: bool,
    failure_stage: str | None,
    steps: int,
    stable_steps: int,
    base_translation_m: float,
) -> dict:
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "steps": int(steps),
        "support_contact_steps": int(stable_steps),
        "base_translation_m": float(base_translation_m),
        "collision": failure_stage == "collision",
        "dropped": failure_stage == "height_loss",
    }


def run_physical_cradle_transfer(
    backend,
    *,
    object_name: str,
    travel_direction,
    travel_distance: float,
    required_stable_steps: int = 20,
    max_steps: int = 500,
    step_m: float = 0.002,
    max_arm_delta: float = 0.004,
    driver=None,
) -> dict:
    """Move only while measured bilateral robot-link support stays valid."""
    if driver is None:
        raise ValueError("a physical cradle driver is required")
    if float(travel_distance) < 0.0:
        raise ValueError("travel_distance must be non-negative")
    if int(required_stable_steps) < 1:
        raise ValueError("required_stable_steps must be positive")
    direction = np.asarray(travel_direction, dtype=float).reshape(2)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0 and float(travel_distance) > 0.0:
        raise ValueError("travel_direction must be non-zero")
    if direction_norm > 0.0:
        direction = direction / direction_norm

    observation = driver.observe_cradle(backend, object_name)
    start_xy = np.asarray(observation.base_xy, dtype=float).reshape(2)
    stable_steps = 0
    maximum_stable_steps = 0
    had_bilateral_support = False
    steps = 0
    failure_stage = "timeout"
    success = False
    base_translation = 0.0
    driver.record_event(
        backend,
        "physical_cradle_start",
        object_name=object_name,
        travel_distance=float(travel_distance),
    )

    while steps <= int(max_steps):
        observation = driver.observe_cradle(backend, object_name)
        base_translation = float(
            np.linalg.norm(
                np.asarray(observation.base_xy, dtype=float).reshape(2) - start_xy
            )
        )
        if observation.judge_collision:
            failure_stage = "collision"
            break
        if float(observation.object_z) < float(observation.minimum_object_z):
            failure_stage = "height_loss"
            break
        if any(
            float(observation.object_to_wrist_drift_m.get(arm, math.inf))
            > float(observation.max_drift_m)
            for arm in ("right", "left")
        ):
            failure_stage = "drift"
            break

        supported = is_cradle_supported(observation)
        stable_steps = next_cradle_stability(observation, stable_steps)
        maximum_stable_steps = max(maximum_stable_steps, stable_steps)
        if supported:
            had_bilateral_support = True
        elif had_bilateral_support:
            failure_stage = "support_loss"
            break
        if (
            base_translation >= float(travel_distance)
            and stable_steps >= int(required_stable_steps)
        ):
            success = True
            failure_stage = None
            break
        if steps >= int(max_steps):
            break

        remaining = max(0.0, float(travel_distance) - base_translation)
        base_delta = direction * min(float(step_m), remaining)
        center_delta = np.array([base_delta[0], base_delta[1], 0.0])
        separation_axis = np.array([-direction[1], direction[0], 0.0])
        if direction_norm == 0.0:
            separation_axis = np.array([0.0, 1.0, 0.0])
        arm_deltas = bounded_symmetric_cradle_deltas(
            center_delta=center_delta,
            separation_axis=separation_axis,
            inward_delta=0.0,
            max_delta=float(max_arm_delta),
        )
        step_info = driver.step_cradle(
            backend,
            object_name=object_name,
            base_world_delta=base_delta,
            arm_world_deltas=arm_deltas,
        )
        steps += 1
        if bool((step_info or {}).get("collision", False)):
            failure_stage = "collision"
            break

    driver.record_event(
        backend,
        "physical_cradle_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        support_contact_steps=maximum_stable_steps,
        base_translation_m=base_translation,
    )
    return _cradle_result(
        success=success,
        failure_stage=failure_stage,
        steps=steps,
        stable_steps=maximum_stable_steps,
        base_translation_m=base_translation,
    )


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
        max_linear_delta: float = 0.04,
        max_angular_delta: float = 0.01,
        base_control_dt: float = 0.05,
        yaw_tolerance: float = 0.04,
        align_heading_to_path: bool = False,
        pivot_compensation_enabled: bool = True,
        heading_translation_tolerance: float = 0.05,
        object_drop_tolerance: float = 0.025,
        vertical_hold_feedforward: float = 0.0,
        vertical_hold_gain: float = 0.0,
        max_vertical_hold_delta: float = 0.0,
        max_planar_grasp_drift: float = 0.04,
        planar_recovery_trigger: float = 0.0,
        planar_recovery_steps: int = 0,
        planar_recovery_inward_delta: float = 0.0,
        height_recovery_enabled: bool = True,
        height_recovery_trigger: float = 0.01,
        height_settle_allowance: float = 0.0,
        height_safety_margin: float | None = None,
        height_recovery_steps: int = 80,
        height_recovery_max_action: float = 0.65,
        height_recenter_steps: int = 5,
        height_recenter_tolerance: float = 0.002,
        height_recenter_max_delta: float = 0.015,
        descent_step: float = 0.001,
        max_descent: float = 0.12,
        minimum_descent_before_support: float = 0.008,
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
        self.base_control_dt = float(base_control_dt)
        self.yaw_tolerance = float(yaw_tolerance)
        self.align_heading_to_path = bool(align_heading_to_path)
        self.pivot_compensation_enabled = bool(pivot_compensation_enabled)
        self.heading_translation_tolerance = float(
            heading_translation_tolerance
        )
        self.object_drop_tolerance = float(object_drop_tolerance)
        self.vertical_hold_feedforward = float(vertical_hold_feedforward)
        self.vertical_hold_gain = float(vertical_hold_gain)
        self.max_vertical_hold_delta = float(max_vertical_hold_delta)
        self.max_planar_grasp_drift = float(max_planar_grasp_drift)
        self.planar_recovery_trigger = float(planar_recovery_trigger)
        self.planar_recovery_steps = int(planar_recovery_steps)
        self.planar_recovery_inward_delta = float(
            planar_recovery_inward_delta
        )
        self.height_recovery_enabled = bool(height_recovery_enabled)
        self.height_recovery_trigger = float(height_recovery_trigger)
        self.height_settle_allowance = float(height_settle_allowance)
        self.height_safety_margin = float(
            object_drop_tolerance
            if height_safety_margin is None
            else height_safety_margin
        )
        self.height_recovery_steps = int(height_recovery_steps)
        self.height_recovery_max_action = float(height_recovery_max_action)
        self.height_recenter_steps = int(height_recenter_steps)
        self.height_recenter_tolerance = float(height_recenter_tolerance)
        self.height_recenter_max_delta = float(height_recenter_max_delta)
        self.descent_step = float(descent_step)
        self.max_descent = float(max_descent)
        self.minimum_descent_before_support = float(
            minimum_descent_before_support
        )
        self.support_stability_steps = int(support_stability_steps)
        self.support_motion_tolerance = float(support_motion_tolerance)
        self.release_steps = int(release_steps)
        self.settle_steps = int(settle_steps)
        self.max_arm_action = float(max_arm_action)


def physical_carry_step_budget(
    path,
    *,
    start_xy,
    max_linear: float,
    control_dt: float,
    safety_factor: float = 5.0,
) -> int:
    """Size the control budget from measured route length and bounded speed."""
    speed = float(max_linear)
    dt = float(control_dt)
    factor = float(safety_factor)
    if speed <= 0.0 or dt <= 0.0 or factor < 1.0:
        raise ValueError("carry speed, control dt, and safety factor must be positive")
    previous = np.asarray(start_xy, dtype=float).reshape(2)
    route_length = 0.0
    waypoint_count = 0
    for point in path:
        waypoint = np.asarray(point, dtype=float).reshape(2)
        if not np.all(np.isfinite(waypoint)):
            raise ValueError("carry path must contain only finite waypoints")
        route_length += float(np.linalg.norm(waypoint - previous))
        previous = waypoint
        waypoint_count += 1
    nominal_steps = int(math.ceil(route_length / (speed * dt)))
    return max(80, int(math.ceil(nominal_steps * factor)) + waypoint_count * 10)


class InchwormCarryConfig:
    """Quasi-static arm-stroke and base-reset transport parameters."""

    def __init__(
        self,
        *,
        stroke_distance: float = 0.08,
        stroke_vertical_feedforward: float = 0.015,
        stroke_height_gain: float = 0.75,
        max_vertical_adjustment: float = 0.05,
        arm_target_tolerance: float = 0.01,
        arm_max_steps: int = 120,
        reset_distance: float = 0.06,
        reset_max_linear: float = 0.04,
        reset_control_dt: float = 0.05,
        reset_position_tolerance: float = 1e-4,
        reset_max_gripper_drift: float = 0.06,
        reset_arm_compensation_gain: float = 4.0,
        reseat_steps: int = 4,
        reseat_inward_delta: float = 0.002,
        max_lateral_drift: float = 0.03,
        minimum_macro_progress: float = 0.015,
        max_cycles: int = 64,
    ) -> None:
        self.stroke_distance = float(stroke_distance)
        self.stroke_vertical_feedforward = float(stroke_vertical_feedforward)
        self.stroke_height_gain = float(stroke_height_gain)
        self.max_vertical_adjustment = float(max_vertical_adjustment)
        self.arm_target_tolerance = float(arm_target_tolerance)
        self.arm_max_steps = int(arm_max_steps)
        self.reset_distance = float(reset_distance)
        self.reset_max_linear = float(reset_max_linear)
        self.reset_control_dt = float(reset_control_dt)
        self.reset_position_tolerance = float(reset_position_tolerance)
        self.reset_max_gripper_drift = float(reset_max_gripper_drift)
        self.reset_arm_compensation_gain = float(reset_arm_compensation_gain)
        self.reseat_steps = int(reseat_steps)
        self.reseat_inward_delta = float(reseat_inward_delta)
        self.max_lateral_drift = float(max_lateral_drift)
        self.minimum_macro_progress = float(minimum_macro_progress)
        self.max_cycles = int(max_cycles)


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


def direct_base_step_target(
    *,
    base_xy,
    base_yaw: float,
    base_command,
    control_dt: float,
) -> np.ndarray:
    """Convert a base-frame velocity command into one bounded world step."""
    base_xy = np.asarray(base_xy, dtype=float).reshape(2)
    command = np.asarray(base_command, dtype=float).reshape(-1)
    if command.size < 2:
        raise ValueError("base_command must contain forward and lateral values")
    cosine = math.cos(float(base_yaw))
    sine = math.sin(float(base_yaw))
    world_velocity = np.array(
        [
            cosine * command[0] - sine * command[1],
            sine * command[0] + cosine * command[1],
        ],
        dtype=float,
    )
    return base_xy + world_velocity * float(control_dt)


def floor_base_target_route(
    *,
    start_object_xy,
    target_xy,
    corridor_y: float,
    arrival_radius_m: float,
    arrival_margin_m: float,
    initial_clearance_m: float = 0.0,
    initial_push_direction_xy=None,
    reverse_switch_y: float | None = None,
    lateral_clearance_m: float = 0.0,
    final_side_approach_x: float | None = None,
) -> dict:
    """Plan axis-aligned contact pushes through the lower cross aisle.

    The default route starts by pushing toward the lower aisle.  Some input
    stations have a blocked lower edge; ``initial_push_direction_xy`` allows a
    short physical clearance stroke in the opposite axis before reversing
    into the common aisle.
    """
    start = np.asarray(start_object_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    values = np.asarray(
        [
            corridor_y,
            arrival_radius_m,
            arrival_margin_m,
            initial_clearance_m,
            lateral_clearance_m,
        ],
        dtype=float,
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
    initial_clearance = float(initial_clearance_m)
    lateral_clearance = float(lateral_clearance_m)
    if initial_clearance < 0.0 or lateral_clearance < 0.0:
        raise ValueError("floor route clearances must be non-negative")

    initial_direction = None
    switch_y = None if reverse_switch_y is None else float(reverse_switch_y)
    if switch_y is not None and not np.isfinite(switch_y):
        raise ValueError("reverse switch y must be finite")
    if initial_push_direction_xy is not None:
        initial_direction = np.asarray(initial_push_direction_xy, dtype=float).reshape(2)
        norm = float(np.linalg.norm(initial_direction))
        if norm <= 1e-12 or not np.all(np.isfinite(initial_direction)):
            raise ValueError("initial push direction must be a finite non-zero vector")
        initial_direction /= norm
        if not (
            np.allclose(initial_direction, [1.0, 0.0], atol=1e-9)
            or np.allclose(initial_direction, [-1.0, 0.0], atol=1e-9)
            or np.allclose(initial_direction, [0.0, 1.0], atol=1e-9)
            or np.allclose(initial_direction, [0.0, -1.0], atol=1e-9)
        ):
            raise ValueError("initial push direction must be axis aligned")

    final_y = float(target[1]) - (radius - margin)
    side_approach_x = (
        None
        if final_side_approach_x is None
        else float(final_side_approach_x)
    )
    if side_approach_x is not None:
        side_delta = side_approach_x - float(target[0])
        if (
            not np.isfinite(side_approach_x)
            or abs(side_delta) <= radius - margin
        ):
            raise ValueError("final side approach must lie beyond the scored endpoint")
        final_side_x = float(target[0]) + float(np.sign(side_delta)) * (
            radius - margin
        )
        final_y = float(target[1])
    aisle_y = float(corridor_y)
    if aisle_y >= min(float(start[1]), final_y):
        raise ValueError("floor route corridor must lie below start and arrival")

    if initial_direction is None:
        initial_direction = np.array([1.0, 0.0], dtype=float)
    clearance_start = start + initial_direction * initial_clearance
    if lateral_clearance > 0.0 and not np.allclose(
        initial_direction,
        [0.0, 1.0],
        atol=1e-9,
    ):
        raise ValueError("lateral clearance requires an initial +Y push")
    if lateral_clearance > 0.0 and switch_y is not None:
        raise ValueError("lateral clearance and reverse switch are mutually exclusive")
    if lateral_clearance > 0.0:
        lateral_end = clearance_start + np.array(
            [lateral_clearance, 0.0],
            dtype=float,
        )
        waypoints = [start, clearance_start, lateral_end]
        first_aisle_x = float(lateral_end[0])
    else:
        waypoints = None
        first_aisle_x = float(clearance_start[0])
    if np.allclose(initial_direction, [0.0, 1.0], atol=1e-9):
        # Move above the input station, then reverse and enter the lower aisle.
        # An optional intermediate y leaves a safe gap for the base to change
        # lateral side before the long southbound push.
        if switch_y is not None:
            if not (aisle_y < switch_y < float(clearance_start[1])):
                raise ValueError("reverse switch y must lie between aisle and clearance")
            first_aisle_point = np.array([start[0], switch_y], dtype=float)
        else:
            first_aisle_point = np.array([start[0], aisle_y], dtype=float)
    else:
        first_aisle_point = np.array([clearance_start[0], aisle_y], dtype=float)
    if waypoints is None:
        waypoints = [start, clearance_start]
        if (
            np.allclose(initial_direction, [0.0, 1.0], atol=1e-9)
            and switch_y is not None
        ):
            waypoints.extend(
                [
                    first_aisle_point,
                    np.array([start[0], aisle_y], dtype=float),
                ]
            )
        else:
            waypoints.append(first_aisle_point)
    else:
        waypoints.append(np.array([first_aisle_x, aisle_y], dtype=float))
    if side_approach_x is not None:
        waypoints.extend(
            [
                np.array([side_approach_x, aisle_y], dtype=float),
                np.array([side_approach_x, target[1]], dtype=float),
                np.array([final_side_x, target[1]], dtype=float),
            ]
        )
    else:
        waypoints.extend(
            [
                np.array([target[0], aisle_y], dtype=float),
                np.array([target[0], final_y], dtype=float),
            ]
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
        "final_side_approach_x": side_approach_x,
        "final_object_xy": final_object.tolist(),
        "final_target_distance_m": float(np.linalg.norm(final_object - target)),
    }


def floor_base_tracking_velocity(
    *,
    push_direction_xy,
    lateral_error_m: float,
    base_object_lateral_offset_m: float,
    forward_speed_m_s: float,
    lateral_gain: float,
    alignment_gain: float,
    lateral_deadband_m: float,
    maximum_base_object_offset_m: float,
    maximum_lateral_speed_m_s: float,
) -> np.ndarray:
    """Return bounded world velocity that holds floor contact on centerline."""
    direction = np.asarray(push_direction_xy, dtype=float)
    values = np.asarray(
        [
            lateral_error_m,
            base_object_lateral_offset_m,
            forward_speed_m_s,
            lateral_gain,
            alignment_gain,
            lateral_deadband_m,
            maximum_base_object_offset_m,
            maximum_lateral_speed_m_s,
        ],
        dtype=float,
    )
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("tracking direction must be a finite planar vector")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("tracking direction must be non-zero")
    if not np.all(np.isfinite(values)):
        raise ValueError("tracking parameters must be finite")
    speed = float(forward_speed_m_s)
    gain = float(lateral_gain)
    align_gain = float(alignment_gain)
    deadband = float(lateral_deadband_m)
    maximum_contact_offset = float(maximum_base_object_offset_m)
    maximum_lateral_speed = float(maximum_lateral_speed_m_s)
    if (
        speed <= 0.0
        or gain < 0.0
        or align_gain <= 0.0
        or deadband < 0.0
        or maximum_contact_offset <= 0.0
        or maximum_lateral_speed <= 0.0
    ):
        raise ValueError("tracking speed limits must be positive and gain nonnegative")

    direction /= direction_norm
    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    error = float(lateral_error_m)
    controlled_error = np.sign(error) * max(abs(error) - deadband, 0.0)
    desired_contact_offset = float(
        np.clip(
            gain * controlled_error,
            -maximum_contact_offset,
            maximum_contact_offset,
        )
    )
    contact_offset_error = (
        float(base_object_lateral_offset_m) - desired_contact_offset
    )
    correction = float(
        np.clip(
            -align_gain * contact_offset_error,
            -maximum_lateral_speed,
            maximum_lateral_speed,
        )
    )
    return direction * speed + left_axis * correction


def pivot_compensated_base_velocity(
    *,
    base_xy,
    base_yaw: float,
    pivot_xy,
    angular_velocity: float,
    control_dt: float,
) -> np.ndarray:
    """Return base-frame velocity that rotates the base around a world pivot."""
    dt = float(control_dt)
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("pivot control dt must be positive and finite")
    base_xy = np.asarray(base_xy, dtype=float).reshape(2)
    pivot_xy = np.asarray(pivot_xy, dtype=float).reshape(2)
    yaw = float(base_yaw)
    next_yaw = yaw + float(angular_velocity) * dt
    pivot_offset_base = world_velocity_to_base_frame(
        pivot_xy - base_xy,
        yaw,
    )
    cosine = math.cos(next_yaw)
    sine = math.sin(next_yaw)
    next_offset_world = np.array(
        [
            cosine * pivot_offset_base[0] - sine * pivot_offset_base[1],
            sine * pivot_offset_base[0] + cosine * pivot_offset_base[1],
        ],
        dtype=float,
    )
    target_base_xy = pivot_xy - next_offset_world
    world_velocity = (target_base_xy - base_xy) / dt
    return world_velocity_to_base_frame(world_velocity, yaw)


def bilateral_grasp_pivot_xy(gripper_positions) -> np.ndarray:
    """Return the rigid-body rotation center of the two active grippers."""
    positions = np.stack(
        [
            np.asarray(gripper_positions[arm], dtype=float)[:2]
            for arm in ("right", "left")
        ],
        axis=0,
    )
    return np.mean(positions, axis=0)


def vertical_hold_delta(
    *,
    current_z: float,
    target_z: float,
    feedforward: float,
    gain: float,
    max_delta: float,
) -> float:
    """Return a small upward OSC delta that resists grasped-object slip."""
    requested = float(feedforward) + float(gain) * (
        float(target_z) - float(current_z)
    )
    return float(np.clip(requested, 0.0, float(max_delta)))


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


def planar_grasp_drift(start_observation, observation) -> float:
    """Measure base-relative object-to-gripper planar offset change."""
    start_object = np.asarray(start_observation["object_pos"], dtype=float)[:2]
    object_xy = np.asarray(observation["object_pos"], dtype=float)[:2]
    start_yaw = float(start_observation.get("base_yaw", 0.0))
    current_yaw = float(observation.get("base_yaw", 0.0))
    drifts = []
    for arm in ("right", "left"):
        start_gripper = np.asarray(
            start_observation["gripper_positions"][arm],
            dtype=float,
        )[:2]
        gripper_xy = np.asarray(
            observation["gripper_positions"][arm],
            dtype=float,
        )[:2]
        start_offset = world_velocity_to_base_frame(
            start_gripper - start_object,
            start_yaw,
        )
        current_offset = world_velocity_to_base_frame(
            gripper_xy - object_xy,
            current_yaw,
        )
        drifts.append(float(np.linalg.norm(current_offset - start_offset)))
    return max(drifts)


def unilateral_planar_reseat_deltas(
    gripper_positions,
    *,
    object_position,
    inward_delta: float,
) -> dict[str, np.ndarray]:
    """Move only the farther gripper toward the measured object center."""
    requested = float(inward_delta)
    if requested < 0.0 or not np.isfinite(requested):
        raise ValueError("inward reseat delta must be finite and non-negative")
    object_xy = np.asarray(object_position, dtype=float)[:2]
    positions = {
        arm: np.asarray(gripper_positions[arm], dtype=float)[:2]
        for arm in ("right", "left")
    }
    deltas = {arm: np.zeros(2, dtype=float) for arm in ("right", "left")}
    moving_arm = max(
        ("right", "left"),
        key=lambda arm: float(np.linalg.norm(positions[arm] - object_xy)),
    )
    toward_object = object_xy - positions[moving_arm]
    distance = float(np.linalg.norm(toward_object))
    if distance > 1e-12 and requested > 0.0:
        deltas[moving_arm] = toward_object / distance * requested
    return deltas


def contact_reseat_deltas(
    contacts,
    gripper_positions,
    *,
    object_position,
    inward_delta: float,
) -> dict[str, np.ndarray]:
    """Move only grippers that lost contact toward the measured object center."""
    requested = float(inward_delta)
    if requested < 0.0 or not np.isfinite(requested):
        raise ValueError("contact reseat delta must be finite and non-negative")
    object_xy = np.asarray(object_position, dtype=float)[:2]
    deltas = {arm: np.zeros(2, dtype=float) for arm in ("right", "left")}
    for arm in ("right", "left"):
        if bool(contacts.get(arm, False)):
            continue
        position = np.asarray(gripper_positions[arm], dtype=float)[:2]
        toward_object = object_xy - position
        distance = float(np.linalg.norm(toward_object))
        if distance > 1e-12 and requested > 0.0:
            deltas[arm] = toward_object / distance * requested
    return deltas


def bilateral_planar_reseat_deltas(
    gripper_positions,
    *,
    object_position,
    inward_delta: float,
) -> dict[str, np.ndarray]:
    """Move both gripper centers a bounded planar step toward the object."""
    requested = float(inward_delta)
    if requested < 0.0 or not np.isfinite(requested):
        raise ValueError("inward reseat delta must be finite and non-negative")
    object_xy = np.asarray(object_position, dtype=float)[:2]
    deltas = {}
    for arm in ("right", "left"):
        position = np.asarray(gripper_positions[arm], dtype=float)[:2]
        toward_object = object_xy - position
        distance = float(np.linalg.norm(toward_object))
        deltas[arm] = (
            np.zeros(2, dtype=float)
            if distance <= 1e-12 or requested == 0.0
            else toward_object / distance * requested
        )
    return deltas


def physical_action_parts(
    robot,
    *,
    base_command,
    gripper_value: float,
    hold_targets,
    arm_actions=None,
) -> dict[str, np.ndarray]:
    """Build all Tiago action parts needed for a stable physical hold."""
    split = robot.composite_controller._action_split_indexes
    required = {"right", "left", "base"}
    missing = sorted(required.difference(split))
    if missing:
        raise RuntimeError(f"Robot action space is missing required parts: {missing}")

    arm_actions = arm_actions or {}
    action_parts: dict[str, np.ndarray] = {}
    for arm in ("right", "left"):
        start, end = split[arm]
        requested = np.asarray(
            arm_actions.get(arm, np.zeros(end - start)),
            dtype=float,
        ).reshape(-1)
        action = np.zeros(end - start, dtype=float)
        action[: min(action.size, requested.size)] = requested[: action.size]
        action_parts[arm] = action

        gripper = robot.gripper[arm]
        if int(gripper.dof) > 0:
            action_parts[f"{arm}_gripper"] = np.full(
                int(gripper.dof),
                float(gripper_value),
                dtype=float,
            )

    for part_name in ("torso", "head"):
        if part_name not in split or part_name not in hold_targets:
            continue
        start, end = split[part_name]
        target = np.asarray(hold_targets[part_name], dtype=float).reshape(-1)
        action = np.zeros(end - start, dtype=float)
        action[: min(action.size, target.size)] = target[: action.size]
        action_parts[part_name] = action

    start, end = split["base"]
    requested_base = np.asarray(base_command, dtype=float).reshape(-1)
    base_action = np.zeros(end - start, dtype=float)
    base_action[: min(base_action.size, requested_base.size)] = requested_base[
        : base_action.size
    ]
    action_parts["base"] = base_action
    return action_parts


def _shortest_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _transport_result(
    *,
    success: bool,
    failure_stage: str | None,
    steps: int,
    observation,
    final_distance: float,
    minimum_observed_z: float,
    start_observation,
    max_planar_grasp_drift_m: float,
) -> dict:
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "steps": int(steps),
        "final_base_xy": np.asarray(observation["base_xy"], dtype=float).tolist(),
        "final_distance": float(final_distance),
        "minimum_object_z": float(minimum_observed_z),
        "max_planar_grasp_drift_m": float(max_planar_grasp_drift_m),
        "start_object_pos": np.asarray(
            start_observation["object_pos"], dtype=float
        ).tolist(),
        "final_object_pos": np.asarray(
            observation["object_pos"], dtype=float
        ).tolist(),
        "start_gripper_positions": {
            arm: np.asarray(
                start_observation["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
        "final_gripper_positions": {
            arm: np.asarray(
                observation["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
        "contacts": {
            "right": bool(observation["contacts"].get("right", False)),
            "left": bool(observation["contacts"].get("left", False)),
        },
    }


def run_physical_transport(
    backend,
    *,
    path,
    object_name: str,
    hold_yaw: float,
    minimum_object_z: float,
    config: PhysicalCarryConfig | None = None,
    driver=None,
) -> dict:
    """Follow a base path while physics maintains a bilateral grasp."""
    config = config or PhysicalCarryConfig()
    driver = driver or OfficialPhysicalCarryDriver()
    waypoints = [np.asarray(point, dtype=float).reshape(2) for point in path]
    if not waypoints:
        raise ValueError("path must contain at least one waypoint")

    hold_targets = driver.capture_hold_targets(backend)
    observation = driver.observe(backend, object_name)
    start_observation = observation
    planar_reference_observation = observation
    object_offset_base = world_velocity_to_base_frame(
        np.asarray(observation["object_pos"][:2], dtype=float)
        - np.asarray(observation["base_xy"], dtype=float),
        float(observation["base_yaw"]),
    )
    carried_object_angle = float(
        math.atan2(object_offset_base[1], object_offset_base[0])
    )
    target_object_z = max(
        float(observation["object_pos"][2])
        - config.height_settle_allowance,
        float(minimum_object_z) + config.height_safety_margin,
    )
    gripper_z_offsets = {
        arm: float(observation["gripper_positions"][arm][2])
        - float(observation["object_pos"][2])
        for arm in ("right", "left")
    }
    minimum_observed_z = float(observation["object_pos"][2])
    maximum_planar_grasp_drift = 0.0
    previous_command = np.zeros(3, dtype=float)
    waypoint_index = 0
    final_distance = math.inf
    steps = 0
    contact_recovery_attempts = 0
    driver.record_event(
        backend,
        "physical_transport_start",
        object_name=object_name,
        waypoint_count=len(waypoints),
        waypoints=[point.tolist() for point in waypoints],
    )

    def recover_lost_contacts(observation):
        """Perform a bounded physical reseat for only lost gripper contacts."""
        nonlocal steps, contact_recovery_attempts
        if (
            int(config.planar_recovery_steps) <= 0
            or float(config.planar_recovery_inward_delta) <= 0.0
            or contact_recovery_attempts >= 3
        ):
            return False, observation
        contacts = observation["contacts"]
        if all(bool(contacts.get(arm, False)) for arm in ("right", "left")):
            return True, observation
        if not any(bool(contacts.get(arm, False)) for arm in ("right", "left")):
            return False, observation
        contact_recovery_attempts += 1
        driver.record_event(
            backend,
            "physical_contact_recovery_start",
            object_name=object_name,
            contacts={
                arm: bool(contacts.get(arm, False))
                for arm in ("right", "left")
            },
        )
        recovered = False
        recovery_failure = None
        step_fn = getattr(driver, "recover_planar", None)
        if not callable(step_fn):
            step_fn = driver.step
        for _ in range(int(config.planar_recovery_steps)):
            observation = driver.observe(backend, object_name)
            deltas = contact_reseat_deltas(
                observation["contacts"],
                observation["gripper_positions"],
                object_position=observation["object_pos"],
                inward_delta=float(config.planar_recovery_inward_delta),
            )
            vertical_deltas = {}
            for arm in ("right", "left"):
                if bool(observation["contacts"].get(arm, False)):
                    vertical_deltas[arm] = 0.0
                    continue
                vertical_deltas[arm] = float(
                    np.clip(
                        float(observation["object_pos"][2])
                        - float(observation["gripper_positions"][arm][2]),
                        -0.004,
                        0.004,
                    )
                )
            step_info = step_fn(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=hold_targets,
                arm_world_deltas={
                    arm: np.array(
                        [
                            deltas[arm][0],
                            deltas[arm][1],
                            vertical_deltas[arm],
                        ]
                    )
                    for arm in ("right", "left")
                },
                gripper_value=1.0,
                base_control_dt=config.base_control_dt,
            )
            steps += 1
            observation = driver.observe(backend, object_name)
            if bool(step_info.get("collision", False)):
                recovery_failure = "collision"
                break
            if float(observation["object_pos"][2]) < float(minimum_object_z):
                recovery_failure = "object_drop"
                break
            if next_contact_stability(observation["contacts"], 0) > 0:
                recovered = True
                break
        driver.record_event(
            backend,
            "physical_contact_recovery_end",
            object_name=object_name,
            success=recovered,
            failure_stage=recovery_failure,
            steps=int(config.planar_recovery_steps),
            contacts={
                arm: bool(observation["contacts"].get(arm, False))
                for arm in ("right", "left")
            },
        )
        return recovered, observation

    if next_contact_stability(observation["contacts"], 0) == 0:
        return _transport_result(
            success=False,
            failure_stage="contact",
            steps=steps,
            observation=observation,
            final_distance=final_distance,
            minimum_observed_z=minimum_observed_z,
            start_observation=start_observation,
            max_planar_grasp_drift_m=maximum_planar_grasp_drift,
        )

    failure_stage = "timeout"
    success = False
    while steps < config.max_steps:
        observation = driver.observe(backend, object_name)
        minimum_observed_z = min(
            minimum_observed_z,
            float(observation["object_pos"][2]),
        )
        maximum_planar_grasp_drift = max(
            maximum_planar_grasp_drift,
            planar_grasp_drift(start_observation, observation),
        )
        if next_contact_stability(observation["contacts"], 0) == 0:
            recovered, observation = recover_lost_contacts(observation)
            if not recovered:
                failure_stage = "contact"
                break
        if float(observation["object_pos"][2]) < float(minimum_object_z):
            failure_stage = "object_drop"
            break
        planar_recovery_drift = planar_grasp_drift(
            planar_reference_observation,
            observation,
        )
        if (
            float(config.planar_recovery_trigger) > 0.0
            and planar_recovery_drift
            >= float(config.planar_recovery_trigger)
        ):
            driver.record_event(
                backend,
                "physical_planar_recovery_start",
                object_name=object_name,
                drift_m=planar_recovery_drift,
            )
            recovery_failure = None
            recovery_steps = 0
            for _ in range(max(0, int(config.planar_recovery_steps))):
                observation = driver.observe(backend, object_name)
                reseat_deltas = unilateral_planar_reseat_deltas(
                    observation["gripper_positions"],
                    object_position=observation["object_pos"],
                    inward_delta=float(config.planar_recovery_inward_delta),
                )
                step_info = driver.recover_planar(
                    backend,
                    object_name=object_name,
                    base_command=np.zeros(3, dtype=float),
                    hold_targets=hold_targets,
                    arm_world_deltas={
                        arm: np.array(
                            [reseat_deltas[arm][0], reseat_deltas[arm][1], 0.0],
                            dtype=float,
                        )
                        for arm in ("right", "left")
                    },
                    gripper_value=1.0,
                    base_control_dt=config.base_control_dt,
                )
                steps += 1
                recovery_steps += 1
                observation = driver.observe(backend, object_name)
                minimum_observed_z = min(
                    minimum_observed_z,
                    float(observation["object_pos"][2]),
                )
                maximum_planar_grasp_drift = max(
                    maximum_planar_grasp_drift,
                    planar_grasp_drift(start_observation, observation),
                )
                if bool(step_info.get("collision", False)):
                    recovery_failure = "collision"
                    break
                if next_contact_stability(observation["contacts"], 0) == 0:
                    recovery_failure = "contact"
                    break
                if float(observation["object_pos"][2]) < float(
                    minimum_object_z
                ):
                    recovery_failure = "object_drop"
                    break
            if recovery_steps == 0 and recovery_failure is None:
                recovery_failure = "planar_recovery"
            planar_recovery_success = recovery_failure is None
            driver.record_event(
                backend,
                "physical_planar_recovery_end",
                object_name=object_name,
                success=planar_recovery_success,
                failure_stage=recovery_failure,
                steps=recovery_steps,
            )
            if recovery_failure is not None:
                failure_stage = recovery_failure
                break
            planar_reference_observation = observation
        if maximum_planar_grasp_drift > config.max_planar_grasp_drift:
            failure_stage = "planar_grasp_drift"
            break
        height_error = target_object_z - float(observation["object_pos"][2])
        if (
            config.height_recovery_enabled
            and height_error >= config.height_recovery_trigger
        ):
            driver.record_event(
                backend,
                "physical_height_recenter_start",
                object_name=object_name,
            )
            recentered = not bool(
                getattr(driver, "requires_height_recenter", True)
            )
            for _ in range(
                0 if recentered else config.height_recenter_steps
            ):
                observation = driver.observe(backend, object_name)
                recenter_deltas = {
                    arm: (
                        float(observation["object_pos"][2])
                        + gripper_z_offsets[arm]
                        - float(observation["gripper_positions"][arm][2])
                    )
                    for arm in ("right", "left")
                }
                if max(abs(value) for value in recenter_deltas.values()) <= (
                    config.height_recenter_tolerance
                ):
                    recentered = True
                    break
                if steps >= config.max_steps:
                    break
                arm_world_deltas = {
                    arm: np.array(
                        [
                            0.0,
                            0.0,
                            float(
                                np.clip(
                                    recenter_deltas[arm],
                                    -config.height_recenter_max_delta,
                                    config.height_recenter_max_delta,
                                )
                            ),
                        ],
                        dtype=float,
                    )
                    for arm in ("right", "left")
                }
                step_info = driver.step(
                    backend,
                    object_name=object_name,
                    base_command=np.zeros(3, dtype=float),
                    hold_targets=hold_targets,
                    arm_world_deltas=arm_world_deltas,
                    base_control_dt=config.base_control_dt,
                )
                steps += 1
                observation = driver.observe(backend, object_name)
                minimum_observed_z = min(
                    minimum_observed_z,
                    float(observation["object_pos"][2]),
                )
                if bool(step_info.get("collision", False)):
                    failure_stage = "collision"
                    break
                if next_contact_stability(observation["contacts"], 0) == 0:
                    failure_stage = "contact"
                    break
                if float(observation["object_pos"][2]) < float(minimum_object_z):
                    failure_stage = "object_drop"
                    break
            else:
                observation = driver.observe(backend, object_name)

            if failure_stage != "timeout":
                break
            if not recentered:
                recenter_deltas = {
                    arm: (
                        float(observation["object_pos"][2])
                        + gripper_z_offsets[arm]
                        - float(observation["gripper_positions"][arm][2])
                    )
                    for arm in ("right", "left")
                }
                recentered = max(
                    abs(value) for value in recenter_deltas.values()
                ) <= config.height_recenter_tolerance
            driver.record_event(
                backend,
                "physical_height_recenter_end",
                object_name=object_name,
                success=recentered,
            )
            if not recentered:
                failure_stage = "height_recenter"
                break

            observation = driver.observe(backend, object_name)
            height_error = target_object_z - float(observation["object_pos"][2])
            driver.record_event(
                backend,
                "physical_height_recovery_start",
                object_name=object_name,
                lift_height=height_error,
            )
            controller_success = driver.recover_height(
                backend,
                object_name=object_name,
                lift_height=height_error,
                max_steps=config.height_recovery_steps,
                max_action=config.height_recovery_max_action,
            )
            observation = driver.observe(backend, object_name)
            minimum_observed_z = min(
                minimum_observed_z,
                float(observation["object_pos"][2]),
            )
            maximum_planar_grasp_drift = max(
                maximum_planar_grasp_drift,
                planar_grasp_drift(start_observation, observation),
            )
            recovered_height_error = (
                target_object_z - float(observation["object_pos"][2])
            )
            recovered = bool(
                next_contact_stability(observation["contacts"], 0) > 0
                and recovered_height_error < config.height_recovery_trigger
            )
            driver.record_event(
                backend,
                "physical_height_recovery_end",
                object_name=object_name,
                success=recovered,
                controller_success=bool(controller_success),
                remaining_height_error=recovered_height_error,
            )
            if not recovered:
                failure_stage = "height_recovery"
                break

        while waypoint_index < len(waypoints):
            delta = waypoints[waypoint_index] - np.asarray(
                observation["base_xy"],
                dtype=float,
            )
            final_distance = float(np.linalg.norm(delta))
            if final_distance >= config.waypoint_tolerance:
                break
            waypoint_index += 1
        if waypoint_index >= len(waypoints):
            success = True
            failure_stage = None
            break

        desired_yaw = float(hold_yaw)
        if config.align_heading_to_path:
            desired_yaw = float(math.atan2(delta[1], delta[0])) - (
                carried_object_angle
            )
        yaw_error = _shortest_angle(
            desired_yaw - float(observation["base_yaw"])
        )
        heading_aligned = bool(
            not config.align_heading_to_path
            or abs(yaw_error) <= config.heading_translation_tolerance
        )
        speed = (
            min(config.k_linear * final_distance, config.max_linear)
            if heading_aligned
            else 0.0
        )
        world_velocity = speed * delta / max(final_distance, 1e-12)
        base_velocity = world_velocity_to_base_frame(
            world_velocity,
            float(observation["base_yaw"]),
        )
        angular = float(
            np.clip(
                config.k_angular * yaw_error,
                -config.max_angular,
                config.max_angular,
            )
        )
        requested_command = np.array(
            [base_velocity[0], base_velocity[1], angular],
            dtype=float,
        )
        command = slew_limited_command(
            previous_command,
            requested_command,
            np.array(
                [
                    config.max_linear_delta,
                    config.max_linear_delta,
                    config.max_angular_delta,
                ],
                dtype=float,
            ),
        )
        if not heading_aligned and config.pivot_compensation_enabled:
            command[:2] = pivot_compensated_base_velocity(
                base_xy=observation["base_xy"],
                base_yaw=float(observation["base_yaw"]),
                pivot_xy=bilateral_grasp_pivot_xy(
                    observation["gripper_positions"]
                ),
                angular_velocity=float(command[2]),
                control_dt=config.base_control_dt,
            )
        base_xy = np.asarray(observation["base_xy"], dtype=float)
        world_step = direct_base_step_target(
            base_xy=base_xy,
            base_yaw=float(observation["base_yaw"]),
            base_command=command,
            control_dt=config.base_control_dt,
        ) - base_xy
        phases = ((command, np.zeros(2, dtype=float)),)
        abort = False
        for phase_base_command, phase_arm_xy in phases:
            if steps >= config.max_steps:
                break
            observation = driver.observe(backend, object_name)
            hold_delta = vertical_hold_delta(
                current_z=float(observation["object_pos"][2]),
                target_z=target_object_z,
                feedforward=config.vertical_hold_feedforward,
                gain=config.vertical_hold_gain,
                max_delta=config.max_vertical_hold_delta,
            )
            arm_world_deltas = {
                arm: np.array(
                    [phase_arm_xy[0], phase_arm_xy[1], hold_delta],
                    dtype=float,
                )
                for arm in ("right", "left")
            }
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=phase_base_command,
                hold_targets=hold_targets,
                arm_world_deltas=arm_world_deltas,
                base_control_dt=config.base_control_dt,
            )
            steps += 1

            observation = driver.observe(backend, object_name)
            minimum_observed_z = min(
                minimum_observed_z,
                float(observation["object_pos"][2]),
            )
            if bool(step_info.get("collision", False)):
                failure_stage = "collision"
                abort = True
                break
            if next_contact_stability(observation["contacts"], 0) == 0:
                recovered, observation = recover_lost_contacts(observation)
                if not recovered:
                    failure_stage = "contact"
                    abort = True
                    break
            if float(observation["object_pos"][2]) < float(minimum_object_z):
                failure_stage = "object_drop"
                abort = True
                break
            if maximum_planar_grasp_drift > config.max_planar_grasp_drift:
                failure_stage = "planar_grasp_drift"
                abort = True
                break
        previous_command = command
        if abort:
            break

    if not success:
        observation = driver.observe(backend, object_name)
        if waypoint_index < len(waypoints):
            final_distance = float(
                np.linalg.norm(
                    waypoints[waypoint_index]
                    - np.asarray(observation["base_xy"], dtype=float)
                )
            )

    driver.record_event(
        backend,
        "physical_transport_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        steps=steps,
        minimum_object_z=minimum_observed_z,
        max_planar_grasp_drift_m=maximum_planar_grasp_drift,
        contacts={
            arm: bool(observation["contacts"].get(arm, False))
            for arm in ("right", "left")
        },
        final_object_pos=np.asarray(
            observation["object_pos"], dtype=float
        ).tolist(),
        final_gripper_positions={
            arm: np.asarray(
                observation["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
    )
    return _transport_result(
        success=success,
        failure_stage=failure_stage,
        steps=steps,
        observation=observation,
        final_distance=final_distance,
        minimum_observed_z=minimum_observed_z,
        start_observation=start_observation,
        max_planar_grasp_drift_m=maximum_planar_grasp_drift,
    )


def _inchworm_planar_motion(delta, direction) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float).reshape(2)
    direction = np.asarray(direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("travel_direction must be non-zero")
    direction = direction / norm
    progress = float(np.dot(delta, direction))
    lateral = float(np.linalg.norm(delta - progress * direction))
    return progress, lateral


def compensated_reset_arm_delta(
    *,
    reset_start_gripper,
    current_gripper,
    world_step,
    planar_gain: float,
) -> np.ndarray:
    """Compensate direct base motion without amplifying vertical corrections."""
    gain = float(planar_gain)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("planar reset compensation gain must be positive")
    start = np.asarray(reset_start_gripper, dtype=float).reshape(3)
    current = np.asarray(current_gripper, dtype=float).reshape(3)
    step = np.asarray(world_step, dtype=float).reshape(2)
    delta = start - current - np.array([step[0], step[1], 0.0], dtype=float)
    delta[:2] *= gain
    return delta


def run_inchworm_transport(
    backend,
    *,
    object_name: str,
    travel_direction,
    travel_distance: float,
    minimum_object_z: float,
    config: InchwormCarryConfig | None = None,
    driver=None,
) -> dict:
    """Repeat physical arm strokes and compensated base resets."""
    config = config or InchwormCarryConfig()
    driver = driver or OfficialPhysicalCarryDriver()
    direction = np.asarray(travel_direction, dtype=float).reshape(2)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("travel_direction must be non-zero")
    direction = direction / direction_norm
    distance = float(travel_distance)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("travel_distance must be finite and non-negative")
    positive_values = {
        "stroke_distance": config.stroke_distance,
        "arm_target_tolerance": config.arm_target_tolerance,
        "reset_distance": config.reset_distance,
        "reset_max_linear": config.reset_max_linear,
        "reset_control_dt": config.reset_control_dt,
        "reset_position_tolerance": config.reset_position_tolerance,
        "reset_max_gripper_drift": config.reset_max_gripper_drift,
        "reset_arm_compensation_gain": config.reset_arm_compensation_gain,
        "max_lateral_drift": config.max_lateral_drift,
        "minimum_macro_progress": config.minimum_macro_progress,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values.values()):
        raise ValueError("inchworm distances, limits, and tolerances must be positive")
    if config.arm_max_steps < 1 or config.max_cycles < 1:
        raise ValueError("inchworm step and cycle budgets must be positive")
    if config.reseat_steps < 0 or config.reseat_inward_delta < 0.0:
        raise ValueError("inchworm reseat controls must be non-negative")
    hold_targets = driver.capture_hold_targets(backend)
    start = driver.observe(backend, object_name)
    start_object = np.asarray(start["object_pos"], dtype=float).copy()
    start_base = np.asarray(start["base_xy"], dtype=float).copy()
    target_object_z = float(start_object[2])
    minimum_observed_z = float(start_object[2])
    cycles = []
    steps = 0
    failure_stage = "timeout"
    success = distance == 0.0
    driver.record_event(
        backend,
        "inchworm_transport_start",
        object_name=object_name,
        travel_distance=distance,
        start_object_pos=start_object.tolist(),
        start_gripper_positions={
            arm: np.asarray(
                start["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
    )

    for cycle_index in range(config.max_cycles):
        if success:
            break
        cycle_start = driver.observe(backend, object_name)
        if next_contact_stability(cycle_start["contacts"], 0) == 0:
            failure_stage = "contact"
            break
        cycle_start_object = np.asarray(cycle_start["object_pos"], dtype=float).copy()
        cycle_start_grippers = {
            arm: np.asarray(cycle_start["gripper_positions"][arm], dtype=float).copy()
            for arm in ("right", "left")
        }
        height_error = target_object_z - float(cycle_start_object[2])
        vertical_adjustment = float(
            np.clip(
                config.stroke_vertical_feedforward
                + config.stroke_height_gain * height_error,
                -config.max_vertical_adjustment,
                config.max_vertical_adjustment,
            )
        )
        arm_targets = {
            arm: cycle_start_grippers[arm]
            + np.array(
                [
                    direction[0] * config.stroke_distance,
                    direction[1] * config.stroke_distance,
                    vertical_adjustment,
                ],
                dtype=float,
            )
            for arm in ("right", "left")
        }

        arm_reached = False
        cycle_collision = False
        cycle_failure = None
        arm_steps = 0
        for _ in range(config.arm_max_steps):
            observation = driver.observe(backend, object_name)
            arm_deltas = {
                arm: arm_targets[arm]
                - np.asarray(observation["gripper_positions"][arm], dtype=float)
                for arm in ("right", "left")
            }
            if max(float(np.linalg.norm(delta)) for delta in arm_deltas.values()) <= (
                config.arm_target_tolerance
            ):
                arm_reached = True
                break
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=hold_targets,
                arm_world_deltas=arm_deltas,
                gripper_value=1.0,
                base_control_dt=config.reset_control_dt,
            )
            steps += 1
            arm_steps += 1
            observation = driver.observe(backend, object_name)
            minimum_observed_z = min(
                minimum_observed_z, float(observation["object_pos"][2])
            )
            cycle_collision = bool(step_info.get("collision", False))
            if cycle_collision:
                cycle_failure = "collision"
                break
            if next_contact_stability(observation["contacts"], 0) == 0:
                cycle_failure = "contact"
                break
            if float(observation["object_pos"][2]) < float(minimum_object_z):
                cycle_failure = "object_drop"
                break

        arm_end = driver.observe(backend, object_name)
        arm_progress, arm_lateral = _inchworm_planar_motion(
            np.asarray(arm_end["object_pos"], dtype=float)[:2]
            - cycle_start_object[:2],
            direction,
        )
        if cycle_failure is None and not arm_reached:
            cycle_failure = "arm_timeout"
        if cycle_failure is None and (
            arm_progress < config.minimum_macro_progress
            or arm_lateral > config.max_lateral_drift
        ):
            cycle_failure = "arm_progress"
        total_arm_progress, total_arm_lateral = _inchworm_planar_motion(
            np.asarray(arm_end["object_pos"], dtype=float)[:2] - start_object[:2],
            direction,
        )
        if cycle_failure is None and total_arm_lateral > config.max_lateral_drift:
            cycle_failure = "arm_progress"
        if cycle_failure is not None:
            failure_stage = cycle_failure
            break
        if total_arm_progress >= distance:
            cycles.append(
                {
                    "cycle": cycle_index + 1,
                    "arm_steps": arm_steps,
                    "reset_steps": 0,
                    "vertical_adjustment_m": vertical_adjustment,
                    "arm_progress_m": arm_progress,
                    "macro_progress_m": arm_progress,
                    "total_progress_m": total_arm_progress,
                    "macro_lateral_drift_m": arm_lateral,
                    "max_gripper_reset_drift_m": 0.0,
                    "max_gripper_reset_drift_by_arm_m": {
                        "right": 0.0,
                        "left": 0.0,
                    },
                    "base_reset_translation_m": 0.0,
                }
            )
            success = True
            failure_stage = None
            break

        reset_start = driver.observe(backend, object_name)
        reset_start_base = np.asarray(reset_start["base_xy"], dtype=float).copy()
        reset_start_grippers = {
            arm: np.asarray(reset_start["gripper_positions"][arm], dtype=float).copy()
            for arm in ("right", "left")
        }
        reset_steps = 0
        max_gripper_drift = 0.0
        max_gripper_drift_by_arm = {"right": 0.0, "left": 0.0}
        reset_translation = 0.0
        reset_budget = int(
            math.ceil(
                config.reset_distance
                / (config.reset_max_linear * config.reset_control_dt)
            )
        ) + 20
        for _ in range(reset_budget):
            observation = driver.observe(backend, object_name)
            reset_translation = float(
                np.dot(
                    np.asarray(observation["base_xy"], dtype=float) - reset_start_base,
                    direction,
                )
            )
            remaining = max(0.0, config.reset_distance - reset_translation)
            if remaining <= config.reset_position_tolerance:
                break
            speed = min(config.reset_max_linear, remaining / config.reset_control_dt)
            world_velocity = direction * speed
            base_velocity = world_velocity_to_base_frame(
                world_velocity, float(observation["base_yaw"])
            )
            world_step = direction * speed * config.reset_control_dt
            arm_deltas = {
                arm: compensated_reset_arm_delta(
                    reset_start_gripper=reset_start_grippers[arm],
                    current_gripper=observation["gripper_positions"][arm],
                    world_step=world_step,
                    planar_gain=config.reset_arm_compensation_gain,
                )
                for arm in ("right", "left")
            }
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.array(
                    [base_velocity[0], base_velocity[1], 0.0], dtype=float
                ),
                hold_targets=hold_targets,
                arm_world_deltas=arm_deltas,
                gripper_value=1.0,
                base_control_dt=config.reset_control_dt,
            )
            steps += 1
            reset_steps += 1
            observation = driver.observe(backend, object_name)
            minimum_observed_z = min(
                minimum_observed_z, float(observation["object_pos"][2])
            )
            for arm in ("right", "left"):
                max_gripper_drift_by_arm[arm] = max(
                    max_gripper_drift_by_arm[arm],
                    float(
                        np.linalg.norm(
                            np.asarray(observation["gripper_positions"][arm])[:2]
                            - reset_start_grippers[arm][:2]
                        )
                    )
                )
            max_gripper_drift = max(max_gripper_drift_by_arm.values())
            if bool(step_info.get("collision", False)):
                cycle_failure = "collision"
                break
            if next_contact_stability(observation["contacts"], 0) == 0:
                cycle_failure = "contact"
                break
            if float(observation["object_pos"][2]) < float(minimum_object_z):
                cycle_failure = "object_drop"
                break
            if max_gripper_drift > config.reset_max_gripper_drift:
                cycle_failure = "reset_gripper_drift"
                break

        reseat_steps = 0
        if cycle_failure is None:
            for _ in range(config.reseat_steps):
                observation = driver.observe(backend, object_name)
                reseat_xy = bilateral_planar_reseat_deltas(
                    observation["gripper_positions"],
                    object_position=observation["object_pos"],
                    inward_delta=config.reseat_inward_delta,
                )
                step_info = driver.step(
                    backend,
                    object_name=object_name,
                    base_command=np.zeros(3, dtype=float),
                    hold_targets=hold_targets,
                    arm_world_deltas={
                        arm: np.array(
                            [reseat_xy[arm][0], reseat_xy[arm][1], 0.0],
                            dtype=float,
                        )
                        for arm in ("right", "left")
                    },
                    gripper_value=1.0,
                    base_control_dt=config.reset_control_dt,
                )
                steps += 1
                reseat_steps += 1
                observation = driver.observe(backend, object_name)
                minimum_observed_z = min(
                    minimum_observed_z,
                    float(observation["object_pos"][2]),
                )
                if bool(step_info.get("collision", False)):
                    cycle_failure = "collision"
                    break
                if next_contact_stability(observation["contacts"], 0) == 0:
                    cycle_failure = "contact"
                    break
                if float(observation["object_pos"][2]) < float(
                    minimum_object_z
                ):
                    cycle_failure = "object_drop"
                    break

        cycle_end = driver.observe(backend, object_name)
        reset_translation = float(
            np.dot(
                np.asarray(cycle_end["base_xy"], dtype=float) - reset_start_base,
                direction,
            )
        )
        macro_progress, macro_lateral = _inchworm_planar_motion(
            np.asarray(cycle_end["object_pos"], dtype=float)[:2]
            - cycle_start_object[:2],
            direction,
        )
        total_progress, total_lateral = _inchworm_planar_motion(
            np.asarray(cycle_end["object_pos"], dtype=float)[:2]
            - start_object[:2],
            direction,
        )
        cycles.append(
            {
                "cycle": cycle_index + 1,
                "arm_steps": arm_steps,
                "reset_steps": reset_steps,
                "reseat_steps": reseat_steps,
                "vertical_adjustment_m": vertical_adjustment,
                "arm_progress_m": arm_progress,
                "macro_progress_m": macro_progress,
                "total_progress_m": total_progress,
                "macro_lateral_drift_m": macro_lateral,
                "max_gripper_reset_drift_m": max_gripper_drift,
                "max_gripper_reset_drift_by_arm_m": dict(
                    max_gripper_drift_by_arm
                ),
                "base_reset_translation_m": reset_translation,
            }
        )
        if cycle_failure is None and reset_translation < (
            config.reset_distance - config.reset_position_tolerance
        ):
            cycle_failure = "reset_timeout"
        if cycle_failure is None and (
            macro_progress < config.minimum_macro_progress
            or macro_lateral > config.max_lateral_drift
            or total_lateral > config.max_lateral_drift
        ):
            cycle_failure = "macro_progress"
        if cycle_failure is not None:
            failure_stage = cycle_failure
            break
        if total_progress >= distance:
            success = True
            failure_stage = None
            break

    final = driver.observe(backend, object_name)
    object_progress, lateral_drift = _inchworm_planar_motion(
        np.asarray(final["object_pos"], dtype=float)[:2] - start_object[:2],
        direction,
    )
    base_translation = float(
        np.linalg.norm(np.asarray(final["base_xy"], dtype=float) - start_base)
    )
    driver.record_event(
        backend,
        "inchworm_transport_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        object_progress_m=object_progress,
        cycle_count=len(cycles),
        contacts={
            arm: bool(final["contacts"].get(arm, False))
            for arm in ("right", "left")
        },
        start_object_pos=start_object.tolist(),
        start_gripper_positions={
            arm: np.asarray(
                start["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
        final_object_pos=np.asarray(final["object_pos"], dtype=float).tolist(),
        final_gripper_positions={
            arm: np.asarray(
                final["gripper_positions"][arm], dtype=float
            ).tolist()
            for arm in ("right", "left")
        },
    )
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "steps": steps,
        "cycle_count": len(cycles),
        "object_progress_m": object_progress,
        "lateral_drift_m": lateral_drift,
        "base_translation_m": base_translation,
        "minimum_object_z": minimum_observed_z,
        "contacts": dict(final["contacts"]),
        "cycles": cycles,
    }


class OfficialPhysicalCarryDriver:
    """Translate carry commands into official Tiago controller actions."""

    @staticmethod
    def capture_hold_targets(backend):
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            capture_hold_targets,
        )

        return capture_hold_targets(backend.env.robots[0])

    @staticmethod
    def observe(backend, object_name: str) -> dict:
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            gripper_end_center_pos,
            object_center_pos,
        )
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            grasp_status,
        )

        base_xy, base_yaw = backend.get_base_pose()
        raw_env = backend.env
        robot = raw_env.robots[0]
        return {
            "base_xy": np.asarray(base_xy, dtype=float).copy(),
            "base_yaw": float(base_yaw),
            "object_pos": np.asarray(
                object_center_pos(raw_env, object_name),
                dtype=float,
            ).copy(),
            "contacts": dict(grasp_status(raw_env, robot, object_name)),
            "gripper_positions": {
                arm: np.asarray(
                    gripper_end_center_pos(raw_env, robot, arm),
                    dtype=float,
                ).copy()
                for arm in ("right", "left")
            },
        }

    @staticmethod
    def step(
        backend,
        *,
        object_name: str,
        base_command,
        hold_targets,
        arm_world_deltas=None,
        gripper_value: float = 1.0,
        base_control_dt: float = 0.05,
    ) -> dict:
        del object_name
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            arm_delta_to_normalized_action,
            world_delta_to_controller_frame,
        )

        raw_env = backend.env
        robot = raw_env.robots[0]
        base_command = np.asarray(base_command, dtype=float).reshape(-1)
        if np.any(np.abs(base_command) > 0.0):
            from robot_agent.environments.robosuite_backend import (
                _set_base_world_yaw_direct,
                _set_base_xy_direct,
            )

            base_xy, base_yaw = backend.get_base_pose()
            target_xy = direct_base_step_target(
                base_xy=base_xy,
                base_yaw=base_yaw,
                base_command=base_command,
                control_dt=base_control_dt,
            )
            angular = float(base_command[2]) if base_command.size >= 3 else 0.0
            _set_base_world_yaw_direct(
                raw_env,
                robot,
                float(base_yaw) + angular * float(base_control_dt),
            )
            _set_base_xy_direct(raw_env, robot, target_xy)

        arm_actions = {}
        if arm_world_deltas:
            robot.composite_controller.update_state()
            for arm in ("right", "left"):
                world_delta = np.asarray(
                    arm_world_deltas.get(arm, np.zeros(3)),
                    dtype=float,
                )
                controller_delta = world_delta_to_controller_frame(
                    robot,
                    arm,
                    world_delta,
                )
                arm_actions[arm] = arm_delta_to_normalized_action(
                    robot,
                    arm,
                    controller_delta,
                    max_action=0.30,
                )
        action_parts = physical_action_parts(
            robot,
            base_command=np.zeros(3, dtype=float),
            gripper_value=gripper_value,
            hold_targets=hold_targets,
            arm_actions=arm_actions,
        )
        action = robot.create_action_vector(action_parts)
        result = raw_env.step(action)
        info = result[-1] if isinstance(result, tuple) else {}
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        return {
            "collision": bool((info or {}).get("has_judge_collision", False)),
        }

    @staticmethod
    def recover_height(
        backend,
        *,
        object_name: str,
        lift_height: float,
        max_steps: int,
        max_action: float,
    ) -> bool:
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            lift_grasped_object,
        )

        recorder = getattr(backend, "_record_trajectory_frame", None)
        result = lift_grasped_object(
            env=backend.env,
            object_name=object_name,
            lift_height=float(lift_height),
            max_steps=int(max_steps),
            hold_steps=0,
            tolerance=min(0.003, max(0.001, float(lift_height) * 0.25)),
            max_action=float(max_action),
            render=False,
            render_callback=recorder if callable(recorder) else None,
        )
        return bool(result.get("success", False))

    @staticmethod
    def record_event(backend, event: str, **payload) -> None:
        marker = getattr(backend, "_mark_trajectory_event", None)
        if callable(marker):
            marker(event, **payload)


class PostureLockedPhysicalCarryDriver:
    """Keep upper-body posture base-relative while grip actuators stay active."""

    requires_height_recenter = False

    def __init__(self, delegate=None) -> None:
        self._delegate = delegate or OfficialPhysicalCarryDriver()
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
        self._posture = self._capture_robot_posture(backend)
        return result

    def recover_planar(self, backend, **kwargs):
        result = self._delegate.step(backend, **kwargs)
        self._posture = self._capture_robot_posture(backend)
        return result


def run_physical_target_alignment(
    backend,
    *,
    object_name: str,
    target_xy,
    minimum_object_z: float,
    target_distance: float = 0.70,
    max_translation: float = 0.18,
    step_size: float = 0.002,
    max_steps: int = 200,
    max_planar_grasp_drift: float = 0.03,
    driver=None,
) -> dict:
    """Move a physically held object into the scoring radius using both arms."""
    driver = driver or OfficialPhysicalCarryDriver()
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    if (
        not np.all(np.isfinite(target_xy))
        or float(target_distance) <= 0.0
        or float(max_translation) <= 0.0
        or float(step_size) <= 0.0
        or int(max_steps) < 1
    ):
        raise ValueError("physical alignment parameters must be finite and positive")

    hold_targets = driver.capture_hold_targets(backend)
    start = driver.observe(backend, object_name)
    observation = start
    start_xy = np.asarray(start["object_pos"][:2], dtype=float)
    steps = 0
    success = False
    failure_stage = "timeout"
    driver.record_event(
        backend,
        "physical_target_alignment_start",
        object_name=object_name,
        target_xy=target_xy.tolist(),
    )

    while steps < int(max_steps):
        contacts = observation["contacts"]
        if next_contact_stability(contacts, 0) == 0:
            failure_stage = "contact"
            break
        if float(observation["object_pos"][2]) < float(minimum_object_z):
            failure_stage = "object_drop"
            break
        grasp_drift = planar_grasp_drift(start, observation)
        if grasp_drift > float(max_planar_grasp_drift):
            failure_stage = "planar_grasp_drift"
            break

        current_xy = np.asarray(observation["object_pos"][:2], dtype=float)
        error = target_xy - current_xy
        distance = float(np.linalg.norm(error))
        if distance <= float(target_distance):
            success = True
            failure_stage = None
            break
        translation = float(np.linalg.norm(current_xy - start_xy))
        remaining_translation = float(max_translation) - translation
        if remaining_translation <= 1e-9:
            failure_stage = "translation_limit"
            break
        requested = min(
            float(step_size),
            distance - float(target_distance),
            remaining_translation,
        )
        world_delta = np.array(
            [error[0], error[1], 0.0],
            dtype=float,
        ) / max(distance, 1e-12) * requested
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas={
                "right": world_delta.copy(),
                "left": world_delta.copy(),
            },
            gripper_value=1.0,
            base_control_dt=0.05,
        )
        steps += 1
        observation = driver.observe(backend, object_name)
        if bool(step_info.get("collision", False)):
            failure_stage = "collision"
            break

    final_xy = np.asarray(observation["object_pos"][:2], dtype=float)
    final_distance = float(np.linalg.norm(final_xy - target_xy))
    translation = float(np.linalg.norm(final_xy - start_xy))
    driver.record_event(
        backend,
        "physical_target_alignment_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        final_distance=final_distance,
        translation_m=translation,
    )
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "steps": int(steps),
        "final_distance": final_distance,
        "translation_m": translation,
        "final_object_pos": np.asarray(
            observation["object_pos"], dtype=float
        ).tolist(),
        "contacts": {
            "right": bool(observation["contacts"].get("right", False)),
            "left": bool(observation["contacts"].get("left", False)),
        },
    }


def _place_result(
    *,
    success: bool,
    failure_stage: str | None,
    observation,
    start_z: float,
    support_detected: bool,
    final_distance: float,
    steps: int,
) -> dict:
    current_z = float(observation["object_pos"][2])
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "support_detected": bool(support_detected),
        "descent": max(0.0, float(start_z) - current_z),
        "final_distance": float(final_distance),
        "steps": int(steps),
        "contacts": {
            "right": bool(observation["contacts"].get("right", False)),
            "left": bool(observation["contacts"].get("left", False)),
        },
    }


def run_physical_place(
    backend,
    *,
    object_name: str,
    target_xy,
    config: PhysicalCarryConfig | None = None,
    driver=None,
) -> dict:
    """Lower a held object onto support, release, and measure its final pose."""
    config = config or PhysicalCarryConfig()
    driver = driver or OfficialPhysicalCarryDriver()
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    hold_targets = driver.capture_hold_targets(backend)
    observation = driver.observe(backend, object_name)
    start_z = float(observation["object_pos"][2])
    previous_z = start_z
    final_distance = float(
        np.linalg.norm(np.asarray(observation["object_pos"][:2]) - target_xy)
    )
    steps = 0
    support_steps = 0
    support_detected = False
    driver.record_event(
        backend,
        "physical_place_start",
        object_name=object_name,
        target_xy=target_xy.tolist(),
    )

    if next_contact_stability(observation["contacts"], 0) == 0:
        return _place_result(
            success=False,
            failure_stage="contact",
            observation=observation,
            start_z=start_z,
            support_detected=False,
            final_distance=final_distance,
            steps=steps,
        )

    descent_steps = max(
        1,
        int(math.ceil(config.max_descent / max(config.descent_step, 1e-12))),
    )
    downward = {
        arm: np.array([0.0, 0.0, -config.descent_step], dtype=float)
        for arm in ("right", "left")
    }
    failure_stage = None
    for _ in range(descent_steps):
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas=downward,
            gripper_value=1.0,
        )
        steps += 1
        observation = driver.observe(backend, object_name)
        current_z = float(observation["object_pos"][2])
        if bool(step_info.get("collision", False)):
            failure_stage = "collision"
            break
        if next_contact_stability(observation["contacts"], 0) == 0:
            # Some factory surfaces are reached by a single physical impact:
            # the object drops onto the support before the gripper contacts
            # disappear.  Confirm a large drop has settled before accepting
            # that as a valid set-down; small contact loss remains a failure.
            descent = max(0.0, start_z - current_z)
            driver.record_event(
                backend,
                "physical_place_contact_loss",
                object_name=object_name,
                start_z=float(start_z),
                current_z=float(current_z),
                previous_z=float(previous_z),
                descent=float(descent),
                contacts={
                    arm: bool(observation["contacts"].get(arm, False))
                    for arm in ("right", "left")
                },
            )
            impact_threshold = max(
                0.05,
                float(config.minimum_descent_before_support) * 4.0,
            )
            if descent >= impact_threshold:
                impact_z = current_z
                stable_steps = 0
                support_tolerance = max(
                    float(config.support_motion_tolerance),
                    0.002,
                )
                for _ in range(config.support_stability_steps):
                    settle_info = driver.step(
                        backend,
                        object_name=object_name,
                        base_command=np.zeros(3, dtype=float),
                        hold_targets=hold_targets,
                        arm_world_deltas={
                            arm: np.zeros(3, dtype=float)
                            for arm in ("right", "left")
                        },
                        gripper_value=1.0,
                    )
                    steps += 1
                    observation = driver.observe(backend, object_name)
                    if bool(settle_info.get("collision", False)):
                        failure_stage = "collision"
                        break
                    settled_z = float(observation["object_pos"][2])
                    if abs(settled_z - impact_z) <= support_tolerance:
                        stable_steps += 1
                    else:
                        break
                if (
                    failure_stage is None
                    and stable_steps >= config.support_stability_steps
                ):
                    support_detected = True
                    break
            if failure_stage is None:
                failure_stage = "contact"
            break

        descent = max(0.0, start_z - current_z)
        vertical_motion = abs(current_z - previous_z)
        if (
            descent >= config.minimum_descent_before_support
            and vertical_motion <= config.support_motion_tolerance
        ):
            support_steps += 1
        else:
            support_steps = 0
        previous_z = current_z
        if support_steps >= config.support_stability_steps:
            support_detected = True
            break

    if failure_stage is None and not support_detected:
        failure_stage = "support"

    if failure_stage is not None:
        final_distance = float(
            np.linalg.norm(np.asarray(observation["object_pos"][:2]) - target_xy)
        )
        driver.record_event(
            backend,
            "physical_place_end",
            object_name=object_name,
            success=False,
            failure_stage=failure_stage,
            support_detected=support_detected,
        )
        return _place_result(
            success=False,
            failure_stage=failure_stage,
            observation=observation,
            start_z=start_z,
            support_detected=support_detected,
            final_distance=final_distance,
            steps=steps,
        )

    zero_arms = {
        arm: np.zeros(3, dtype=float)
        for arm in ("right", "left")
    }
    for _ in range(config.release_steps + config.settle_steps):
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas=zero_arms,
            gripper_value=-1.0,
        )
        steps += 1
        observation = driver.observe(backend, object_name)
        if bool(step_info.get("collision", False)):
            failure_stage = "collision"
            break

    final_distance = float(
        np.linalg.norm(np.asarray(observation["object_pos"][:2]) - target_xy)
    )
    if failure_stage is None and final_distance >= 0.8:
        failure_stage = "target_distance"
    if failure_stage is None and all(
        bool(observation["contacts"].get(arm, False))
        for arm in ("right", "left")
    ):
        failure_stage = "release"
    success = failure_stage is None
    driver.record_event(
        backend,
        "physical_place_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        support_detected=support_detected,
        final_distance=final_distance,
    )
    return _place_result(
        success=success,
        failure_stage=failure_stage,
        observation=observation,
        start_z=start_z,
        support_detected=support_detected,
        final_distance=final_distance,
        steps=steps,
    )


def run_scored_physical_release(
    backend,
    *,
    object_name: str,
    target_xy,
    release_steps: int = 40,
    settle_steps: int = 40,
    max_target_distance: float = 0.8,
    driver=None,
    before_release_fn=None,
) -> dict:
    """Open both grippers only after an attached object enters scoring range."""
    if int(release_steps) < 1 or int(settle_steps) < 0:
        raise ValueError("release_steps must be positive and settle_steps non-negative")
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    driver = driver or OfficialPhysicalCarryDriver()
    observation = driver.observe(backend, object_name)
    initial_distance = float(
        np.linalg.norm(
            np.asarray(observation["object_pos"], dtype=float)[:2] - target_xy
        )
    )
    if initial_distance >= float(max_target_distance):
        return {
            "success": False,
            "failure_stage": "target_distance",
            "initial_distance": initial_distance,
            "final_distance": initial_distance,
            "steps": 0,
            "contacts": dict(observation["contacts"]),
            "release_started": False,
        }

    driver.record_event(
        backend,
        "scored_attachment_release_start",
        object_name=object_name,
        initial_distance=initial_distance,
    )
    hold_targets = driver.capture_hold_targets(backend)
    if before_release_fn is not None:
        before_release_fn()

    zero_arms = {
        arm: np.zeros(3, dtype=float)
        for arm in ("right", "left")
    }
    failure_stage = None
    steps = 0
    for _ in range(int(release_steps) + int(settle_steps)):
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas=zero_arms,
            gripper_value=-1.0,
        )
        steps += 1
        observation = driver.observe(backend, object_name)
        if bool((step_info or {}).get("collision", False)):
            failure_stage = "collision"
            break

    final_distance = float(
        np.linalg.norm(
            np.asarray(observation["object_pos"], dtype=float)[:2] - target_xy
        )
    )
    contacts = {
        arm: bool(observation["contacts"].get(arm, False))
        for arm in ("right", "left")
    }
    if failure_stage is None and final_distance >= float(max_target_distance):
        failure_stage = "target_distance"
    if failure_stage is None and all(contacts.values()):
        failure_stage = "release"
    success = failure_stage is None
    driver.record_event(
        backend,
        "scored_attachment_release_end",
        object_name=object_name,
        success=success,
        failure_stage=failure_stage,
        final_distance=final_distance,
    )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "final_object_pos": np.asarray(
            observation["object_pos"], dtype=float
        ).tolist(),
        "steps": steps,
        "contacts": contacts,
        "release_started": True,
    }


def _navigation_retract_targets(
    *,
    base_xy,
    base_yaw: float,
    forward_m: float,
    lateral_m: float,
    target_z: float,
) -> dict[str, np.ndarray]:
    base = np.asarray(base_xy, dtype=float)
    values = np.asarray(
        [base_yaw, forward_m, lateral_m, target_z],
        dtype=float,
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
            [float(forward_m), local_lateral],
            dtype=float,
        )
        targets[arm] = np.array(
            [planar[0], planar[1], float(target_z)],
            dtype=float,
        )
    return targets


def _floor_push_staging_targets(
    *,
    object_xy,
    current_base_xy,
    push_direction_xy,
    base_standoff_m: float,
    orientation_clearance_m: float,
    lateral_offset_m: float | None,
    maximum_lateral_offset_m: float,
) -> dict:
    object_position = np.asarray(object_xy, dtype=float)
    base_position = np.asarray(current_base_xy, dtype=float)
    direction = np.asarray(push_direction_xy, dtype=float)
    if any(
        value.shape != (2,)
        for value in (object_position, base_position, direction)
    ):
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
    standoff = float(base_standoff_m)
    orientation_clearance = float(orientation_clearance_m)
    maximum_lateral = float(maximum_lateral_offset_m)
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (standoff, orientation_clearance, maximum_lateral)
    ):
        raise ValueError("floor push geometry parameters must be positive")

    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    if lateral_offset_m is None:
        raw_lateral = float(np.dot(base_position - object_position, left_axis))
        lateral_offset = float(
            np.clip(raw_lateral, -maximum_lateral, maximum_lateral)
        )
    else:
        lateral_offset = float(lateral_offset_m)
        if not np.isfinite(lateral_offset) or abs(lateral_offset) > maximum_lateral:
            raise ValueError("requested floor push lateral offset is invalid")
    stage_base_xy = (
        object_position - direction * standoff + left_axis * lateral_offset
    )
    orientation_base_xy = stage_base_xy - direction * orientation_clearance
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
    }


def _floor_base_reposition_targets(
    *,
    object_xy,
    current_base_xy,
    next_push_direction_xy,
    retreat_clearance_m: float,
    base_standoff_m: float,
    lateral_offset_m: float = 0.0,
    reverse_heading: bool = False,
) -> dict:
    object_position = np.asarray(object_xy, dtype=float)
    base_position = np.asarray(current_base_xy, dtype=float)
    direction = np.asarray(next_push_direction_xy, dtype=float)
    if any(
        value.shape != (2,)
        for value in (object_position, base_position, direction)
    ):
        raise ValueError("floor reposition positions must be planar vectors")
    if not all(
        np.all(np.isfinite(value))
        for value in (object_position, base_position, direction)
    ):
        raise ValueError("floor reposition positions must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("next floor push direction must be non-zero")
    direction /= direction_norm
    retreat_clearance = float(retreat_clearance_m)
    standoff = float(base_standoff_m)
    lateral_offset = float(lateral_offset_m)
    if (
        not np.isfinite(retreat_clearance)
        or not np.isfinite(standoff)
        or retreat_clearance <= standoff
        or standoff <= 0.0
        or not np.isfinite(lateral_offset)
    ):
        raise ValueError("retreat clearance must exceed positive standoff")

    outward = base_position - object_position
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-12:
        raise ValueError("floor reposition base must not coincide with object")
    retreat = object_position + outward / outward_norm * retreat_clearance
    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    stage = object_position - direction * standoff + left_axis * lateral_offset
    corner = retreat + (stage - object_position)
    target_yaw = float(np.arctan2(direction[1], direction[0]))
    if reverse_heading:
        target_yaw = float(
            np.arctan2(
                np.sin(target_yaw + np.pi),
                np.cos(target_yaw + np.pi),
            )
        )
    return {
        "direction": direction,
        "retreat_base_xy": retreat,
        "corner_base_xy": corner,
        "stage_base_xy": stage,
        "target_yaw": target_yaw,
    }


def _object_body_ids(raw_env, object_name: str) -> set[int]:
    model = raw_env.sim.model
    descendants = {int(raw_env.obj_body_id[object_name])}
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


def _object_all_robot_contacts(raw_env, object_name: str) -> tuple[str, ...]:
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


def _navigation_retract_for_floor_push(
    backend,
    *,
    forward_m: float,
    lateral_m: float,
    target_z: float,
    max_steps: int = 240,
) -> dict:
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        ScriptedGraspConfig,
    )

    base_xy, base_yaw = backend.get_base_pose()
    targets = _navigation_retract_targets(
        base_xy=base_xy,
        base_yaw=base_yaw,
        forward_m=forward_m,
        lateral_m=lateral_m,
        target_z=target_z,
    )
    driver = OfficialScriptedGraspDriver()
    helpers = driver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    # The arms are open and already clear of the tote after the physical drop;
    # accept a bounded 10 cm retreat error while still rejecting any collision.
    config = ScriptedGraspConfig(max_action=0.30, position_tolerance=0.10)
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "navigation_retract_start",
            targets={arm: target.tolist() for arm, target in targets.items()},
        )
    reached = bool(
        driver._move_to_targets(
            backend,
            targets,
            config,
            max_steps=int(max_steps),
            gripper_value=-1.0,
            tolerance=config.position_tolerance,
        )
    )
    final_positions = {
        arm: np.asarray(
            helpers["gripper_position"](raw_env, robot, arm),
            dtype=float,
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


def _extract_floor_push_object(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    macro_count: int,
    distance_m: float,
    world_direction,
    table_object_z: float,
    stroke_m: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    minimum_macro_progress_m: float = 0.02,
) -> dict:
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    direction = np.asarray(world_direction, dtype=float).reshape(2)
    direction /= float(np.linalg.norm(direction))
    macros = []
    completed = 0
    failure_stage = None

    for macro_index in range(int(macro_count)):
        macro_start = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=float,
        ).copy()
        transport = run_inchworm_transport(
            backend,
            object_name=object_name,
            travel_direction=direction,
            travel_distance=float(distance_m),
            minimum_object_z=float(table_object_z) + float(minimum_lift_m),
            config=InchwormCarryConfig(
                stroke_distance=float(stroke_m),
                stroke_vertical_feedforward=0.0,
                stroke_height_gain=0.0,
                reset_distance=float(reset_m),
                reset_max_gripper_drift=0.03,
                reset_arm_compensation_gain=1.0,
                reseat_steps=0,
                minimum_macro_progress=float(minimum_macro_progress_m),
                max_cycles=64,
            ),
        )
        place = None
        if bool(transport.get("success", False)):
            setdown_xy = np.asarray(
                raw_env.sim.data.body_xpos[body_id][:2],
                dtype=float,
            ).copy()
            place = run_physical_place(
                backend,
                object_name=object_name,
                target_xy=setdown_xy,
                config=PhysicalCarryConfig(max_descent=float(place_max_descent_m)),
            )
        contacts = dict(place.get("contacts", {})) if isinstance(place, Mapping) else {}
        released = bool(
            isinstance(place, Mapping)
            and place.get("success", False)
            and contacts
            and not any(bool(contacts.get(arm, False)) for arm in ("right", "left"))
        )
        success = bool(
            transport.get("success", False)
            and isinstance(place, Mapping)
            and place.get("success", False)
            and place.get("support_detected", False)
            and released
        )
        macro_end = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=float,
        ).copy()
        macros.append(
            {
                "macro": macro_index + 1,
                "success": success,
                "transport": transport,
                "place": place,
                "released": released,
                "start_object_position": macro_start.tolist(),
                "end_object_position": macro_end.tolist(),
            }
        )
        if not success:
            failure_stage = f"macro_{macro_index + 1}"
            break
        completed += 1
        competition_driver._physical_hold = None
        if completed >= int(macro_count):
            break
        if not competition_driver.move(
            source,
            carrying=False,
            object_name=object_name,
        ):
            failure_stage = f"regrasp_{macro_index + 1}:move"
            break
        grasp = competition_driver.grasp(source, object_name)
        macros[-1]["next_grasp"] = grasp
        if not verified_floor_route_grasp(grasp):
            failure_stage = f"regrasp_{macro_index + 1}:grasp"
            break

    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    return {
        "success": failure_stage is None and completed == int(macro_count),
        "failure_stage": failure_stage,
        "requested_macro_count": int(macro_count),
        "completed_macro_count": completed,
        "macros": macros,
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "attachment_activations": 0,
        "object_pose_writes": 0,
    }


def _l4_open_arm_sweep_plan(
    *,
    home_gripper_positions,
    stroke_distance_m: float,
    tip_after_push: bool = False,
) -> dict:
    """Build a left-arm westward push while the right arm remains neutral."""
    stroke = float(stroke_distance_m)
    if not np.isfinite(stroke) or stroke <= 0.0:
        raise ValueError("L4 sweep distance must be finite and positive")
    homes = {
        arm: np.asarray(home_gripper_positions[arm], dtype=float).reshape(3)
        for arm in ("right", "left")
    }
    push = {
        "name": "push",
        "max_steps": 600,
        "targets": {
            "left": homes["left"]
            + np.array([-stroke, 0.0, 0.0], dtype=float)
        },
        "relative_targets": {},
        "gripper_value": -1.0,
    }
    if tip_after_push:
        trailing_phases = (
            {
                "name": "close",
                "max_steps": 80,
                "targets": {},
                "relative_targets": {},
                "gripper_value": 1.0,
            },
            {
                "name": "tip_lift",
                "max_steps": 240,
                "targets": {},
                "relative_targets": {
                    "left": np.array([0.0, 0.0, 0.15], dtype=float)
                },
                "gripper_value": 1.0,
            },
            {
                "name": "release",
                "max_steps": 80,
                "targets": {},
                "relative_targets": {},
                "gripper_value": -1.0,
            },
            {
                "name": "settle",
                "max_steps": 120,
                "targets": {},
                "relative_targets": {},
                "gripper_value": -1.0,
            },
        )
    else:
        trailing_phases = (
            {
                "name": "retract",
                "max_steps": 240,
                "targets": {"left": homes["left"].copy()},
                "relative_targets": {},
                "gripper_value": -1.0,
            },
            {
                "name": "settle",
                "max_steps": 120,
                "targets": {},
                "relative_targets": {},
                "gripper_value": -1.0,
            },
        )
    return {
        "base_command": np.zeros(3, dtype=float),
        "phases": (push, *trailing_phases),
    }


def _l4_left_arm_realign_base_target(*, current_base_xy, object_xy) -> np.ndarray:
    """Align base y with the tipped container without reducing x clearance."""
    current_base = np.asarray(current_base_xy, dtype=float).reshape(2)
    object_position = np.asarray(object_xy, dtype=float).reshape(2)
    return np.array([current_base[0], object_position[1]], dtype=float)


def _l4_initial_base_prepush_profile(*, current_base_xy) -> dict:
    """Use the measured collision-free west prepush with a small edge margin."""
    current_base = np.asarray(current_base_xy, dtype=float).reshape(2)
    return {
        "target_base_xy": current_base - np.array([0.17, 0.0], dtype=float),
        "waypoint_tolerance_m": 0.005,
    }


def _l4_lower_bilateral_west_push_plan(
    *,
    home_gripper_positions,
    object_position,
    east_standoff_m: float,
    west_overshoot_m: float,
    clearance_height_m: float,
    contact_height_m: float,
) -> tuple[dict, ...]:
    """Stage both open grippers east of the box and push its wall west."""
    homes = {
        arm: np.asarray(home_gripper_positions[arm], dtype=float).reshape(3)
        for arm in ("right", "left")
    }
    object_position = np.asarray(object_position, dtype=float).reshape(3)
    east_standoff = float(east_standoff_m)
    west_overshoot = float(west_overshoot_m)
    clearance = float(clearance_height_m)
    contact_height = float(contact_height_m)
    if not all(
        np.isfinite(value) and value > 0.0
        for value in (
            east_standoff,
            west_overshoot,
            clearance,
            contact_height,
        )
    ):
        raise ValueError("L4 push distances must be finite and positive")
    raised = {
        arm: homes[arm] + np.array([0.0, 0.0, clearance])
        for arm in ("right", "left")
    }
    staged = {
        arm: np.array(
            [
                object_position[0] + east_standoff,
                homes[arm][1],
                raised[arm][2],
            ],
            dtype=float,
        )
        for arm in ("right", "left")
    }
    lowered = {
        arm: np.array(
            [staged[arm][0], staged[arm][1], object_position[2] + contact_height],
            dtype=float,
        )
        for arm in ("right", "left")
    }
    pushed = {
        arm: np.array(
            [object_position[0] - west_overshoot, lowered[arm][1], lowered[arm][2]],
            dtype=float,
        )
        for arm in ("right", "left")
    }

    def phase(name: str, right, left, max_steps: int = 180) -> dict:
        return {
            "name": name,
            "targets": {
                "right": np.asarray(right, dtype=float).copy(),
                "left": np.asarray(left, dtype=float).copy(),
            },
            "max_steps": int(max_steps),
        }

    return (
        phase("raise_both", raised["right"], raised["left"]),
        phase("stage_east", staged["right"], staged["left"], 240),
        phase("lower_both", lowered["right"], lowered["left"], 240),
        phase("push_west", pushed["right"], pushed["left"], 480),
    )


def _extract_l4_lower_container_to_floor(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    macro_count: int,
    distance_m: float,
    world_direction,
    table_object_z: float,
    stroke_m: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    minimum_macro_progress_m: float = 0.02,
) -> dict:
    """Release the lower container and push its exposed east wall west."""
    del source, macro_count, distance_m, world_direction, stroke_m, reset_m
    del minimum_lift_m, place_max_descent_m, minimum_macro_progress_m
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    marker = getattr(backend, "_mark_trajectory_event", None)
    driver = OfficialPhysicalCarryDriver()
    steps = 0
    phase_results = []
    failure_stage = None
    floor_transition = False
    collision = False

    hold_targets = driver.capture_hold_targets(backend)
    previous_position = np.asarray(
        raw_env.sim.data.body_xpos[body_id], dtype=float
    ).copy()
    release_stable_steps = 0
    release_steps = 0
    object_position = previous_position.copy()
    for _ in range(300):
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas={
                arm: np.zeros(3, dtype=float)
                for arm in ("right", "left")
            },
            gripper_value=-1.0,
        )
        steps += 1
        release_steps += 1
        collision = bool(step_info.get("collision", False))
        object_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        ).copy()
        contacts = _object_all_robot_contacts(raw_env, object_name)
        motion = float(np.linalg.norm(object_position - previous_position))
        release_stable_steps = (
            release_stable_steps + 1
            if not contacts and motion <= 0.002
            else 0
        )
        previous_position = object_position
        if collision or release_stable_steps >= 20:
            break
    if collision:
        failure_stage = "collision"
    elif release_stable_steps < 20:
        failure_stage = "release_settle"

    if failure_stage is None:
        for push_cycle in range(1, 3):
            observation = driver.observe(backend, object_name)
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            plan = _l4_lower_bilateral_west_push_plan(
                home_gripper_positions=observation["gripper_positions"],
                object_position=object_position,
                east_standoff_m=0.42,
                west_overshoot_m=0.35,
                clearance_height_m=0.12,
                contact_height_m=0.09,
            )
            for phase in plan:
                phase_start = np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).copy()
                hold_targets = driver.capture_hold_targets(backend)
                reached = False
                phase_steps = 0
                for _ in range(int(phase["max_steps"])):
                    observation = driver.observe(backend, object_name)
                    arm_deltas = {
                        arm: np.asarray(phase["targets"][arm], dtype=float)
                        - np.asarray(
                            observation["gripper_positions"][arm], dtype=float
                        )
                        for arm in ("right", "left")
                    }
                    if max(
                        float(np.linalg.norm(delta))
                        for delta in arm_deltas.values()
                    ) <= 0.01:
                        reached = True
                        break
                    step_info = driver.step(
                        backend,
                        object_name=object_name,
                        base_command=np.zeros(3, dtype=float),
                        hold_targets=hold_targets,
                        arm_world_deltas=arm_deltas,
                        gripper_value=-1.0,
                    )
                    steps += 1
                    phase_steps += 1
                    collision = bool(step_info.get("collision", False))
                    object_position = np.asarray(
                        raw_env.sim.data.body_xpos[body_id], dtype=float
                    ).copy()
                    floor_transition = bool(
                        float(object_position[2])
                        < float(table_object_z) - 0.30
                    )
                    if collision or floor_transition:
                        break
                phase_end = np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).copy()
                phase_name = f"{phase['name']}_{push_cycle}"
                phase_results.append(
                    {
                        "name": phase_name,
                        "steps": phase_steps,
                        "target_reached": reached,
                        "start_object_position": phase_start.tolist(),
                        "end_object_position": phase_end.tolist(),
                    }
                )
                if callable(marker):
                    marker(
                        "l4_lower_push_phase_end",
                        object_name=object_name,
                        phase=phase_name,
                        target_reached=reached,
                        steps=phase_steps,
                        object_position=phase_end.tolist(),
                    )
                if collision or floor_transition:
                    break
            if collision or floor_transition:
                break

    settled_floor_steps = 0
    if floor_transition and not collision:
        hold_targets = driver.capture_hold_targets(backend)
        previous_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        ).copy()
        for _ in range(300):
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=hold_targets,
                arm_world_deltas={
                    arm: np.zeros(3, dtype=float)
                    for arm in ("right", "left")
                },
                gripper_value=-1.0,
            )
            steps += 1
            collision = bool(step_info.get("collision", False))
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
            contacts = _object_all_robot_contacts(raw_env, object_name)
            motion = float(np.linalg.norm(object_position - previous_position))
            settled_floor_steps = (
                settled_floor_steps + 1
                if not contacts
                and motion <= 0.002
                and float(object_position[2]) < float(table_object_z) - 0.30
                else 0
            )
            previous_position = object_position
            if collision or settled_floor_steps >= 20:
                break

    competition_driver._physical_hold = None
    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    if collision:
        failure_stage = "collision"
    elif not floor_transition and failure_stage is None:
        failure_stage = "south_edge_push"
    elif floor_transition and settled_floor_steps < 20 and failure_stage is None:
        failure_stage = "floor_settle"
    success = bool(
        floor_transition
        and settled_floor_steps >= 20
        and failure_stage is None
        and not collision
    )
    if callable(marker):
        marker(
            "l4_lower_closed_carry_end",
            object_name=object_name,
            success=success,
            collision=collision,
            failure_stage=failure_stage,
            floor_transition_detected=floor_transition,
            steps=steps,
            end_object_position=end_object.tolist(),
        )
    return {
        "success": success,
        "failure_stage": (
            None
            if success
            else failure_stage or "south_edge_release"
        ),
        "collision": collision,
        "release_steps": release_steps,
        "release_stable_steps": release_stable_steps,
        "phases": phase_results,
        "floor_settle_steps": settled_floor_steps,
        "floor_transition_detected": floor_transition,
        "steps": steps,
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "attachment_activations": 0,
        "object_pose_writes": 0,
    }


def _extract_l4_container_to_floor(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    macro_count: int,
    distance_m: float,
    world_direction,
    table_object_z: float,
    stroke_m: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    minimum_macro_progress_m: float = 0.02,
) -> dict:
    """Open the L4 grasp and push the container over the table's west edge."""
    del source, macro_count, distance_m, world_direction, stroke_m, reset_m
    del minimum_lift_m, place_max_descent_m, minimum_macro_progress_m
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "l4_table_edge_push_start",
            object_name=object_name,
            direction=[-1.0, 0.0],
            method="physical_open_gripper_arm_sweep",
        )

    driver = OfficialPhysicalCarryDriver()
    hold_targets = driver.capture_hold_targets(backend)
    push_steps = 0
    released_steps = 0
    floor_transition = False
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    staged = False
    realigned = False
    sweep_started = False
    phase_results = []
    if not collision:
        for _ in range(60):
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=hold_targets,
                arm_world_deltas=None,
                gripper_value=-1.0,
            )
            push_steps += 1
            collision = bool(step_info.get("collision", False))
            if collision:
                break
        if not collision:
            current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
            prepush = _l4_initial_base_prepush_profile(
                current_base_xy=current_base
            )
            staged = bool(
                backend.follow_path(
                    [prepush["target_base_xy"]],
                    max_steps=1800,
                    waypoint_tolerance=prepush["waypoint_tolerance_m"],
                )
            )
            collision = bool(getattr(raw_env, "has_judge_collision", False))
        sweep_started = bool(staged and not collision)
    if sweep_started:
        for sweep_index in range(1, 3):
            hold_targets = driver.capture_hold_targets(backend)
            observation = driver.observe(backend, object_name)
            home_gripper_positions = {
                arm: np.asarray(
                    observation["gripper_positions"][arm], dtype=float
                ).copy()
                for arm in ("right", "left")
            }
            plan = _l4_open_arm_sweep_plan(
                home_gripper_positions=home_gripper_positions,
                stroke_distance_m=0.30,
                tip_after_push=sweep_index == 2,
            )
            if callable(marker):
                marker(
                    "l4_arm_sweep_pose",
                    object_name=object_name,
                    sweep_index=sweep_index,
                    object_position=np.asarray(
                        observation["object_pos"], dtype=float
                    ).tolist(),
                    gripper_positions={
                        arm: np.asarray(
                            observation["gripper_positions"][arm], dtype=float
                        ).tolist()
                        for arm in ("right", "left")
                    },
                )
            for phase in plan["phases"]:
                phase_start = np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).copy()
                executed_steps = 0
                target_reached = False
                phase_observation = driver.observe(backend, object_name)
                phase_targets = {
                    arm: np.asarray(target, dtype=float).copy()
                    for arm, target in phase["targets"].items()
                }
                for arm, relative_target in phase["relative_targets"].items():
                    phase_targets[arm] = np.asarray(
                        phase_observation["gripper_positions"][arm], dtype=float
                    ) + np.asarray(relative_target, dtype=float)
                for _ in range(int(phase["max_steps"])):
                    observation = driver.observe(backend, object_name)
                    arm_deltas = {
                        arm: np.asarray(target, dtype=float)
                        - np.asarray(
                            observation["gripper_positions"][arm], dtype=float
                        )
                        for arm, target in phase_targets.items()
                    }
                    if arm_deltas and max(
                        float(np.linalg.norm(delta))
                        for delta in arm_deltas.values()
                    ) <= 0.01:
                        target_reached = True
                        break
                    step_info = driver.step(
                        backend,
                        object_name=object_name,
                        base_command=plan["base_command"],
                        hold_targets=hold_targets,
                        arm_world_deltas=arm_deltas or None,
                        gripper_value=phase["gripper_value"],
                    )
                    push_steps += 1
                    executed_steps += 1
                    collision = bool(step_info.get("collision", False))
                    object_position = np.asarray(
                        raw_env.sim.data.body_xpos[body_id], dtype=float
                    ).copy()
                    contacts = _object_all_robot_contacts(raw_env, object_name)
                    released_steps = released_steps + 1 if not contacts else 0
                    floor_transition = bool(
                        floor_transition
                        or float(object_position[2])
                        < float(table_object_z) - 0.30
                    )
                    if collision or (floor_transition and released_steps >= 20):
                        break
                phase_end = np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).copy()
                final_observation = driver.observe(backend, object_name)
                phase_results.append(
                    {
                        "name": f"{phase['name']}_{sweep_index}",
                        "steps": executed_steps,
                        "target_reached": target_reached,
                        "start_object_position": phase_start.tolist(),
                        "end_object_position": phase_end.tolist(),
                        "final_gripper_positions": {
                            arm: np.asarray(
                                final_observation["gripper_positions"][arm],
                                dtype=float,
                            ).tolist()
                            for arm in ("right", "left")
                        },
                    }
                )
                if collision or (floor_transition and released_steps >= 20):
                    break
            if collision or (floor_transition and released_steps >= 20):
                break
            if sweep_index == 1:
                current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
                current_object = np.asarray(
                    raw_env.sim.data.body_xpos[body_id], dtype=float
                ).copy()
                realign_target = _l4_left_arm_realign_base_target(
                    current_base_xy=current_base,
                    object_xy=current_object[:2],
                )
                realigned = bool(
                    backend.follow_path(
                        [realign_target],
                        max_steps=1200,
                        waypoint_tolerance=0.03,
                    )
                )
                collision = bool(getattr(raw_env, "has_judge_collision", False))
                if not realigned or collision:
                    break

    if hasattr(competition_driver, "_physical_hold"):
        competition_driver._physical_hold = None
    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    success = bool(
        floor_transition
        and sweep_started
        and released_steps >= 20
        and not collision
    )
    if callable(marker):
        marker(
            "l4_table_edge_push_end",
            object_name=object_name,
            success=success,
            collision=collision,
            staged=staged,
            realigned=realigned,
            sweep_started=sweep_started,
            floor_transition_detected=floor_transition,
            push_steps=push_steps,
            phases=phase_results,
            end_object_position=end_object.tolist(),
        )
    return {
        "success": success,
        "failure_stage": None if success else "table_edge_push",
        "collision": collision,
        "staged": staged,
        "realigned": realigned,
        "sweep_started": sweep_started,
        "floor_transition_detected": floor_transition,
        "push_steps": push_steps,
        "phases": phase_results,
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "attachment_activations": 0,
        "object_pose_writes": 0,
    }


def _extract_green_tote_to_floor(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    macro_count: int,
    distance_m: float,
    world_direction,
    table_object_z: float,
    stroke_m: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    minimum_macro_progress_m: float = 0.02,
) -> dict:
    """Carry the green tote to the table edge, release, then push on the floor."""
    del competition_driver, source, macro_count, distance_m, world_direction
    del stroke_m, reset_m, minimum_lift_m, place_max_descent_m
    del minimum_macro_progress_m
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    marker = getattr(backend, "_mark_trajectory_event", None)
    driver = OfficialPhysicalCarryDriver()
    if callable(marker):
        marker(
            "table_edge_push_start",
            object_name=object_name,
            direction=[0.0, 1.0],
            method="physical_open_gripper_push_from_grasp_height",
        )
    hold_targets = driver.capture_hold_targets(backend)
    push_step_m = 0.02
    # The hidden green support is only 0.48 m deep, but the tote rolls while
    # the open fingers push it.  A diagonal second phase clears the support's
    # outer corner; continuing straight in +Y makes the tote roll back.
    max_steps = 1000
    straight_steps = 340
    floor_transition = False
    collision = False
    release_steps = 0
    for _ in range(max_steps):
        push_y = push_step_m
        push_x = 0.0
        if release_steps >= straight_steps:
            push_x = push_step_m
            push_y = push_step_m * 0.75
        arm_deltas = {
            "right": np.array([push_x, push_y, 0.0], dtype=float),
            "left": np.array([push_x, push_y, 0.0], dtype=float),
        }
        if release_steps >= straight_steps:
            # Once the tote reaches the corner, use the outer left finger as
            # a controlled wedge and keep the opposite finger out of the way.
            arm_deltas["right"] = np.zeros(3, dtype=float)
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=np.zeros(3, dtype=float),
            hold_targets=hold_targets,
            arm_world_deltas=arm_deltas,
            gripper_value=-1.0,
        )
        release_steps += 1
        collision = bool(step_info.get("collision", False))
        object_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        ).copy()
        if collision:
            break
        if float(object_position[2]) < float(table_object_z) - 0.30:
            floor_transition = True
            break

    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    success = bool(floor_transition and not collision)
    if callable(marker):
        marker(
            "table_edge_push_end",
            object_name=object_name,
            success=success,
            collision=collision,
            floor_transition_detected=floor_transition,
            steps=release_steps,
            end_object_position=end_object.tolist(),
        )
    return {
        "success": success,
        "failure_stage": None if success else "table_edge_push",
        "place": None,
        "floor_transition_detected": floor_transition,
        "collision": collision,
        "steps": release_steps,
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "attachment_activations": 0,
        "object_pose_writes": 0,
    }


def _extract_blue_tote_to_floor(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    macro_count: int,
    distance_m: float,
    world_direction,
    table_object_z: float,
    stroke_m: float,
    reset_m: float,
    minimum_lift_m: float,
    place_max_descent_m: float,
    minimum_macro_progress_m: float = 0.02,
) -> dict:
    """Reposition north of a side-table tote and physically push it south."""
    del competition_driver, source, macro_count, distance_m, world_direction
    del stroke_m, reset_m, minimum_lift_m, place_max_descent_m
    del minimum_macro_progress_m
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        ScriptedGraspConfig,
        apply_object_grasp_profile,
    )
    from robot_agent.skills.competition_navigation import orient_base

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    marker = getattr(backend, "_mark_trajectory_event", None)
    collision = False
    failure_stage = None
    push_steps = 0
    if callable(marker):
        marker(
            "side_table_north_push_start",
            object_name=object_name,
            direction=[0.0, -1.0],
            method="physical_north_side_open_gripper_push",
        )

    retract = _navigation_retract_for_floor_push(
        backend,
        forward_m=0.20,
        lateral_m=0.15,
        target_z=1.55,
    )
    retract_safe = bool(
        not retract.get("collision", False)
        and (
            retract.get("success", False)
            or float(retract.get("maximum_error_m", math.inf)) < 0.30
        )
    )
    if not retract_safe:
        failure_stage = "release_retract"

    object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
    detour_x = max(float(object_position[0]) + 1.20, 1.40)
    north_y = max(float(object_position[1]) + 1.20, 9.70)
    staged = False
    if failure_stage is None:
        staged = bool(
            backend.follow_path(
                [
                    np.array([detour_x, current_base[1]], dtype=float),
                    np.array([detour_x, north_y], dtype=float),
                    np.array([object_position[0], north_y], dtype=float),
                ],
                max_steps=3000,
                waypoint_tolerance=0.03,
            )
        )
        collision = bool(getattr(raw_env, "has_judge_collision", False))
        if not staged or collision:
            failure_stage = "north_stage"

    oriented = False
    if failure_stage is None:
        oriented = bool(orient_base(backend, -math.pi / 2.0))
        collision = bool(getattr(raw_env, "has_judge_collision", False))
        if not oriented or collision:
            failure_stage = "north_orient"

    scripted = OfficialScriptedGraspDriver()
    targets_reached = False
    floor_transition = False
    if failure_stage is None:
        object_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=float,
        ).copy()
        config = apply_object_grasp_profile(
            ScriptedGraspConfig(
                position_tolerance=0.06,
                approach_tolerance=0.06,
                pregrasp_steps=500,
                max_action=0.30,
            ),
            object_name,
        )
        targets = {
            "right": np.array(
                [object_position[0] + 0.16, object_position[1] + 0.32, object_position[2] + 0.04],
                dtype=float,
            ),
            "left": np.array(
                [object_position[0] - 0.16, object_position[1] + 0.32, object_position[2] + 0.04],
                dtype=float,
            ),
        }
        targets_reached = bool(
            scripted._move_to_targets(
                backend,
                targets,
                config,
                max_steps=500,
                gripper_value=-1.0,
                tolerance=0.06,
            )
        )
        collision = bool(getattr(raw_env, "has_judge_collision", False))
        object_position = np.asarray(
            raw_env.sim.data.body_xpos[body_id],
            dtype=float,
        ).copy()
        floor_transition = bool(
            float(object_position[2]) < float(table_object_z) - 0.30
        )
        if (not targets_reached and not floor_transition) or collision:
            failure_stage = "north_push_targets"

    if failure_stage is None and not floor_transition:
        driver = OfficialPhysicalCarryDriver()
        hold_targets = driver.capture_hold_targets(backend)
        push_start_y = float(raw_env.sim.data.body_xpos[body_id][1])
        for step in range(1200):
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.zeros(3, dtype=float),
                hold_targets=hold_targets,
                arm_world_deltas={
                    "right": np.array([0.0, -0.002, 0.0], dtype=float),
                    "left": np.array([0.0, -0.002, 0.0], dtype=float),
                },
                gripper_value=-1.0,
            )
            push_steps = step + 1
            collision = bool(step_info.get("collision", False))
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id],
                dtype=float,
            ).copy()
            if collision:
                failure_stage = "north_push_collision"
                break
            floor_transition = bool(
                float(object_position[2]) < float(table_object_z) - 0.30
            )
            if floor_transition:
                break
            if push_start_y - float(object_position[1]) > 0.90:
                failure_stage = "north_push_progress"
                break
        if failure_stage is None and not floor_transition:
            failure_stage = "floor_transition"

    end_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    success = bool(floor_transition and not collision and failure_stage is None)
    if callable(marker):
        marker(
            "side_table_north_push_end",
            object_name=object_name,
            success=success,
            failure_stage=failure_stage,
            collision=collision,
            staged=staged,
            oriented=oriented,
            targets_reached=targets_reached,
            push_steps=push_steps,
            floor_transition_detected=floor_transition,
            end_object_position=end_object.tolist(),
        )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "collision": collision,
        "retract": retract,
        "retract_safe": retract_safe,
        "staged": staged,
        "oriented": oriented,
        "targets_reached": targets_reached,
        "push_steps": push_steps,
        "floor_transition_detected": floor_transition,
        "navigation_ready": success,
        "start_object_position": start_object.tolist(),
        "end_object_position": end_object.tolist(),
        "attachment_activations": 0,
        "object_pose_writes": 0,
    }


def verified_floor_route_grasp(result: Mapping) -> bool:
    contacts = result.get("contacts", {}) if isinstance(result, Mapping) else {}
    return bool(
        isinstance(result, Mapping)
        and result.get("success", False)
        and result.get("lift_success", False)
        and all(bool(contacts.get(arm, False)) for arm in ("right", "left"))
    )


def _reposition_base_for_floor_push(
    backend,
    object_name: str,
    *,
    direction_xy,
    retreat_clearance_m: float,
    base_standoff_m: float,
    lateral_offset_m: float,
    retract_forward_m: float,
    retract_lateral_m: float,
    retract_target_z: float,
    minimum_retract_z_m: float,
    skip_retract: bool = False,
    reverse_pusher: bool = False,
    positive_x_detour_m: float | None = None,
) -> dict:
    from robot_agent.skills.competition_navigation import orient_base

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    targets = _floor_base_reposition_targets(
        object_xy=object_position[:2],
        current_base_xy=np.asarray(backend.get_base_pose()[0], dtype=float),
        next_push_direction_xy=direction_xy,
        retreat_clearance_m=retreat_clearance_m,
        base_standoff_m=base_standoff_m,
        lateral_offset_m=lateral_offset_m,
        reverse_heading=reverse_pusher,
    )
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_base_reposition_start",
            object_name=object_name,
            direction=targets["direction"].tolist(),
            retreat_base_xy=targets["retreat_base_xy"].tolist(),
            corner_base_xy=targets["corner_base_xy"].tolist(),
        )
    # Route above the released tote before changing lateral side.  The former
    # diagonal corner path could sweep the torso through the tote and move it
    # without registering a judge collision.
    if float(targets["direction"][1]) > 0.5:
        clearance_y = min(
            float(targets["stage_base_xy"][1]),
            float(object_position[1]) - 0.75,
        )
    else:
        clearance_y = max(
            float(targets["stage_base_xy"][1]),
            float(object_position[1]) + 0.75,
        )
    orientation_stage_xy = np.array(
        [targets["stage_base_xy"][0], clearance_y],
        dtype=float,
    )
    if positive_x_detour_m is not None:
        detour_x = float(positive_x_detour_m)
        if not np.isfinite(detour_x):
            raise ValueError("positive x detour must be finite")
        current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
        safe_reposition_path = [
            np.array([detour_x, current_base[1]], dtype=float),
            np.array([detour_x, clearance_y], dtype=float),
            orientation_stage_xy,
        ]
    else:
        safe_reposition_path = [
            targets["retreat_base_xy"],
            np.array([targets["retreat_base_xy"][0], clearance_y], dtype=float),
            orientation_stage_xy,
        ]
    retreat_reached = bool(
        backend.follow_path(
            safe_reposition_path,
            max_steps=2400,
            waypoint_tolerance=0.03,
        )
    )
    # Rotate while the torso is still at least 0.75 m from the tote, on the
    # side opposite the requested push.  Rotating after entering the final
    # standoff sweeps the square torso through the tote or the target table.
    oriented = bool(retreat_reached and orient_base(backend, targets["target_yaw"]))
    if oriented and not skip_retract:
        retract = _navigation_retract_for_floor_push(
            backend,
            forward_m=retract_forward_m,
            lateral_m=retract_lateral_m,
            target_z=retract_target_z,
        )
    elif oriented:
        # The first floor stage already lifted and opened both arms.  Repeating
        # an XY arm sweep here can touch the released tote and move it without
        # a judge collision, so later route stages keep the existing posture.
        retract = {
            "success": True,
            "collision": bool(getattr(raw_env, "has_judge_collision", False)),
            "skipped": True,
        }
    else:
        retract = None
    positions = retract.get("final_positions", {}) if isinstance(retract, Mapping) else {}
    minimum_z = min(
        (
            float(np.asarray(position, dtype=float)[2])
            for position in positions.values()
            if np.asarray(position, dtype=float).shape == (3,)
        ),
        default=float("-inf"),
    )
    retract_safe = bool(
        isinstance(retract, Mapping)
        and not retract.get("collision", False)
        and (retract.get("success", False) or minimum_z >= float(minimum_retract_z_m))
    )
    refined_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    refined = _floor_base_reposition_targets(
        object_xy=refined_object[:2],
        current_base_xy=np.asarray(backend.get_base_pose()[0], dtype=float),
        next_push_direction_xy=direction_xy,
        retreat_clearance_m=retreat_clearance_m,
        base_standoff_m=base_standoff_m,
        lateral_offset_m=lateral_offset_m,
    )
    stage_reached = bool(
        retract_safe
        and backend.follow_path(
            [refined["stage_base_xy"]],
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
        )
    return {
        "success": success,
        "collision": collision,
        "retreat_reached": retreat_reached,
        "oriented": oriented,
        "retract": retract,
        "retract_minimum_z_m": minimum_z,
        "retract_clearance_safe": retract_safe,
        "stage_reached": stage_reached,
    }


def _physical_base_push_segment(
    backend,
    object_name: str,
    *,
    direction_xy,
    distance_m: float,
    base_speed_m_s: float,
    tracking_gain: float,
    alignment_gain: float,
    tracking_deadband_m: float,
    maximum_base_object_offset_m: float,
    maximum_lateral_speed_m_s: float,
    minimum_base_x_m: float | None,
    maximum_lateral_drift_m: float,
    max_steps: int,
    stop_target_xy=None,
    stop_target_distance_m: float | None = None,
) -> dict:
    direction = np.asarray(direction_xy, dtype=float).reshape(2)
    direction /= float(np.linalg.norm(direction))
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    left_axis = np.array([-direction[1], direction[0]], dtype=float)
    nominal_lateral_offset = float(
        np.dot(start_base_xy - start_object[:2], left_axis)
    )
    driver = OfficialPhysicalCarryDriver()
    hold_targets = driver.capture_hold_targets(backend)
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_base_push_segment_start",
            object_name=object_name,
            direction=direction.tolist(),
            requested_distance_m=float(distance_m),
        )

    maximum_contact_steps = 0
    stable_contact_steps = 0
    no_contact_steps = 0
    object_progress = 0.0
    lateral_drift = 0.0
    lateral_error = 0.0
    base_object_lateral_offset = 0.0
    base_progress = 0.0
    target_reached = False
    target_distance = math.inf
    if (stop_target_xy is None) != (stop_target_distance_m is None):
        raise ValueError("floor segment target and stop distance must be provided together")
    stop_target = (
        None
        if stop_target_xy is None
        else np.asarray(stop_target_xy, dtype=float).reshape(2)
    )
    if stop_target is not None:
        if (
            not np.all(np.isfinite(stop_target))
            or not np.isfinite(float(stop_target_distance_m))
            or float(stop_target_distance_m) <= 0.0
        ):
            raise ValueError("floor segment stop target must be finite and positive")
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    failure_stage = "collision" if collision else None
    steps = 0
    observations = []
    if failure_stage is None:
        for step in range(int(max_steps)):
            _, base_yaw = backend.get_base_pose()
            world_velocity = floor_base_tracking_velocity(
                push_direction_xy=direction,
                lateral_error_m=lateral_error,
                base_object_lateral_offset_m=base_object_lateral_offset,
                forward_speed_m_s=base_speed_m_s,
                lateral_gain=tracking_gain,
                alignment_gain=alignment_gain,
                lateral_deadband_m=tracking_deadband_m,
                maximum_base_object_offset_m=maximum_base_object_offset_m,
                maximum_lateral_speed_m_s=maximum_lateral_speed_m_s,
            )
            if minimum_base_x_m is not None:
                minimum_x = float(minimum_base_x_m)
                if not np.isfinite(minimum_x):
                    raise ValueError("minimum base x must be finite")
                current_base_x = float(backend.get_base_pose()[0][0])
                # Keep a small active margin instead of waiting for the torso
                # box to touch the production-line proxy.  This remains a
                # commanded base velocity; no robot or object state is set.
                if current_base_x < minimum_x + 0.03:
                    world_velocity[0] = max(
                        float(world_velocity[0]),
                        float(maximum_lateral_speed_m_s),
                    )
            base_velocity = world_velocity_to_base_frame(world_velocity, base_yaw)
            step_info = driver.step(
                backend,
                object_name=object_name,
                base_command=np.array([base_velocity[0], base_velocity[1], 0.0]),
                hold_targets=hold_targets,
                arm_world_deltas=None,
                gripper_value=-1.0,
                base_control_dt=0.05,
            )
            steps = step + 1
            collision = bool(step_info.get("collision", False))
            contacts = _object_all_robot_contacts(raw_env, object_name)
            stable_contact_steps = stable_contact_steps + 1 if contacts else 0
            maximum_contact_steps = max(maximum_contact_steps, stable_contact_steps)
            no_contact_steps = 0 if contacts else no_contact_steps + 1
            object_position = np.asarray(
                raw_env.sim.data.body_xpos[body_id],
                dtype=float,
            ).copy()
            if stop_target is not None:
                target_distance = float(
                    np.linalg.norm(object_position[:2] - stop_target)
                )
                target_reached = bool(
                    target_distance <= float(stop_target_distance_m)
                )
            object_delta = object_position[:2] - start_object[:2]
            object_progress = float(np.dot(object_delta, direction))
            lateral_error = float(np.dot(object_delta, left_axis))
            lateral_drift = abs(lateral_error)
            base_delta = np.asarray(backend.get_base_pose()[0], dtype=float) - start_base_xy
            base_progress = float(np.dot(base_delta, direction))
            base_object_lateral_offset = float(
                np.dot(
                    np.asarray(backend.get_base_pose()[0], dtype=float)
                    - object_position[:2],
                    left_axis,
                )
                - nominal_lateral_offset
            )
            if step % 25 == 0 or collision or object_progress >= float(distance_m):
                observations.append(
                    {
                        "step": steps,
                        "object_position": object_position.tolist(),
                        "object_progress_m": object_progress,
                        "lateral_drift_m": lateral_drift,
                        "contacts": list(contacts),
                        "judge_collision": collision,
                    }
                )
            if collision:
                failure_stage = "collision"
                break
            if lateral_drift > float(maximum_lateral_drift_m):
                failure_stage = "lateral_drift"
                break
            if target_reached and maximum_contact_steps >= 20:
                failure_stage = None
                break
            if object_progress >= float(distance_m) and maximum_contact_steps >= 20:
                failure_stage = None
                break
            if maximum_contact_steps >= 20 and no_contact_steps >= 80:
                failure_stage = "contact_lost"
                break
        else:
            failure_stage = "timeout"

    success = bool(
        failure_stage is None
        and (object_progress >= float(distance_m) or target_reached)
        and maximum_contact_steps >= 20
        and not collision
    )
    final_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
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
        "target_reached": target_reached,
        "target_distance_m": target_distance,
        "start_object_position": start_object.tolist(),
        "end_object_position": final_object.tolist(),
        "direction": direction.tolist(),
        "requested_distance_m": float(distance_m),
        "observations": observations,
    }


def _run_floor_corridor_push(
    backend,
    *,
    object_name: str,
    push_direction,
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
    route_target_xy,
    route_corridor_y: float,
    initial_clearance_m: float,
    route_arrival_radius_m: float,
    route_arrival_margin_m: float,
    route_reposition_clearance_m: float,
    route_reposition_lateral_offset_m: float,
    route_south_reposition_lateral_offset_m: float | None,
    route_south_tracking_gain: float | None,
    route_final_approach_reverse_pusher: bool,
    route_final_side_approach_x: float | None,
    route_reverse_switch_y: float | None,
    route_lateral_clearance_m: float,
    route_minimum_retract_z_m: float,
    route_minimum_base_x_m: float | None,
    tracking_gain: float,
    alignment_gain: float,
    tracking_deadband_m: float,
    maximum_base_object_offset_m: float,
    maximum_lateral_speed_m_s: float,
) -> dict:
    del torso_drop_m, face_offset_m, hand_separation_m, hand_height_m
    del precontact_clearance_m, push_distance_m
    if not base_pusher:
        raise ValueError("verified floor corridor route requires the base pusher")
    from robot_agent.skills.competition_navigation import orient_base

    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    direction = np.asarray(push_direction, dtype=float).reshape(2)
    direction /= float(np.linalg.norm(direction))
    if float(initial_clearance_m) > 0.0 and direction[0] > 0.5:
        # The upper input AABB is immediately behind the dropped tote.  Move
        # the base out in +X first, then move above the station before any
        # orientation or arm-retraction command is issued.
        current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
        safe_base = current_base.copy()
        safe_base[0] += 0.80
        safe_base[1] = max(safe_base[1], float(start_object[1]) + 0.10)
        if not bool(
            backend.follow_path(
                [
                    np.array([safe_base[0], current_base[1]], dtype=float),
                    safe_base,
                ],
                max_steps=1800,
                waypoint_tolerance=0.03,
            )
        ):
            return {
                "success": False,
                "failure_stage": "safe_base_clearance",
                "collision": bool(getattr(raw_env, "has_judge_collision", False)),
            }
    targets = _floor_push_staging_targets(
        object_xy=start_object[:2],
        current_base_xy=np.asarray(backend.get_base_pose()[0], dtype=float),
        push_direction_xy=direction,
        base_standoff_m=base_standoff_m,
        orientation_clearance_m=orientation_clearance_m,
        lateral_offset_m=lateral_offset_m,
        maximum_lateral_offset_m=maximum_lateral_offset_m,
    )
    marker = getattr(backend, "_mark_trajectory_event", None)
    if callable(marker):
        marker(
            "floor_corridor_push_start",
            object_name=object_name,
            push_direction=targets["direction"].tolist(),
            stage_base_xy=targets["stage_base_xy"].tolist(),
        )
    blue_side_table_route = "blue_tote_b01" in str(object_name).lower()
    if blue_side_table_route:
        current_base = np.asarray(backend.get_base_pose()[0], dtype=float)
        detour_x = max(float(start_object[0]) + 1.20, 1.40)
        safe_y = float(targets["stage_base_xy"][1])
        escape_reached = bool(
            backend.follow_path(
                [
                    np.array([detour_x, current_base[1]], dtype=float),
                    np.array([detour_x, safe_y], dtype=float),
                ],
                max_steps=2400,
                waypoint_tolerance=0.03,
            )
        )
        orientation_reached = escape_reached
        oriented = bool(
            orientation_reached and orient_base(backend, targets["target_yaw"])
        )
    else:
        escape_reached = bool(
            backend.follow_path(
                [targets["escape_base_xy"]],
                max_steps=1200,
                waypoint_tolerance=0.03,
            )
        )
        orientation_reached = bool(
            escape_reached
            and backend.follow_path(
                [targets["orientation_base_xy"]],
                max_steps=1200,
                waypoint_tolerance=0.03,
            )
        )
        oriented = bool(
            orientation_reached and orient_base(backend, targets["target_yaw"])
        )
    if oriented and blue_side_table_route:
        retract = {
            "success": True,
            "collision": False,
            "skipped": True,
            "reason": "side_table_extractor_open_posture",
        }
    elif oriented:
        retract = _navigation_retract_for_floor_push(
            backend,
            forward_m=oriented_retract_forward_m,
            lateral_m=oriented_retract_lateral_m,
            target_z=oriented_retract_target_z,
        )
    else:
        retract = None
    interaction_start = np.asarray(
        raw_env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    route = floor_base_target_route(
        start_object_xy=interaction_start[:2],
        target_xy=route_target_xy,
        corridor_y=route_corridor_y,
        arrival_radius_m=route_arrival_radius_m,
        arrival_margin_m=route_arrival_margin_m,
        initial_clearance_m=initial_clearance_m,
        initial_push_direction_xy=direction,
        reverse_switch_y=route_reverse_switch_y,
        lateral_clearance_m=route_lateral_clearance_m,
        final_side_approach_x=route_final_side_approach_x,
    )
    if not np.allclose(
        route["segments"][0]["direction"],
        targets["direction"],
        atol=1e-9,
    ):
        raise ValueError("initial push direction does not match target route")
    if isinstance(retract, Mapping) and retract.get("success", False):
        targets = _floor_push_staging_targets(
            object_xy=interaction_start[:2],
            current_base_xy=np.asarray(backend.get_base_pose()[0], dtype=float),
            push_direction_xy=push_direction,
            base_standoff_m=base_standoff_m,
            orientation_clearance_m=orientation_clearance_m,
            lateral_offset_m=lateral_offset_m,
            maximum_lateral_offset_m=maximum_lateral_offset_m,
        )
    stage_reached = bool(
        isinstance(retract, Mapping)
        and retract.get("success", False)
        and backend.follow_path(
            [targets["stage_base_xy"]],
            max_steps=1200,
            waypoint_tolerance=0.03,
        )
    )
    collision = bool(getattr(raw_env, "has_judge_collision", False))
    if not escape_reached:
        failure_stage = "escape_stage_base"
    elif not orientation_reached:
        failure_stage = "orientation_stage_base"
    elif not oriented:
        failure_stage = "orient"
    elif not isinstance(retract, Mapping) or not retract.get("success", False):
        failure_stage = "oriented_retract"
    elif not stage_reached:
        failure_stage = "stage_base"
    elif collision:
        failure_stage = "collision"
    else:
        failure_stage = None

    route_segments = []
    total_steps = 0
    total_contacts = 0
    maximum_lateral_drift = 0.0
    success = failure_stage is None
    for route_index, planned_segment in enumerate(route["segments"], start=1):
        if not success:
            break
        direction = np.asarray(planned_segment["direction"], dtype=float)
        current_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
        desired_end = np.asarray(planned_segment["end_object_xy"], dtype=float)
        remaining = float(np.dot(desired_end - current_object[:2], direction))
        if remaining <= 0.01:
            route_segments.append(
                {
                    "index": route_index,
                    "success": True,
                    "skipped": True,
                    "requested_distance_m": remaining,
                }
            )
            continue
        reposition = None
        previous_direction = (
            np.asarray(route["segments"][route_index - 2]["direction"], dtype=float)
            if route_index > 1
            else None
        )
        continuing_same_direction = bool(
            previous_direction is not None
            and np.allclose(direction, previous_direction, atol=1e-9)
        )
        if route_index > 1 and not continuing_same_direction:
            final_approach_reverse_pusher = bool(
                route_final_approach_reverse_pusher
                and route_index == len(route["segments"])
            )
            reposition_lateral_offset = route_reposition_lateral_offset_m
            if (
                direction[1] < -0.5
                and float(route_lateral_clearance_m) > 0.0
            ):
                # After the eastward clearance stroke the tote and robot are
                # safely outside the line proxy.  Center the southbound push;
                # retaining the positive-X offset loads the tote's right
                # corner and creates a persistent westward drift.
                reposition_lateral_offset = 0.0
            if blue_side_table_route and direction[1] < -0.5:
                # Bracketing runs at 0.00 m and 0.05 m produced opposite
                # lateral drifts.  Use their midpoint for the long southbound
                # stroke instead of relaxing the collision guard.
                reposition_lateral_offset = float(
                    route_south_reposition_lateral_offset_m or 0.0
                )
            if (
                direction[1] > 0.5
                and float(route_lateral_clearance_m) > 0.0
            ):
                # The final northbound stroke already starts on the output's
                # target x.  Centering preserves that alignment; a positive
                # route offset places the base west of the tote and steers it
                # away from the target.
                reposition_lateral_offset = 0.0
            if route_index == 2 and route_reverse_switch_y is not None:
                # The production-line proxy ends at x=11.901 m and the fixed
                # torso box has a 0.25 m half-width.  A 0.10 m positive-X
                # offset preserves clearance at the measured tote pose while
                # keeping the flat pusher face engaged instead of its corner.
                reposition_lateral_offset = max(
                    float(reposition_lateral_offset),
                    0.10,
                )
            reposition = _reposition_base_for_floor_push(
                backend,
                object_name,
                direction_xy=direction,
                retreat_clearance_m=route_reposition_clearance_m,
                base_standoff_m=base_standoff_m,
                lateral_offset_m=reposition_lateral_offset,
                retract_forward_m=oriented_retract_forward_m,
                retract_lateral_m=oriented_retract_lateral_m,
                retract_target_z=oriented_retract_target_z,
                minimum_retract_z_m=route_minimum_retract_z_m,
                skip_retract=True,
                reverse_pusher=final_approach_reverse_pusher,
                positive_x_detour_m=(
                    max(float(current_object[0]) + 0.80, 1.40)
                    if blue_side_table_route and direction[1] < -0.5
                    else None
                ),
            )
            if not reposition["success"]:
                failure_stage = f"route_{route_index}:reposition"
                collision = bool(reposition["collision"])
                success = False
                route_segments.append(
                    {
                        "index": route_index,
                        "success": False,
                        "failure_stage": failure_stage,
                        "reposition": reposition,
                    }
                )
                break
        southbound_clearance_segment = bool(
            direction[1] < -0.5
            and (
                float(route_lateral_clearance_m) > 0.0
                or blue_side_table_route
            )
        )
        westbound_clearance_segment = bool(
            direction[0] < -0.5
            and float(route_lateral_clearance_m) > 0.0
        )
        segment = _physical_base_push_segment(
            backend,
            object_name,
            direction_xy=direction,
            distance_m=remaining,
            base_speed_m_s=(
                0.030 if westbound_clearance_segment else base_speed_m_s
            ),
            tracking_gain=(
                float(route_south_tracking_gain)
                if direction[1] < -0.5
                and route_south_tracking_gain is not None
                else tracking_gain
            ),
            alignment_gain=(
                0.20 if southbound_clearance_segment else alignment_gain
            ),
            tracking_deadband_m=tracking_deadband_m,
            maximum_base_object_offset_m=maximum_base_object_offset_m,
            maximum_lateral_speed_m_s=(
                0.015
                if southbound_clearance_segment
                else maximum_lateral_speed_m_s
            ),
            minimum_base_x_m=(
                route_minimum_base_x_m
                if direction[1] < -0.5
                else None
            ),
            maximum_lateral_drift_m=(
                1.00
                if blue_side_table_route and direction[0] > 0.5
                else 0.30
            ),
            max_steps=max_steps,
            stop_target_xy=(
                route["target_xy"]
                if route_index == len(route["segments"])
                else None
            ),
            stop_target_distance_m=(
                float(route_arrival_radius_m) - float(route_arrival_margin_m)
                if route_index == len(route["segments"])
                else None
            ),
        )
        segment["index"] = route_index
        segment["reposition"] = reposition
        route_segments.append(segment)
        total_steps += int(segment["steps"])
        total_contacts += int(segment["physical_contact_steps"])
        maximum_lateral_drift = max(
            maximum_lateral_drift,
            float(segment["lateral_drift_m"]),
        )
        collision = bool(segment["collision"])
        if not segment["success"]:
            success = False
            failure_stage = f"route_{route_index}:{segment['failure_stage'] or 'unknown'}"

    final_object = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()
    final_distance = float(
        np.linalg.norm(final_object[:2] - np.asarray(route["target_xy"], dtype=float))
    )
    if success and final_distance >= float(route_arrival_radius_m):
        success = False
        failure_stage = "target_distance"
    if callable(marker):
        marker(
            "floor_corridor_push_end",
            object_name=object_name,
            success=success,
            failure_stage=failure_stage,
            physical_contact_steps=total_contacts,
            final_target_distance_m=final_distance,
        )
    return {
        "success": success,
        "failure_stage": failure_stage,
        "pusher": "base",
        "escape_stage_reached": escape_reached,
        "orientation_stage_reached": orientation_reached,
        "oriented": oriented,
        "oriented_retract": retract,
        "stage_reached": stage_reached,
        "collision": collision,
        "steps": total_steps,
        "physical_contact_steps": total_contacts,
        "lateral_drift_m": maximum_lateral_drift,
        "start_object_position": start_object.tolist(),
        "interaction_start_object_position": interaction_start.tolist(),
        "end_object_position": final_object.tolist(),
        "route_plan": route,
        "route_segments": route_segments,
        "final_target_distance_m": final_distance,
    }


def run_physical_floor_route(
    backend,
    *,
    competition_driver,
    source: str,
    object_name: str,
    target_xy,
    table_object_z: float,
    _extract_and_setdown=None,
    _navigation_retract=None,
    _floor_push=None,
) -> dict:
    """Run the verified L1 grasp-setdown and contact-push route."""
    if _extract_and_setdown is not None:
        extract_and_setdown = _extract_and_setdown
    elif str(object_name).lower() == "green_tote_b01_upper":
        extract_and_setdown = _extract_green_tote_to_floor
    elif str(object_name).lower() == "blue_container_h01_back_lower":
        extract_and_setdown = _extract_l4_lower_container_to_floor
    elif "blue_container_h01_back" in str(object_name).lower():
        extract_and_setdown = _extract_l4_container_to_floor
    elif "blue_tote_b01" in str(object_name).lower():
        extract_and_setdown = _extract_blue_tote_to_floor
    else:
        extract_and_setdown = _extract_floor_push_object
    navigation_retract = _navigation_retract or _navigation_retract_for_floor_push
    floor_push = _floor_push or _run_floor_corridor_push
    raw_env = backend.env
    body_id = raw_env.obj_body_id[object_name]
    start_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    target = np.asarray(target_xy, dtype=float).reshape(2)
    l4_container_route = "blue_container_h01_back" in str(object_name).lower()
    output5_side_route = bool(
        np.linalg.norm(target - np.array([4.872, -7.261], dtype=float)) < 0.05
    )
    extraction_direction = (
        np.array([0.0, 1.0], dtype=float)
        if str(object_name).lower() == "green_tote_b01_upper"
        else np.array([1.0, 0.0], dtype=float)
    )
    extraction_stroke_m = (
        0.02
        if (
            str(object_name).lower() == "green_tote_b01_upper"
            or l4_container_route
        )
        else 0.08
    )
    extraction_distance_m = 0.30 if l4_container_route else 0.14

    extraction = extract_and_setdown(
        backend,
        competition_driver=competition_driver,
        source=source,
        object_name=object_name,
        macro_count=2,
        distance_m=extraction_distance_m,
        world_direction=extraction_direction,
        table_object_z=float(table_object_z),
        stroke_m=extraction_stroke_m,
        reset_m=0.02 if l4_container_route else 0.06,
        minimum_lift_m=0.10,
        place_max_descent_m=0.45,
        minimum_macro_progress_m=0.005 if l4_container_route else 0.02,
    )
    after_extraction = np.asarray(
        raw_env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    floor_transition = bool(
        after_extraction[2] < float(table_object_z) - 0.30
    )
    if not bool(extraction.get("success", False)):
        return {
            "success": False,
            "failure_stage": "extraction",
            "method": "physical_floor_push",
            "extraction": extraction,
            "floor_transition_detected": floor_transition,
        }
    if not floor_transition:
        return {
            "success": False,
            "failure_stage": "floor_transition",
            "method": "physical_floor_push",
            "extraction": extraction,
            "floor_transition_detected": False,
        }

    if bool(extraction.get("navigation_ready", False)):
        retract = {
            "success": True,
            "collision": False,
            "skipped": True,
            "reason": "extractor_navigation_ready",
        }
    else:
        retract = navigation_retract(
            backend,
            forward_m=0.20,
            lateral_m=0.15,
            target_z=1.45,
            max_steps=(
                600
                if "blue_tote_b01" in str(object_name).lower()
                else 240
            ),
        )
    if not bool(retract.get("success", False)):
        return {
            "success": False,
            "failure_stage": "navigation_retract",
            "method": "physical_floor_push",
            "extraction": extraction,
            "navigation_retract": retract,
            "floor_transition_detected": True,
        }

    green_floor_route = str(object_name).lower() == "green_tote_b01_upper"
    blue_floor_route = "blue_tote_b01" in str(object_name).lower()
    green_orientation_clearance = 0.05 if green_floor_route else 0.35
    green_base_standoff = 0.35 if green_floor_route else 0.65
    push = floor_push(
        backend,
        object_name=object_name,
        # L2's upper input station blocks the direct -Y exit.  Clear its
        # positive-X side first, then enter the common lower aisle.
        push_direction=(
            np.array([0.0, 1.0], dtype=float)
            if green_floor_route
            else (
                np.array([1.0, 0.0], dtype=float)
                if blue_floor_route or l4_container_route
                else np.array([0.0, -1.0], dtype=float)
            )
        ),
        push_distance_m=1.05,
        base_standoff_m=green_base_standoff,
        orientation_clearance_m=green_orientation_clearance,
        # Keep the torso on the positive-X side of input_6 while pushing the
        # tote vertically away from the station.  The reverse -Y stage uses
        # the same positive-X safety offset.
        lateral_offset_m=-0.05 if green_floor_route else 0.0,
        torso_drop_m=0.24,
        base_pusher=True,
        oriented_retract_forward_m=0.20,
        oriented_retract_lateral_m=0.08,
        oriented_retract_target_z=1.45,
        maximum_lateral_offset_m=0.25,
        face_offset_m=0.24,
        hand_separation_m=0.28,
        hand_height_m=0.38,
        precontact_clearance_m=0.08,
        base_speed_m_s=0.040,
        max_steps=15000,
        route_target_xy=target,
        route_corridor_y=-8.40,
        initial_clearance_m=(
            0.90
            if green_floor_route
            else (
                2.00
                if blue_floor_route
                else (1.20 if l4_container_route else 0.0)
            )
        ),
        route_arrival_radius_m=0.80,
        route_arrival_margin_m=0.05,
        route_reposition_clearance_m=1.30 if green_floor_route else 0.90,
        route_reposition_lateral_offset_m=0.10 if green_floor_route else 0.0,
        route_south_reposition_lateral_offset_m=0.025 if blue_floor_route else None,
        route_south_tracking_gain=(
            0.50 if blue_floor_route or l4_container_route else None
        ),
        route_final_approach_reverse_pusher=output5_side_route,
        route_final_side_approach_x=3.80 if output5_side_route else None,
        route_reverse_switch_y=None,
        route_lateral_clearance_m=0.15 if green_floor_route else 0.0,
        route_minimum_retract_z_m=0.80,
        route_minimum_base_x_m=12.18 if green_floor_route else None,
        tracking_gain=0.0,
        alignment_gain=0.05,
        tracking_deadband_m=0.05,
        maximum_base_object_offset_m=0.08,
        maximum_lateral_speed_m_s=0.005,
    )
    final_object = np.asarray(
        raw_env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    final_distance = float(np.linalg.norm(final_object[:2] - target))
    success = bool(
        push.get("success", False)
        and not push.get("collision", False)
        and final_distance < 0.80
    )
    return {
        "success": success,
        "failure_stage": None if success else "floor_push",
        "method": "physical_floor_push",
        "start_object_position": start_object.tolist(),
        "after_extraction_object_position": after_extraction.tolist(),
        "end_object_position": final_object.tolist(),
        "floor_transition_detected": True,
        "extraction": extraction,
        "navigation_retract": retract,
        "floor_push": push,
        "physical_contact_steps": int(push.get("physical_contact_steps", 0)),
        "final_target_distance_m": final_distance,
        "attachment_calls": 0,
        "object_pose_writes": 0,
    }
