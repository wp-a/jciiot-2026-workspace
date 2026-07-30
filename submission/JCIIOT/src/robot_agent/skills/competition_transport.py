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
        heading_translation_tolerance: float = 0.05,
        object_drop_tolerance: float = 0.025,
        vertical_hold_feedforward: float = 0.0,
        vertical_hold_gain: float = 0.0,
        max_vertical_hold_delta: float = 0.0,
        max_planar_grasp_drift: float = 0.04,
        height_recovery_trigger: float = 0.01,
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
        self.heading_translation_tolerance = float(
            heading_translation_tolerance
        )
        self.object_drop_tolerance = float(object_drop_tolerance)
        self.vertical_hold_feedforward = float(vertical_hold_feedforward)
        self.vertical_hold_gain = float(vertical_hold_gain)
        self.max_vertical_hold_delta = float(max_vertical_hold_delta)
        self.max_planar_grasp_drift = float(max_planar_grasp_drift)
        self.height_recovery_trigger = float(height_recovery_trigger)
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
        reset_max_gripper_drift: float = 0.03,
        max_lateral_drift: float = 0.03,
        minimum_macro_progress: float = 0.02,
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
    object_offset_base = world_velocity_to_base_frame(
        np.asarray(observation["object_pos"][:2], dtype=float)
        - np.asarray(observation["base_xy"], dtype=float),
        float(observation["base_yaw"]),
    )
    carried_object_angle = float(
        math.atan2(object_offset_base[1], object_offset_base[0])
    )
    target_object_z = float(minimum_object_z) + config.object_drop_tolerance
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
    driver.record_event(
        backend,
        "physical_transport_start",
        object_name=object_name,
        waypoint_count=len(waypoints),
    )

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
            failure_stage = "contact"
            break
        if float(observation["object_pos"][2]) < float(minimum_object_z):
            failure_stage = "object_drop"
            break
        if maximum_planar_grasp_drift > config.max_planar_grasp_drift:
            failure_stage = "planar_grasp_drift"
            break
        height_error = target_object_z - float(observation["object_pos"][2])
        if height_error >= config.height_recovery_trigger:
            driver.record_event(
                backend,
                "physical_height_recenter_start",
                object_name=object_name,
            )
            recentered = False
            for _ in range(config.height_recenter_steps):
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
        if not heading_aligned:
            command[:2] = 0.0
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
        "max_lateral_drift": config.max_lateral_drift,
        "minimum_macro_progress": config.minimum_macro_progress,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values.values()):
        raise ValueError("inchworm distances, limits, and tolerances must be positive")
    if config.arm_max_steps < 1 or config.max_cycles < 1:
        raise ValueError("inchworm step and cycle budgets must be positive")

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
                arm: (
                    reset_start_grippers[arm]
                    - np.asarray(observation["gripper_positions"][arm], dtype=float)
                    - np.array([world_step[0], world_step[1], 0.0], dtype=float)
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
            max_gripper_drift = max(
                max_gripper_drift,
                max(
                    float(
                        np.linalg.norm(
                            np.asarray(observation["gripper_positions"][arm])[:2]
                            - reset_start_grippers[arm][:2]
                        )
                    )
                    for arm in ("right", "left")
                ),
            )
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
                "vertical_adjustment_m": vertical_adjustment,
                "arm_progress_m": arm_progress,
                "macro_progress_m": macro_progress,
                "total_progress_m": total_progress,
                "macro_lateral_drift_m": macro_lateral,
                "max_gripper_reset_drift_m": max_gripper_drift,
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
        if result:
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
