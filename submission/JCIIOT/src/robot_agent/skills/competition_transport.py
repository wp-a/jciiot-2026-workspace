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
    if arm not in name:
        return False
    if "finger" in name:
        return False
    if any(token in name for token in ("wrist", "palm", "gripper")):
        return True
    return any(f"arm_{arm}_{index}" in name for index in (4, 5, 6))


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
        object_drop_tolerance: float = 0.025,
        vertical_hold_feedforward: float = 0.0,
        vertical_hold_gain: float = 0.0,
        max_vertical_hold_delta: float = 0.0,
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
        self.object_drop_tolerance = float(object_drop_tolerance)
        self.vertical_hold_feedforward = float(vertical_hold_feedforward)
        self.vertical_hold_gain = float(vertical_hold_gain)
        self.max_vertical_hold_delta = float(max_vertical_hold_delta)
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
) -> dict:
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "steps": int(steps),
        "final_base_xy": np.asarray(observation["base_xy"], dtype=float).tolist(),
        "final_distance": float(final_distance),
        "minimum_object_z": float(minimum_observed_z),
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
    target_object_z = float(minimum_object_z) + config.object_drop_tolerance
    gripper_z_offsets = {
        arm: float(observation["gripper_positions"][arm][2])
        - float(observation["object_pos"][2])
        for arm in ("right", "left")
    }
    minimum_observed_z = float(observation["object_pos"][2])
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
        )

    failure_stage = "timeout"
    success = False
    while steps < config.max_steps:
        observation = driver.observe(backend, object_name)
        minimum_observed_z = min(
            minimum_observed_z,
            float(observation["object_pos"][2]),
        )
        if next_contact_stability(observation["contacts"], 0) == 0:
            failure_stage = "contact"
            break
        if float(observation["object_pos"][2]) < float(minimum_object_z):
            failure_stage = "object_drop"
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

        speed = min(config.k_linear * final_distance, config.max_linear)
        world_velocity = speed * delta / max(final_distance, 1e-12)
        base_velocity = world_velocity_to_base_frame(
            world_velocity,
            float(observation["base_yaw"]),
        )
        yaw_error = _shortest_angle(float(hold_yaw) - float(observation["base_yaw"]))
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
        base_xy = np.asarray(observation["base_xy"], dtype=float)
        world_step = direct_base_step_target(
            base_xy=base_xy,
            base_yaw=float(observation["base_yaw"]),
            base_command=command,
            control_dt=config.base_control_dt,
        ) - base_xy
        phases = (
            (np.zeros(3, dtype=float), world_step),
            (command, -world_step),
        )
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
    )
    return _transport_result(
        success=success,
        failure_stage=failure_stage,
        steps=steps,
        observation=observation,
        final_distance=final_distance,
        minimum_observed_z=minimum_observed_z,
    )


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
