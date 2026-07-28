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
        max_linear: float = 0.40,
        max_angular: float = 0.08,
        max_linear_delta: float = 0.04,
        max_angular_delta: float = 0.01,
        base_control_dt: float = 0.05,
        yaw_tolerance: float = 0.04,
        object_drop_tolerance: float = 0.025,
        vertical_hold_feedforward: float = 0.0004,
        vertical_hold_gain: float = 0.8,
        max_vertical_hold_delta: float = 0.003,
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
    for _ in range(config.max_steps):
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
        hold_delta = vertical_hold_delta(
            current_z=float(observation["object_pos"][2]),
            target_z=target_object_z,
            feedforward=config.vertical_hold_feedforward,
            gain=config.vertical_hold_gain,
            max_delta=config.max_vertical_hold_delta,
        )
        arm_world_deltas = {
            arm: np.array([0.0, 0.0, hold_delta], dtype=float)
            for arm in ("right", "left")
        }
        step_info = driver.step(
            backend,
            object_name=object_name,
            base_command=command,
            hold_targets=hold_targets,
            arm_world_deltas=arm_world_deltas,
            base_control_dt=config.base_control_dt,
        )
        steps += 1
        previous_command = command

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
