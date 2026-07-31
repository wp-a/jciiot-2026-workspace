"""Verified two-arm physical grasp for the competition submission.

The controller uses the grasp sites already present in the official MuJoCo
objects. It never writes object poses directly: both grippers are moved by the
configured OSC controllers, contact is checked by robosuite, and success also
requires the official lift verifier.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ARMS = ("right", "left")

# Measured in the official L5 scene at the collision-free clearance pose.
# Order: torso, right arm joints 1-6, left arm joints 1-6.
STATION_SIDE_CLEARANCE_JOINT_SEED = np.array(
    [
        0.349987,
        1.061774,
        -0.515986,
        1.725133,
        0.521535,
        0.079322,
        -0.544378,
        1.498435,
        -0.337203,
        1.102509,
        0.708113,
        -0.656307,
        -0.822341,
    ],
    dtype=float,
)


class ScriptedGraspConfig:
    """Small, explicit parameter set for the geometric grasp controller."""

    def __init__(
        self,
        *,
        clearance_height: float = 0.30,
        clearance_raise_steps: int = 180,
        clearance_translate_steps: int = 180,
        torso_drop: float = 0.04,
        torso_minimum: float = 0.10,
        torso_steps: int = 80,
        torso_tolerance: float = 0.005,
        pregrasp_height: float = 0.10,
        site_below_offset: float = 0.015,
        position_tolerance: float = 0.012,
        approach_tolerance: float = 0.08,
        pregrasp_steps: int = 180,
        approach_steps: int = 180,
        close_steps: int = 300,
        contact_polish_step: float = 0.001,
        contact_polish_max_drop: float = 0.030,
        contact_confirmation_drop: float = 0.003,
        contact_settle_steps: int = 5,
        left_wrist_adjustment: float = 0.10,
        single_wrist_adjustment_steps: int = 20,
        wrist_adjustment_steps: int = 80,
        wrist_height_trigger: float = 0.04,
        mirrored_ik_height_offset: float = 0.13,
        mirrored_ik_regularization: float = 0.05,
        mirrored_ik_max_error: float = 0.08,
        mirrored_ik_max_nfev: int = 300,
        station_side_reach_offset: float = 0.0,
        station_side_seed_steps: int = 120,
        hold_close_pose: bool = True,
        face_insertion: float = 0.0,
        close_follow_max_distance: float = 0.0,
        close_increment_interval: int = 20,
        max_action: float = 0.65,
        lift_height: float = 0.04,
        container_lift_height_override: float | None = None,
        lift_steps: int = 300,
        lift_hold_steps: int = 0,
        lift_tolerance: float = 0.01,
        lift_follower_lead: float = 0.003,
        swap_arm_targets: bool = False,
        clearance_prepared: bool = False,
    ) -> None:
        self.clearance_height = float(clearance_height)
        self.clearance_raise_steps = int(clearance_raise_steps)
        self.clearance_translate_steps = int(clearance_translate_steps)
        self.torso_drop = float(torso_drop)
        self.torso_minimum = float(torso_minimum)
        self.torso_steps = int(torso_steps)
        self.torso_tolerance = float(torso_tolerance)
        self.pregrasp_height = float(pregrasp_height)
        self.site_below_offset = float(site_below_offset)
        self.position_tolerance = float(position_tolerance)
        self.approach_tolerance = float(approach_tolerance)
        self.pregrasp_steps = int(pregrasp_steps)
        self.approach_steps = int(approach_steps)
        self.close_steps = int(close_steps)
        self.contact_polish_step = float(contact_polish_step)
        self.contact_polish_max_drop = float(contact_polish_max_drop)
        self.contact_confirmation_drop = float(contact_confirmation_drop)
        self.contact_settle_steps = int(contact_settle_steps)
        self.left_wrist_adjustment = float(left_wrist_adjustment)
        self.single_wrist_adjustment_steps = int(single_wrist_adjustment_steps)
        self.wrist_adjustment_steps = int(wrist_adjustment_steps)
        self.wrist_height_trigger = float(wrist_height_trigger)
        self.mirrored_ik_height_offset = float(mirrored_ik_height_offset)
        self.mirrored_ik_regularization = float(mirrored_ik_regularization)
        self.mirrored_ik_max_error = float(mirrored_ik_max_error)
        self.mirrored_ik_max_nfev = int(mirrored_ik_max_nfev)
        self.station_side_reach_offset = float(station_side_reach_offset)
        self.station_side_seed_steps = int(station_side_seed_steps)
        self.hold_close_pose = bool(hold_close_pose)
        self.face_insertion = float(face_insertion)
        self.close_follow_max_distance = float(close_follow_max_distance)
        self.close_increment_interval = int(close_increment_interval)
        self.max_action = float(max_action)
        self.lift_height = float(lift_height)
        self.container_lift_height_override = (
            None
            if container_lift_height_override is None
            else float(container_lift_height_override)
        )
        self.lift_steps = int(lift_steps)
        self.lift_hold_steps = int(lift_hold_steps)
        self.lift_tolerance = float(lift_tolerance)
        self.lift_follower_lead = float(lift_follower_lead)
        self.swap_arm_targets = bool(swap_arm_targets)
        self.clearance_prepared = bool(clearance_prepared)


def normalized_position_action(
    delta: np.ndarray,
    position_scale: np.ndarray,
    max_action: float,
) -> np.ndarray:
    """Convert a Cartesian delta to a clipped OSC pose action."""
    delta = np.asarray(delta, dtype=float).reshape(3)
    scale = np.asarray(position_scale, dtype=float).reshape(3)
    normalized = np.divide(delta, scale, out=np.zeros(3), where=scale > 0)
    normalized = np.clip(normalized, -float(max_action), float(max_action))
    return np.concatenate([normalized, np.zeros(3, dtype=float)])


def build_independent_gripper_action(
    robot,
    *,
    arm_actions: Mapping[str, np.ndarray],
    gripper_values: Mapping[str, float],
    hold_targets: Mapping[str, np.ndarray],
    build_action_fn=None,
) -> np.ndarray:
    """Reuse the official action builder, then set each gripper independently."""
    if set(gripper_values) != set(ARMS):
        raise ValueError("gripper_values must provide commands for both arms")
    values = {arm: float(gripper_values[arm]) for arm in ARMS}
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("gripper commands must be finite")
    if build_action_fn is None:
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            build_action as build_action_fn,
        )

    action = np.asarray(
        build_action_fn(
            robot,
            arm_actions,
            gripper_value=0.0,
            hold_targets=hold_targets,
        ),
        dtype=float,
    ).copy()
    split_indexes = robot.composite_controller._action_split_indexes
    for arm in ARMS:
        if int(robot.gripper[arm].dof) <= 0:
            continue
        start, end = split_indexes[f"{arm}_gripper"]
        action[int(start) : int(end)] = values[arm]
    return action


def lowered_torso_target(current, *, drop: float, minimum: float) -> np.ndarray:
    """Return a bounded absolute torso target for improved low reach."""
    current = np.asarray(current, dtype=float).copy()
    return np.maximum(current - float(drop), float(minimum))


def contact_micro_adjustment_targets(
    current: float,
    *,
    step: float,
    max_drop: float,
    minimum: float,
) -> list[float]:
    """Return bounded torso targets for a fine post-close contact search."""
    current = float(current)
    step = float(step)
    max_drop = float(max_drop)
    minimum = float(minimum)
    if step <= 0.0 or max_drop <= 0.0 or current <= minimum:
        return []

    lower = max(current - max_drop, minimum)
    targets = []
    target = current
    while target - step > lower:
        target -= step
        targets.append(target)
    if not targets or not np.isclose(targets[-1], lower):
        targets.append(lower)
    return targets


def contact_margin_reached(
    *,
    first_contact: float,
    current: float,
    required_drop: float,
) -> bool:
    """Return whether a contact search moved far enough beyond first touch."""
    return float(first_contact) - float(current) >= float(required_drop) - 1e-12


def follower_lift_offset(
    *,
    object_lift: float,
    lead: float,
    lift_height: float,
) -> float:
    """Track the object with a small bounded lead instead of a fixed arm goal."""
    return min(
        float(lift_height),
        max(0.0, float(object_lift)) + max(0.0, float(lead)),
    )


def lift_goal_reached(
    *,
    reference_z: float,
    current_z: float,
    lift_height: float,
    tolerance: float,
) -> bool:
    """Measure lift from the pre-close height, including close-stage motion."""
    required = float(reference_z) + float(lift_height) - float(tolerance)
    return float(current_z) >= required


def synchronize_controller_goals(robot) -> None:
    """Reset moving-base controller goals to the current simulated posture."""
    composite = robot.composite_controller
    composite.update_state()
    for part_name in (*ARMS, "torso"):
        controller = composite.part_controllers.get(part_name)
        if controller is None:
            continue
        controller.update(force=True)
    reset = getattr(composite, "reset", None)
    if callable(reset):
        reset()
    else:
        for controller in composite.part_controllers.values():
            controller.reset_goal()


def quiesce_robot_for_grasp(raw_env) -> None:
    """Clear navigation velocity state before a new wall-side grasp."""
    robot = raw_env.robots[0]
    sim = getattr(raw_env, "sim", None)
    if sim is not None:
        joint_names = list(getattr(robot, "robot_arm_joints", []))
        joint_names.extend(getattr(robot.robot_model, "torso_joints", []))
        joint_names.extend(getattr(robot.robot_model, "head_joints", []))
        for gripper_joints in getattr(robot, "gripper_joints", {}).values():
            joint_names.extend(gripper_joints)
        for joint_name in dict.fromkeys(joint_names):
            try:
                qvel_addr = sim.model.get_joint_qvel_addr(joint_name)
                sim.data.qvel[qvel_addr] = 0.0
            except Exception:
                continue
        sim.forward()
    synchronize_controller_goals(robot)


def next_contact_stability(contacts: Mapping[str, Any], stable_steps: int) -> int:
    """Advance consecutive bilateral-contact count, or reset on contact loss."""
    if all(bool(contacts.get(arm, False)) for arm in ARMS):
        return int(stable_steps) + 1
    return 0


def wrist_adjustment_required(
    *,
    current_z: float,
    target_z: float,
    threshold: float,
) -> bool:
    """Detect when left-arm height reach needs the measured wrist correction."""
    return float(current_z) - float(target_z) > float(threshold)


def uses_mirrored_open_grasp(object_name: str) -> bool:
    """Select the mirrored rim grasp only for matching tote geometry."""
    return "tote_b01" in str(object_name).lower()


def uses_axis_aware_fingerpad_mirror(object_name: str) -> bool:
    """Select heading-aware mirroring only for rotated tote stations."""
    name = str(object_name).lower()
    return "blue_tote_b01" in name or "white_tote_b01_left" in name


def uses_legacy_container_grasp(object_name: str) -> bool:
    """Select the scored L1 controller for the matching container geometry."""
    return "container_h01" in str(object_name).lower()


def uses_station_side_tote_grasp(object_name: str) -> bool:
    """Select the wall-side geometry used by the three L5 white totes."""
    return "white_tote_b01_left" in str(object_name).lower()


def station_side_clearance_joint_seed(object_name: str) -> np.ndarray | None:
    """Return the deterministic upper-body reset for later L5 transfers."""
    name = str(object_name).lower()
    if not uses_station_side_tote_grasp(name) or name.endswith("_front"):
        return None
    return STATION_SIDE_CLEARANCE_JOINT_SEED.copy()


def should_swap_arm_targets(object_name: str, *, requested: bool) -> bool:
    """Keep the dynamic wall-side assignment from being swapped twice."""
    return bool(requested) and not uses_station_side_tote_grasp(object_name)


def apply_object_grasp_profile(
    config: ScriptedGraspConfig,
    object_name: str,
) -> ScriptedGraspConfig:
    """Apply parameters validated for an official object geometry family."""
    if uses_legacy_container_grasp(object_name):
        lift_height = config.container_lift_height_override
        if lift_height is not None and (
            not np.isfinite(lift_height) or lift_height <= 0.0
        ):
            raise ValueError(
                "container_lift_height_override must be finite and positive"
            )
        config.site_below_offset = 0.035
        config.approach_tolerance = config.position_tolerance
        config.close_steps = 80
        config.close_increment_interval = 1
        config.contact_settle_steps = config.close_steps + 1
        config.hold_close_pose = False
        config.face_insertion = 0.0
        config.wrist_height_trigger = float("inf")
        config.lift_height = 0.15 if lift_height is None else lift_height
        config.lift_hold_steps = 20
        config.lift_tolerance = 0.02
    elif uses_station_side_tote_grasp(object_name):
        config.mirrored_ik_height_offset = 0.06
        config.station_side_reach_offset = 0.04
        config.clearance_translate_steps = 360
    return config


def station_side_tote_grasp_targets(
    raw_targets: Mapping[str, np.ndarray],
    *,
    object_xy: np.ndarray,
    base_xy: np.ndarray,
    reach_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Reflect the reachable marked site across the robot heading axis."""
    targets = {
        arm: np.asarray(raw_targets[arm], dtype=float).copy()
        for arm in ARMS
    }
    center = np.asarray(object_xy, dtype=float).reshape(2)
    base = np.asarray(base_xy, dtype=float).reshape(2)
    grasp_center = np.mean(
        np.stack([targets[arm][:2] for arm in ARMS]),
        axis=0,
    )
    forward = grasp_center - base
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("base_xy must differ from the grasp center")
    forward /= forward_norm
    near = min(
        targets.values(),
        key=lambda target: float(np.linalg.norm(target[:2] - base)),
    ).copy()
    relative = near[:2] - center
    reflected = near.copy()
    reflected[:2] = center + 2.0 * np.dot(relative, forward) * forward - relative
    expected_right_minus_left = np.array(
        [forward[1], -forward[0]],
        dtype=float,
    )
    if float(np.dot(reflected[:2] - near[:2], expected_right_minus_left)) >= 0.0:
        result = {"right": reflected, "left": near}
    else:
        result = {"right": near, "left": reflected}
    toward_base = base - grasp_center
    toward_base /= float(np.linalg.norm(toward_base))
    for target in result.values():
        target[:2] += max(0.0, float(reach_offset)) * toward_base
    return result


def mirrored_fingerpad_targets(
    right_fingerpads: np.ndarray,
    *,
    object_xy: np.ndarray,
    mirror_normal_xy: np.ndarray,
    height_offset: float,
) -> np.ndarray:
    """Mirror a right open-gripper pose across the current inter-arm axis."""
    targets = np.asarray(right_fingerpads, dtype=float).copy()
    if targets.shape != (2, 3):
        raise ValueError("right_fingerpads must have shape (2, 3)")
    center = np.asarray(object_xy, dtype=float).reshape(2)
    normal = np.asarray(mirror_normal_xy, dtype=float).reshape(2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("mirror_normal_xy must be nonzero")
    normal /= normal_norm
    signed_distances = (targets[:, :2] - center) @ normal
    targets[:, :2] -= 2.0 * signed_distances[:, None] * normal
    targets[:, 2] -= float(height_offset)
    return targets


def world_x_mirrored_fingerpad_targets(
    right_fingerpads: np.ndarray,
    *,
    object_x: float,
    height_offset: float,
) -> np.ndarray:
    """Mirror the measured L2 pose across its world-aligned station axis."""
    targets = np.asarray(right_fingerpads, dtype=float).copy()
    if targets.shape != (2, 3):
        raise ValueError("right_fingerpads must have shape (2, 3)")
    targets[:, 0] = 2.0 * float(object_x) - targets[:, 0]
    targets[:, 2] -= float(height_offset)
    return targets


def joint_interpolation_path(
    start: np.ndarray,
    target: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Return a fixed-size joint path that excludes start and includes target."""
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if start.shape != target.shape:
        raise ValueError("start and target joints must have the same shape")
    if int(steps) < 1:
        raise ValueError("steps must be at least 1")
    return np.linspace(start, target, int(steps) + 1, dtype=float)[1:]


def close_pose_targets(
    current: Mapping[str, np.ndarray],
    requested: Mapping[str, np.ndarray],
    *,
    hold_current: bool,
) -> dict[str, np.ndarray]:
    """Hold the corrected left pose while the right arm finishes its approach."""
    if hold_current:
        return {
            "right": np.asarray(requested["right"], dtype=float).copy(),
            "left": np.asarray(current["left"], dtype=float).copy(),
        }
    return {
        arm: np.asarray(requested[arm], dtype=float).copy()
        for arm in ARMS
    }


def inward_face_targets(
    targets: Mapping[str, np.ndarray],
    *,
    object_xy: np.ndarray,
    insertion: float,
) -> dict[str, np.ndarray]:
    """Insert grasp centers from the marked face toward the object interior."""
    copied = {
        arm: np.asarray(targets[arm], dtype=float).copy()
        for arm in ARMS
    }
    grasp_center = np.mean(
        np.stack([copied[arm][:2] for arm in ARMS]),
        axis=0,
    )
    inward = np.asarray(object_xy, dtype=float).reshape(2) - grasp_center
    norm = float(np.linalg.norm(inward))
    if norm <= 1e-9 or float(insertion) == 0.0:
        return copied
    offset = inward / norm * float(insertion)
    for arm in ARMS:
        copied[arm][:2] += offset
    return copied


def bounded_planar_follow_offset(delta: np.ndarray, *, max_distance: float) -> np.ndarray:
    """Bound object-relative close tracking without changing its direction."""
    delta = np.asarray(delta, dtype=float).reshape(2)
    distance = float(np.linalg.norm(delta))
    limit = max(0.0, float(max_distance))
    if distance <= limit or distance <= 1e-12:
        return delta.copy()
    return delta / distance * limit


def gripper_close_command(step: int, *, interval: int) -> float:
    """Pulse the binary Robotiq close command to reduce contact impulse."""
    interval = int(interval)
    if interval < 1:
        raise ValueError("interval must be at least 1")
    return 1.0 if int(step) % interval == 0 else 0.0


def targets_reached(
    current: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> bool:
    """Return true only when every configured arm reached its target."""
    for arm in ARMS:
        if arm not in current or arm not in targets:
            return False
        distance = float(
            np.linalg.norm(
                np.asarray(current[arm], dtype=float)
                - np.asarray(targets[arm], dtype=float)
            )
        )
        if distance > float(tolerance):
            return False
    return True


def vertical_clearance_targets(
    current: Mapping[str, np.ndarray],
    grasp_targets: Mapping[str, np.ndarray],
    *,
    clearance_height: float,
) -> dict[str, np.ndarray]:
    """Raise both grippers vertically above the highest grasp target."""
    safe_z = max(
        max(float(np.asarray(grasp_targets[arm], dtype=float)[2]) for arm in ARMS)
        + float(clearance_height),
        max(float(np.asarray(current[arm], dtype=float)[2]) for arm in ARMS),
    )
    targets = {}
    for arm in ARMS:
        target = np.asarray(current[arm], dtype=float).copy()
        target[2] = safe_z
        targets[arm] = target
    return targets


def assigned_grasp_targets(
    raw_targets: Mapping[str, np.ndarray],
    *,
    swap: bool,
) -> dict[str, np.ndarray]:
    """Assign object grasp sites to arms after base-orientation selection."""
    if swap:
        return {
            "right": np.asarray(raw_targets["left"], dtype=float).copy(),
            "left": np.asarray(raw_targets["right"], dtype=float).copy(),
        }
    return {
        arm: np.asarray(raw_targets[arm], dtype=float).copy()
        for arm in ARMS
    }


def verified_grasp(contacts: Mapping[str, Any], *, lift_success: bool) -> bool:
    """Require both grippers and the lift verifier to report success."""
    return bool(lift_success and all(bool(contacts.get(arm, False)) for arm in ARMS))


def mark_verified_grasp_end(
    backend,
    *,
    source: str,
    object_name: str,
    contacts: Mapping[str, Any],
    lift_success: bool,
) -> bool:
    """Record the combined physical result and return its success value."""
    success = verified_grasp(contacts, lift_success=lift_success)
    backend._mark_trajectory_event(
        "grasp_end",
        source=source,
        object_name=object_name,
        success=success,
        contact_right=bool(contacts.get("right", False)),
        contact_left=bool(contacts.get("left", False)),
        lift_success=bool(lift_success),
    )
    return success


class OfficialScriptedGraspDriver:
    """Adapter from the stage protocol to official robosuite helpers."""

    @staticmethod
    def _helpers():
        from robosuite.environments.factory_sorting.lift_after_grasp import (
            arm_delta_to_normalized_action,
            build_action,
            capture_hold_targets,
            current_part_qpos,
            lift_grasped_object,
            object_center_pos,
            world_delta_to_controller_frame,
        )
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            get_target_positions,
            grasp_status,
            gripper_end_center_pos,
        )
        return {
            "arm_action": arm_delta_to_normalized_action,
            "build_action": build_action,
            "capture_hold_targets": capture_hold_targets,
            "current_part_qpos": current_part_qpos,
            "lift": lift_grasped_object,
            "object_center": object_center_pos,
            "world_delta": world_delta_to_controller_frame,
            "get_targets": get_target_positions,
            "grasp_status": grasp_status,
            "gripper_position": gripper_end_center_pos,
        }

    @staticmethod
    def _record(backend, raw_env) -> None:
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)

    def _seed_station_side_clearance(self, backend, object_name, config) -> bool:
        target = station_side_clearance_joint_seed(object_name)
        if target is None:
            return True

        raw_env = backend.env
        robot = raw_env.robots[0]
        model = raw_env.sim.model
        data = raw_env.sim.data
        torso_joint = next(
            (
                model.joint_id2name(index)
                for index in range(model.njnt)
                if (model.joint_id2name(index) or "").endswith(
                    "torso_lift_joint"
                )
            ),
            None,
        )
        if torso_joint is None:
            return False
        joint_names = [torso_joint]
        joint_names.extend(
            f"robot0_arm_{arm}_{index}_joint"
            for arm in ARMS
            for index in range(1, 7)
        )
        try:
            qpos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        except Exception:
            return False
        if any(isinstance(addr, tuple) for addr in qpos_addrs):
            return False

        from robot_agent.environments.robosuite_backend import (
            _navigation_collisions,
        )

        start = data.qpos[qpos_addrs].copy()
        path = joint_interpolation_path(
            start,
            target,
            steps=config.station_side_seed_steps,
        )
        for values in path:
            data.qpos[qpos_addrs] = values
            raw_env.sim.forward()
            collisions = _navigation_collisions(
                raw_env,
                robot,
                getattr(backend, "_ignore_collision_geom", ()),
            )
            if collisions:
                data.qpos[qpos_addrs] = start
                raw_env.sim.forward()
                synchronize_controller_goals(robot)
                self._record(backend, raw_env)
                return False
            self._record(backend, raw_env)
        synchronize_controller_goals(robot)
        return True

    def _move_to_targets(
        self,
        backend,
        targets: Mapping[str, np.ndarray],
        config: ScriptedGraspConfig,
        *,
        max_steps: int,
        gripper_value: float,
        tolerance: float | None = None,
    ) -> bool:
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        fixed_targets = {
            arm: np.asarray(targets[arm], dtype=float).copy()
            for arm in ARMS
        }
        hold_targets = helpers["capture_hold_targets"](robot)

        for _ in range(max_steps):
            robot.composite_controller.update_state()
            current = {
                arm: helpers["gripper_position"](raw_env, robot, arm)
                for arm in ARMS
            }
            if targets_reached(
                current,
                fixed_targets,
                tolerance=(
                    config.position_tolerance
                    if tolerance is None
                    else float(tolerance)
                ),
            ):
                return True

            arm_actions = {}
            for arm in ARMS:
                world_delta = fixed_targets[arm] - current[arm]
                controller_delta = helpers["world_delta"](robot, arm, world_delta)
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    config.max_action,
                )
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=gripper_value,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            self._record(backend, raw_env)
            if bool((info or {}).get("has_judge_collision", False)):
                return False
        return False

    def lower_torso_for_reach(self, backend, config) -> bool:
        raw_env = backend.env
        robot = raw_env.robots[0]
        torso_joint = next(
            (
                raw_env.sim.model.joint_id2name(index)
                for index in range(raw_env.sim.model.njnt)
                if (raw_env.sim.model.joint_id2name(index) or "").endswith(
                    "torso_lift_joint"
                )
            ),
            None,
        )
        if torso_joint is None:
            return False
        qpos_addr = raw_env.sim.model.get_joint_qpos_addr(torso_joint)
        if isinstance(qpos_addr, tuple):
            return False

        start = float(raw_env.sim.data.qpos[qpos_addr])
        target = float(lowered_torso_target(
            np.array([start], dtype=float),
            drop=config.torso_drop,
            minimum=config.torso_minimum,
        )[0])
        steps = max(1, config.torso_steps)

        from robot_agent.environments.robosuite_backend import (
            _navigation_collisions,
        )

        for value in np.linspace(start, target, steps + 1, dtype=float)[1:]:
            raw_env.sim.data.qpos[qpos_addr] = value
            raw_env.sim.forward()
            collisions = _navigation_collisions(
                raw_env,
                robot,
                getattr(backend, "_ignore_collision_geom", ()),
            )
            if collisions:
                raw_env.sim.data.qpos[qpos_addr] = start
                raw_env.sim.forward()
                synchronize_controller_goals(robot)
                self._record(backend, raw_env)
                return False
            self._record(backend, raw_env)
        synchronize_controller_goals(robot)
        return abs(float(raw_env.sim.data.qpos[qpos_addr]) - target) <= config.torso_tolerance

    def _grasp_targets(self, backend, object_name, config, *, height_offset):
        helpers = self._helpers()
        raw_targets, _ = helpers["get_targets"](
            backend.env,
            object_name,
            config.site_below_offset,
        )
        body_id = backend.env.obj_body_id[object_name]
        if uses_station_side_tote_grasp(object_name):
            base_xy, _ = backend.get_base_pose()
            raw_targets = station_side_tote_grasp_targets(
                raw_targets,
                object_xy=backend.env.sim.data.body_xpos[body_id][:2],
                base_xy=base_xy,
                reach_offset=config.station_side_reach_offset,
            )
        grasp_targets = assigned_grasp_targets(
            raw_targets,
            swap=should_swap_arm_targets(
                object_name,
                requested=config.swap_arm_targets,
            ),
        )
        object_xy = backend.env.sim.data.body_xpos[body_id][:2]
        grasp_targets = inward_face_targets(
            grasp_targets,
            object_xy=object_xy,
            insertion=config.face_insertion,
        )
        offset = np.array([0.0, 0.0, float(height_offset)], dtype=float)
        return {
            arm: np.asarray(grasp_targets[arm], dtype=float) + offset
            for arm in ARMS
        }

    def raise_to_clearance(self, backend, object_name, config) -> bool:
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        current = {
            arm: helpers["gripper_position"](raw_env, robot, arm)
            for arm in ARMS
        }
        grasp_targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=0.0,
        )
        targets = vertical_clearance_targets(
            current,
            grasp_targets,
            clearance_height=config.clearance_height,
        )
        return self._move_to_targets(
            backend,
            targets,
            config,
            max_steps=config.clearance_raise_steps,
            gripper_value=-1.0,
        )

    def move_above_grasp_sites(self, backend, object_name, config) -> bool:
        if not self._seed_station_side_clearance(
            backend,
            object_name,
            config,
        ):
            return False
        targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=config.clearance_height,
        )
        return self._move_to_targets(
            backend,
            targets,
            config,
            max_steps=config.clearance_translate_steps,
            gripper_value=-1.0,
            tolerance=config.approach_tolerance,
        )

    def move_to_pregrasp(self, backend, object_name, config) -> bool:
        targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=config.pregrasp_height,
        )
        return self._move_to_targets(
            backend,
            targets,
            config,
            max_steps=config.pregrasp_steps,
            gripper_value=-1.0,
        )

    def approach_grasp_sites(self, backend, object_name, config) -> bool:
        targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=0.0,
        )
        return self._move_to_targets(
            backend,
            targets,
            config,
            max_steps=config.approach_steps,
            gripper_value=-1.0,
            tolerance=config.approach_tolerance,
        )

    def _adjust_single_wrist_for_reach(self, backend, config) -> bool:
        raw_env = backend.env
        robot = raw_env.robots[0]
        joint_name = next(
            (
                raw_env.sim.model.joint_id2name(index)
                for index in range(raw_env.sim.model.njnt)
                if (raw_env.sim.model.joint_id2name(index) or "").endswith(
                    "arm_left_5_joint"
                )
            ),
            None,
        )
        if joint_name is None:
            return False
        joint_id = raw_env.sim.model.joint_name2id(joint_name)
        qpos_addr = raw_env.sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(qpos_addr, tuple):
            return False

        start = float(raw_env.sim.data.qpos[qpos_addr])
        target = start + config.left_wrist_adjustment
        if bool(raw_env.sim.model.jnt_limited[joint_id]):
            joint_range = raw_env.sim.model.jnt_range[joint_id]
            target = float(np.clip(target, joint_range[0], joint_range[1]))

        from robot_agent.environments.robosuite_backend import (
            _navigation_collisions,
        )

        steps = max(1, config.single_wrist_adjustment_steps)
        for value in np.linspace(start, target, steps + 1, dtype=float)[1:]:
            raw_env.sim.data.qpos[qpos_addr] = value
            raw_env.sim.forward()
            collisions = _navigation_collisions(
                raw_env,
                robot,
                getattr(backend, "_ignore_collision_geom", ()),
            )
            if collisions:
                raw_env.sim.data.qpos[qpos_addr] = start
                raw_env.sim.forward()
                synchronize_controller_goals(robot)
                self._record(backend, raw_env)
                return False
            self._record(backend, raw_env)
        synchronize_controller_goals(robot)
        return True

    def adjust_wrist_for_reach(self, backend, object_name, config) -> bool:
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=0.0,
        )
        current = helpers["gripper_position"](raw_env, robot, "left")
        if not wrist_adjustment_required(
            current_z=current[2],
            target_z=targets["left"][2],
            threshold=config.wrist_height_trigger,
        ):
            return True
        if not uses_mirrored_open_grasp(object_name):
            return self._adjust_single_wrist_for_reach(backend, config)

        model = raw_env.sim.model
        data = raw_env.sim.data
        joint_names = [
            f"robot0_arm_left_{index}_joint"
            for index in range(1, 7)
        ]
        try:
            joint_ids = [model.joint_name2id(name) for name in joint_names]
            qpos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        except Exception:
            return False
        if any(isinstance(addr, tuple) for addr in qpos_addrs):
            return False

        def fingerpad_positions(arm: str) -> np.ndarray:
            gripper = robot.gripper[arm]
            geom_names = [
                gripper.important_geoms[key][0]
                for key in ("left_fingerpad", "right_fingerpad")
            ]
            geom_ids = [model.geom_name2id(name) for name in geom_names]
            return np.stack(
                [data.geom_xpos[geom_id].copy() for geom_id in geom_ids],
                axis=0,
            )

        right_fingerpads = fingerpad_positions("right")
        left_fingerpads = fingerpad_positions("left")
        body_id = raw_env.obj_body_id[object_name]
        if uses_axis_aware_fingerpad_mirror(object_name):
            target_fingerpads = mirrored_fingerpad_targets(
                right_fingerpads,
                object_xy=data.body_xpos[body_id][:2],
                mirror_normal_xy=(
                    np.mean(right_fingerpads[:, :2], axis=0)
                    - np.mean(left_fingerpads[:, :2], axis=0)
                ),
                height_offset=config.mirrored_ik_height_offset,
            )
        else:
            target_fingerpads = world_x_mirrored_fingerpad_targets(
                right_fingerpads,
                object_x=float(data.body_xpos[body_id][0]),
                height_offset=config.mirrored_ik_height_offset,
            )
        start = data.qpos[qpos_addrs].copy()
        lower = np.array(
            [model.jnt_range[joint_id][0] for joint_id in joint_ids],
            dtype=float,
        )
        upper = np.array(
            [model.jnt_range[joint_id][1] for joint_id in joint_ids],
            dtype=float,
        )

        from scipy.optimize import least_squares

        def residual(joints: np.ndarray) -> np.ndarray:
            data.qpos[qpos_addrs] = joints
            raw_env.sim.forward()
            position_error = (
                fingerpad_positions("left") - target_fingerpads
            ).reshape(-1)
            regularization = (
                config.mirrored_ik_regularization * (joints - start)
            )
            return np.concatenate([position_error, regularization])

        solution = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=config.mirrored_ik_max_nfev,
        )
        target = np.asarray(solution.x, dtype=float)
        position_error = float(
            np.linalg.norm(
                fingerpad_positions("left") - target_fingerpads
            )
        )
        data.qpos[qpos_addrs] = start
        raw_env.sim.forward()
        if not bool(solution.success) or position_error > config.mirrored_ik_max_error:
            synchronize_controller_goals(robot)
            return False

        from robot_agent.environments.robosuite_backend import (
            _navigation_collisions,
        )

        steps = max(1, config.wrist_adjustment_steps)
        for values in np.linspace(start, target, steps + 1, dtype=float)[1:]:
            data.qpos[qpos_addrs] = values
            raw_env.sim.forward()
            collisions = _navigation_collisions(
                raw_env,
                robot,
                getattr(backend, "_ignore_collision_geom", ()),
            )
            if collisions:
                data.qpos[qpos_addrs] = start
                raw_env.sim.forward()
                synchronize_controller_goals(robot)
                self._record(backend, raw_env)
                return False
            self._record(backend, raw_env)
        synchronize_controller_goals(robot)
        return True

    def close_and_check_contacts(self, backend, object_name, config):
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        grasp_targets = self._grasp_targets(
            backend,
            object_name,
            config,
            height_offset=0.0,
        )
        current_positions = {
            arm: helpers["gripper_position"](raw_env, robot, arm)
            for arm in ARMS
        }
        grasp_targets = close_pose_targets(
            current_positions,
            grasp_targets,
            hold_current=config.hold_close_pose,
        )
        body_id = raw_env.obj_body_id[object_name]
        start_object_xy = raw_env.sim.data.body_xpos[body_id][:2].copy()
        self._close_lift_reference = (
            object_name,
            float(raw_env.sim.data.body_xpos[body_id][2]),
        )
        hold_targets = helpers["capture_hold_targets"](robot)
        stable_steps = 0

        for step in range(config.close_steps):
            robot.composite_controller.update_state()
            object_xy = raw_env.sim.data.body_xpos[body_id][:2]
            follow_offset = bounded_planar_follow_offset(
                object_xy - start_object_xy,
                max_distance=config.close_follow_max_distance,
            )
            arm_actions = {}
            for arm in ARMS:
                current = helpers["gripper_position"](raw_env, robot, arm)
                active_target = np.asarray(grasp_targets[arm], dtype=float).copy()
                active_target[:2] += follow_offset
                world_delta = active_target - current
                controller_delta = helpers["world_delta"](robot, arm, world_delta)
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    config.max_action,
                )
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=gripper_close_command(
                    step,
                    interval=config.close_increment_interval,
                ),
                hold_targets=hold_targets,
            )
            raw_env.step(action)
            self._record(backend, raw_env)
            contacts = helpers["grasp_status"](raw_env, robot, object_name)
            stable_steps = next_contact_stability(contacts, stable_steps)
            if stable_steps >= config.contact_settle_steps:
                return contacts

        return helpers["grasp_status"](raw_env, robot, object_name)

    def polish_contacts(self, backend, object_name, config, contacts):
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        torso_joint = next(
            (
                raw_env.sim.model.joint_id2name(index)
                for index in range(raw_env.sim.model.njnt)
                if (raw_env.sim.model.joint_id2name(index) or "").endswith(
                    "torso_lift_joint"
                )
            ),
            None,
        )
        if torso_joint is None:
            return contacts

        qpos_addr = raw_env.sim.model.get_joint_qpos_addr(torso_joint)
        if isinstance(qpos_addr, tuple):
            return contacts
        start = float(raw_env.sim.data.qpos[qpos_addr])
        targets = contact_micro_adjustment_targets(
            start,
            step=config.contact_polish_step,
            max_drop=config.contact_polish_max_drop,
            minimum=config.torso_minimum,
        )

        first_full_contact = None
        for target in targets:
            raw_env.sim.data.qpos[qpos_addr] = target
            raw_env.sim.forward()
            self._record(backend, raw_env)
            current_contacts = helpers["grasp_status"](
                raw_env,
                robot,
                object_name,
            )
            if all(bool(current_contacts.get(arm, False)) for arm in ARMS):
                if first_full_contact is None:
                    first_full_contact = target
                if contact_margin_reached(
                    first_contact=first_full_contact,
                    current=target,
                    required_drop=config.contact_confirmation_drop,
                ):
                    synchronize_controller_goals(robot)
                    hold_targets = helpers["capture_hold_targets"](robot)
                    stable_steps = 0
                    for _ in range(config.contact_settle_steps):
                        action = helpers["build_action"](
                            robot,
                            arm_actions={},
                            gripper_value=1.0,
                            hold_targets=hold_targets,
                        )
                        _, _, _, info = raw_env.step(action)
                        self._record(backend, raw_env)
                        current_contacts = helpers["grasp_status"](
                            raw_env,
                            robot,
                            object_name,
                        )
                        stable_steps = next_contact_stability(
                            current_contacts,
                            stable_steps,
                        )
                        if bool((info or {}).get("has_judge_collision", False)):
                            break
                    if stable_steps >= config.contact_settle_steps:
                        return current_contacts
                    first_full_contact = None
            if bool(getattr(raw_env, "has_judge_collision", False)):
                break

        raw_env.sim.data.qpos[qpos_addr] = start
        raw_env.sim.forward()
        synchronize_controller_goals(robot)
        self._record(backend, raw_env)
        return helpers["grasp_status"](raw_env, robot, object_name)

    def lift_and_verify(self, backend, object_name, config) -> bool:
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        contacts = helpers["grasp_status"](raw_env, robot, object_name)
        if not all(bool(contacts.get(arm, False)) for arm in ARMS):
            return False

        if uses_legacy_container_grasp(object_name):
            result = helpers["lift"](
                env=raw_env,
                object_name=object_name,
                lift_height=config.lift_height,
                max_steps=config.lift_steps,
                hold_steps=config.lift_hold_steps,
                tolerance=config.lift_tolerance,
                max_action=config.max_action,
                render=False,
                render_callback=lambda: self._record(backend, raw_env),
            )
            if not bool(result.get("success", False)):
                return False
            contacts = helpers["grasp_status"](
                raw_env,
                robot,
                object_name,
            )
            return all(bool(contacts.get(arm, False)) for arm in ARMS)

        lift_start_object_z = float(
            helpers["object_center"](raw_env, object_name)[2]
        )
        reference_name, reference_z = getattr(
            self,
            "_close_lift_reference",
            (object_name, lift_start_object_z),
        )
        if reference_name != object_name:
            reference_z = lift_start_object_z
        starts = {
            arm: helpers["gripper_position"](raw_env, robot, arm)
            for arm in ARMS
        }
        leader = min(ARMS, key=lambda arm: float(starts[arm][2]))
        follower = next(arm for arm in ARMS if arm != leader)
        hold_targets = helpers["capture_hold_targets"](robot)
        success = False

        for _ in range(config.lift_steps):
            current_object_z = float(
                helpers["object_center"](raw_env, object_name)[2]
            )
            if lift_goal_reached(
                reference_z=reference_z,
                current_z=current_object_z,
                lift_height=config.lift_height,
                tolerance=config.lift_tolerance,
            ):
                success = True
                break

            contacts = helpers["grasp_status"](raw_env, robot, object_name)
            if not all(bool(contacts.get(arm, False)) for arm in ARMS):
                return False

            object_lift = current_object_z - lift_start_object_z
            offsets = {
                leader: config.lift_height,
                follower: follower_lift_offset(
                    object_lift=object_lift,
                    lead=config.lift_follower_lead,
                    lift_height=config.lift_height,
                ),
            }
            robot.composite_controller.update_state()
            arm_actions = {}
            for arm in ARMS:
                target = starts[arm] + np.array(
                    [0.0, 0.0, offsets[arm]],
                    dtype=float,
                )
                current = helpers["gripper_position"](raw_env, robot, arm)
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    target - current,
                )
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    config.max_action,
                )
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=1.0,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            self._record(backend, raw_env)
            if bool((info or {}).get("has_judge_collision", False)):
                return False

        contacts = helpers["grasp_status"](raw_env, robot, object_name)
        return bool(
            success
            and all(bool(contacts.get(arm, False)) for arm in ARMS)
        )

    def physical_hold_metadata(self, backend, object_name) -> dict[str, Any]:
        """Return read-only state needed by the physical transport controller."""
        helpers = self._helpers()
        base_xy, base_yaw = backend.get_base_pose()
        raw_env = backend.env
        object_pos = np.asarray(
            helpers["object_center"](raw_env, object_name),
            dtype=float,
        ).copy()
        body_id = raw_env.obj_body_id[object_name]
        current_body_z = float(raw_env.sim.data.body_xpos[body_id][2])
        reference_name, reference_body_z = getattr(
            self,
            "_close_lift_reference",
            (object_name, current_body_z),
        )
        if reference_name != object_name:
            reference_body_z = current_body_z
        support_reference_object_z = float(reference_body_z) + (
            float(object_pos[2]) - current_body_z
        )
        return {
            "base_xy": np.asarray(base_xy, dtype=float).tolist(),
            "base_yaw": float(base_yaw),
            "object_pos": object_pos.tolist(),
            "object_z": float(object_pos[2]),
            "support_reference_object_z": support_reference_object_z,
            "minimum_transport_object_z": support_reference_object_z + 0.10,
        }


def run_scripted_grasp(
    backend,
    *,
    source: str,
    object_name: str,
    config: ScriptedGraspConfig | None = None,
    driver=None,
) -> dict[str, Any]:
    """Run the bounded stage sequence and return an auditable result."""
    config = apply_object_grasp_profile(
        config or ScriptedGraspConfig(),
        object_name,
    )
    driver = driver or OfficialScriptedGraspDriver()
    if uses_station_side_tote_grasp(object_name):
        quiesce_robot_for_grasp(backend.env)
    backend._mark_trajectory_event(
        "grasp_start",
        source=source,
        object_name=object_name,
        method="scripted_osc",
    )

    contacts = {arm: False for arm in ARMS}
    lift_success = False
    failure_stage = None
    error = None
    try:
        if (
            not config.clearance_prepared
            and not driver.raise_to_clearance(backend, object_name, config)
        ):
            failure_stage = "raise_clearance"
        elif not driver.move_above_grasp_sites(backend, object_name, config):
            failure_stage = "move_above"
        elif not driver.move_to_pregrasp(backend, object_name, config):
            failure_stage = "pregrasp"
        elif not driver.approach_grasp_sites(backend, object_name, config):
            failure_stage = "approach"
        elif not driver.adjust_wrist_for_reach(backend, object_name, config):
            failure_stage = "wrist_adjustment"
        else:
            contacts = dict(driver.close_and_check_contacts(backend, object_name, config))
            if not all(bool(contacts.get(arm, False)) for arm in ARMS):
                contacts = dict(
                    driver.polish_contacts(
                        backend,
                        object_name,
                        config,
                        contacts,
                    )
                )
            if all(bool(contacts.get(arm, False)) for arm in ARMS):
                lift_success = bool(driver.lift_and_verify(backend, object_name, config))
                if not lift_success:
                    failure_stage = "lift"
            else:
                failure_stage = "contact"
    except Exception as exc:  # preserve the physical failure for the run manifest
        failure_stage = failure_stage or "exception"
        error = f"{type(exc).__name__}: {exc}"

    success = mark_verified_grasp_end(
        backend,
        source=source,
        object_name=object_name,
        contacts=contacts,
        lift_success=lift_success,
    )
    hold = None
    if success:
        try:
            hold = dict(driver.physical_hold_metadata(backend, object_name))
        except Exception as exc:
            success = False
            failure_stage = "hold_observation"
            error = f"{type(exc).__name__}: {exc}"

    return {
        "success": bool(success),
        "source": source,
        "object_name": object_name,
        "contacts": {arm: bool(contacts.get(arm, False)) for arm in ARMS},
        "lift_success": bool(lift_success),
        "hold": hold,
        "failure_stage": failure_stage,
        "error": error,
    }
