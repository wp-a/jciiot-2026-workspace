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
        wrist_adjustment_steps: int = 20,
        wrist_height_trigger: float = 0.04,
        hold_close_pose: bool = True,
        face_insertion: float = 0.0,
        close_follow_max_distance: float = 0.0,
        close_increment_interval: int = 20,
        max_action: float = 0.65,
        lift_height: float = 0.05,
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
        self.wrist_adjustment_steps = int(wrist_adjustment_steps)
        self.wrist_height_trigger = float(wrist_height_trigger)
        self.hold_close_pose = bool(hold_close_pose)
        self.face_insertion = float(face_insertion)
        self.close_follow_max_distance = float(close_follow_max_distance)
        self.close_increment_interval = int(close_increment_interval)
        self.max_action = float(max_action)
        self.lift_height = float(lift_height)
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


def synchronize_controller_goals(robot) -> None:
    """Reset moving-base controller goals to the current simulated posture."""
    composite = robot.composite_controller
    composite.update_state()
    for part_name in (*ARMS, "torso"):
        controller = composite.part_controllers.get(part_name)
        if controller is None:
            continue
        controller.update(force=True)
        controller.reset_goal()


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
            object_center_pos,
            world_delta_to_controller_frame,
        )
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            get_target_positions,
            grasp_status,
            gripper_end_center_pos,
        )
        from robosuite.environments.factory_sorting.transport_attachment import (
            capture_transport_attachment,
        )

        return {
            "arm_action": arm_delta_to_normalized_action,
            "build_action": build_action,
            "capture_hold_targets": capture_hold_targets,
            "current_part_qpos": current_part_qpos,
            "object_center": object_center_pos,
            "world_delta": world_delta_to_controller_frame,
            "get_targets": get_target_positions,
            "grasp_status": grasp_status,
            "gripper_position": gripper_end_center_pos,
            "attach": capture_transport_attachment,
        }

    @staticmethod
    def _record(backend, raw_env) -> None:
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)

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
        grasp_targets = assigned_grasp_targets(
            raw_targets,
            swap=config.swap_arm_targets,
        )
        body_id = backend.env.obj_body_id[object_name]
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

        steps = max(1, config.wrist_adjustment_steps)
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
        hold_targets = helpers["capture_hold_targets"](robot)

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
            if all(bool(contacts.get(arm, False)) for arm in ARMS):
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

        start_object_z = float(
            helpers["object_center"](raw_env, object_name)[2]
        )
        target_object_z = start_object_z + config.lift_height
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
            if current_object_z >= target_object_z - config.lift_tolerance:
                success = True
                break

            contacts = helpers["grasp_status"](raw_env, robot, object_name)
            if not all(bool(contacts.get(arm, False)) for arm in ARMS):
                return False

            object_lift = current_object_z - start_object_z
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

    def attach_for_transport(self, backend, object_name) -> None:
        helpers = self._helpers()
        helpers["attach"](backend.env, object_name)
        backend._held_crate_name = object_name
        backend._held_crate_body_id = backend.env.obj_body_id.get(object_name)
        self._record(backend, backend.env)


def run_scripted_grasp(
    backend,
    *,
    source: str,
    object_name: str,
    config: ScriptedGraspConfig | None = None,
    driver=None,
) -> dict[str, Any]:
    """Run the bounded stage sequence and return an auditable result."""
    config = config or ScriptedGraspConfig()
    driver = driver or OfficialScriptedGraspDriver()
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
    if success:
        try:
            driver.attach_for_transport(backend, object_name)
        except Exception as exc:
            success = False
            failure_stage = "transport_attachment"
            error = f"{type(exc).__name__}: {exc}"

    return {
        "success": bool(success),
        "source": source,
        "object_name": object_name,
        "contacts": {arm: bool(contacts.get(arm, False)) for arm in ARMS},
        "lift_success": bool(lift_success),
        "failure_stage": failure_stage,
        "error": error,
    }
