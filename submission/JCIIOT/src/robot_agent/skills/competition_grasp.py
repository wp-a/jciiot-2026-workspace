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
        site_below_offset: float = 0.035,
        position_tolerance: float = 0.012,
        approach_tolerance: float = 0.055,
        pregrasp_steps: int = 180,
        approach_steps: int = 180,
        close_steps: int = 300,
        contact_polish_step: float = 0.001,
        contact_polish_max_drop: float = 0.030,
        contact_confirmation_drop: float = 0.003,
        max_action: float = 0.65,
        lift_height: float = 0.05,
        lift_steps: int = 300,
        lift_hold_steps: int = 0,
        lift_tolerance: float = 0.01,
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
        self.max_action = float(max_action)
        self.lift_height = float(lift_height)
        self.lift_steps = int(lift_steps)
        self.lift_hold_steps = int(lift_hold_steps)
        self.lift_tolerance = float(lift_tolerance)
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
            "lift": lift_grasped_object,
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
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        current = helpers["current_part_qpos"](robot, "torso")
        if current is None:
            return False
        target = lowered_torso_target(
            current,
            drop=config.torso_drop,
            minimum=config.torso_minimum,
        )
        hold_targets = helpers["capture_hold_targets"](robot)
        hold_targets["torso"] = target

        for _ in range(config.torso_steps):
            current = helpers["current_part_qpos"](robot, "torso")
            if current is not None and float(np.max(np.abs(current - target))) <= config.torso_tolerance:
                return True
            action = helpers["build_action"](
                robot,
                arm_actions={},
                gripper_value=-1.0,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            self._record(backend, raw_env)
            if bool((info or {}).get("has_judge_collision", False)):
                return False
        return False

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

    def close_and_check_contacts(self, backend, object_name, config):
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        raw_targets, _ = helpers["get_targets"](
            raw_env,
            object_name,
            config.site_below_offset,
        )
        grasp_targets = assigned_grasp_targets(
            raw_targets,
            swap=config.swap_arm_targets,
        )
        hold_targets = helpers["capture_hold_targets"](robot)

        for _ in range(config.close_steps):
            robot.composite_controller.update_state()
            arm_actions = {}
            for arm in ARMS:
                current = helpers["gripper_position"](raw_env, robot, arm)
                world_delta = np.asarray(grasp_targets[arm], dtype=float) - current
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
                gripper_value=1.0,
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
                    return current_contacts
            if bool(getattr(raw_env, "has_judge_collision", False)):
                break

        raw_env.sim.data.qpos[qpos_addr] = start
        raw_env.sim.forward()
        self._record(backend, raw_env)
        return helpers["grasp_status"](raw_env, robot, object_name)

    def lift_and_verify(self, backend, object_name, config) -> bool:
        helpers = self._helpers()
        result = helpers["lift"](
            env=backend.env,
            object_name=object_name,
            lift_height=config.lift_height,
            max_steps=config.lift_steps,
            hold_steps=config.lift_hold_steps,
            tolerance=config.lift_tolerance,
            max_action=config.max_action,
            render=False,
            render_callback=lambda: self._record(backend, backend.env),
        )
        if not bool(result.get("success", False)):
            return False
        contacts = helpers["grasp_status"](
            backend.env,
            backend.env.robots[0],
            object_name,
        )
        return all(bool(contacts.get(arm, False)) for arm in ARMS)

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
