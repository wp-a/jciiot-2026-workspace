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
        pregrasp_height: float = 0.10,
        site_below_offset: float = 0.035,
        position_tolerance: float = 0.012,
        pregrasp_steps: int = 180,
        approach_steps: int = 180,
        close_steps: int = 80,
        max_action: float = 0.65,
        lift_height: float = 0.15,
        lift_steps: int = 300,
        lift_hold_steps: int = 20,
        lift_tolerance: float = 0.02,
    ) -> None:
        self.pregrasp_height = float(pregrasp_height)
        self.site_below_offset = float(site_below_offset)
        self.position_tolerance = float(position_tolerance)
        self.pregrasp_steps = int(pregrasp_steps)
        self.approach_steps = int(approach_steps)
        self.close_steps = int(close_steps)
        self.max_action = float(max_action)
        self.lift_height = float(lift_height)
        self.lift_steps = int(lift_steps)
        self.lift_hold_steps = int(lift_hold_steps)
        self.lift_tolerance = float(lift_tolerance)


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

    def _move(
        self,
        backend,
        object_name: str,
        config: ScriptedGraspConfig,
        *,
        height_offset: float,
        max_steps: int,
        gripper_value: float,
    ) -> bool:
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        grasp_targets, _ = helpers["get_targets"](
            raw_env,
            object_name,
            config.site_below_offset,
        )
        offset = np.array([0.0, 0.0, float(height_offset)], dtype=float)
        targets = {arm: np.asarray(grasp_targets[arm], dtype=float) + offset for arm in ARMS}
        hold_targets = helpers["capture_hold_targets"](robot)

        for _ in range(max_steps):
            robot.composite_controller.update_state()
            current = {
                arm: helpers["gripper_position"](raw_env, robot, arm)
                for arm in ARMS
            }
            if targets_reached(current, targets, tolerance=config.position_tolerance):
                return True

            arm_actions = {}
            for arm in ARMS:
                world_delta = targets[arm] - current[arm]
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

    def move_to_pregrasp(self, backend, object_name, config) -> bool:
        return self._move(
            backend,
            object_name,
            config,
            height_offset=config.pregrasp_height,
            max_steps=config.pregrasp_steps,
            gripper_value=-1.0,
        )

    def approach_grasp_sites(self, backend, object_name, config) -> bool:
        return self._move(
            backend,
            object_name,
            config,
            height_offset=0.0,
            max_steps=config.approach_steps,
            gripper_value=-1.0,
        )

    def close_and_check_contacts(self, backend, object_name, config):
        helpers = self._helpers()
        raw_env = backend.env
        robot = raw_env.robots[0]
        grasp_targets, _ = helpers["get_targets"](
            raw_env,
            object_name,
            config.site_below_offset,
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
        if not driver.move_to_pregrasp(backend, object_name, config):
            failure_stage = "pregrasp"
        elif not driver.approach_grasp_sites(backend, object_name, config):
            failure_stage = "approach"
        else:
            contacts = dict(driver.close_and_check_contacts(backend, object_name, config))
            if not all(bool(contacts.get(arm, False)) for arm in ARMS):
                failure_stage = "contact"
            else:
                lift_success = bool(driver.lift_and_verify(backend, object_name, config))
                if not lift_success:
                    failure_stage = "lift"
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
