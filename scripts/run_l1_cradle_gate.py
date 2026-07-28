#!/usr/bin/env python3
"""Run and validate the L1 physical cradle-transfer research gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


CRADLE_GATE_THRESHOLDS = {
    "lift_m": 0.13,
    "support_contact_steps": 20,
    "base_translation_m": 0.50,
}

_REQUIRED_FIELDS = (
    "physical_grasp",
    "lift_m",
    "support_contact_steps",
    "base_translation_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "dropped",
    "infrastructure_error",
)


def cradle_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the hard L1 physical-transfer gate."""
    failures = [key for key in _REQUIRED_FIELDS if key not in record]

    def reject_numeric(key: str, *, minimum: float | None = None) -> None:
        if key not in record:
            return
        value = record[key]
        if isinstance(value, bool):
            failures.append(key)
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failures.append(key)
            return
        if not np.isfinite(numeric):
            failures.append(key)
            return
        if minimum is not None and numeric < minimum:
            failures.append(key)

    if record.get("physical_grasp") is not True:
        failures.append("physical_grasp")
    reject_numeric("lift_m", minimum=CRADLE_GATE_THRESHOLDS["lift_m"])
    reject_numeric(
        "support_contact_steps",
        minimum=CRADLE_GATE_THRESHOLDS["support_contact_steps"],
    )
    reject_numeric(
        "base_translation_m",
        minimum=CRADLE_GATE_THRESHOLDS["base_translation_m"],
    )

    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        reject_numeric(key)
        if key in record:
            try:
                if float(record[key]) != 0.0:
                    failures.append(key)
            except (TypeError, ValueError):
                pass

    if record.get("dropped") is not False:
        failures.append("dropped")
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")

    return list(dict.fromkeys(failures))


def cradle_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only a complete record that passes every hard condition."""
    return not cradle_gate_failures(record)


_PUSH_REQUIRED_FIELDS = (
    "physical_contact_steps",
    "object_translation_m",
    "base_translation_m",
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "infrastructure_error",
)


def push_gate_failures(record: Mapping[str, object]) -> list[str]:
    """Return failed evidence fields for the hard L1 physical-push gate."""
    failures = [key for key in _PUSH_REQUIRED_FIELDS if key not in record]

    def numeric(key: str) -> float | None:
        if key not in record or isinstance(record[key], bool):
            return None
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    minimums = {
        "physical_contact_steps": 20.0,
        "object_translation_m": 0.50,
        "base_translation_m": 0.30,
    }
    for key, minimum in minimums.items():
        value = numeric(key)
        if value is None or value < minimum:
            failures.append(key)
    for key in ("attachment_calls", "object_pose_writes", "collision_frames"):
        value = numeric(key)
        if value is None or value != 0.0:
            failures.append(key)
    if record.get("infrastructure_error") is not None:
        failures.append("infrastructure_error")
    return list(dict.fromkeys(failures))


def push_gate_accepted(record: Mapping[str, object]) -> bool:
    """Accept only complete, collision-free physical-push evidence."""
    return not push_gate_failures(record)


def has_bilateral_object_contact(
    contacts: Mapping[str, tuple[str, ...]],
) -> bool:
    """Return whether both arms have at least one physical object contact."""
    return all(bool(contacts.get(arm)) for arm in ("right", "left"))


def opposed_wall_clearance_targets(
    current_positions: Mapping[str, np.ndarray],
    *,
    separation_axis: np.ndarray,
    clearance_m: float,
) -> dict[str, np.ndarray]:
    """Move both end effectors outward before descending around opposed walls."""
    distance = float(clearance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("clearance_m must be a finite non-negative value")
    axis = np.asarray(separation_axis, dtype=float).reshape(3).copy()
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("separation_axis must have a horizontal component")
    axis /= norm
    right = np.asarray(current_positions["right"], dtype=float).reshape(3)
    left = np.asarray(current_positions["left"], dtype=float).reshape(3)
    if float(np.dot(right - left, axis)) < 0.0:
        axis *= -1.0
    return {
        "right": right + axis * distance,
        "left": left - axis * distance,
    }


def opposed_wall_squeeze_targets(
    current_positions: Mapping[str, np.ndarray],
    *,
    separation_axis: np.ndarray,
    squeeze_m: float,
) -> dict[str, np.ndarray]:
    """Move both end effectors inward to apply opposed wall pressure."""
    outward = opposed_wall_clearance_targets(
        current_positions,
        separation_axis=separation_axis,
        clearance_m=squeeze_m,
    )
    return {
        arm: 2.0 * np.asarray(current_positions[arm], dtype=float) - outward[arm]
        for arm in ("right", "left")
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_candidate(candidate_root: Path) -> Path:
    app_dir = candidate_root / "JCIIOT"
    if not (app_dir / "app.py").is_file():
        raise FileNotFoundError(f"official app not found: {app_dir / 'app.py'}")
    paths = (
        app_dir / "src",
        app_dir,
        app_dir / "robomimic",
        app_dir / "robosuite",
        app_dir / "robosuite" / "robosuite",
    )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.chdir(app_dir)
    return app_dir


def _load_scene(app_dir: Path, task: Mapping[str, Any], seed: int):
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.environments import RobosuiteBackend

    map_dir = (
        app_dir
        / "robosuite"
        / "robosuite"
        / "environments"
        / "factory_sorting"
        / "generated_maps"
    )
    prefix = str(task["scene_prefix"])
    semantic_path = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid_path = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    scene_data, grid = load_map_files(semantic_path, grid_path)
    scene_context = SceneContext.from_semantic_map(scene_data)
    backend = RobosuiteBackend(
        env_name=str(task["env_name"]),
        camera="birdview",
        headless=True,
        drive_mode="direct",
        seed=int(seed),
    )
    backend._scene_context = scene_context
    backend.reset()
    return backend, scene_context, grid


def _object_body_ids(raw_env, object_name: str) -> set[int]:
    model = raw_env.sim.model
    root = int(raw_env.obj_body_id[object_name])
    descendants = {root}
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


def object_robot_contacts(raw_env, object_name: str) -> dict[str, tuple[str, ...]]:
    """Return robot geoms that currently contact the named physical object."""
    model = raw_env.sim.model
    object_bodies = _object_body_ids(raw_env, object_name)
    result = {"right": set(), "left": set()}
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
        if not name:
            continue
        lowered = name.lower()
        if lowered.startswith("gripper0_left_") or "_left_collision" in lowered:
            result["left"].add(name)
        elif lowered.startswith("gripper0_right_") or lowered.startswith("robot0_arm_"):
            result["right"].add(name)
    return {arm: tuple(sorted(names)) for arm, names in result.items()}


def geometry_snapshot(raw_env, object_name: str) -> dict[str, Any]:
    """Capture world geometry needed to design a physical support transition."""
    model = raw_env.sim.model
    data = raw_env.sim.data
    object_bodies = _object_body_ids(raw_env, object_name)
    selected = []
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        if not name:
            continue
        lowered = name.lower()
        is_object = int(model.geom_bodyid[geom_id]) in object_bodies
        is_support_link = any(
            token in lowered
            for token in (
                "arm_5_collision",
                "arm_6_collision",
                "arm_5_left_collision",
                "arm_6_left_collision",
                "gripper0_right_hand_collision",
                "gripper0_left_hand_collision",
            )
        )
        if not (is_object or is_support_link):
            continue
        selected.append(
            {
                "name": name,
                "is_object": is_object,
                "world_position": np.asarray(data.geom_xpos[geom_id], dtype=float).tolist(),
                "world_rotation": np.asarray(data.geom_xmat[geom_id], dtype=float)
                .reshape(3, 3)
                .tolist(),
                "size": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
                "type": int(model.geom_type[geom_id]),
            }
        )
    body_id = raw_env.obj_body_id[object_name]
    return {
        "object_position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "geometries": selected,
    }


def _hold_probe(backend, object_name: str, *, steps: int) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import _is_allowed_cradle_geom

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    hold_targets = helpers["capture_hold_targets"](robot)
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0
    observations = []
    for step in range(int(steps)):
        robot.composite_controller.update_state()
        action = helpers["build_action"](
            robot,
            arm_actions={},
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        contacts = object_robot_contacts(raw_env, object_name)
        supported = all(
            any(_is_allowed_cradle_geom(name, arm) for name in contacts[arm])
            for arm in ("right", "left")
        )
        stable_support_steps = stable_support_steps + 1 if supported else 0
        maximum_support_steps = max(maximum_support_steps, stable_support_steps)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        body_id = raw_env.obj_body_id[object_name]
        observations.append(
            {
                "step": step + 1,
                "object_z": float(raw_env.sim.data.body_xpos[body_id][2]),
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "bilateral_support": supported,
                "judge_collision": collision,
            }
        )
    return {
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "observations": observations,
    }


def _inward_support_probe(
    backend,
    object_name: str,
    *,
    inward_m: float,
    move_steps: int,
    hold_steps: int,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import _is_allowed_cradle_geom

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
    start_geometry = geometry_snapshot(raw_env, object_name)
    starts = {
        arm: np.asarray(
            helpers["gripper_position"](raw_env, robot, arm),
            dtype=float,
        )
        for arm in ("right", "left")
    }
    targets = {}
    for arm, start in starts.items():
        direction = object_position - start
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise RuntimeError(f"cannot resolve inward direction for {arm} arm")
        targets[arm] = start + direction / norm * float(inward_m)

    hold_targets = helpers["capture_hold_targets"](robot)
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0
    observations = []
    total_steps = int(move_steps) + int(hold_steps)
    for step in range(total_steps):
        robot.composite_controller.update_state()
        arm_actions = {}
        if step < int(move_steps):
            for arm in ("right", "left"):
                current = helpers["gripper_position"](raw_env, robot, arm)
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    targets[arm] - current,
                )
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    0.30,
                )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        contacts = object_robot_contacts(raw_env, object_name)
        supported = all(
            any(_is_allowed_cradle_geom(name, arm) for name in contacts[arm])
            for arm in ("right", "left")
        )
        stable_support_steps = stable_support_steps + 1 if supported else 0
        maximum_support_steps = max(maximum_support_steps, stable_support_steps)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        current_positions = {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }
        observations.append(
            {
                "step": step + 1,
                "phase": "inward" if step < int(move_steps) else "hold",
                "object_z": float(raw_env.sim.data.body_xpos[body_id][2]),
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "bilateral_support": supported,
                "judge_collision": collision,
                "eef_positions": {
                    arm: current_positions[arm].tolist()
                    for arm in ("right", "left")
                },
            }
        )
        if collision:
            break
    return {
        "requested_inward_m": float(inward_m),
        "move_steps": int(move_steps),
        "hold_steps": int(hold_steps),
        "start_eef_positions": {arm: starts[arm].tolist() for arm in starts},
        "target_eef_positions": {arm: targets[arm].tolist() for arm in targets},
        "start_geometry": start_geometry,
        "end_geometry": geometry_snapshot(raw_env, object_name),
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "observations": observations,
    }


def _physical_push_probe(
    backend,
    object_name: str,
    *,
    table_object_z: float,
    push_distance_m: float,
    max_push_steps: int,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import OfficialScriptedGraspDriver
    from robot_agent.skills.competition_transport import (
        OfficialPhysicalCarryDriver,
        world_velocity_to_base_frame,
    )

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    hold_targets = helpers["capture_hold_targets"](robot)
    observations = []
    collision_steps = 0
    start_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
    start_object_xy = np.asarray(
        raw_env.sim.data.body_xpos[body_id][:2],
        dtype=float,
    )
    direction = start_object_xy - start_base_xy
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        raise RuntimeError("base and object centers coincide before physical push")
    direction /= direction_norm

    def eef_positions() -> dict[str, np.ndarray]:
        return {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }

    shift_starts = eef_positions()
    shift_targets = {
        arm: position
        + np.array([direction[0] * 0.20, direction[1] * 0.20, 0.0])
        for arm, position in shift_starts.items()
    }
    for step in range(180):
        robot.composite_controller.update_state()
        current = eef_positions()
        arm_actions = {}
        for arm in ("right", "left"):
            controller_delta = helpers["world_delta"](
                robot,
                arm,
                shift_targets[arm] - current[arm],
            )
            arm_actions[arm] = helpers["arm_action"](
                robot,
                arm,
                controller_delta,
                0.30,
            )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        observations.append(
            {
                "phase": "arm_shift_outward",
                "step": step + 1,
                "base_xy": np.asarray(backend.get_base_pose()[0], dtype=float).tolist(),
                "object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id],
                    dtype=float,
                ).tolist(),
                "contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
                "judge_collision": collision,
            }
        )
        if collision:
            break
        if all(
            float(np.linalg.norm(shift_targets[arm] - current[arm])) <= 0.012
            for arm in ("right", "left")
        ):
            break
    if collision_steps:
        return {
            "success": False,
            "failure_stage": "arm_shift_outward",
            "physical_contact_steps": 0,
            "object_translation_m": float(
                np.linalg.norm(raw_env.sim.data.body_xpos[body_id][:2] - start_object_xy)
            ),
            "base_translation_m": 0.0,
            "collision_steps": collision_steps,
            "observations": observations,
        }

    lower_starts = eef_positions()
    lower_targets = {
        arm: position + np.array([0.0, 0.0, -0.25], dtype=float)
        for arm, position in lower_starts.items()
    }
    lower_success = False
    for step in range(180):
        robot.composite_controller.update_state()
        current = eef_positions()
        arm_actions = {}
        for arm in ("right", "left"):
            controller_delta = helpers["world_delta"](
                robot,
                arm,
                lower_targets[arm] - current[arm],
            )
            arm_actions[arm] = helpers["arm_action"](
                robot,
                arm,
                controller_delta,
                0.30,
            )
        action = helpers["build_action"](
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        _, _, _, info = raw_env.step(action)
        recorder = getattr(backend, "_record_trajectory_frame", None)
        if callable(recorder):
            recorder(_env=raw_env)
        collision = bool((info or {}).get("has_judge_collision", False))
        collision_steps += int(collision)
        body_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
        observations.append(
            {
                "phase": "lower_to_table",
                "step": step + 1,
                "base_xy": np.asarray(backend.get_base_pose()[0], dtype=float).tolist(),
                "object_position": body_position.tolist(),
                "contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
                "judge_collision": collision,
            }
        )
        if collision:
            break
        if body_position[2] <= float(table_object_z) + 0.06:
            lower_success = True
            break

    if not lower_success:
        return {
            "success": False,
            "failure_stage": "lower_to_table",
            "physical_contact_steps": 0,
            "object_translation_m": float(
                np.linalg.norm(raw_env.sim.data.body_xpos[body_id][:2] - start_object_xy)
            ),
            "base_translation_m": float(
                np.linalg.norm(np.asarray(backend.get_base_pose()[0]) - start_base_xy)
            ),
            "collision_steps": collision_steps,
            "observations": observations,
        }

    _, base_yaw = backend.get_base_pose()
    base_velocity_xy = world_velocity_to_base_frame(
        direction * 0.04,
        base_yaw,
    )
    base_command = np.array(
        [base_velocity_xy[0], base_velocity_xy[1], 0.0],
        dtype=float,
    )
    stable_contact_steps = 0
    maximum_contact_steps = 0
    object_translation = 0.0
    base_translation = 0.0
    failure_stage = "timeout"
    success = False
    carry_driver = OfficialPhysicalCarryDriver()
    arm_push_delta = np.array(
        [direction[0] * 0.002, direction[1] * 0.002, 0.0],
        dtype=float,
    )
    for step in range(int(max_push_steps)):
        step_info = carry_driver.step(
            backend,
            object_name=object_name,
            base_command=base_command,
            hold_targets=hold_targets,
            arm_world_deltas={
                "right": arm_push_delta,
                "left": arm_push_delta,
            },
            gripper_value=1.0,
            base_control_dt=0.05,
        )
        collision = bool(step_info.get("collision", False))
        collision_steps += int(collision)
        contacts = object_robot_contacts(raw_env, object_name)
        has_contact = any(contacts[arm] for arm in ("right", "left"))
        stable_contact_steps = stable_contact_steps + 1 if has_contact else 0
        maximum_contact_steps = max(maximum_contact_steps, stable_contact_steps)
        base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
        object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
        base_translation = float(np.linalg.norm(base_xy - start_base_xy))
        object_translation = float(
            np.linalg.norm(object_position[:2] - start_object_xy)
        )
        observations.append(
            {
                "phase": "physical_push",
                "step": step + 1,
                "base_xy": base_xy.tolist(),
                "object_position": object_position.tolist(),
                "base_translation_m": base_translation,
                "object_translation_m": object_translation,
                "contacts": {arm: list(contacts[arm]) for arm in contacts},
                "judge_collision": collision,
            }
        )
        if collision:
            from robot_agent.environments.robosuite_backend import (
                _navigation_collisions,
            )

            observations[-1]["judge_collision_pairs"] = [
                list(pair)
                for pair in _navigation_collisions(
                    raw_env,
                    robot,
                    getattr(backend, "_ignore_collision_geom", ()),
                )
            ]
            failure_stage = "collision"
            break
        if (
            base_translation >= 0.30
            and object_translation >= float(push_distance_m)
            and maximum_contact_steps >= 20
        ):
            success = True
            failure_stage = None
            break
        if base_translation >= 0.30:
            failure_stage = "object_distance"
            break

    return {
        "success": success,
        "failure_stage": failure_stage,
        "physical_contact_steps": maximum_contact_steps,
        "object_translation_m": object_translation,
        "base_translation_m": base_translation,
        "collision_steps": collision_steps,
        "push_direction": direction.tolist(),
        "observations": observations,
        "final_geometry": geometry_snapshot(raw_env, object_name),
    }


def _center_regrasp_probe(
    backend,
    object_name: str,
    *,
    table_object_z: float,
    center_shift_m: float,
    wall_clearance_m: float,
    wall_squeeze_m: float,
    hold_steps: int,
) -> dict[str, Any]:
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        gripper_close_command,
    )
    from robot_agent.skills.competition_transport import _is_allowed_cradle_geom

    helpers = OfficialScriptedGraspDriver._helpers()
    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[object_name]
    hold_targets = helpers["capture_hold_targets"](robot)
    observations = []
    stage_results = []
    stable_support_steps = 0
    maximum_support_steps = 0
    collision_steps = 0

    def eef_positions() -> dict[str, np.ndarray]:
        return {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm),
                dtype=float,
            )
            for arm in ("right", "left")
        }

    def execute_stage(
        name: str,
        targets: Mapping[str, np.ndarray],
        *,
        max_steps: int,
        gripper_value: float,
        close_schedule: bool = False,
        stop_object_z: float | None = None,
        stop_object_z_at_least: float | None = None,
        stop_bilateral_support_steps: int | None = None,
        stop_bilateral_contact_steps: int | None = None,
        stop_grasp_contact_steps: int | None = None,
        support_seek_down_step: float = 0.0,
        support_seek_down_limit: float = 0.0,
        require_target: bool = True,
    ) -> bool:
        nonlocal stable_support_steps
        nonlocal maximum_support_steps
        nonlocal collision_steps
        reached = False
        collision = False
        stage_support_steps = 0
        stage_contact_steps = 0
        support_stop_met = False
        contact_stop_met = False
        object_stop_met = False
        grasp_stop_met = False
        stage_grasp_steps = 0
        active_targets = {
            arm: np.asarray(targets[arm], dtype=float).copy()
            for arm in ("right", "left")
        }
        initial_target_z = {
            arm: float(active_targets[arm][2]) for arm in ("right", "left")
        }
        for local_step in range(int(max_steps)):
            robot.composite_controller.update_state()
            current = eef_positions()
            arm_actions = {}
            for arm in ("right", "left"):
                controller_delta = helpers["world_delta"](
                    robot,
                    arm,
                    active_targets[arm] - current[arm],
                )
                arm_actions[arm] = helpers["arm_action"](
                    robot,
                    arm,
                    controller_delta,
                    0.30,
                )
            command = (
                gripper_close_command(local_step, interval=1)
                if close_schedule
                else float(gripper_value)
            )
            action = helpers["build_action"](
                robot,
                arm_actions=arm_actions,
                gripper_value=command,
                hold_targets=hold_targets,
            )
            _, _, _, info = raw_env.step(action)
            recorder = getattr(backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder(_env=raw_env)
            contacts = object_robot_contacts(raw_env, object_name)
            bilateral_contact = has_bilateral_object_contact(contacts)
            arm_supported = {
                arm: any(
                    _is_allowed_cradle_geom(geom, arm)
                    for geom in contacts[arm]
                )
                for arm in ("right", "left")
            }
            supported = all(arm_supported.values())
            grasp_contacts = helpers["grasp_status"](
                raw_env,
                robot,
                object_name,
            )
            grasped = all(
                bool(grasp_contacts.get(arm)) for arm in ("right", "left")
            )
            stage_grasp_steps = stage_grasp_steps + 1 if grasped else 0
            stage_support_steps = stage_support_steps + 1 if supported else 0
            stage_contact_steps = stage_contact_steps + 1 if bilateral_contact else 0
            stable_support_steps = stable_support_steps + 1 if supported else 0
            maximum_support_steps = max(maximum_support_steps, stable_support_steps)
            collision = bool((info or {}).get("has_judge_collision", False))
            collision_steps += int(collision)
            object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
            observations.append(
                {
                    "stage": name,
                    "step": local_step + 1,
                    "object_position": object_position.tolist(),
                    "eef_positions": {
                        arm: current[arm].tolist() for arm in ("right", "left")
                    },
                    "contacts": {arm: list(contacts[arm]) for arm in contacts},
                    "bilateral_support": supported,
                    "bilateral_grasp": grasped,
                    "judge_collision": collision,
                }
            )
            if collision:
                break
            if float(support_seek_down_step) > 0.0:
                for arm in ("right", "left"):
                    if arm_supported[arm]:
                        continue
                    active_targets[arm][2] = max(
                        initial_target_z[arm] - float(support_seek_down_limit),
                        active_targets[arm][2] - float(support_seek_down_step),
                    )
            if stop_object_z is not None and object_position[2] <= float(stop_object_z):
                reached = True
                object_stop_met = True
                break
            if (
                stop_object_z_at_least is not None
                and object_position[2] >= float(stop_object_z_at_least)
            ):
                reached = True
                object_stop_met = True
                break
            if (
                stop_bilateral_support_steps is not None
                and stage_support_steps >= int(stop_bilateral_support_steps)
            ):
                reached = True
                support_stop_met = True
                break
            if (
                stop_bilateral_contact_steps is not None
                and stage_contact_steps >= int(stop_bilateral_contact_steps)
            ):
                reached = True
                contact_stop_met = True
                break
            if (
                stop_grasp_contact_steps is not None
                and stage_grasp_steps >= int(stop_grasp_contact_steps)
            ):
                reached = True
                grasp_stop_met = True
                break
            reached = all(
                float(np.linalg.norm(active_targets[arm] - current[arm])) <= 0.012
                for arm in ("right", "left")
            )
            if (
                reached
                and require_target
                and not close_schedule
                and stop_bilateral_support_steps is None
                and stop_bilateral_contact_steps is None
                and stop_grasp_contact_steps is None
                and stop_object_z is None
                and stop_object_z_at_least is None
            ):
                break
        if stop_bilateral_support_steps is not None:
            reached = support_stop_met
        if stop_bilateral_contact_steps is not None:
            reached = contact_stop_met
        if stop_grasp_contact_steps is not None:
            reached = grasp_stop_met
        if stop_object_z is not None or stop_object_z_at_least is not None:
            reached = object_stop_met
        if not require_target and not collision:
            reached = True
        stage_results.append(
            {
                "stage": name,
                "success": bool(reached and not collision),
                "collision": collision,
                "steps": sum(1 for item in observations if item["stage"] == name),
                "final_object_position": np.asarray(
                    raw_env.sim.data.body_xpos[body_id],
                    dtype=float,
                ).tolist(),
                "final_contacts": {
                    arm: list(names)
                    for arm, names in object_robot_contacts(raw_env, object_name).items()
                },
            }
        )
        return bool(reached and not collision)

    current = eef_positions()
    lower_targets = {
        arm: position + np.array([0.0, 0.0, -0.25], dtype=float)
        for arm, position in current.items()
    }
    lowered_to_table = execute_stage(
        "lower_to_table",
        lower_targets,
        max_steps=180,
        gripper_value=1.0,
        stop_object_z=float(table_object_z) + 0.006,
    )
    if (
        not lowered_to_table
        and not stage_results[-1]["collision"]
        and float(raw_env.sim.data.body_xpos[body_id][2])
        <= float(table_object_z) + 0.06
    ):
        lowered_to_table = True
    if not lowered_to_table and not stage_results[-1]["collision"]:
        torso_target = hold_targets.get("torso")
        if torso_target is not None:
            torso_target = np.asarray(torso_target, dtype=float).copy()
            torso_target[0] = max(0.05, float(torso_target[0]) - 0.06)
            hold_targets["torso"] = torso_target
            current = eef_positions()
            torso_lower_targets = {
                arm: position + np.array([0.0, 0.0, -0.10], dtype=float)
                for arm, position in current.items()
            }
            lowered_to_table = execute_stage(
                "lower_torso_assist",
                torso_lower_targets,
                max_steps=140,
                gripper_value=1.0,
                stop_object_z=float(table_object_z) + 0.006,
            )
    if not lowered_to_table:
        return {
            "success": False,
            "failure_stage": stage_results[-1]["stage"],
            "support_contact_steps": maximum_support_steps,
            "collision_steps": collision_steps,
            "stages": stage_results,
            "observations": observations,
        }

    table_hold = eef_positions()
    if not execute_stage(
        "open_on_table",
        table_hold,
        max_steps=40,
        gripper_value=-1.0,
        require_target=False,
    ):
        failure_stage = "open_on_table"
    else:
        raw_targets, _ = helpers["get_targets"](raw_env, object_name, 0.0)
        separation_axis = np.asarray(raw_targets["right"] - raw_targets["left"], dtype=float)
        separation_axis[2] = 0.0
        separation_axis /= float(np.linalg.norm(separation_axis))
        clearance_height = 0.18
        current = eef_positions()
        clearance_targets = {
            arm: position + np.array([0.0, 0.0, clearance_height])
            for arm, position in current.items()
        }
        if not execute_stage(
            "raise_open_clearance",
            clearance_targets,
            max_steps=180,
            gripper_value=-1.0,
        ):
            return {
                "success": False,
                "failure_stage": "raise_open_clearance",
                "support_contact_steps": maximum_support_steps,
                "collision_steps": collision_steps,
                "stages": stage_results,
                "observations": observations,
            }
        current = eef_positions()
        retreat_targets = opposed_wall_clearance_targets(
            current,
            separation_axis=separation_axis,
            clearance_m=wall_clearance_m,
        )
        if not execute_stage(
            "retreat_from_walls",
            retreat_targets,
            max_steps=100,
            gripper_value=-1.0,
        ):
            failure_stage = "retreat_from_walls"
        else:
            current = eef_positions()
            midpoint = (current["right"] + current["left"]) * 0.5
            object_position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
            center_delta = object_position - midpoint
            center_delta[2] = 0.0
            center_delta -= separation_axis * float(
                np.dot(center_delta, separation_axis)
            )
            norm = float(np.linalg.norm(center_delta))
            if norm > float(center_shift_m):
                center_delta *= float(center_shift_m) / norm
            center_targets = {
                arm: position + center_delta for arm, position in current.items()
            }
            if not execute_stage(
                "translate_to_center",
                center_targets,
                max_steps=140,
                gripper_value=-1.0,
            ):
                failure_stage = "translate_to_center"
            else:
                current = eef_positions()
                approach_targets = {
                    arm: np.array(
                        [
                            position[0],
                            position[1],
                            float(table_object_z) + 0.115,
                        ],
                        dtype=float,
                    )
                    for arm, position in current.items()
                }
                if not execute_stage(
                    "approach_center_walls",
                    approach_targets,
                    max_steps=220,
                    gripper_value=-1.0,
                    stop_bilateral_contact_steps=1,
                ):
                    failure_stage = "approach_center_walls"
                else:
                    squeeze_targets = opposed_wall_squeeze_targets(
                        eef_positions(),
                        separation_axis=separation_axis,
                        squeeze_m=wall_squeeze_m,
                    )
                    if not execute_stage(
                        "squeeze_center_walls",
                        squeeze_targets,
                        max_steps=120,
                        gripper_value=-1.0,
                        stop_bilateral_contact_steps=1,
                    ):
                        failure_stage = "squeeze_center_walls"
                    else:
                        current = eef_positions()
                        lift_targets = {
                            arm: position + np.array([0.0, 0.0, 0.17])
                            for arm, position in current.items()
                        }
                        if not execute_stage(
                            "lift_center_squeeze",
                            lift_targets,
                            max_steps=180,
                            gripper_value=-1.0,
                            stop_object_z_at_least=float(table_object_z) + 0.14,
                        ):
                            failure_stage = "lift_center_squeeze"
                        else:
                            elevated_targets = eef_positions()
                            if not execute_stage(
                                "hold_center_squeeze",
                                elevated_targets,
                                max_steps=max(20, int(hold_steps)),
                                gripper_value=-1.0,
                                require_target=False,
                            ):
                                failure_stage = "hold_center_squeeze"
                            else:
                                final_contacts = object_robot_contacts(
                                    raw_env,
                                    object_name,
                                )
                                failure_stage = (
                                    None
                                    if has_bilateral_object_contact(final_contacts)
                                    else "final_contact"
                                )

    final_object_z = float(raw_env.sim.data.body_xpos[body_id][2])
    return {
        "success": failure_stage is None,
        "failure_stage": failure_stage,
        "lift_m": final_object_z - float(table_object_z),
        "support_contact_steps": maximum_support_steps,
        "collision_steps": collision_steps,
        "stages": stage_results,
        "observations": observations,
        "final_geometry": geometry_snapshot(raw_env, object_name),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    candidate_root = args.candidate_root.resolve()
    app_dir = _configure_candidate(candidate_root)
    official_commit = _git_commit(candidate_root)
    if official_commit != args.expected_official_commit:
        raise RuntimeError(
            f"official commit mismatch: expected {args.expected_official_commit}, "
            f"got {official_commit}"
        )
    tasks = json.loads(
        (app_dir / "knowledge" / "task_config.json").read_text(encoding="utf-8")
    )["tasks"]
    task = dict(tasks[0])
    backend = None
    started = time.perf_counter()
    record: dict[str, Any] = {
        "physical_grasp": False,
        "lift_m": 0.0,
        "support_contact_steps": 0,
        "base_translation_m": 0.0,
        "attachment_calls": 0,
        "object_pose_writes": 0,
        "collision_frames": 0,
        "dropped": True,
        "infrastructure_error": None,
        "seed": int(args.seed),
        "official_commit": official_commit,
        "mode": "post_lift_hold_probe",
    }
    try:
        backend, scene_context, grid = _load_scene(app_dir, task, args.seed)
        from robot_agent.workflows.competition_flow import OfficialCompetitionDriver

        driver = OfficialCompetitionDriver(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
        )
        candidates = driver.rank_objects(str(task["source"]), task["object"])
        if not candidates:
            raise RuntimeError("no graspable L1 candidate resolved")
        object_name = candidates[0]
        record["object_name"] = object_name
        body_id = backend.env.obj_body_id[object_name]
        backend.start_recording()
        backend._record_trajectory_frame()
        moved = driver.move(
            str(task["source"]),
            carrying=False,
            object_name=object_name,
        )
        record["move_success"] = bool(moved)
        if moved:
            pre_grasp_z = float(backend.env.sim.data.body_xpos[body_id][2])
            pre_grasp_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            grasp = driver.grasp(str(task["source"]), object_name)
            post_grasp_z = float(backend.env.sim.data.body_xpos[body_id][2])
            record["grasp_result"] = grasp
            record["physical_grasp"] = bool(
                grasp.get("success")
                and grasp.get("lift_success")
                and all(bool(value) for value in grasp.get("contacts", {}).values())
            )
            record["pre_grasp_object_z"] = pre_grasp_z
            record["post_grasp_object_z"] = post_grasp_z
            record["lift_m"] = post_grasp_z - pre_grasp_z
            record["post_grasp_geometry"] = geometry_snapshot(
                backend.env,
                object_name,
            )
            if record["physical_grasp"]:
                if args.physical_push:
                    probe = _physical_push_probe(
                        backend,
                        object_name,
                        table_object_z=pre_grasp_z,
                        push_distance_m=args.push_distance_m,
                        max_push_steps=args.max_push_steps,
                    )
                    record["mode"] = "physical_push_probe"
                    record["physical_contact_steps"] = int(
                        probe.get("physical_contact_steps", 0)
                    )
                    record["object_translation_m"] = float(
                        probe.get("object_translation_m", 0.0)
                    )
                elif args.center_regrasp:
                    probe = _center_regrasp_probe(
                        backend,
                        object_name,
                        table_object_z=pre_grasp_z,
                        center_shift_m=args.regrasp_center_shift_m,
                        wall_clearance_m=args.regrasp_wall_clearance_m,
                        wall_squeeze_m=args.regrasp_wall_squeeze_m,
                        hold_steps=args.hold_steps,
                    )
                    record["mode"] = "table_assisted_center_regrasp"
                    record["physical_grasp"] = bool(probe.get("success", False))
                    record["lift_m"] = float(probe.get("lift_m", 0.0))
                elif args.inward_probe_m > 0.0:
                    probe = _inward_support_probe(
                        backend,
                        object_name,
                        inward_m=args.inward_probe_m,
                        move_steps=args.inward_steps,
                        hold_steps=args.hold_steps,
                    )
                    record["mode"] = "inward_support_probe"
                else:
                    probe = _hold_probe(backend, object_name, steps=args.hold_steps)
                record["hold_probe"] = probe
                record["support_contact_steps"] = int(
                    probe.get("support_contact_steps", 0)
                )
            final_z = float(backend.env.sim.data.body_xpos[body_id][2])
            record["final_geometry"] = geometry_snapshot(
                backend.env,
                object_name,
            )
            final_base_xy = np.asarray(backend.get_base_pose()[0], dtype=float)
            record["final_object_z"] = final_z
            record["base_translation_m"] = float(
                np.linalg.norm(final_base_xy - pre_grasp_base_xy)
            )
            record["dropped"] = bool(
                final_z < pre_grasp_z + CRADLE_GATE_THRESHOLDS["lift_m"]
            )
        backend._record_trajectory_frame()
        trajectory_path = args.trajectory.resolve()
        backend.save_trajectory(trajectory_path)
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        record["trajectory"] = str(trajectory_path)
        record["collision_frames"] = sum(
            int(bool(frame.get("has_collision", False)))
            for frame in trajectory.get("frames", [])
            if isinstance(frame, dict)
        )
    except Exception as exc:
        record["infrastructure_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
    record["elapsed_s"] = round(time.perf_counter() - started, 6)
    if record.get("mode") == "physical_push_probe":
        record["gate_failures"] = push_gate_failures(record)
    else:
        record["gate_failures"] = cradle_gate_failures(record)
    record["accepted"] = not record["gate_failures"]
    _atomic_json(args.output.resolve(), record)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--inward-probe-m", type=float, default=0.0)
    parser.add_argument("--inward-steps", type=int, default=40)
    parser.add_argument("--center-regrasp", action="store_true")
    parser.add_argument("--regrasp-center-shift-m", type=float, default=0.24)
    parser.add_argument("--regrasp-wall-clearance-m", type=float, default=0.10)
    parser.add_argument("--regrasp-wall-squeeze-m", type=float, default=0.025)
    parser.add_argument("--physical-push", action="store_true")
    parser.add_argument("--push-distance-m", type=float, default=0.50)
    parser.add_argument("--max-push-steps", type=int, default=400)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run_probe(parse_args())
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    raise SystemExit(0 if result["accepted"] else 1)
