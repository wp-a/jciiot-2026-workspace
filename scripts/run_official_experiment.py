#!/usr/bin/env python3
"""Run a deterministic competition workflow and score its official trajectory."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.perturbation_protocol import (
        PerturbationSample,
        sample_perturbation,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from perturbation_protocol import (
        PerturbationSample,
        sample_perturbation,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_success(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "1", "true", "yes", "ok", "success", "succeeded"
    }


def _object_name_matches(name: str, candidates: list[str]) -> bool:
    if not candidates:
        return True
    return any(name == item or name in item or item in name for item in candidates)


def _object_position(positions: dict, object_name: str):
    value = positions.get(object_name)
    if value is None:
        for candidate, candidate_value in positions.items():
            if object_name in str(candidate) or str(candidate) in object_name:
                value = candidate_value
                break
    if value is None or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def audit_trajectory(
    *,
    task_index: int,
    task: dict[str, Any],
    trajectory: dict[str, Any],
    trajectory_path: str,
    score_details: dict[str, Any],
    target_center_xy: list[float],
    official_commit: str,
    workspace_commit: str,
    seed: int,
    elapsed_s: float,
    execution_result: dict[str, Any],
    source_center_xy: list[float] | None = None,
    perturbation: dict[str, Any] | None = None,
    perturbation_application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates_value = task.get("object", [])
    if isinstance(candidates_value, str):
        candidates = [candidates_value]
    else:
        candidates = [str(name) for name in candidates_value if name]

    events = trajectory.get("events", [])
    if not isinstance(events, list):
        events = []
    successful_events = []
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "grasp_end":
            continue
        source_matches = not event.get("source") or str(event.get("source")) == str(task["source"])
        object_name = str(event.get("object_name") or "")
        if source_matches and _object_name_matches(object_name, candidates) and _event_success(event.get("success")):
            successful_events.append(event)

    frames = trajectory.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    collision_frames = sum(
        1 for frame in frames
        if isinstance(frame, dict) and bool(frame.get("has_collision", False))
    )

    final_positions = {}
    if frames and isinstance(frames[-1], dict):
        final_positions = frames[-1].get("object_positions", {}) or {}
    scored_names = [
        str(event.get("object_name")) for event in successful_events
        if event.get("object_name")
    ]
    if not scored_names:
        scored_names = candidates

    distances = {}
    departure_axes = {}
    for object_name in scored_names:
        position = _object_position(final_positions, object_name)
        if position is None:
            continue
        distances[object_name] = math.hypot(
            position[0] - float(target_center_xy[0]),
            position[1] - float(target_center_xy[1]),
        )
        if source_center_xy is not None:
            departure_axes[object_name] = max(
                abs(position[0] - float(source_center_xy[0])),
                abs(position[1] - float(source_center_xy[1])),
            )

    final_distance = max(distances.values()) if distances else None
    required_grasps = 3 if str(task.get("level")) == "L5" else 1
    manifest = {
        "status": "complete",
        "task_index": int(task_index),
        "level": task.get("level"),
        "scene": task.get("env_name"),
        "seed": int(seed),
        "official_commit": official_commit,
        "workspace_commit": workspace_commit,
        "trajectory": str(trajectory_path),
        "trajectory_frames": len(frames),
        "official_score": int(score_details.get("total", 0)),
        "max_score": int(task.get("max_score", 0)),
        "score_details": _json_safe(score_details),
        "successful_grasp_events": len(successful_events),
        "required_grasp_events": required_grasps,
        "collision_frames": collision_frames,
        "final_target_distance_m": final_distance,
        "final_target_distances_m": distances,
        "departure_axis_m": departure_axes,
        "elapsed_s": round(float(elapsed_s), 6),
        "execution_result": _json_safe(execution_result),
        "finished_at": _utc_now(),
    }
    if perturbation is not None:
        manifest["perturbation"] = _json_safe(perturbation)
    if perturbation_application is not None:
        manifest["perturbation_application"] = _json_safe(
            perturbation_application
        )
    return manifest


def acceptance_met(manifest: dict[str, Any], *, required_score: int) -> bool:
    distance = manifest.get("final_target_distance_m")
    execution = manifest.get("execution_result", {})
    return bool(
        manifest.get("status") == "complete"
        and int(manifest.get("official_score", 0)) >= int(required_score)
        and int(manifest.get("successful_grasp_events", 0))
        >= int(manifest.get("required_grasp_events", 1))
        and int(manifest.get("collision_frames", 0)) == 0
        and distance is not None
        and float(distance) < 0.80
        and bool(execution.get("success", False))
    )


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(data), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _load_scene(app_dir: Path, task: dict[str, Any], seed: int):
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
    prefix = task["scene_prefix"]
    semantic_path = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid_path = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    scene_data, grid = load_map_files(semantic_path, grid_path)
    scene_context = SceneContext.from_semantic_map(scene_data)
    backend = RobosuiteBackend(
        env_name=task["env_name"],
        camera="birdview",
        headless=True,
        drive_mode="direct",
        seed=seed,
    )
    backend._scene_context = scene_context
    backend.reset()
    return backend, scene_context, grid


def _score_trajectory(task_index: int, trajectory_path: Path) -> dict[str, Any]:
    import streamlit as st

    app = importlib.import_module("app")
    st.session_state["_last_trajectory"] = str(trajectory_path.resolve())
    return _json_safe(app._score_steps(task_index))


def _primary_object_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if item:
                return str(item)
    return ""


def resolve_scored_object(
    task: dict[str, Any],
    requested_name: str | None = None,
) -> str:
    value = task.get("object", [])
    if isinstance(value, str):
        candidates = [value]
    else:
        candidates = [str(item) for item in value if item]
    if not candidates:
        raise ValueError("task contains no scored object candidates")
    if requested_name is None:
        return candidates[0]
    requested = str(requested_name)
    if requested not in candidates:
        raise ValueError(
            f"perturbation object is not a scored candidate: {requested}"
        )
    return requested


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat_wxyz(quat) -> float:
    w, x, y, z = np.asarray(quat, dtype=float).reshape(4)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(float(yaw) / 2.0), 0.0, 0.0, math.sin(float(yaw) / 2.0)],
        dtype=float,
    )


def _quat_multiply_wxyz(left, right) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=float).reshape(4)
    w0, x0, y0, z0 = np.asarray(right, dtype=float).reshape(4)
    value = np.array(
        [
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("invalid zero-norm object quaternion")
    return value / norm


def _descendant_body_ids(model, root_body_id: int) -> list[int]:
    parents = np.asarray(model.body_parentid, dtype=int).reshape(-1)
    selected = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if body_id not in selected and int(parent_id) in selected:
                selected.add(body_id)
                changed = True
    return sorted(selected)


def _object_joint_name(raw_env, object_name: str) -> str:
    metadata = getattr(raw_env, "material_metadata", {}).get(object_name, {})
    return str(metadata.get("joint_name") or f"{object_name}_free")


def _set_base_pose_for_research(backend, target_xy, target_yaw: float) -> None:
    from robot_agent.environments.robosuite_backend import (
        _set_base_world_yaw_direct,
        _set_base_xy_direct,
    )

    raw_env = backend.env
    robot = raw_env.robots[0]
    _set_base_world_yaw_direct(raw_env, robot, float(target_yaw))
    _set_base_xy_direct(raw_env, robot, np.asarray(target_xy, dtype=float))

    for joint_name in robot.robot_model.base_joints:
        try:
            address = raw_env.sim.model.get_joint_qvel_addr(joint_name)
        except Exception:
            continue
        if isinstance(address, tuple):
            raw_env.sim.data.qvel[address[0]:address[1]] = 0.0
        else:
            raw_env.sim.data.qvel[address] = 0.0
    raw_env.sim.forward()


def apply_perturbation(
    backend,
    task: dict[str, Any],
    sample: PerturbationSample,
    *,
    base_pose_setter=None,
) -> dict[str, Any]:
    """Apply and measure one research-only initial-state perturbation."""
    object_name = resolve_scored_object(task, sample.object_name)
    raw_env = backend.env
    sim = raw_env.sim
    model = sim.model
    data = sim.data

    try:
        body_id = int(raw_env.obj_body_id[object_name])
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"object body is not registered: {object_name}") from exc
    joint_name = _object_joint_name(raw_env, object_name)
    before_qpos = np.asarray(data.get_joint_qpos(joint_name), dtype=float).copy()
    if before_qpos.shape != (7,):
        raise ValueError(
            f"expected 7D free-joint qpos for {object_name}, got {before_qpos.shape}"
        )

    before_base_xy, before_base_yaw = backend.get_base_pose()
    before_base_xy = np.asarray(before_base_xy, dtype=float).reshape(2)
    before_base_yaw = float(before_base_yaw)

    body_ids = _descendant_body_ids(model, body_id)
    geom_ids = [
        int(geom_id)
        for geom_id, geom_body_id in enumerate(
            np.asarray(model.geom_bodyid, dtype=int).reshape(-1)
        )
        if int(geom_body_id) in body_ids
    ]
    before_masses = np.asarray(model.body_mass[body_ids], dtype=float).copy()
    before_friction = np.asarray(model.geom_friction[geom_ids], dtype=float).copy()

    nominal_noop = all(
        abs(float(value)) <= 1e-15
        for value in (
            sample.object_dx_m,
            sample.object_dy_m,
            sample.object_dyaw_rad,
            sample.base_dx_m,
            sample.base_dy_m,
            sample.base_dyaw_rad,
            sample.mass_scale - 1.0,
            sample.friction_scale - 1.0,
        )
    )

    if not nominal_noop:
        target_qpos = before_qpos.copy()
        target_qpos[0] += float(sample.object_dx_m)
        target_qpos[1] += float(sample.object_dy_m)
        target_qpos[3:7] = _quat_multiply_wxyz(
            _yaw_quat_wxyz(sample.object_dyaw_rad),
            before_qpos[3:7],
        )
        data.set_joint_qpos(joint_name, target_qpos)
        data.set_joint_qvel(joint_name, np.zeros(6, dtype=float))
        model.body_mass[body_ids] = before_masses * float(sample.mass_scale)
        if geom_ids:
            model.geom_friction[geom_ids] = (
                before_friction * float(sample.friction_scale)
            )

        setter = base_pose_setter or _set_base_pose_for_research
        setter(
            backend,
            before_base_xy
            + np.array([sample.base_dx_m, sample.base_dy_m], dtype=float),
            before_base_yaw + float(sample.base_dyaw_rad),
        )
        sim.forward()

    after_qpos = np.asarray(data.get_joint_qpos(joint_name), dtype=float).copy()
    after_base_xy, after_base_yaw = backend.get_base_pose()
    after_base_xy = np.asarray(after_base_xy, dtype=float).reshape(2)
    after_base_yaw = float(after_base_yaw)

    measured_object_delta = after_qpos[:2] - before_qpos[:2]
    measured_object_dyaw = _wrap_angle(
        _yaw_from_quat_wxyz(after_qpos[3:7])
        - _yaw_from_quat_wxyz(before_qpos[3:7])
    )
    measured_base_delta = after_base_xy - before_base_xy
    measured_base_dyaw = _wrap_angle(after_base_yaw - before_base_yaw)

    object_xy_error = measured_object_delta - np.array(
        [sample.object_dx_m, sample.object_dy_m],
        dtype=float,
    )
    base_xy_error = measured_base_delta - np.array(
        [sample.base_dx_m, sample.base_dy_m],
        dtype=float,
    )
    object_yaw_error = _wrap_angle(
        measured_object_dyaw - float(sample.object_dyaw_rad)
    )
    base_yaw_error = _wrap_angle(
        measured_base_dyaw - float(sample.base_dyaw_rad)
    )
    position_tolerance_m = 0.001
    yaw_tolerance_rad = math.radians(0.1)
    valid = bool(
        float(np.linalg.norm(object_xy_error)) <= position_tolerance_m
        and float(np.linalg.norm(base_xy_error)) <= position_tolerance_m
        and abs(object_yaw_error) <= yaw_tolerance_rad
        and abs(base_yaw_error) <= yaw_tolerance_rad
    )
    if not valid:
        raise RuntimeError(
            "measured perturbation differs from request: "
            f"object_xy_error={object_xy_error.tolist()}, "
            f"object_yaw_error={object_yaw_error}, "
            f"base_xy_error={base_xy_error.tolist()}, "
            f"base_yaw_error={base_yaw_error}"
        )

    return {
        "valid": valid,
        "nominal_noop": nominal_noop,
        "object_name": object_name,
        "joint_name": joint_name,
        "body_ids": body_ids,
        "geom_ids": geom_ids,
        "before_object_xy": before_qpos[:2].tolist(),
        "after_object_xy": after_qpos[:2].tolist(),
        "measured_object_dx_m": float(measured_object_delta[0]),
        "measured_object_dy_m": float(measured_object_delta[1]),
        "measured_object_dyaw_rad": measured_object_dyaw,
        "before_base_xy": before_base_xy.tolist(),
        "after_base_xy": after_base_xy.tolist(),
        "measured_base_dx_m": float(measured_base_delta[0]),
        "measured_base_dy_m": float(measured_base_delta[1]),
        "measured_base_dyaw_rad": measured_base_dyaw,
        "before_body_mass": before_masses.tolist(),
        "after_body_mass": np.asarray(
            model.body_mass[body_ids], dtype=float
        ).tolist(),
        "before_geom_friction": before_friction.tolist(),
        "after_geom_friction": np.asarray(
            model.geom_friction[geom_ids], dtype=float
        ).tolist(),
        "position_tolerance_m": position_tolerance_m,
        "yaw_tolerance_rad": yaw_tolerance_rad,
    }


def execute_task(
    *,
    execution_mode: str,
    backend,
    scene_context,
    grid,
    task: dict[str, Any],
    task_index: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Execute through either the historical flow or official RobotAgent."""
    if execution_mode == "flow":
        from robot_agent.workflows.competition_flow import run_official_task

        return run_official_task(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            task=task,
            max_attempts=max_attempts,
        )
    if execution_mode != "agent":
        raise ValueError(f"unsupported execution mode: {execution_mode}")

    from robot_agent.core.agent import RobotAgent

    object_name = _primary_object_name(task.get("object", ""))
    scene_metadata = {
        "task_index": int(task_index),
        "env_name": str(task.get("env_name", "")),
        "scene_name": getattr(scene_context, "scene_name", ""),
        "map_name": getattr(scene_context, "map_name", ""),
        "map_prefix": str(task.get("scene_prefix", "")),
        "input_object_map": (
            {str(task["source"]): object_name} if object_name else {}
        ),
    }
    agent = RobotAgent(
        backend=backend,
        scene_context=scene_context,
        grid=grid,
        scene_metadata=scene_metadata,
        knowledge_enabled=False,
    )
    output = agent.run(
        f"Execute official {task.get('level', '')} transport task "
        f"from {task.get('source', '')} to {task.get('target', '')}."
    )
    if hasattr(output, "as_dict"):
        return _json_safe(output.as_dict())
    if isinstance(output, dict):
        return _json_safe(output)
    raise TypeError("RobotAgent.run returned an unsupported result")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    candidate_root = args.candidate_root.resolve()
    app_dir = _configure_candidate(candidate_root)
    official_commit = _git_commit(candidate_root)
    if official_commit != args.expected_official_commit:
        raise RuntimeError(
            f"official commit mismatch: expected {args.expected_official_commit}, got {official_commit}"
        )

    task_config = json.loads((app_dir / "knowledge" / "task_config.json").read_text(encoding="utf-8"))
    tasks = task_config.get("tasks", [])
    task = dict(tasks[args.task_index])
    backend = None
    perturbation = None
    perturbation_application = None
    started = time.perf_counter()
    trajectory_path = args.trajectory.resolve()
    try:
        backend, scene_context, grid = _load_scene(app_dir, task, args.seed)
        perturbation_object = resolve_scored_object(
            task,
            requested_name=args.perturbation_object,
        )
        sample = sample_perturbation(
            tier=args.perturbation_tier,
            seed=args.seed,
            task_index=args.task_index,
            object_name=perturbation_object,
        )
        perturbation = sample.as_dict()
        perturbation_application = apply_perturbation(backend, task, sample)
        backend.start_recording()
        backend._record_trajectory_frame()

        execution_result = execute_task(
            execution_mode=args.execution_mode,
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            task=task,
            task_index=args.task_index,
            max_attempts=args.max_attempts,
        )
        backend._record_trajectory_frame()
        saved_path = Path(backend.save_trajectory(trajectory_path)).resolve()
        trajectory = json.loads(saved_path.read_text(encoding="utf-8"))
        score_details = _score_trajectory(args.task_index, saved_path)
        source = scene_context.input_ports[task["source"]].center[:2].tolist()
        target = scene_context.output_ports[task["target"]].center[:2].tolist()
        return audit_trajectory(
            task_index=args.task_index,
            task=task,
            trajectory=trajectory,
            trajectory_path=str(saved_path),
            score_details=score_details,
            source_center_xy=source,
            target_center_xy=target,
            official_commit=official_commit,
            workspace_commit=args.workspace_commit,
            seed=args.seed,
            elapsed_s=time.perf_counter() - started,
            execution_result=execution_result,
            perturbation=perturbation,
            perturbation_application=perturbation_application,
        )
    except Exception as exc:
        if backend is not None:
            try:
                backend.save_trajectory(trajectory_path)
            except Exception:
                pass
        return {
            "status": "error",
            "task_index": args.task_index,
            "seed": args.seed,
            "official_commit": official_commit,
            "workspace_commit": args.workspace_commit,
            "trajectory": str(trajectory_path),
            "official_score": 0,
            "collision_frames": None,
            "successful_grasp_events": 0,
            "perturbation": _json_safe(perturbation),
            "perturbation_application": _json_safe(perturbation_application),
            "elapsed_s": round(time.perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": _utc_now(),
        }
    finally:
        if backend is not None:
            backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--workspace-commit", required=True)
    parser.add_argument("--task-index", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-score", type=int)
    parser.add_argument(
        "--execution-mode",
        choices=("flow", "agent"),
        default="flow",
    )
    parser.add_argument(
        "--perturbation-tier",
        choices=("nominal", "small", "medium", "stress"),
        default="nominal",
    )
    parser.add_argument("--perturbation-object")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    started_at = _utc_now()
    report = run_experiment(args)
    report["runner"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": started_at,
        "execution_mode": args.execution_mode,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(_json_safe(report), ensure_ascii=True, indent=2))
    required_score = args.required_score
    if required_score is None:
        required_score = int(report.get("max_score", 0))
    return 0 if acceptance_met(report, required_score=required_score) else 1


if __name__ == "__main__":
    raise SystemExit(main())
