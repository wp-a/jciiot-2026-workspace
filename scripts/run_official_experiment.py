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
    started = time.perf_counter()
    trajectory_path = args.trajectory.resolve()
    try:
        backend, scene_context, grid = _load_scene(app_dir, task, args.seed)
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
