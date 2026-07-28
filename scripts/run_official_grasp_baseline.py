#!/usr/bin/env python3
"""Measure the official checkpoint's physical grasp-and-lift baseline.

This research runner deliberately bypasses ``RobosuiteBackend.grasp_object_physics``
because that method synchronizes object qpos and creates a transport attachment.
Only the official wrapped policy rollout and lift helper are used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_OFFICIAL_COMMIT = "0dcdddf18a9e694569aa1433cdfc04eb097fed78"
EXPECTED_CHECKPOINT_SHA256 = (
    "ef5910f6a9f6309b5ced617762dffeb1169a8b0cfcea892d158e6b483252169f"
)
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class GraspJob:
    task_index: int
    level: str
    scene_prefix: str
    env_name: str
    source: str
    object_name: str
    seed: int


class UnsupportedPolicyTargetError(RuntimeError):
    """The official policy interface lacks sites required for this object."""


def _object_names(task: dict[str, Any]) -> list[str]:
    value = task.get("object", [])
    if isinstance(value, str):
        value = [value]
    names = [str(item) for item in value if item]
    if not names:
        raise ValueError(f"task {task.get('level', '<unknown>')} has no objects")
    return names


def build_jobs(
    tasks: Iterable[dict[str, Any]],
    seeds: Iterable[int],
) -> list[GraspJob]:
    tasks = list(tasks)
    jobs = []
    for seed in seeds:
        for task_index, task in enumerate(tasks):
            for object_name in _object_names(task):
                jobs.append(
                    GraspJob(
                        task_index=task_index,
                        level=str(task["level"]),
                        scene_prefix=str(task["scene_prefix"]),
                        env_name=str(task["env_name"]),
                        source=str(task["source"]),
                        object_name=object_name,
                        seed=int(seed),
                    )
                )
    return jobs


def physical_grasp_success(
    record: dict[str, Any],
    *,
    required_lift_m: float = 0.15,
    tolerance_m: float = 0.02,
) -> bool:
    grasps = record.get("grasp_status") or {}
    lift_threshold = float(required_lift_m) - float(tolerance_m)
    return bool(
        grasps.get("left")
        and grasps.get("right")
        and float(record.get("lifted_m", float("-inf"))) >= lift_threshold
        and not record.get("collision", False)
        and not record.get("infrastructure_error")
    )


def summarize(records: Iterable[dict[str, Any]], *, planned_runs: int) -> dict[str, Any]:
    records = list(records)
    if planned_runs < len(records):
        raise ValueError("planned_runs cannot be smaller than recorded runs")
    successes = sum(bool(record.get("physical_success")) for record in records)
    collisions = sum(bool(record.get("collision")) for record in records)
    infrastructure_errors = sum(bool(record.get("infrastructure_error")) for record in records)
    return {
        "planned_runs": int(planned_runs),
        "recorded_runs": len(records),
        "missing_runs": int(planned_runs) - len(records),
        "successful_runs": successes,
        "success_rate": successes / planned_runs if planned_runs else 0.0,
        "collision_runs": collisions,
        "infrastructure_errors": infrastructure_errors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_materialized_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with path.open("rb") as handle:
        header = handle.read(len(LFS_HEADER))
    if header == LFS_HEADER:
        raise ValueError(f"checkpoint is a Git LFS pointer, not model weights: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def has_required_grasp_sites(raw_env: Any, object_name: str) -> bool:
    for arm in ("left", "right"):
        try:
            raw_env.sim.model.site_name2id(f"{object_name}_{arm}_grasp_site")
        except Exception:
            return False
    return True


def initialized_object_center(
    runtime: dict[str, Any],
    env: Any,
    object_name: str,
) -> list[float]:
    """Advance the wrapped env before reading MuJoCo world-space sites."""
    raw = runtime["base_robosuite_env"](env)
    runtime["current_wrapped_policy_obs"](env)
    raw.sim.forward()
    try:
        center = runtime["object_center_pos"](raw, object_name)
    except RuntimeError:
        body_ids = getattr(raw, "obj_body_id", {})
        if object_name not in body_ids:
            raise
        center = raw.sim.data.body_xpos[body_ids[object_name]]
    return [float(value) for value in center]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_app_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (path, path / "JCIIOT")
    for candidate in candidates:
        if (candidate / "app.py").is_file():
            return candidate
    raise FileNotFoundError(f"could not find JCIIOT/app.py below {path}")


def _configure_imports(app_dir: Path) -> None:
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


def _load_tasks(task_config: Path) -> list[dict[str, Any]]:
    data = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 5:
        raise ValueError("official task_config.json must contain five tasks")
    return tasks


def _semantic_station(app_dir: Path, job: GraspJob) -> dict[str, Any]:
    map_path = (
        app_dir
        / "robosuite"
        / "robosuite"
        / "environments"
        / "factory_sorting"
        / "generated_maps"
        / f"{job.scene_prefix}_scene_regenerated_semantic_map.json"
    )
    data = json.loads(map_path.read_text(encoding="utf-8"))
    station = (data.get("input_ports") or {}).get(job.source)
    if not station:
        raise KeyError(f"source {job.source!r} missing from {map_path}")
    if len(station.get("approach", [])) != 2 or len(station.get("center", [])) != 2:
        raise ValueError(f"source {job.source!r} lacks a 2D approach or center")
    return station


def _eval_namespace(job: GraspJob, xy: list[float], yaw: float) -> argparse.Namespace:
    return argparse.Namespace(
        factory_scene=job.env_name,
        robot_base_pos=[float(xy[0]), float(xy[1]), 0.0],
        robot_base_ori=[0.0, 0.0, float(yaw)],
        renderer="mjviewer",
        camera="robot0_robotview",
        camera_height=128,
        camera_width=128,
        controller=None,
        gripper_types="Robotiq140Gripper",
        seed=job.seed,
    )


def _close_env(env: Any) -> None:
    if env is not None and hasattr(env, "close"):
        env.close()


def _runtime_imports() -> dict[str, Any]:
    from robosuite.environments.factory_sorting.lift_after_grasp import (
        lift_grasped_object,
        object_center_pos,
    )
    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        base_robosuite_env,
        current_wrapped_policy_obs,
        fingerpad_contact_status,
        grasp_status,
        load_factory_sorting_policy,
        make_eval_env,
        run_factory_sorting_grasp_in_wrapped_env,
    )

    return locals()


def _probe_object_center(
    runtime: dict[str, Any],
    job: GraspJob,
    approach: list[float],
    station_center: list[float],
    config: Any,
    checkpoint_dict: dict[str, Any],
) -> tuple[list[float], bool]:
    preliminary_yaw = math.atan2(
        float(station_center[1]) - float(approach[1]),
        float(station_center[0]) - float(approach[0]),
    )
    env = None
    try:
        env = runtime["make_eval_env"](
            _eval_namespace(job, approach, preliminary_yaw),
            config=config,
            ckpt_dict=checkpoint_dict,
            render=False,
        )
        raw = runtime["base_robosuite_env"](env)
        center = initialized_object_center(runtime, env, job.object_name)
        return center, has_required_grasp_sites(raw, job.object_name)
    finally:
        _close_env(env)


def _run_job(
    *,
    app_dir: Path,
    runtime: dict[str, Any],
    policy: Any,
    config: Any,
    checkpoint_dict: dict[str, Any],
    job: GraspJob,
    grasp_params: dict[str, Any],
    lift_params: dict[str, Any],
    render: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        **asdict(job),
        "started_at": _utc_now(),
        "infrastructure_error": None,
        "failure_stage": None,
    }
    env = None
    try:
        station = _semantic_station(app_dir, job)
        approach = [float(value) for value in station["approach"]]
        station_center = [float(value) for value in station["center"]]
        object_center, policy_target_supported = _probe_object_center(
            runtime,
            job,
            approach,
            station_center,
            config,
            checkpoint_dict,
        )
        yaw = math.atan2(
            object_center[1] - approach[1],
            object_center[0] - approach[0],
        )
        record.update(
            semantic_approach_xy=approach,
            probe_object_center_xyz=object_center,
            derived_yaw=yaw,
            policy_target_supported=policy_target_supported,
        )
        if not policy_target_supported:
            raise UnsupportedPolicyTargetError(
                f"official grasp sites are missing for {job.object_name}"
            )

        env = runtime["make_eval_env"](
            _eval_namespace(job, approach, yaw),
            config=config,
            ckpt_dict=checkpoint_dict,
            render=render,
        )
        raw = runtime["base_robosuite_env"](env)
        start_center = initialized_object_center(runtime, env, job.object_name)
        record["start_object_center_xyz"] = start_center
        record["probe_replay_position_error_m"] = math.dist(
            object_center, record["start_object_center_xyz"]
        )

        grasp_result = runtime["run_factory_sorting_grasp_in_wrapped_env"](
            env=env,
            policy=policy,
            eval_steps=int(grasp_params["eval_steps"]),
            debug_policy=bool(grasp_params.get("debug_policy", False)),
            debug_every=int(grasp_params.get("debug_every", 25)),
            object_name=job.object_name,
            post_hold_steps=int(grasp_params["post_hold_steps"]),
            initial_view_steps=int(grasp_params["initial_view_steps"]),
            camera="robot0_robotview",
            render=render,
            render_sleep=0.0,
        )
        robot = raw.robots[0]
        record["policy_grasp_success"] = bool(grasp_result.get("success"))
        record["pre_lift_grasp_status"] = runtime["grasp_status"](
            raw, robot, job.object_name
        )
        record["pre_lift_fingerpad_contacts"] = runtime["fingerpad_contact_status"](
            raw, robot, job.object_name
        )
        if not record["policy_grasp_success"]:
            record["failure_stage"] = "grasp"

        lift_result = runtime["lift_grasped_object"](
            env=env,
            object_name=job.object_name,
            lift_height=float(lift_params["lift_height"]),
            max_steps=int(lift_params["max_steps"]),
            hold_steps=int(lift_params["hold_steps"]),
            tolerance=float(lift_params["tolerance"]),
            max_action=float(lift_params["max_action"]),
            render=render,
            render_sleep=0.0,
        )
        raw.sim.forward()
        final_center = runtime["object_center_pos"](raw, job.object_name)
        record["lift_result"] = lift_result
        record["final_object_center_xyz"] = [float(value) for value in final_center]
        record["lifted_m"] = float(final_center[2] - start_center[2])
        record["grasp_status"] = runtime["grasp_status"](raw, robot, job.object_name)
        record["fingerpad_contacts"] = runtime["fingerpad_contact_status"](
            raw, robot, job.object_name
        )
        record["collision"] = bool(getattr(raw, "has_judge_collision", False))
        if not bool(lift_result.get("success")) and record["failure_stage"] is None:
            record["failure_stage"] = "lift"
        if record["collision"]:
            record["failure_stage"] = "collision"
    except UnsupportedPolicyTargetError as exc:
        record["failure_stage"] = "unsupported_official_target"
        record["unsupported_reason"] = str(exc)
    except Exception as exc:
        record["infrastructure_error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["failure_stage"] = record.get("failure_stage") or "infrastructure"
    finally:
        _close_env(env)

    record.setdefault("collision", False)
    record.setdefault("lifted_m", 0.0)
    record.setdefault("grasp_status", {"left": False, "right": False})
    record["physical_success"] = physical_grasp_success(
        record,
        required_lift_m=float(lift_params["lift_height"]),
        tolerance_m=float(lift_params["tolerance"]),
    )
    if record["physical_success"]:
        record["failure_stage"] = None
    record["elapsed_s"] = round(time.monotonic() - started, 6)
    record["finished_at"] = _utc_now()
    return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--expected-official-commit", default=EXPECTED_OFFICIAL_COMMIT)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app_dir = _resolve_app_dir(args.app_dir)
    source_commit = _git_commit(app_dir)
    if source_commit != args.expected_official_commit:
        raise RuntimeError(
            f"official source commit mismatch: {source_commit} != "
            f"{args.expected_official_commit}"
        )
    checkpoint = validate_materialized_checkpoint(args.checkpoint)
    if checkpoint["sha256"] != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: {checkpoint['sha256']} != "
            f"{args.expected_checkpoint_sha256}"
        )

    task_config = app_dir / "knowledge" / "task_config.json"
    robot_params_path = app_dir / "knowledge" / "robot_params.json"
    tasks = _load_tasks(task_config)
    task_indices = args.task_index if args.task_index is not None else list(range(5))
    if any(index < 0 or index >= len(tasks) for index in task_indices):
        raise ValueError("--task-index must be between 0 and 4")
    selected_tasks = [tasks[index] for index in task_indices]
    jobs = build_jobs(selected_tasks, seeds=args.seed or [0])
    jobs = [
        GraspJob(
            task_index=task_indices[job.task_index],
            level=job.level,
            scene_prefix=job.scene_prefix,
            env_name=job.env_name,
            source=job.source,
            object_name=job.object_name,
            seed=job.seed,
        )
        for job in jobs
    ]

    _configure_imports(app_dir)
    runtime = _runtime_imports()
    policy, config, checkpoint_dict = runtime["load_factory_sorting_policy"](
        checkpoint=args.checkpoint,
        device=args.device,
        verbose=False,
    )
    robot_params = json.loads(robot_params_path.read_text(encoding="utf-8"))
    records = []
    for job in jobs:
        record = _run_job(
            app_dir=app_dir,
            runtime=runtime,
            policy=policy,
            config=config,
            checkpoint_dict=checkpoint_dict,
            job=job,
            grasp_params=robot_params["grasp_policy"],
            lift_params=robot_params["lift"],
            render=args.render,
        )
        records.append(record)
        _atomic_write_json(
            args.output,
            {
                "version": "official_physical_grasp_baseline_v1",
                "official_commit": source_commit,
                "checkpoint": checkpoint,
                "jobs": [asdict(item) for item in jobs],
                "records": records,
                "summary": summarize(records, planned_runs=len(jobs)),
            },
        )
        print(
            f"{job.level} {job.object_name} seed={job.seed}: "
            f"physical_success={record['physical_success']} "
            f"stage={record['failure_stage']}"
        )
    return 0 if not any(record.get("infrastructure_error") for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
