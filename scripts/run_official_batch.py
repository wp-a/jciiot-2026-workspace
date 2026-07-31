#!/usr/bin/env python3
"""Run resumable multi-seed official experiments and summarize stability."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.run_official_experiment import acceptance_met, write_json_atomic
except ModuleNotFoundError:
    from run_official_experiment import acceptance_met, write_json_atomic


TASK_LEVELS = {0: "L1", 1: "L2", 2: "L3", 3: "L4", 4: "L5"}
TASK_MAX_SCORES = {0: 10, 1: 15, 2: 20, 3: 25, 4: 30}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BatchJob:
    task_index: int
    seed: int
    output_dir: Path
    perturbation_tier: str = "nominal"

    @property
    def level(self) -> str:
        return TASK_LEVELS[self.task_index]

    @property
    def label(self) -> str:
        tier = str(self.perturbation_tier).strip().lower()
        tier_label = "" if tier == "nominal" else f"-{tier}"
        return f"{self.level.lower()}{tier_label}-seed-{self.seed}"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifests" / f"manifest-{self.label}.json"

    @property
    def trajectory_path(self) -> Path:
        return self.output_dir / "trajectories" / f"trajectory-{self.label}.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "logs" / f"{self.label}.log"


def build_jobs(
    *,
    task_indices: Iterable[int],
    seeds: Iterable[int],
    output_dir: Path,
    perturbation_tier: str = "nominal",
) -> list[BatchJob]:
    jobs = []
    for task_index in task_indices:
        if task_index not in TASK_LEVELS:
            raise ValueError(f"invalid task index: {task_index}")
        for seed in seeds:
            jobs.append(
                BatchJob(
                    task_index=int(task_index),
                    seed=int(seed),
                    output_dir=output_dir,
                    perturbation_tier=str(perturbation_tier),
                )
            )
    return jobs


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_terminal_manifest(path: Path) -> bool:
    manifest = _read_manifest(path)
    return bool(manifest and manifest.get("status") in {"complete", "error"})


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _is_full_score(manifest: dict[str, Any]) -> bool:
    required_score = int(manifest.get("max_score") or 0)
    if required_score <= 0:
        task_index = manifest.get("task_index")
        required_score = TASK_MAX_SCORES.get(task_index, 0)
    return required_score > 0 and acceptance_met(
        manifest, required_score=required_score
    )


def summarize_manifests(
    manifests: Iterable[dict[str, Any]], *, planned_runs: int
) -> dict[str, Any]:
    records = list(manifests)
    by_level: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        level = str(record.get("level") or TASK_LEVELS.get(record.get("task_index"), "unknown"))
        by_level.setdefault(level, []).append(record)

    levels = {}
    for level, level_records in sorted(by_level.items()):
        total = len(level_records)
        full_score_runs = sum(_is_full_score(record) for record in level_records)
        collision_runs = sum(
            int(record.get("collision_frames") or 0) > 0 for record in level_records
        )
        collision_frames = sum(
            int(record.get("collision_frames") or 0) for record in level_records
        )
        error_runs = sum(record.get("status") == "error" for record in level_records)
        grasp_verified_runs = sum(
            int(record.get("successful_grasp_events") or 0)
            >= int(record.get("required_grasp_events") or 1)
            for record in level_records
        )
        scores = [float(record.get("official_score") or 0) for record in level_records]
        elapsed = [
            float(record["elapsed_s"])
            for record in level_records
            if record.get("elapsed_s") is not None
        ]
        distances = [
            float(record["final_target_distance_m"])
            for record in level_records
            if record.get("final_target_distance_m") is not None
        ]
        levels[level] = {
            "completed_runs": total,
            "full_score_runs": full_score_runs,
            "full_score_rate": round(full_score_runs / total, 6),
            "full_score_rate_wilson_95": _wilson_interval(full_score_runs, total),
            "collision_runs": collision_runs,
            "collision_rate": round(collision_runs / total, 6),
            "collision_frames": collision_frames,
            "error_runs": error_runs,
            "grasp_verified_runs": grasp_verified_runs,
            "score_mean": _mean(scores),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "elapsed_mean_s": _mean(elapsed),
            "elapsed_max_s": max(elapsed) if elapsed else None,
            "target_distance_mean_m": _mean(distances),
            "target_distance_max_m": max(distances) if distances else None,
        }

    full_score_runs = sum(_is_full_score(record) for record in records)
    return {
        "planned_runs": int(planned_runs),
        "completed_runs": len(records),
        "remaining_runs": max(0, int(planned_runs) - len(records)),
        "full_score_runs": full_score_runs,
        "full_score_rate": round(full_score_runs / len(records), 6) if records else 0.0,
        "levels": levels,
    }


def _experiment_command(job: BatchJob, args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.runner),
        "--candidate-root",
        str(args.candidate_root),
        "--expected-official-commit",
        args.expected_official_commit,
        "--workspace-commit",
        args.workspace_commit,
        "--task-index",
        str(job.task_index),
        "--seed",
        str(job.seed),
        "--max-attempts",
        str(args.max_attempts),
        "--trajectory",
        str(job.trajectory_path),
        "--output",
        str(job.manifest_path),
        "--required-score",
        str(TASK_MAX_SCORES[job.task_index]),
        "--perturbation-tier",
        job.perturbation_tier,
    ]


def _run_job(
    job: BatchJob, args: argparse.Namespace, egl_device: str
) -> tuple[BatchJob, dict[str, Any]]:
    for path in (job.manifest_path.parent, job.trajectory_path.parent, job.log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = egl_device
    command = _experiment_command(job, args)
    with job.log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=args.working_directory,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    manifest = _read_manifest(job.manifest_path)
    if manifest is None:
        manifest = {
            "status": "error",
            "task_index": job.task_index,
            "level": job.level,
            "seed": job.seed,
            "official_score": 0,
            "max_score": TASK_MAX_SCORES[job.task_index],
            "successful_grasp_events": 0,
            "required_grasp_events": 3 if job.level == "L5" else 1,
            "collision_frames": None,
            "error": f"runner exited {process.returncode} without a valid manifest",
            "finished_at": _utc_now(),
        }
        write_json_atomic(job.manifest_path, manifest)
    return job, manifest


def _write_summary(
    *, jobs: list[BatchJob], manifests: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    summary = summarize_manifests(manifests, planned_runs=len(jobs))
    summary.update(
        {
            "status": "complete" if len(manifests) == len(jobs) else "running",
            "official_commit": args.expected_official_commit,
            "workspace_commit": args.workspace_commit,
            "task_indices": list(args.task_indices),
            "seeds": list(args.seeds),
            "perturbation_tier": args.perturbation_tier,
            "max_workers": args.max_workers,
            "updated_at": _utc_now(),
        }
    )
    write_json_atomic(args.output_dir / "batch-summary.json", summary)
    return summary


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    jobs = build_jobs(
        task_indices=args.task_indices,
        seeds=args.seeds,
        output_dir=args.output_dir,
        perturbation_tier=args.perturbation_tier,
    )
    manifests = [
        manifest
        for job in jobs
        if is_terminal_manifest(job.manifest_path)
        for manifest in [_read_manifest(job.manifest_path)]
        if manifest is not None
    ]
    summary = _write_summary(jobs=jobs, manifests=manifests, args=args)
    pending = [job for job in jobs if not is_terminal_manifest(job.manifest_path)]
    if not pending:
        return summary

    devices = args.egl_devices or ["0"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_run_job, job, args, devices[index % len(devices)]): job
            for index, job in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            job, manifest = future.result()
            manifests.append(manifest)
            summary = _write_summary(jobs=jobs, manifests=manifests, args=args)
            print(
                json.dumps(
                    {
                        "job": job.label,
                        "status": manifest.get("status"),
                        "score": manifest.get("official_score"),
                        "completed": summary["completed_runs"],
                        "planned": summary["planned_runs"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return summary


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--workspace-commit", required=True)
    parser.add_argument("--task-indices", type=_parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=_parse_int_list, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--egl-devices", type=lambda value: value.split(","), default=["0"])
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--perturbation-tier",
        choices=("nominal", "small", "medium", "stress"),
        default="nominal",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_official_experiment.py"),
    )
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if not args.seeds:
        parser.error("--seeds must not be empty")

    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return int(
        summary["completed_runs"] != summary["planned_runs"]
        or summary["full_score_runs"] != summary["planned_runs"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
