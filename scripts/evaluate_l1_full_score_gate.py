#!/usr/bin/env python3
"""Evaluate the repeated nominal and perturbed official L1 score gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.run_official_experiment import acceptance_met, write_json_atomic
except ModuleNotFoundError:
    from run_official_experiment import acceptance_met, write_json_atomic


def _is_l1_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("task_index") == 0 and str(manifest.get("level")) == "L1"


def _has_zero_collision_evidence(manifest: dict[str, Any]) -> bool:
    value = manifest.get("collision_frames")
    return value is not None and int(value) == 0


def _is_full_score(manifest: dict[str, Any]) -> bool:
    if not _is_l1_manifest(manifest):
        return False
    try:
        return acceptance_met(manifest, required_score=10)
    except (TypeError, ValueError):
        return False


def _terminal_failure_stage(manifest: dict[str, Any]) -> str:
    execution = manifest.get("execution_result")
    if isinstance(execution, dict):
        failures = execution.get("failures")
        if isinstance(failures, list):
            for failure in reversed(failures):
                if isinstance(failure, dict) and failure.get("failure_stage"):
                    return str(failure["failure_stage"])
    if manifest.get("status") == "error" or manifest.get("error"):
        return "runner_error"
    return "unknown"


def _perturbation_is_valid(manifest: dict[str, Any]) -> bool:
    perturbation = manifest.get("perturbation")
    application = manifest.get("perturbation_application")
    return bool(
        isinstance(perturbation, dict)
        and str(perturbation.get("tier")) in {"small", "medium", "stress"}
        and isinstance(application, dict)
        and application.get("valid") is True
        and application.get("nominal_noop") is False
    )


def _nominal_is_valid(manifest: dict[str, Any]) -> bool:
    perturbation = manifest.get("perturbation")
    application = manifest.get("perturbation_application")
    return bool(
        isinstance(perturbation, dict)
        and perturbation.get("tier") == "nominal"
        and isinstance(application, dict)
        and application.get("valid") is True
        and application.get("nominal_noop") is True
    )


def evaluate_l1_gate(
    nominal_manifests: Iterable[dict[str, Any]],
    perturbation_manifests: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    nominal = list(nominal_manifests)
    perturbed = list(perturbation_manifests)
    all_manifests = nominal + perturbed

    nominal_full_score_runs = sum(_is_full_score(item) for item in nominal)
    perturbation_full_score_runs = sum(_is_full_score(item) for item in perturbed)
    collision_runs = sum(
        item.get("collision_frames") is not None
        and int(item.get("collision_frames")) > 0
        for item in all_manifests
    )
    missing_collision_evidence = sum(
        item.get("collision_frames") is None for item in all_manifests
    )
    invalid_task_manifests = sum(
        not _is_l1_manifest(item) for item in all_manifests
    )
    invalid_nominal_manifests = sum(not _nominal_is_valid(item) for item in nominal)
    invalid_perturbation_manifests = sum(
        not _perturbation_is_valid(item) for item in perturbed
    )
    failure_stages = Counter(
        _terminal_failure_stage(item)
        for item in all_manifests
        if not _is_full_score(item)
    )

    gate_passed = bool(
        len(nominal) == 5
        and nominal_full_score_runs == 5
        and len(perturbed) == 20
        and perturbation_full_score_runs >= 18
        and collision_runs == 0
        and missing_collision_evidence == 0
        and invalid_task_manifests == 0
        and invalid_nominal_manifests == 0
        and invalid_perturbation_manifests == 0
        and all(_has_zero_collision_evidence(item) for item in all_manifests)
    )
    return {
        "nominal_runs": len(nominal),
        "nominal_full_score_runs": nominal_full_score_runs,
        "perturbation_runs": len(perturbed),
        "perturbation_full_score_runs": perturbation_full_score_runs,
        "collision_runs": collision_runs,
        "missing_collision_evidence": missing_collision_evidence,
        "invalid_task_manifests": invalid_task_manifests,
        "invalid_nominal_manifests": invalid_nominal_manifests,
        "invalid_perturbation_manifests": invalid_perturbation_manifests,
        "failure_stages": dict(sorted(failure_stages.items())),
        "nominal_manifest_paths": [
            str(item["manifest_path"])
            for item in nominal
            if item.get("manifest_path")
        ],
        "perturbation_manifest_paths": [
            str(item["manifest_path"])
            for item in perturbed
            if item.get("manifest_path")
        ],
        "gate_passed": gate_passed,
    }


def _load_manifests(directories: Iterable[Path]) -> list[dict[str, Any]]:
    manifests = []
    for directory in directories:
        paths = sorted(Path(directory).rglob("manifest-*.json"))
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"manifest is not a JSON object: {path}")
            value["manifest_path"] = str(path.resolve())
            manifests.append(value)
    return manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--perturbation-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_l1_gate(
        _load_manifests(args.nominal_dir),
        _load_manifests(args.perturbation_dir),
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
