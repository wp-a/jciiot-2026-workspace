#!/usr/bin/env python3
"""Validate tracked task facts against the locked official checkout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    """Raised when tracked workspace facts differ from the official baseline."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"expected JSON object in {path}")
    return data


def _index_tasks(tasks: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(tasks, list) or not tasks:
        raise ValidationError(f"{label} tasks must be a non-empty array")

    indexed: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("level"), str):
            raise ValidationError(f"{label} task is missing a string level")
        level = task["level"]
        if level in indexed:
            raise ValidationError(f"duplicate {label} task level: {level}")
        indexed[level] = task
    return indexed


def _require_equal(level: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValidationError(
            f"{level} {field} mismatch: workspace={actual!r} official={expected!r}"
        )


def _station_center(
    semantic_map: dict[str, Any], level: str, collection: str, station: str
) -> list[float]:
    stations = semantic_map.get(collection)
    if not isinstance(stations, dict) or station not in stations:
        raise ValidationError(f"{level} {station} missing from semantic map {collection}")
    entry = stations[station]
    center = entry.get("center") if isinstance(entry, dict) else None
    if not isinstance(center, list) or len(center) < 2:
        raise ValidationError(f"{level} {station} has no two-dimensional center")
    try:
        return [float(center[0]), float(center[1])]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{level} {station} center is not numeric") from exc


def _require_coordinates(
    level: str, field: str, actual: Any, expected: list[float]
) -> None:
    if not isinstance(actual, list) or len(actual) != 2:
        raise ValidationError(f"{level} {field} must contain two coordinates")
    try:
        matches = all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(actual, expected)
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValidationError(
            f"{level} {field} mismatch: workspace={actual!r} official={expected!r}"
        )


def validate_workspace(workspace: Path) -> list[str]:
    workspace = workspace.resolve()
    tasks_data = _load_json(workspace / "config" / "tasks.json")
    lock_data = _load_json(workspace / "config" / "upstream-lock.json")

    repository = lock_data.get("repository")
    if not isinstance(repository, dict):
        raise ValidationError("upstream lock is missing repository metadata")
    locked_commit = repository.get("commit")
    _require_equal(
        "workspace",
        "official_commit",
        tasks_data.get("official_commit"),
        locked_commit,
    )

    vendor_rel = repository.get("local_path")
    if not isinstance(vendor_rel, str) or not vendor_rel:
        raise ValidationError("upstream lock is missing repository.local_path")
    vendor = workspace / vendor_rel
    official_config = _load_json(
        vendor / "JCIIOT" / "knowledge" / "task_config.json"
    )

    workspace_tasks = _index_tasks(tasks_data.get("tasks"), "workspace")
    official_tasks = _index_tasks(official_config.get("tasks"), "official")
    _require_equal(
        "workspace",
        "task levels",
        set(workspace_tasks),
        set(official_tasks),
    )

    map_dir = (
        vendor
        / "JCIIOT"
        / "robosuite"
        / "robosuite"
        / "environments"
        / "factory_sorting"
        / "generated_maps"
    )
    validated: list[str] = []
    for official_task in official_config["tasks"]:
        level = official_task["level"]
        tracked = workspace_tasks[level]
        _require_equal(level, "scene", tracked.get("scene"), official_task.get("env_name"))
        _require_equal(level, "source", tracked.get("source"), official_task.get("source"))
        _require_equal(level, "target", tracked.get("target"), official_task.get("target"))

        official_objects = official_task.get("object")
        if isinstance(official_objects, str):
            official_objects = [official_objects]
        _require_equal(level, "objects", tracked.get("objects"), official_objects)
        _require_equal(
            level, "max_score", tracked.get("max_score"), official_task.get("max_score")
        )

        scene_prefix = official_task.get("scene_prefix")
        if not isinstance(scene_prefix, str) or not scene_prefix:
            raise ValidationError(f"{level} official task is missing scene_prefix")
        semantic_map = _load_json(
            map_dir / f"{scene_prefix}_scene_regenerated_semantic_map.json"
        )
        source_center = _station_center(
            semantic_map, level, "input_ports", official_task["source"]
        )
        target_center = _station_center(
            semantic_map, level, "output_ports", official_task["target"]
        )
        _require_coordinates(
            level, "source_center_xy", tracked.get("source_center_xy"), source_center
        )
        _require_coordinates(
            level, "target_center_xy", tracked.get("target_center_xy"), target_center
        )
        validated.append(level)

    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare tracked task facts with the locked official checkout."
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        levels = validate_workspace(args.workspace)
    except ValidationError as exc:
        print(f"official task validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"official task facts valid: {len(levels)} tasks ({', '.join(levels)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
