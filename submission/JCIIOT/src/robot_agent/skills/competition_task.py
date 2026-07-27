"""Deterministic adapter from the official agent entry point to competition flow."""

from __future__ import annotations

import json
from pathlib import Path

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill
from robot_agent.workflows.competition_flow import run_official_task


def task_index_from_metadata(metadata: dict) -> int:
    """Return the official zero-based task index without guessing a fallback."""
    value = metadata.get("task_index")
    scene = metadata.get("scene")
    if value is None and isinstance(scene, dict):
        value = scene.get("task_index")

    if isinstance(value, bool):
        raise ValueError("task_index must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    raise ValueError("task_index is missing or invalid")


def load_official_task(
    task_index: int,
    config_path: Path | None = None,
) -> dict:
    """Load one task from the immutable official task configuration."""
    path = config_path or (
        Path(__file__).resolve().parents[3]
        / "knowledge"
        / "task_config.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task_config.tasks must be a list")
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task_index out of range: {task_index}")

    task = tasks[task_index]
    required = {"level", "source", "target", "object", "max_score"}
    if not isinstance(task, dict) or not required.issubset(task):
        raise ValueError(f"malformed task entry at index {task_index}")
    return dict(task)


class CompetitionTaskSkill(BaseSkill):
    """Execute the selected official task through the verified state machine."""

    def __init__(self, *, backend, scene_context, grid) -> None:
        super().__init__(
            name="competition_task",
            description="Execute the selected official competition task",
            keywords=("competition", "official", "task"),
        )
        self._backend = backend
        self._scene_context = scene_context
        self._grid = grid

    def can_handle(self, task: str) -> bool:
        return bool(str(task).strip())

    def run(self, context: ExecutionContext) -> SkillResult:
        try:
            task_index = task_index_from_metadata(context.metadata)
            task = load_official_task(task_index)
        except (IndexError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Official task resolution failed: {exc}",
                payload={
                    "error_code": "task_resolution_failed",
                    "error_type": type(exc).__name__,
                },
            )

        try:
            workflow = run_official_task(
                backend=self._backend,
                scene_context=self._scene_context,
                grid=self._grid,
                task=task,
                max_attempts=1,
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Competition workflow failed: {exc}",
                payload={
                    "task_index": task_index,
                    "level": task["level"],
                    "error_code": "workflow_exception",
                    "error_type": type(exc).__name__,
                },
            )

        if not isinstance(workflow, dict):
            return SkillResult(
                skill_name=self.name,
                success=False,
                message="Competition workflow returned an invalid result",
                payload={
                    "task_index": task_index,
                    "level": task["level"],
                    "error_code": "workflow_invalid",
                },
            )

        success = bool(workflow.get("success", False))
        payload = {
            "task_index": task_index,
            "level": task["level"],
            "workflow": workflow,
        }
        if not success:
            payload["error_code"] = "workflow_failed"
        return SkillResult(
            skill_name=self.name,
            success=success,
            message=(
                f"Official {task['level']} workflow completed"
                if success
                else f"Official {task['level']} workflow failed"
            ),
            payload=payload,
        )
