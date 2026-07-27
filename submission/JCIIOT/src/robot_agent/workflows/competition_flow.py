"""Deterministic, verified per-object competition workflow."""

from __future__ import annotations

from typing import Any, Iterable


def object_aligned_approach(
    *,
    station_center,
    station_approach,
    object_xy,
) -> list[float]:
    """Translate a station approach point by the object's planar offset."""
    return [
        float(station_approach[0]) + float(object_xy[0]) - float(station_center[0]),
        float(station_approach[1]) + float(object_xy[1]) - float(station_center[1]),
    ]


class CompetitionFlow:
    """Execute bounded object transfers through a small state machine."""

    def __init__(self, driver, *, max_attempts: int = 1) -> None:
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1")
        self.driver = driver
        self.max_attempts = int(max_attempts)

    @staticmethod
    def _event(object_name: str, state: str, attempt: int, **extra: Any) -> dict:
        event = {
            "object_name": object_name,
            "state": state,
            "attempt": int(attempt),
        }
        event.update(extra)
        return event

    def run(
        self,
        *,
        source: str,
        target: str,
        object_names: Iterable[str],
    ) -> dict[str, Any]:
        names = [str(name) for name in object_names if str(name)]
        if not names:
            raise ValueError("object_names must contain at least one object")

        states = {name: "pending" for name in names}
        history: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for object_name in names:
            for attempt in range(1, self.max_attempts + 1):
                states[object_name] = "pending"
                history.append(self._event(object_name, "pending", attempt))

                if not self.driver.move(
                    source,
                    carrying=False,
                    object_name=object_name,
                ):
                    stage = "move_source"
                else:
                    states[object_name] = "approached"
                    history.append(self._event(object_name, "approached", attempt))
                    grasp = self.driver.grasp(source, object_name)
                    if not bool(grasp.get("success", False)):
                        stage = "grasp"
                    else:
                        states[object_name] = "grasped"
                        history.append(self._event(object_name, "grasped", attempt))
                        if not bool(grasp.get("lift_success", False)):
                            stage = "lift"
                        else:
                            states[object_name] = "lifted"
                            history.append(self._event(object_name, "lifted", attempt))
                            if not self.driver.move(
                                target,
                                carrying=True,
                                object_name=object_name,
                            ):
                                stage = "transport"
                            else:
                                states[object_name] = "transported"
                                history.append(self._event(object_name, "transported", attempt))
                                if not self.driver.place(target, object_name):
                                    stage = "place"
                                else:
                                    states[object_name] = "placed"
                                    history.append(self._event(object_name, "placed", attempt))
                                    if not self.driver.verify(target, object_name):
                                        stage = "verify"
                                    else:
                                        states[object_name] = "verified"
                                        history.append(self._event(object_name, "verified", attempt))
                                        break

                failure = {
                    "object_name": object_name,
                    "attempt": attempt,
                    "failure_stage": stage,
                }
                failures.append(failure)
                if attempt == self.max_attempts:
                    states[object_name] = "failed"
                    history.append(self._event(object_name, "failed", attempt, failure_stage=stage))

            if states[object_name] != "verified":
                break

        return {
            "success": all(state == "verified" for state in states.values()),
            "source": source,
            "target": target,
            "states": states,
            "history": history,
            "failures": failures,
        }


class OfficialCompetitionDriver:
    """Connect the state machine to the allowed official skill interfaces."""

    def __init__(
        self,
        *,
        backend,
        scene_context,
        grid,
        path_spacing: float = 0.35,
        grasp_config=None,
    ) -> None:
        from robot_agent.skills.move import MoveSkill
        from robot_agent.skills.place_down import PlaceDownSkill

        self.backend = backend
        self.scene_context = scene_context
        self.grasp_config = grasp_config
        self.move_skill = MoveSkill(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            path_spacing=path_spacing,
        )
        self.place_skill = PlaceDownSkill(
            backend=backend,
            scene_context=scene_context,
        )

    @staticmethod
    def _context(task: str, **inputs):
        from robot_agent.core.types import ExecutionContext

        return ExecutionContext(task=task, metadata={"inputs": inputs})

    def move(
        self,
        target: str,
        *,
        carrying: bool,
        object_name: str | None = None,
    ) -> bool:
        resolved_target = target
        if not carrying and object_name:
            try:
                station = self.scene_context.input_ports[target]
                station_approach = (
                    station.approach
                    if station.approach is not None
                    else self.scene_context.approach_xy(target)
                )
                body_id = self.backend.env.obj_body_id[object_name]
                object_xy = self.backend.env.sim.data.body_xpos[body_id][:2]
                aligned_xy = object_aligned_approach(
                    station_center=station.center,
                    station_approach=station_approach,
                    object_xy=object_xy,
                )
                resolved_target = f"{aligned_xy[0]:.6f}, {aligned_xy[1]:.6f}"
            except (AttributeError, KeyError, TypeError, ValueError):
                resolved_target = target

        result = self.move_skill.run(
            self._context(
                f"move to {resolved_target}",
                target=resolved_target,
                carrying=bool(carrying),
            )
        )
        return bool(result.success)

    def grasp(self, source: str, object_name: str) -> dict[str, Any]:
        from robot_agent.skills.competition_grasp import run_scripted_grasp

        return run_scripted_grasp(
            self.backend,
            source=source,
            object_name=object_name,
            config=self.grasp_config,
        )

    def place(self, target: str, object_name: str) -> bool:
        result = self.place_skill.run(
            self._context(
                f"place {object_name} at {target}",
                target=target,
                object_name=object_name,
            )
        )
        return bool(result.success)

    def verify(self, target: str, object_name: str) -> bool:
        import numpy as np

        station = self.scene_context.output_ports.get(target)
        if station is None:
            return False
        try:
            qpos_addr = self.backend._get_object_joint_addr(object_name)
            if isinstance(qpos_addr, tuple):
                object_xy = np.asarray(
                    self.backend.env.sim.data.qpos[qpos_addr[0]:qpos_addr[0] + 2],
                    dtype=float,
                )
            else:
                return False
            target_xy = np.asarray(station.center[:2], dtype=float)
            distance = float(np.linalg.norm(object_xy - target_xy))
            recorder = getattr(self.backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder()
            return distance < 0.80
        except Exception:
            return False


def run_official_task(
    *,
    backend,
    scene_context,
    grid,
    task: dict[str, Any],
    max_attempts: int = 1,
    grasp_config=None,
) -> dict[str, Any]:
    """Execute one official task entry without asking an LLM for actions."""
    raw_objects = task.get("object", [])
    if isinstance(raw_objects, str):
        candidates = [raw_objects]
    else:
        candidates = [str(name) for name in raw_objects if name]
    object_names = candidates if task.get("level") == "L5" else candidates[:1]

    driver = OfficialCompetitionDriver(
        backend=backend,
        scene_context=scene_context,
        grid=grid,
        grasp_config=grasp_config,
    )
    return CompetitionFlow(driver, max_attempts=max_attempts).run(
        source=str(task["source"]),
        target=str(task["target"]),
        object_names=object_names,
    )
