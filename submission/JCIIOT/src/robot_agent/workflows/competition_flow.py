"""Deterministic, verified per-object competition workflow."""

from __future__ import annotations

from typing import Any, Iterable


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
        self._grasp_yaw: float | None = None
        self._swap_arm_targets = False
        self._clearance_prepared = False
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

    def _move_to(self, target: str, *, carrying: bool) -> bool:
        result = self.move_skill.run(
            self._context(
                f"move to {target}",
                target=target,
                carrying=bool(carrying),
            )
        )
        return bool(result.success)

    def _grasp_pose(self, source: str, object_name: str) -> dict:
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            get_target_positions,
        )
        from robot_agent.skills.competition_navigation import (
            grasp_aligned_base_pose,
        )

        station = self.scene_context.input_ports[source]
        body_id = self.backend.env.obj_body_id[object_name]
        object_xy = self.backend.env.sim.data.body_xpos[body_id][:2]
        raw_targets, _ = get_target_positions(
            self.backend.env,
            object_name,
            0.035,
        )
        station_approach = (
            station.approach
            if station.approach is not None
            else self.scene_context.approach_xy(source)
        )
        return grasp_aligned_base_pose(
            object_xy=object_xy,
            right_site_xy=raw_targets["right"][:2],
            left_site_xy=raw_targets["left"][:2],
            station_center=station.center,
            station_approach=station_approach,
        )

    def rank_objects(self, source: str, object_names: Iterable[str]) -> list[str]:
        from robot_agent.skills.competition_navigation import (
            select_grasp_candidate,
        )

        names = [str(name) for name in object_names]
        station = self.scene_context.input_ports[source]
        station_approach = (
            station.approach
            if station.approach is not None
            else self.scene_context.approach_xy(source)
        )
        entries = []
        for name in names:
            pose = self._grasp_pose(source, name)
            body_id = self.backend.env.obj_body_id[name]
            object_xy = self.backend.env.sim.data.body_xpos[body_id][:2]
            entries.append(
                {
                    "name": name,
                    "base_xy": pose["base_xy"],
                    "object_xy": object_xy,
                }
            )

        ranked = []
        remaining = entries
        while remaining:
            selected = select_grasp_candidate(
                remaining,
                station_approach=station_approach,
            )
            ranked.append(selected)
            remaining = [entry for entry in remaining if entry["name"] != selected]
        return ranked

    def _prepare_grasp_clearance(self, object_name: str) -> bool:
        from robot_agent.skills.competition_grasp import (
            OfficialScriptedGraspDriver,
            ScriptedGraspConfig,
        )

        config = self.grasp_config or ScriptedGraspConfig()
        driver = OfficialScriptedGraspDriver()
        return bool(driver.raise_to_clearance(self.backend, object_name, config))

    def move(
        self,
        target: str,
        *,
        carrying: bool,
        object_name: str | None = None,
    ) -> bool:
        resolved_target = target
        staging_target: str | None = None
        active_grasp_pose = None
        orient_for_grasp = False
        if not carrying and object_name:
            self._clearance_prepared = False
            try:
                grasp_pose = self._grasp_pose(target, object_name)
                base_xy = grasp_pose["base_xy"]
                staging_xy = grasp_pose["staging_xy"]
                active_grasp_pose = grasp_pose
                resolved_target = f"{base_xy[0]:.6f}, {base_xy[1]:.6f}"
                staging_target = f"{staging_xy[0]:.6f}, {staging_xy[1]:.6f}"
                self._grasp_yaw = float(grasp_pose["yaw"])
                self._swap_arm_targets = bool(grasp_pose["swap_arm_targets"])
                orient_for_grasp = True
            except (AttributeError, KeyError, TypeError, ValueError):
                resolved_target = target

        if orient_for_grasp and self._grasp_yaw is not None:
            from robot_agent.skills.competition_navigation import orient_base

            if staging_target is None or not self._move_to(
                staging_target,
                carrying=False,
            ):
                return False
            if not orient_base(self.backend, self._grasp_yaw):
                return False
            if staging_target == resolved_target:
                return True
            if object_name is None or not self._prepare_grasp_clearance(object_name):
                return False
            self._clearance_prepared = True
        if not self._move_to(resolved_target, carrying=carrying):
            return False
        if active_grasp_pose is not None:
            from robot_agent.skills.competition_navigation import (
                SAFE_GRASP_YAW_CORRECTION,
                bounded_yaw_step,
                grasp_orientation_from_base,
                orient_base,
            )

            reached_xy, reached_yaw = self.backend.get_base_pose()
            orientation = grasp_orientation_from_base(
                base_xy=reached_xy,
                right_site_xy=active_grasp_pose["right_site_xy"],
                left_site_xy=active_grasp_pose["left_site_xy"],
            )
            self._grasp_yaw = bounded_yaw_step(
                current_yaw=reached_yaw,
                target_yaw=float(orientation["yaw"]),
                max_step=SAFE_GRASP_YAW_CORRECTION,
            )
            self._swap_arm_targets = bool(orientation["swap_arm_targets"])
            return bool(orient_base(self.backend, self._grasp_yaw))
        return True

    def grasp(self, source: str, object_name: str) -> dict[str, Any]:
        from robot_agent.skills.competition_grasp import (
            ScriptedGraspConfig,
            run_scripted_grasp,
        )

        config = self.grasp_config or ScriptedGraspConfig()
        config.swap_arm_targets = self._swap_arm_targets
        config.clearance_prepared = self._clearance_prepared

        return run_scripted_grasp(
            self.backend,
            source=source,
            object_name=object_name,
            config=config,
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

    driver = OfficialCompetitionDriver(
        backend=backend,
        scene_context=scene_context,
        grid=grid,
        grasp_config=grasp_config,
    )
    ranked_candidates = driver.rank_objects(str(task["source"]), candidates)
    object_names = (
        ranked_candidates
        if task.get("level") == "L5"
        else ranked_candidates[:1]
    )
    return CompetitionFlow(driver, max_attempts=max_attempts).run(
        source=str(task["source"]),
        target=str(task["target"]),
        object_names=object_names,
    )
