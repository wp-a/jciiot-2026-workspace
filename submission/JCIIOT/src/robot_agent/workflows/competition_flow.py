"""Deterministic, verified per-object competition workflow."""

from __future__ import annotations

from typing import Any, Iterable


def auxiliary_source_detour(*, target: str, carrying: bool) -> list[float] | None:
    """Use the verified upper corridor when approaching upper-row inputs."""
    if (
        str(target) in {"input_1", "input_2", "aux_input_1"}
        and not bool(carrying)
    ):
        return [12.4, 7.2]
    return None


def physical_output_available(output_names: Iterable[str], target: str) -> bool:
    """Return whether the physics backend registered the semantic output."""
    target = str(target)
    return any(
        str(name) == target or str(name).startswith(f"{target}_")
        for name in output_names
    )


def delivery_inset_target(
    *,
    center,
    approach,
    inset: float = 0.15,
):
    """Move a semantic approach point toward its center by a bounded inset."""
    import numpy as np

    center = np.asarray(center, dtype=float).reshape(2)
    approach = np.asarray(approach, dtype=float).reshape(2)
    direction = center - approach
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        return approach.copy()
    return approach + direction / distance * min(float(inset), distance)


def delivery_slot_target(center, object_name: str | None):
    """Assign the three L5 totes distinct scored positions on one output."""
    import numpy as np

    target = np.asarray(center, dtype=float).reshape(2).copy()
    name = str(object_name or "").lower()
    if "white_tote_b01_left" not in name:
        return target
    if name.endswith("_front"):
        target[0] -= 0.60
    elif name.endswith("_back"):
        target[0] += 0.60
    return target


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
        self.backend = backend
        self.scene_context = scene_context
        self.grasp_config = grasp_config
        self._grasp_yaw: float | None = None
        self._swap_arm_targets = False
        self._clearance_prepared = False
        self._physical_hold: dict[str, Any] | None = None
        self._last_transport: dict[str, Any] | None = None
        self._last_alignment: dict[str, Any] | None = None
        self._last_place: dict[str, Any] | None = None
        self.move_skill = MoveSkill(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            path_spacing=path_spacing,
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

    def _physical_output_available(self, target: str) -> bool:
        ports = getattr(self.backend.env, "output_ports", {})
        names = ports.keys() if hasattr(ports, "keys") else ports
        return physical_output_available(names, target)

    def _grasp_pose(self, source: str, object_name: str) -> dict:
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            get_target_positions,
        )
        from robot_agent.skills.competition_navigation import (
            grasp_aligned_base_pose,
            station_side_grasp_pose,
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
        pose = grasp_aligned_base_pose(
            object_xy=object_xy,
            right_site_xy=raw_targets["right"][:2],
            left_site_xy=raw_targets["left"][:2],
            station_center=station.center,
            station_approach=station_approach,
        )
        if str(source) == "input_1" and "white_tote_b01_left" in object_name.lower():
            pose = station_side_grasp_pose(
                grasp_center_xy=pose["grasp_center_xy"],
                right_site_xy=pose["right_site_xy"],
                left_site_xy=pose["left_site_xy"],
                station_center=station.center,
                station_approach=station_approach,
            )
        return pose

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
            try:
                pose = self._grasp_pose(source, name)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
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
            apply_object_grasp_profile,
        )

        config = apply_object_grasp_profile(
            self.grasp_config or ScriptedGraspConfig(),
            object_name,
        )
        driver = OfficialScriptedGraspDriver()
        return bool(driver.raise_to_clearance(self.backend, object_name, config))

    def move(
        self,
        target: str,
        *,
        carrying: bool,
        object_name: str | None = None,
    ) -> bool:
        if carrying:
            return self._move_physically_while_holding(target, object_name)

        resolved_target = target
        staging_target: str | None = None
        active_grasp_pose = None
        orient_for_grasp = False
        if object_name:
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

            detour = auxiliary_source_detour(
                target=target,
                carrying=carrying,
            )
            if detour is not None and not self._move_to(
                f"{detour[0]:.6f}, {detour[1]:.6f}",
                carrying=False,
            ):
                return False
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

    def _move_physically_while_holding(
        self,
        target: str,
        object_name: str | None,
    ) -> bool:
        import numpy as np

        from robot_agent.skills.competition_transport import (
            PostureLockedPhysicalCarryDriver,
            PhysicalCarryConfig,
            physical_carry_step_budget,
            run_physical_transport,
            transport_base_goal,
        )

        if not object_name or not self._physical_hold:
            return False
        station = self.scene_context.output_ports.get(target)
        if station is None:
            return False

        base_xy, _ = self.backend.get_base_pose()
        hold = self._physical_hold
        object_pos = np.asarray(hold["object_pos"], dtype=float)
        object_target_xy = delivery_slot_target(
            station.center[:2],
            object_name,
        )
        goal_xy = transport_base_goal(
            object_target_xy=object_target_xy,
            base_xy=base_xy,
            base_yaw=float(hold["base_yaw"]),
            object_xy=object_pos[:2],
        )
        path = self.move_skill._plan(
            np.asarray(base_xy, dtype=float),
            np.asarray(goal_xy, dtype=float),
        )
        if not path:
            self._last_transport = {
                "success": False,
                "failure_stage": "path",
            }
            return False
        max_linear = 0.04
        control_dt = 0.05
        config = PhysicalCarryConfig(
            max_steps=physical_carry_step_budget(
                path,
                start_xy=base_xy,
                max_linear=max_linear,
                control_dt=control_dt,
            ),
            max_linear=max_linear,
            max_angular=0.04,
            max_linear_delta=0.005,
            max_angular_delta=0.01,
            base_control_dt=control_dt,
            align_heading_to_path=True,
            heading_translation_tolerance=0.05,
            max_planar_grasp_drift=0.12,
            height_recovery_trigger=0.01,
            planar_recovery_trigger=0.015,
            planar_recovery_steps=4,
            planar_recovery_inward_delta=0.002,
        )
        self._last_transport = run_physical_transport(
            self.backend,
            path=path,
            object_name=object_name,
            hold_yaw=float(hold["base_yaw"]),
            minimum_object_z=(
                float(
                    hold.get(
                        "minimum_transport_object_z",
                        float(hold["object_z"])
                        - config.object_drop_tolerance,
                    )
                )
            ),
            config=config,
            driver=PostureLockedPhysicalCarryDriver(),
        )
        return bool(self._last_transport.get("success", False))

    def grasp(self, source: str, object_name: str) -> dict[str, Any]:
        from robot_agent.skills.competition_grasp import (
            ScriptedGraspConfig,
            run_scripted_grasp,
        )

        config = self.grasp_config or ScriptedGraspConfig()
        config.swap_arm_targets = self._swap_arm_targets
        config.clearance_prepared = self._clearance_prepared

        result = run_scripted_grasp(
            self.backend,
            source=source,
            object_name=object_name,
            config=config,
        )
        if bool(result.get("success", False)):
            self._physical_hold = dict(result.get("hold") or {})
        else:
            self._physical_hold = None
        return result

    def place(self, target: str, object_name: str) -> bool:
        from robot_agent.skills.competition_transport import (
            run_physical_place,
            run_physical_target_alignment,
        )

        station = self.scene_context.output_ports.get(target)
        if station is None or not self._physical_hold:
            return False
        target_xy = delivery_slot_target(station.center[:2], object_name)
        minimum_object_z = float(
            self._physical_hold.get(
                "minimum_transport_object_z",
                float(self._physical_hold["object_z"]) - 0.025,
            )
        )
        self._last_alignment = run_physical_target_alignment(
            self.backend,
            object_name=object_name,
            target_xy=target_xy,
            minimum_object_z=minimum_object_z,
        )
        if not bool(self._last_alignment.get("success", False)):
            return False
        self._last_place = run_physical_place(
            self.backend,
            object_name=object_name,
            target_xy=target_xy,
        )
        if bool(self._last_place.get("success", False)):
            self._physical_hold = None
        return bool(self._last_place.get("success", False))

    def verify(self, target: str, object_name: str) -> bool:
        import numpy as np

        station = self.scene_context.output_ports.get(target)
        if station is None:
            return False
        try:
            body_id = self.backend.env.obj_body_id[object_name]
            object_xy = np.asarray(
                self.backend.env.sim.data.body_xpos[body_id][:2],
                dtype=float,
            )
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
