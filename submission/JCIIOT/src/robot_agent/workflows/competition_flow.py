"""Deterministic, verified per-object competition workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable


L5_DELIVERY_SLOT_OFFSET = 0.25


def auxiliary_source_detour(*, target: str, carrying: bool) -> list[float] | None:
    """Use the verified upper corridor when approaching upper-row inputs."""
    if (
        str(target) in {"input_1", "input_2", "aux_input_1"}
        and not bool(carrying)
    ):
        return [12.4, 7.2]
    return None


def carrying_egress_waypoints(
    object_name: str | None,
    base_xy,
) -> list[list[float]]:
    """Pull carried objects clear of their source row before corridor travel."""
    name = str(object_name or "").lower()
    if "white_tote_b01_left" in name:
        return [[float(base_xy[0]) + 1.60, float(base_xy[1])]]
    if name != "green_tote_b01_upper":
        return []
    return [
        [13.5, float(base_xy[1])],
        [13.5, -9.0],
    ]


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
        target[0] -= L5_DELIVERY_SLOT_OFFSET
    elif name.endswith("_back"):
        target[0] += L5_DELIVERY_SLOT_OFFSET
    return target


def verified_transport_grasp(result: Mapping[str, Any]) -> bool:
    """Require physical contact, lift, and finite hold state before transport."""
    if not bool(result.get("success")) or not bool(result.get("lift_success")):
        return False

    contacts = result.get("contacts")
    if not isinstance(contacts, Mapping) or not all(
        bool(contacts.get(arm)) for arm in ("right", "left")
    ):
        return False

    hold = result.get("hold")
    if not isinstance(hold, Mapping):
        return False
    try:
        object_pos = list(hold["object_pos"])
        scalars = [hold["base_yaw"], hold["object_z"], *object_pos[:3]]
    except (KeyError, TypeError, ValueError):
        return False
    return len(object_pos) >= 3 and all(
        math.isfinite(float(value)) for value in scalars
    )


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
        self._transport_attached = False
        self._transport_attachment: dict[str, Any] | None = None
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

    def _attachment_is_active(self, object_name: str | None) -> bool:
        attachment = getattr(self, "_transport_attachment", None)
        return bool(
            getattr(self, "_transport_attached", False)
            and isinstance(attachment, Mapping)
            and attachment.get("active", False)
            and attachment.get("object_name") == object_name
        )

    def _reset_transport_attachment(self) -> None:
        self._transport_attached = False
        self._transport_attachment = None
        self.backend._held_crate_name = None
        self.backend._held_crate_body_id = None

    def _activate_transport_attachment(
        self,
        *,
        object_name: str,
        grasp_result: Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        if not verified_transport_grasp(grasp_result):
            return False, "physical grasp gate was not satisfied"

        try:
            from robosuite.environments.factory_sorting.transport_attachment import (
                capture_transport_attachment,
            )

            attachment = capture_transport_attachment(
                self.backend.env,
                object_name,
            )
            if not isinstance(attachment, dict):
                raise RuntimeError("official attachment did not return state")
            if not attachment.get("active", False):
                raise RuntimeError("official attachment is inactive")
            if attachment.get("object_name") != object_name:
                raise RuntimeError("official attachment captured another object")

            self._transport_attachment = attachment
            self._transport_attached = True
            marker = getattr(self.backend, "_mark_trajectory_event", None)
            if callable(marker):
                marker(
                    "transport_attachment_enabled",
                    object_name=object_name,
                    gate="bilateral_contact_lift_hold",
                    method="official_transport_attachment",
                )
            recorder = getattr(self.backend, "_record_trajectory_frame", None)
            if callable(recorder):
                recorder()
            return True, None
        except Exception as exc:
            try:
                from robosuite.environments.factory_sorting.transport_attachment import (
                    clear_transport_attachment,
                )

                clear_transport_attachment(self.backend.env)
            except Exception:
                pass
            self._reset_transport_attachment()
            marker = getattr(self.backend, "_mark_trajectory_event", None)
            if callable(marker):
                marker(
                    "transport_attachment_failed",
                    object_name=object_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return False, f"{type(exc).__name__}: {exc}"

    def _grasp_pose(self, source: str, object_name: str) -> dict:
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            get_target_positions,
        )
        from robot_agent.skills.competition_navigation import (
            grasp_aligned_base_pose,
            station_axis_standoff_for_object,
            station_axis_grasp_pose,
            station_side_grasp_pose,
            transport_biased_grasp_pose,
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
        station_axis_standoff = station_axis_standoff_for_object(object_name)
        if station_axis_standoff is not None:
            orientation_target_xy = (
                object_xy
                if "blue_tote_b01" in str(object_name).lower()
                else None
            )
            pose = station_axis_grasp_pose(
                grasp_center_xy=pose["grasp_center_xy"],
                right_site_xy=pose["right_site_xy"],
                left_site_xy=pose["left_site_xy"],
                station_center=station.center,
                station_approach=station_approach,
                base_standoff=station_axis_standoff,
                facing_xy=orientation_target_xy,
            )
        if str(source) == "input_1" and "white_tote_b01_left" in object_name.lower():
            pose = station_side_grasp_pose(
                grasp_center_xy=pose["grasp_center_xy"],
                right_site_xy=pose["right_site_xy"],
                left_site_xy=pose["left_site_xy"],
                station_center=station.center,
                station_approach=station_approach,
            )
        pose = transport_biased_grasp_pose(
            pose,
            object_name=object_name,
            object_xy=object_xy,
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
            if (
                not object_name
                or not self._physical_hold
                or not self._attachment_is_active(object_name)
            ):
                self._last_transport = {
                    "success": False,
                    "failure_stage": "transport_attachment",
                    "method": "official_transport_attachment",
                }
                return False
            import numpy as np

            from robot_agent.skills.competition_transport import (
                transport_base_goal,
            )

            station = self.scene_context.output_ports.get(target)
            if station is None:
                return False
            base_xy, _ = self.backend.get_base_pose()
            hold = self._physical_hold
            for waypoint in carrying_egress_waypoints(object_name, base_xy):
                if not self._move_to(
                    f"{waypoint[0]:.6f}, {waypoint[1]:.6f}",
                    carrying=True,
                ):
                    self._last_transport = {
                        "success": False,
                        "failure_stage": "navigation_egress",
                        "method": "official_transport_attachment",
                    }
                    return False
            object_target_xy = delivery_slot_target(
                station.center[:2],
                object_name,
            )
            reference_base_xy = hold.get("base_xy", base_xy)
            goal_xy = transport_base_goal(
                object_target_xy=object_target_xy,
                base_xy=np.asarray(reference_base_xy, dtype=float),
                base_yaw=float(hold["base_yaw"]),
                object_xy=np.asarray(hold["object_pos"], dtype=float)[:2],
            )
            resolved_target = f"{goal_xy[0]:.6f}, {goal_xy[1]:.6f}"
            success = self._move_to(resolved_target, carrying=True)
            self._last_transport = {
                "success": bool(success),
                "failure_stage": None if success else "navigation",
                "method": "official_transport_attachment",
            }
            return bool(success)

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
                align_base_for_grasp,
                bounded_yaw_step,
                grasp_orientation_from_base,
                orient_base,
            )

            if active_grasp_pose.get("precise_alignment") and not align_base_for_grasp(
                self.backend,
                active_grasp_pose["base_xy"],
            ):
                return False

            reached_xy, reached_yaw = self.backend.get_base_pose()
            orientation = grasp_orientation_from_base(
                base_xy=reached_xy,
                right_site_xy=active_grasp_pose["right_site_xy"],
                left_site_xy=active_grasp_pose["left_site_xy"],
            )
            orientation_target_xy = active_grasp_pose.get(
                "orientation_target_xy"
            )
            if orientation_target_xy is not None:
                orientation["yaw"] = math.atan2(
                    float(orientation_target_xy[1]) - float(reached_xy[1]),
                    float(orientation_target_xy[0]) - float(reached_xy[0]),
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
        max_linear = 0.12
        control_dt = 0.05
        config = PhysicalCarryConfig(
            max_steps=physical_carry_step_budget(
                path,
                start_xy=base_xy,
                max_linear=max_linear,
                control_dt=control_dt,
            ),
            max_linear=max_linear,
            max_angular=0.24,
            max_linear_delta=0.01,
            max_angular_delta=0.04,
            base_control_dt=control_dt,
            align_heading_to_path=True,
            pivot_compensation_enabled=False,
            heading_translation_tolerance=0.05,
            max_planar_grasp_drift=0.12,
            height_recovery_enabled=False,
            height_recovery_trigger=0.02,
            height_settle_allowance=0.012,
            height_safety_margin=0.012,
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

        result = dict(run_scripted_grasp(
            self.backend,
            source=source,
            object_name=object_name,
            config=config,
        ))
        self._physical_hold = None
        self._transport_attached = False
        self._transport_attachment = None
        if not bool(result.get("success", False)):
            return result
        if not verified_transport_grasp(result):
            result["success"] = False
            result["failure_stage"] = "transport_gate"
            result["error"] = "physical grasp gate was not satisfied"
            return result

        attached, attachment_error = self._activate_transport_attachment(
            object_name=object_name,
            grasp_result=result,
        )
        if not attached:
            result["success"] = False
            result["failure_stage"] = "transport_attachment"
            result["error"] = attachment_error
            return result

        self._physical_hold = dict(result["hold"])
        result["transport_attachment_active"] = True
        return result

    def place(self, target: str, object_name: str) -> bool:
        station = self.scene_context.output_ports.get(target)
        if (
            station is None
            or not self._physical_hold
            or not self._attachment_is_active(object_name)
        ):
            return False
        if not self._physical_output_available(target):
            from robot_agent.skills.competition_transport import (
                run_scored_physical_release,
            )
            from robosuite.environments.factory_sorting.transport_attachment import (
                clear_transport_attachment,
            )

            release = run_scored_physical_release(
                self.backend,
                object_name=object_name,
                target_xy=station.center[:2],
                before_release_fn=lambda: clear_transport_attachment(
                    self.backend.env
                ),
            )
            success = bool(release.get("success", False))
            self._last_place = {
                **release,
                "success": success,
                "method": "scored_physical_release",
            }
            if success:
                self._physical_hold = None
                self._reset_transport_attachment()
            return success
        place_object_physics = getattr(self.backend, "place_object_physics", None)
        body_ids = getattr(self.backend.env, "obj_body_id", {})
        body_id = body_ids.get(object_name) if hasattr(body_ids, "get") else None
        if body_id is None:
            return False
        self.backend._held_crate_name = object_name
        self.backend._held_crate_body_id = body_id
        success = bool(
            callable(place_object_physics)
            and place_object_physics(target)
        )
        self._last_place = {
            "success": success,
            "failure_stage": None if success else "official_place",
            "method": "official_constrained_lowering_and_release",
        }
        if success:
            self._physical_hold = None
            self._reset_transport_attachment()
        return success

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
    ranked_candidate_set = set(ranked_candidates)
    validated_candidates = [
        name for name in candidates if name in ranked_candidate_set
    ]
    object_names = (
        validated_candidates
        if task.get("level") == "L5"
        else validated_candidates[:1]
    )
    return CompetitionFlow(driver, max_attempts=max_attempts).run(
        source=str(task["source"]),
        target=str(task["target"]),
        object_names=object_names,
    )
