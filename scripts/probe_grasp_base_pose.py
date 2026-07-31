#!/usr/bin/env python3
"""Probe one collision-free base pose through the scripted grasp stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_official_experiment as experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--base-x", type=float, required=True)
    parser.add_argument("--base-y", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app_dir = experiment._configure_candidate(args.candidate_root.resolve())
    tasks = json.loads(
        (app_dir / "knowledge" / "task_config.json").read_text(encoding="utf-8")
    )["tasks"]
    task = tasks[args.task_index]
    backend, _, _ = experiment._load_scene(app_dir, task, args.seed)

    from robot_agent.environments.robosuite_backend import (
        _navigation_collisions,
        _set_base_world_yaw_direct,
        _set_base_xy_direct,
    )
    from robot_agent.skills.competition_grasp import (
        OfficialScriptedGraspDriver,
        ScriptedGraspConfig,
        apply_object_grasp_profile,
        run_scripted_grasp,
    )

    diagnostics: list[dict[str, object]] = []
    stage_notes: list[dict[str, object]] = []
    close_trace: list[dict[str, object]] = []
    close_active = False
    original_move = OfficialScriptedGraspDriver._move_to_targets
    original_helpers = OfficialScriptedGraspDriver._helpers
    original_adjust = OfficialScriptedGraspDriver.adjust_wrist_for_reach
    original_close = OfficialScriptedGraspDriver.close_and_check_contacts

    def measured_helpers():
        helpers = original_helpers()
        original_status = helpers["grasp_status"]

        def measured_status(raw_env, active_robot, object_name):
            contacts = original_status(raw_env, active_robot, object_name)
            if close_active:
                close_trace.append(
                    {
                        "sample": len(close_trace) + 1,
                        "contacts": {
                            arm: bool(contacts.get(arm, False))
                            for arm in ("right", "left")
                        },
                        "object_position": raw_env.sim.data.body_xpos[
                            raw_env.obj_body_id[object_name]
                        ].tolist(),
                        "eef_positions": {
                            arm: helpers["gripper_position"](
                                raw_env, active_robot, arm
                            ).tolist()
                            for arm in ("right", "left")
                        },
                    }
                )
            return contacts

        helpers["grasp_status"] = measured_status
        return helpers

    def snapshot(self, active_backend, object_name):
        helpers = self._helpers()
        active_env = active_backend.env
        active_robot = active_env.robots[0]
        fingerpads = {}
        for arm in ("right", "left"):
            geom_names = [
                active_robot.gripper[arm].important_geoms[key][0]
                for key in ("left_fingerpad", "right_fingerpad")
            ]
            fingerpads[arm] = [
                active_env.sim.data.geom_xpos[
                    active_env.sim.model.geom_name2id(name)
                ].tolist()
                for name in geom_names
            ]
        return {
            "object_position": active_env.sim.data.body_xpos[
                active_env.obj_body_id[object_name]
            ].tolist(),
            "eef_positions": {
                arm: helpers["gripper_position"](
                    active_env, active_robot, arm
                ).tolist()
                for arm in ("right", "left")
            },
            "fingerpads": fingerpads,
        }

    def measured_move(self, active_backend, targets, config, **kwargs):
        result = original_move(self, active_backend, targets, config, **kwargs)
        helpers = self._helpers()
        raw_env = active_backend.env
        robot = raw_env.robots[0]
        current = {
            arm: np.asarray(
                helpers["gripper_position"](raw_env, robot, arm), dtype=float
            )
            for arm in ("right", "left")
        }
        requested = {
            arm: np.asarray(targets[arm], dtype=float)
            for arm in ("right", "left")
        }
        diagnostics.append(
            {
                "call": len(diagnostics) + 1,
                "result": bool(result),
                "target": {arm: requested[arm].tolist() for arm in requested},
                "current": {arm: current[arm].tolist() for arm in current},
                "error_m": {
                    arm: float(np.linalg.norm(current[arm] - requested[arm]))
                    for arm in current
                },
            }
        )
        return result

    def measured_adjust(self, active_backend, object_name, config):
        note = {
            "stage": "wrist_adjustment",
            "before": snapshot(self, active_backend, object_name),
        }
        result = original_adjust(self, active_backend, object_name, config)
        note["result"] = bool(result)
        note["after"] = snapshot(self, active_backend, object_name)
        stage_notes.append(note)
        return result

    def measured_close(self, active_backend, object_name, config):
        nonlocal close_active
        note = {
            "stage": "close",
            "before": snapshot(self, active_backend, object_name),
            "requested_targets": {
                arm: target.tolist()
                for arm, target in self._grasp_targets(
                    active_backend,
                    object_name,
                    config,
                    height_offset=0.0,
                ).items()
            },
        }
        close_active = True
        try:
            contacts = original_close(self, active_backend, object_name, config)
        finally:
            close_active = False
        note["reported_contacts"] = {
            arm: bool(contacts.get(arm, False))
            for arm in ("right", "left")
        }
        note["after"] = snapshot(self, active_backend, object_name)
        stage_notes.append(note)
        return contacts

    raw_env = backend.env
    robot = raw_env.robots[0]
    body_id = raw_env.obj_body_id[args.object_name]
    target_xy = np.asarray(raw_env.sim.data.body_xpos[body_id][:2], dtype=float)
    target_yaw = math.atan2(
        target_xy[1] - args.base_y,
        target_xy[0] - args.base_x,
    )
    _set_base_world_yaw_direct(raw_env, robot, target_yaw)
    _set_base_xy_direct(raw_env, robot, np.array([args.base_x, args.base_y]))
    raw_env.sim.forward()
    initial_collisions = _navigation_collisions(
        raw_env,
        robot,
        getattr(backend, "_ignore_collision_geom", ()),
    )
    backend.start_recording()
    OfficialScriptedGraspDriver._move_to_targets = measured_move
    OfficialScriptedGraspDriver._helpers = staticmethod(measured_helpers)
    OfficialScriptedGraspDriver.adjust_wrist_for_reach = measured_adjust
    OfficialScriptedGraspDriver.close_and_check_contacts = measured_close
    try:
        config = apply_object_grasp_profile(
            ScriptedGraspConfig(),
            args.object_name,
        )
        result = run_scripted_grasp(
            backend,
            source=str(task["source"]),
            object_name=args.object_name,
            config=config,
        )
        backend.save_trajectory(args.trajectory)
        report = {
            "task_index": args.task_index,
            "object_name": args.object_name,
            "base_xy": [args.base_x, args.base_y],
            "base_yaw": target_yaw,
            "initial_collisions": initial_collisions,
            "result": result,
            "stage_diagnostics": diagnostics,
            "stage_notes": stage_notes,
            "close_trace": close_trace,
            "trajectory": str(args.trajectory),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    finally:
        OfficialScriptedGraspDriver._move_to_targets = original_move
        OfficialScriptedGraspDriver._helpers = staticmethod(original_helpers)
        OfficialScriptedGraspDriver.adjust_wrist_for_reach = original_adjust
        OfficialScriptedGraspDriver.close_and_check_contacts = original_close
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
