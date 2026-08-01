#!/usr/bin/env python3
"""Run one pre-registered BC-RNN grasp in the full official L1 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from scripts import run_official_experiment as official_runner
    from scripts.evaluate_bc_rnn_offline import sha256_file
except ImportError:
    import run_official_experiment as official_runner
    from evaluate_bc_rnn_offline import sha256_file


EXPECTED_ACTION_DIM = 20


def execute_policy_window(
    *,
    policy,
    observation_fn: Callable[[], dict[str, np.ndarray]],
    step_fn: Callable[[np.ndarray], Any],
    contact_fn: Callable[[], dict[str, bool]],
    object_z_fn: Callable[[], float],
    record_fn: Callable[[], None],
    max_steps: int,
    required_lift_m: float,
    stable_steps: int,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if stable_steps < 1:
        raise ValueError("stable_steps must be positive")
    if required_lift_m <= 0.0:
        raise ValueError("required_lift_m must be positive")

    policy.start_episode()
    initial_object_z = float(object_z_fn())
    stable_count = 0
    clipping_count = 0
    maximum_raw_action = 0.0
    contacts = {"right": False, "left": False}
    final_lift = 0.0

    for step_index in range(max_steps):
        try:
            raw_action = np.asarray(
                policy(ob=observation_fn()),
                dtype=float,
            )
        except Exception as exc:
            return {
                "success": False,
                "failure_stage": "policy",
                "error": f"{type(exc).__name__}: {exc}",
                "steps": step_index,
                "stable_steps": stable_count,
                "contacts": contacts,
                "lift_m": final_lift,
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }
        if raw_action.shape != (EXPECTED_ACTION_DIM,):
            return {
                "success": False,
                "failure_stage": "action_shape",
                "error": f"expected {(EXPECTED_ACTION_DIM,)}, got {raw_action.shape}",
                "steps": step_index,
                "stable_steps": stable_count,
                "contacts": contacts,
                "lift_m": final_lift,
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }
        if not np.all(np.isfinite(raw_action)):
            return {
                "success": False,
                "failure_stage": "action_finite",
                "error": "policy emitted a non-finite action",
                "steps": step_index,
                "stable_steps": stable_count,
                "contacts": contacts,
                "lift_m": final_lift,
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }
        maximum_raw_action = max(
            maximum_raw_action,
            float(np.max(np.abs(raw_action))),
        )
        clipping_count += int(np.any(np.abs(raw_action) > 1.0))
        action = np.clip(raw_action, -1.0, 1.0)
        try:
            step_result = step_fn(action)
            record_fn()
        except Exception as exc:
            return {
                "success": False,
                "failure_stage": "environment_step",
                "error": f"{type(exc).__name__}: {exc}",
                "steps": step_index,
                "stable_steps": stable_count,
                "contacts": contacts,
                "lift_m": final_lift,
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }
        info = (
            step_result[3]
            if isinstance(step_result, tuple)
            and len(step_result) > 3
            and isinstance(step_result[3], dict)
            else {}
        )
        steps_taken = step_index + 1
        if bool(info.get("has_judge_collision", False)):
            return {
                "success": False,
                "failure_stage": "collision",
                "error": None,
                "steps": steps_taken,
                "stable_steps": stable_count,
                "contacts": contacts,
                "lift_m": final_lift,
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }
        contacts = {
            arm: bool(contact_fn().get(arm, False))
            for arm in ("right", "left")
        }
        final_lift = float(object_z_fn()) - initial_object_z
        if all(contacts.values()) and final_lift >= required_lift_m:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= stable_steps:
            return {
                "success": True,
                "failure_stage": None,
                "error": None,
                "steps": steps_taken,
                "stable_steps": stable_count,
                "contacts": contacts,
                "initial_object_z": initial_object_z,
                "final_object_z": float(object_z_fn()),
                "lift_m": final_lift,
                "required_lift_m": float(required_lift_m),
                "clipping_count": clipping_count,
                "max_abs_raw_action": maximum_raw_action,
            }

    return {
        "success": False,
        "failure_stage": "timeout",
        "error": None,
        "steps": max_steps,
        "stable_steps": stable_count,
        "contacts": contacts,
        "initial_object_z": initial_object_z,
        "final_object_z": float(object_z_fn()),
        "lift_m": final_lift,
        "required_lift_m": float(required_lift_m),
        "clipping_count": clipping_count,
        "max_abs_raw_action": maximum_raw_action,
    }


def _driver_class(base_class):
    class BCPolicyGraspDriver(base_class):
        def __init__(self, *, policy, observation_keys, max_steps, stable_steps):
            super().__init__()
            self.policy = policy
            self.observation_keys = tuple(observation_keys)
            self.max_steps = int(max_steps)
            self.stable_steps = int(stable_steps)
            self.last_result: dict[str, Any] | None = None

        def _ensure_rollout(self, backend, object_name, config) -> bool:
            if self.last_result is not None:
                return bool(self.last_result.get("success", False))
            helpers = self._helpers()
            raw_env = backend.env
            robot = raw_env.robots[0]
            body_id = raw_env.obj_body_id[object_name]
            self._close_lift_reference = (
                object_name,
                float(raw_env.sim.data.body_xpos[body_id][2]),
            )

            def observations():
                raw = raw_env._get_observations(force_update=True)
                missing = [key for key in self.observation_keys if key not in raw]
                if missing:
                    raise RuntimeError(f"missing policy observations: {missing}")
                return {
                    key: np.asarray(raw[key]).copy()
                    for key in self.observation_keys
                }

            def step(action):
                result = raw_env.step(action)
                if isinstance(result, tuple) and len(result) > 3:
                    info = dict(result[3] or {})
                    info["has_judge_collision"] = bool(
                        info.get("has_judge_collision", False)
                        or getattr(raw_env, "has_judge_collision", False)
                    )
                    result = (*result[:3], info, *result[4:])
                return result

            self.last_result = execute_policy_window(
                policy=self.policy,
                observation_fn=observations,
                step_fn=step,
                contact_fn=lambda: helpers["grasp_status"](
                    raw_env,
                    robot,
                    object_name,
                ),
                object_z_fn=lambda: float(
                    helpers["object_center"](raw_env, object_name)[2]
                ),
                record_fn=lambda: self._record(backend, raw_env),
                max_steps=self.max_steps,
                required_lift_m=max(
                    1e-6,
                    float(config.lift_height) - float(config.lift_tolerance),
                ),
                stable_steps=self.stable_steps,
            )
            return bool(self.last_result.get("success", False))

        def raise_to_clearance(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

        def move_above_grasp_sites(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

        def adjust_wrist_for_reach(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

        def move_to_pregrasp(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

        def approach_grasp_sites(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

        def close_and_check_contacts(self, backend, object_name, config):
            self._ensure_rollout(backend, object_name, config)
            return dict((self.last_result or {}).get("contacts", {}))

        def polish_contacts(self, backend, object_name, config, contacts):
            del backend, object_name, config
            return dict(contacts)

        def lift_and_verify(self, backend, object_name, config):
            return self._ensure_rollout(backend, object_name, config)

    return BCPolicyGraspDriver


def run_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    import robomimic.utils.file_utils as FileUtils
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = output_dir / "trajectory.json"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "bc-rnn-summary.json"
    checkpoint = Path(args.checkpoint).resolve()
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        device=torch.device(args.device),
        ckpt_path=str(checkpoint),
        verbose=False,
    )
    observation_keys = list(
        checkpoint_dict["shape_metadata"]["all_obs_keys"]
    )

    holder: dict[str, Any] = {}
    original_load_scene = official_runner._load_scene

    def load_scene_with_policy(app_dir, task, seed):
        backend, scene_context, grid = original_load_scene(app_dir, task, seed)
        from robot_agent.skills import competition_grasp as grasp_module

        original_grasp = grasp_module.run_scripted_grasp
        Driver = _driver_class(grasp_module.OfficialScriptedGraspDriver)
        driver = Driver(
            policy=policy,
            observation_keys=observation_keys,
            max_steps=args.max_policy_steps,
            stable_steps=args.stable_steps,
        )

        def run_bc_grasp(
            grasp_backend,
            *,
            source,
            object_name,
            config=None,
            driver=None,
        ):
            del driver
            original_marker = grasp_backend._mark_trajectory_event

            def marked_event(name, *event_args, **details):
                if name == "grasp_start":
                    details["method"] = "bc_rnn_lowdim"
                    details["checkpoint_sha256"] = sha256_file(checkpoint)
                return original_marker(name, *event_args, **details)

            grasp_backend._mark_trajectory_event = marked_event
            try:
                result = dict(
                    original_grasp(
                        grasp_backend,
                        source=source,
                        object_name=object_name,
                        config=config,
                        driver=driver_instance,
                    )
                )
            finally:
                grasp_backend._mark_trajectory_event = original_marker
            result["bc_rnn"] = dict(driver_instance.last_result or {})
            return result

        driver_instance = driver
        grasp_module.run_scripted_grasp = run_bc_grasp
        holder.update(
            grasp_module=grasp_module,
            original_grasp=original_grasp,
            driver=driver,
        )
        return backend, scene_context, grid

    run_args = argparse.Namespace(
        candidate_root=Path(args.candidate_root),
        expected_official_commit=args.expected_official_commit,
        workspace_commit=args.workspace_commit,
        task_index=0,
        seed=args.seed,
        max_attempts=1,
        trajectory=trajectory_path,
        output=manifest_path,
        required_score=10,
        execution_mode="flow",
        perturbation_tier="small",
        perturbation_object=args.perturbation_object,
    )
    try:
        official_runner._load_scene = load_scene_with_policy
        manifest = official_runner.run_experiment(run_args)
    finally:
        official_runner._load_scene = original_load_scene
        grasp_module = holder.get("grasp_module")
        original_grasp = holder.get("original_grasp")
        if grasp_module is not None and original_grasp is not None:
            grasp_module.run_scripted_grasp = original_grasp
    official_runner.write_json_atomic(manifest_path, manifest)

    driver = holder.get("driver")
    policy_result = dict(getattr(driver, "last_result", None) or {})
    grasp_gate = bool(
        policy_result.get("success")
        and policy_result.get("contacts", {}).get("right")
        and policy_result.get("contacts", {}).get("left")
        and policy_result.get("lift_m", 0.0)
        >= policy_result.get("required_lift_m", float("inf"))
        and int(manifest.get("collision_frames") or 0) == 0
    )
    full_workflow_gate = bool(
        grasp_gate
        and int(manifest.get("official_score") or 0) == 10
        and int(manifest.get("successful_grasp_events") or 0) >= 1
    )
    summary = {
        "status": "complete" if full_workflow_gate else "failed",
        "seed": args.seed,
        "perturbation_tier": "small",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_dict.get("variable_state", {}).get(
            "epoch"
        ),
        "observation_keys": observation_keys,
        "policy_result": policy_result,
        "grasp_gate_passed": grasp_gate,
        "full_workflow_gate_passed": full_workflow_gate,
        "manifest": str(manifest_path),
        "trajectory": str(trajectory_path),
        "official_score": manifest.get("official_score"),
        "collision_frames": manifest.get("collision_frames"),
        "successful_grasp_events": manifest.get("successful_grasp_events"),
        "error": manifest.get("error"),
    }
    official_runner.write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--workspace-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--perturbation-object")
    parser.add_argument("--max-policy-steps", type=int, default=400)
    parser.add_argument("--stable-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run_closed_loop(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if summary["full_workflow_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
