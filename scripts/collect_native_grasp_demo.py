#!/usr/bin/env python3
"""Collect one scored, competition-native Tiago grasp demonstration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    from scripts import run_official_experiment as official_runner
except ImportError:
    import run_official_experiment as official_runner


LOW_DIM_OBSERVATION_KEYS = (
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
)
IMAGE_KEY = "robot0_robotview_image"
EXPECTED_ACTION_DIM = 20


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


class GraspWindowRecorder:
    """Record pre-action state and observations between grasp events."""

    def __init__(
        self,
        env,
        *,
        observation_keys: Iterable[str],
        image_size: int = 128,
        camera: str = "robot0_robotview",
    ) -> None:
        self.env = env
        self.observation_keys = tuple(str(key) for key in observation_keys)
        self.image_size = int(image_size)
        self.camera = str(camera)
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if not self.observation_keys:
            raise ValueError("observation_keys must not be empty")
        self.active = False
        self.start_event: dict[str, Any] | None = None
        self.end_event: dict[str, Any] | None = None
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.observations: dict[str, list[np.ndarray]] = {
            key: [] for key in (*self.observation_keys, IMAGE_KEY)
        }

    def handle_event(self, name: str, **details: Any) -> None:
        name = str(name)
        event = {"name": name, **_json_safe(details)}
        if name == "grasp_start":
            if self.start_event is not None:
                raise RuntimeError("one grasp window is allowed per demonstration")
            self.start_event = event
            self.active = True
        elif name == "grasp_end" and self.active:
            self.end_event = event
            self.active = False

    def _capture_observations(self) -> dict[str, np.ndarray]:
        raw = self.env._get_observations(force_update=True)
        missing = [key for key in self.observation_keys if key not in raw]
        if missing:
            raise RuntimeError(
                f"missing observations {missing}; available={sorted(raw)}"
            )
        captured = {
            key: np.asarray(raw[key]).copy()
            for key in self.observation_keys
        }
        image = self.env.sim.render(
            camera_name=self.camera,
            width=self.image_size,
            height=self.image_size,
            depth=False,
        )
        captured[IMAGE_KEY] = np.asarray(image, dtype=np.uint8)[::-1].copy()
        return captured

    def wrap_step(self, step_fn: Callable) -> Callable:
        def recorded_step(action, *args, **kwargs):
            if self.active:
                action_array = np.asarray(action, dtype=float).copy()
                state = np.asarray(
                    self.env.sim.get_state().flatten(),
                    dtype=float,
                ).copy()
                observations = self._capture_observations()
                self.actions.append(action_array)
                self.states.append(state)
                for key, value in observations.items():
                    self.observations[key].append(value)
            return step_fn(action, *args, **kwargs)

        return recorded_step

    def as_demo(self) -> dict[str, Any]:
        if self.actions:
            actions = np.stack(self.actions)
            states = np.stack(self.states)
        else:
            actions = np.empty((0, 0), dtype=float)
            states = np.empty((0, 0), dtype=float)
        obs = {
            key: np.stack(values) if values else np.empty((0,), dtype=float)
            for key, values in self.observations.items()
        }
        return {
            "actions": actions,
            "states": states,
            "obs": obs,
            "start_event": dict(self.start_event or {}),
            "end_event": dict(self.end_event or {}),
        }


def validate_demo(
    demo: dict[str, Any],
    *,
    required_observation_keys: Iterable[str],
    minimum_samples: int = 250,
    action_dim: int = EXPECTED_ACTION_DIM,
) -> dict[str, Any]:
    actions = np.asarray(demo.get("actions"))
    if actions.ndim != 2 or actions.shape[1] != int(action_dim):
        raise ValueError(
            f"expected action dimension {action_dim}, got shape {actions.shape}"
        )
    samples = int(actions.shape[0])
    if samples < int(minimum_samples):
        raise ValueError(
            f"expected at least {minimum_samples} samples, got {samples}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain non-finite values")
    max_abs_action = float(np.max(np.abs(actions)))
    if max_abs_action > 1.000001:
        raise ValueError(f"actions exceed normalized range: {max_abs_action}")

    states = np.asarray(demo.get("states"))
    if states.ndim != 2 or states.shape[0] != samples:
        raise ValueError(
            f"state/action length mismatch: states={states.shape}, actions={actions.shape}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("states contain non-finite values")

    observations = demo.get("obs")
    if not isinstance(observations, dict):
        raise ValueError("obs must be a mapping")
    required = tuple(str(key) for key in required_observation_keys)
    missing = [key for key in (*required, IMAGE_KEY) if key not in observations]
    if missing:
        raise ValueError(f"missing required observations: {missing}")
    observation_shapes = {}
    for key, raw_value in observations.items():
        value = np.asarray(raw_value)
        if value.shape[0] != samples:
            raise ValueError(
                f"observation/action length mismatch for {key}: "
                f"obs={value.shape}, actions={actions.shape}"
            )
        if key != IMAGE_KEY and not np.all(np.isfinite(value)):
            raise ValueError(f"observation {key} contains non-finite values")
        observation_shapes[key] = list(value.shape[1:])

    images = np.asarray(observations[IMAGE_KEY])
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(
            f"images must have shape [T,H,W,3] and uint8 dtype, got "
            f"{images.shape} {images.dtype}"
        )
    image_std = float(np.std(images.astype(np.float32)))
    if not math.isfinite(image_std) or image_std <= 0.0:
        raise ValueError("images are blank or constant")
    if not np.any(np.abs(actions[:, :12]) > 1e-6):
        raise ValueError("demonstration contains no nonzero arm commands")

    end_event = demo.get("end_event") or {}
    grasp_success = bool(
        end_event.get("success")
        and end_event.get("lift_success", end_event.get("success"))
    )
    if not grasp_success:
        raise ValueError("grasp window did not end in verified grasp and lift")
    return {
        "samples": samples,
        "action_dim": int(actions.shape[1]),
        "state_dim": int(states.shape[1]),
        "max_abs_action": max_abs_action,
        "nonzero_action_fraction": float(np.mean(np.abs(actions) > 1e-6)),
        "image_std": image_std,
        "observation_shapes": observation_shapes,
        "grasp_success": True,
    }


def write_robomimic_hdf5(
    output: str | Path,
    demo: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> Path:
    import h5py

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.asarray(demo["states"], dtype=np.float64)
    samples = int(actions.shape[0])
    safe_metadata = _json_safe(metadata)

    with h5py.File(temporary, "w") as handle:
        data = handle.create_group("data")
        data.attrs["total"] = samples
        data.attrs["num_demos"] = 1
        data.attrs["collection_metadata"] = json.dumps(
            safe_metadata,
            ensure_ascii=True,
            sort_keys=True,
        )
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": safe_metadata.get("scene"),
                "type": 1,
                "env_kwargs": {},
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        episode = data.create_group("demo_0")
        episode.attrs["num_samples"] = samples
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("states", data=states)
        episode.create_dataset(
            "rewards",
            data=np.zeros(samples, dtype=np.float32),
        )
        dones = np.zeros(samples, dtype=np.uint8)
        dones[-1] = 1
        episode.create_dataset("dones", data=dones)
        obs_group = episode.create_group("obs")
        for key, raw_value in demo["obs"].items():
            value = np.asarray(raw_value)
            if key == IMAGE_KEY:
                obs_group.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                obs_group.create_dataset(key, data=value)
    temporary.replace(output)
    return output


def _install_recorder(backend, recorder: GraspWindowRecorder) -> Callable[[], None]:
    raw_env = backend.env
    original_step = raw_env.step
    original_marker = backend._mark_trajectory_event

    def marked_event(name: str, *args, **kwargs):
        recorder.handle_event(name, **kwargs)
        return original_marker(name, *args, **kwargs)

    raw_env.step = recorder.wrap_step(original_step)
    backend._mark_trajectory_event = marked_event

    def restore() -> None:
        raw_env.step = original_step
        backend._mark_trajectory_event = original_marker

    return restore


def _action_split_indexes(env) -> dict[str, list[int]]:
    split = env.robots[0].composite_controller._action_split_indexes
    return {
        str(key): [int(value[0]), int(value[1])]
        for key, value in split.items()
    }


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = output_dir / "trajectory.json"
    manifest_path = output_dir / "manifest.json"
    dataset_path = output_dir / "grasp-demo.hdf5"
    summary_path = output_dir / "collector-summary.json"
    holder: dict[str, Any] = {}
    original_load_scene = official_runner._load_scene

    def load_scene_with_recorder(app_dir, task, seed):
        backend, scene_context, grid = original_load_scene(app_dir, task, seed)
        object_name = official_runner.resolve_scored_object(
            task,
            requested_name=args.perturbation_object,
        )
        observation_keys = (
            *LOW_DIM_OBSERVATION_KEYS,
            f"{object_name}_pos",
            f"{object_name}_quat",
        )
        recorder = GraspWindowRecorder(
            backend.env,
            observation_keys=observation_keys,
            image_size=args.image_size,
            camera=args.camera,
        )
        holder.update(
            backend=backend,
            recorder=recorder,
            restore=_install_recorder(backend, recorder),
            observation_keys=observation_keys,
            action_split_indexes=_action_split_indexes(backend.env),
            object_name=object_name,
        )
        return backend, scene_context, grid

    run_args = argparse.Namespace(
        candidate_root=Path(args.candidate_root),
        expected_official_commit=args.expected_official_commit,
        workspace_commit=args.workspace_commit,
        task_index=args.task_index,
        seed=args.seed,
        max_attempts=1,
        trajectory=trajectory_path,
        output=manifest_path,
        required_score=None,
        execution_mode="flow",
        perturbation_tier=args.perturbation_tier,
        perturbation_object=args.perturbation_object,
    )
    try:
        official_runner._load_scene = load_scene_with_recorder
        manifest = official_runner.run_experiment(run_args)
    finally:
        official_runner._load_scene = original_load_scene
        restore = holder.get("restore")
        if callable(restore):
            try:
                restore()
            except Exception:
                pass
    official_runner.write_json_atomic(manifest_path, manifest)

    required_score = int(manifest.get("max_score", 0))
    accepted = official_runner.acceptance_met(
        manifest,
        required_score=required_score,
    )
    recorder = holder.get("recorder")
    demo = recorder.as_demo() if recorder is not None else None
    validation = None
    error = None
    if accepted and demo is not None:
        try:
            validation = validate_demo(
                demo,
                required_observation_keys=holder["observation_keys"],
                minimum_samples=args.minimum_samples,
            )
            metadata = {
                "official_commit": args.expected_official_commit,
                "workspace_commit": args.workspace_commit,
                "level": manifest.get("level"),
                "scene": manifest.get("scene"),
                "task_index": args.task_index,
                "seed": args.seed,
                "object_name": holder["object_name"],
                "perturbation": manifest.get("perturbation"),
                "perturbation_application": manifest.get(
                    "perturbation_application"
                ),
                "official_score": manifest.get("official_score"),
                "collision_frames": manifest.get("collision_frames"),
                "final_target_distance_m": manifest.get(
                    "final_target_distance_m"
                ),
                "start_event": demo["start_event"],
                "end_event": demo["end_event"],
                "action_split_indexes": holder["action_split_indexes"],
                "observation_keys": list(holder["observation_keys"]),
                "image_key": IMAGE_KEY,
                "image_size": args.image_size,
                "validation": validation,
            }
            write_robomimic_hdf5(dataset_path, demo, metadata=metadata)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    elif not accepted:
        error = "full scored workflow did not pass the acceptance gate"
    else:
        error = "recorder was not installed"

    summary = {
        "status": "complete" if dataset_path.is_file() else "failed",
        "dataset_written": dataset_path.is_file(),
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "trajectory": str(trajectory_path),
        "validation": validation,
        "error": error,
    }
    official_runner.write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--workspace-commit", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--perturbation-tier",
        choices=("nominal", "small", "medium", "stress"),
        default="nominal",
    )
    parser.add_argument("--perturbation-object")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera", default="robot0_robotview")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_collection(args)
    print(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2))
    return 0 if summary["dataset_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
