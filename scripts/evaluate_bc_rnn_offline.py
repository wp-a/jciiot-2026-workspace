#!/usr/bin/env python3
"""Evaluate a robomimic BC-RNN against fixed action baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_GROUPS = {
    "right_arm": (0, 6),
    "left_arm": (6, 12),
    "torso": (12, 13),
    "head": (13, 15),
    "base": (15, 18),
    "right_gripper": (18, 19),
    "left_gripper": (19, 20),
}
MODEL_SPLITS = ("train", "valid", "heldout")
BEST_VALIDATION_PATTERN = re.compile(
    r"_best_validation_([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\.pth$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    action_groups: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {predictions.shape} != {targets.shape}"
        )
    if predictions.ndim != 2 or predictions.shape[0] < 1:
        raise ValueError("predictions and targets must be nonempty rank-2 arrays")
    errors = predictions - targets
    groups = {}
    for name, (start, stop) in action_groups.items():
        if start < 0 or stop <= start or stop > predictions.shape[1]:
            raise ValueError(f"invalid action group {name}: {(start, stop)}")
        group_errors = errors[:, start:stop]
        groups[name] = {
            "mse": float(np.mean(np.square(group_errors))),
            "mae": float(np.mean(np.abs(group_errors))),
        }
    return {
        "samples": int(predictions.shape[0]),
        "action_dim": int(predictions.shape[1]),
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "out_of_range_fraction": float(np.mean(np.abs(predictions) > 1.0)),
        "groups": groups,
    }


def select_best_validation_checkpoint(paths: Iterable[str | Path]) -> Path:
    candidates = []
    for raw_path in paths:
        path = Path(raw_path)
        match = BEST_VALIDATION_PATTERN.search(path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise ValueError("no best-validation checkpoint was found")
    return min(candidates, key=lambda item: (item[0], item[1].name))[1]


def _decode_names(raw_values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_values
    ]


def _split_names(handle, split: str) -> list[str]:
    key = f"mask/{split}"
    if key not in handle:
        raise ValueError(f"dataset has no {key}")
    return _decode_names(handle[key][:])


def _demo_targets(handle, demo_names: Iterable[str]) -> np.ndarray:
    arrays = [np.asarray(handle[f"data/{name}/actions"][:]) for name in demo_names]
    if not arrays:
        raise ValueError("split contains no demonstrations")
    return np.concatenate(arrays, axis=0)


def _policy_predictions(policy, handle, demo_names: Iterable[str], obs_keys) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = []
    per_demo = []
    for demo_name in demo_names:
        demo = handle[f"data/{demo_name}"]
        targets = np.asarray(demo["actions"][:])
        policy.start_episode()
        demo_predictions = []
        for index in range(targets.shape[0]):
            observation = {
                key: np.asarray(demo[f"obs/{key}"][index])
                for key in obs_keys
            }
            demo_predictions.append(np.asarray(policy(ob=observation), dtype=float))
        demo_predictions_array = np.stack(demo_predictions)
        predictions.append(demo_predictions_array)
        per_demo.append(
            {
                "demo": demo_name,
                "source_run": str(demo.attrs.get("source_run", "")),
                "split": str(demo.attrs.get("split", "")),
                "metrics": error_metrics(
                    demo_predictions_array,
                    targets,
                    action_groups=ACTION_GROUPS,
                ),
            }
        )
    return np.concatenate(predictions, axis=0), per_demo


def evaluate_checkpoint(
    dataset: str | Path,
    checkpoint: str | Path,
    *,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    import h5py
    import torch
    import robomimic.utils.file_utils as FileUtils

    dataset = Path(dataset).resolve()
    checkpoint = Path(checkpoint).resolve()
    device = torch.device(device_name)
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_path=str(checkpoint),
        verbose=False,
    )
    obs_keys = list(checkpoint_dict["shape_metadata"]["all_obs_keys"])

    with h5py.File(dataset, "r") as handle:
        split_names = {split: _split_names(handle, split) for split in MODEL_SPLITS}
        train_targets = _demo_targets(handle, split_names["train"])
        mean_action = np.mean(train_targets, axis=0)
        split_results = {}
        with torch.no_grad():
            for split in MODEL_SPLITS:
                targets = _demo_targets(handle, split_names[split])
                predictions, per_demo = _policy_predictions(
                    policy,
                    handle,
                    split_names[split],
                    obs_keys,
                )
                zero_predictions = np.zeros_like(targets)
                mean_predictions = np.broadcast_to(mean_action, targets.shape)
                split_results[split] = {
                    "demo_count": len(split_names[split]),
                    "demo_names": split_names[split],
                    "policy": error_metrics(
                        predictions,
                        targets,
                        action_groups=ACTION_GROUPS,
                    ),
                    "zero_baseline": error_metrics(
                        zero_predictions,
                        targets,
                        action_groups=ACTION_GROUPS,
                    ),
                    "mean_baseline": error_metrics(
                        mean_predictions,
                        targets,
                        action_groups=ACTION_GROUPS,
                    ),
                    "per_demo": per_demo,
                }

    heldout = split_results["heldout"]
    better_baseline_mse = min(
        heldout["zero_baseline"]["mse"],
        heldout["mean_baseline"]["mse"],
    )
    policy_mse = heldout["policy"]["mse"]
    relative_improvement = (
        1.0 - policy_mse / better_baseline_mse
        if better_baseline_mse > 0.0
        else float("-inf")
    )
    variable_state = checkpoint_dict.get("variable_state", {})
    return {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": variable_state.get("epoch"),
        "checkpoint_best_valid_loss": variable_state.get("best_valid_loss"),
        "device": str(device),
        "observation_keys": obs_keys,
        "action_groups": {
            key: list(value) for key, value in ACTION_GROUPS.items()
        },
        "training_mean_action": mean_action.tolist(),
        "splits": split_results,
        "heldout_relative_improvement_over_better_constant": float(
            relative_improvement
        ),
        "offline_gate_threshold": 0.25,
        "offline_gate_passed": bool(relative_improvement >= 0.25),
    }


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--models-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = select_best_validation_checkpoint(
            args.models_dir.glob("*.pth")
        )
    results = evaluate_checkpoint(
        args.dataset,
        checkpoint,
        device_name=args.device,
    )
    write_json_atomic(args.output, results)
    print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if results["offline_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
