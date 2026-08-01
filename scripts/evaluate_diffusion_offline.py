#!/usr/bin/env python3
"""Evaluate stochastic robomimic Diffusion Policy checkpoints reproducibly."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.evaluate_bc_rnn_offline import (
        ACTION_GROUPS,
        _demo_targets,
        _policy_predictions,
        _split_names,
        error_metrics,
        select_best_validation_checkpoint,
        sha256_file,
        write_json_atomic,
    )
except ImportError:
    from evaluate_bc_rnn_offline import (
        ACTION_GROUPS,
        _demo_targets,
        _policy_predictions,
        _split_names,
        error_metrics,
        select_best_validation_checkpoint,
        sha256_file,
        write_json_atomic,
    )


DEFAULT_SAMPLING_SEEDS = (20260820, 20260821, 20260822)
EXPECTED_ACTION_DIM = 20
PERIODIC_CHECKPOINT_PATTERN = re.compile(r"^model_epoch_(\d+)\.pth$")
VALIDATION_LOSS_PATTERN = re.compile(
    r"Validation Epoch\s+(\d+)\s*\r?\n\{.*?"
    r'"Loss":\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
    re.DOTALL,
)


def parse_validation_losses(log_text: str) -> dict[int, float]:
    losses = {
        int(epoch): float(loss)
        for epoch, loss in VALIDATION_LOSS_PATTERN.findall(log_text)
    }
    if not losses:
        raise ValueError("no validation losses were found in the training log")
    return losses


def select_periodic_checkpoint_by_validation(
    paths: Iterable[str | Path],
    validation_losses: dict[int, float],
) -> Path:
    candidates = []
    for raw_path in paths:
        path = Path(raw_path)
        match = PERIODIC_CHECKPOINT_PATTERN.match(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch in validation_losses:
                candidates.append((float(validation_losses[epoch]), epoch, path))
    if not candidates:
        raise ValueError("no periodic checkpoint has a logged validation loss")
    return min(candidates, key=lambda item: (item[0], item[1], item[2].name))[2]


def set_sampling_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def summarize_trials(
    trials: Iterable[dict[str, Any]],
    *,
    constant_baseline_mse: float,
    expected_action_dim: int,
) -> dict[str, Any]:
    trials = list(trials)
    if not trials or len(trials) % 2 == 0:
        raise ValueError("an odd, nonzero number of sampling trials is required")
    if not math.isfinite(constant_baseline_mse) or constant_baseline_mse <= 0.0:
        raise ValueError("constant baseline MSE must be finite and positive")

    ordered = sorted(trials, key=lambda trial: float(trial["metrics"]["mse"]))
    median_trial = ordered[len(ordered) // 2]
    median_mse = float(median_trial["metrics"]["mse"])
    all_finite = bool(
        all(
            bool(trial.get("all_finite"))
            and math.isfinite(float(trial["metrics"]["mse"]))
            for trial in trials
        )
    )
    all_action_dims_valid = bool(
        all(int(trial.get("action_dim", -1)) == expected_action_dim for trial in trials)
    )
    relative_improvement = 1.0 - median_mse / constant_baseline_mse

    group_names = tuple(median_trial["metrics"].get("groups", {}))
    median_group_metrics = {
        group: {
            metric: float(
                np.median(
                    [
                        trial["metrics"]["groups"][group][metric]
                        for trial in trials
                    ]
                )
            )
            for metric in ("mse", "mae")
        }
        for group in group_names
    }
    return {
        "trial_count": len(trials),
        "sampling_seeds": [int(trial["sampling_seed"]) for trial in trials],
        "median_trial_seed": int(median_trial["sampling_seed"]),
        "median_heldout_mse": median_mse,
        "median_group_metrics": median_group_metrics,
        "constant_baseline_mse": float(constant_baseline_mse),
        "relative_improvement": float(relative_improvement),
        "all_trials_finite": all_finite,
        "all_action_dims_valid": all_action_dims_valid,
        "offline_gate_passed": bool(
            all_finite
            and all_action_dims_valid
            and median_mse < constant_baseline_mse
        ),
    }


def evaluate_checkpoint(
    dataset: str | Path,
    checkpoint: str | Path,
    *,
    sampling_seeds: Iterable[int] = DEFAULT_SAMPLING_SEEDS,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    import h5py
    import robomimic.utils.file_utils as FileUtils
    import torch

    dataset = Path(dataset).resolve()
    checkpoint = Path(checkpoint).resolve()
    sampling_seeds = tuple(int(seed) for seed in sampling_seeds)
    if len(set(sampling_seeds)) != len(sampling_seeds):
        raise ValueError("sampling seeds must be unique")
    if not sampling_seeds or len(sampling_seeds) % 2 == 0:
        raise ValueError("sampling seeds must contain an odd, nonzero count")

    device = torch.device(device_name)
    set_sampling_seed(sampling_seeds[0], torch)
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_path=str(checkpoint),
        verbose=False,
    )
    observation_keys = list(checkpoint_dict["shape_metadata"]["all_obs_keys"])

    with h5py.File(dataset, "r") as handle:
        train_names = _split_names(handle, "train")
        heldout_names = _split_names(handle, "heldout")
        train_targets = _demo_targets(handle, train_names)
        heldout_targets = _demo_targets(handle, heldout_names)
        mean_action = np.mean(train_targets, axis=0)
        zero_baseline = error_metrics(
            np.zeros_like(heldout_targets),
            heldout_targets,
            action_groups=ACTION_GROUPS,
        )
        mean_baseline = error_metrics(
            np.broadcast_to(mean_action, heldout_targets.shape),
            heldout_targets,
            action_groups=ACTION_GROUPS,
        )

        trials = []
        with torch.no_grad():
            for sampling_seed in sampling_seeds:
                set_sampling_seed(sampling_seed, torch)
                predictions, per_demo = _policy_predictions(
                    policy,
                    handle,
                    heldout_names,
                    observation_keys,
                )
                trial_metrics = error_metrics(
                    predictions,
                    heldout_targets,
                    action_groups=ACTION_GROUPS,
                )
                trials.append(
                    {
                        "sampling_seed": sampling_seed,
                        "all_finite": bool(np.all(np.isfinite(predictions))),
                        "action_dim": int(predictions.shape[1]),
                        "clipped_step_count": int(
                            np.sum(np.any(np.abs(predictions) > 1.0, axis=1))
                        ),
                        "metrics": trial_metrics,
                        "per_demo": per_demo,
                    }
                )

    better_constant_mse = min(zero_baseline["mse"], mean_baseline["mse"])
    summary = summarize_trials(
        trials,
        constant_baseline_mse=better_constant_mse,
        expected_action_dim=EXPECTED_ACTION_DIM,
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
        "observation_keys": observation_keys,
        "action_groups": {key: list(value) for key, value in ACTION_GROUPS.items()},
        "heldout_demo_names": heldout_names,
        "training_mean_action": mean_action.tolist(),
        "zero_baseline": zero_baseline,
        "mean_baseline": mean_baseline,
        "trials": trials,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--models-dir", type=Path)
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=DEFAULT_SAMPLING_SEEDS)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint
    selection: dict[str, Any]
    if checkpoint is None:
        model_paths = list(args.models_dir.glob("*.pth"))
        try:
            checkpoint = select_best_validation_checkpoint(model_paths)
            selection = {"method": "best_validation_checkpoint_filename"}
        except ValueError:
            if args.training_log is None:
                raise ValueError(
                    "--training-log is required for periodic checkpoints"
                )
            validation_losses = parse_validation_losses(
                args.training_log.read_text(encoding="utf-8", errors="replace")
            )
            checkpoint = select_periodic_checkpoint_by_validation(
                model_paths,
                validation_losses,
            )
            checkpoint_epoch = int(
                PERIODIC_CHECKPOINT_PATTERN.match(checkpoint.name).group(1)
            )
            selection = {
                "method": "minimum_logged_validation_loss_at_saved_epoch",
                "training_log": str(args.training_log.resolve()),
                "logged_validation_loss": validation_losses[checkpoint_epoch],
            }
    else:
        selection = {"method": "explicit_checkpoint"}
    results = evaluate_checkpoint(
        args.dataset,
        checkpoint,
        sampling_seeds=args.sampling_seeds,
        device_name=args.device,
    )
    results["checkpoint_selection"] = selection
    write_json_atomic(args.output, results)
    print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if results["summary"]["offline_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
