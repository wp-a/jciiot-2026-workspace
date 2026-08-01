#!/usr/bin/env python3
"""Merge registered competition-native grasp demos with fixed split masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_SPLITS = ("train", "valid", "heldout")
EXCLUDED_SPLIT = "excluded_duplicate"
EXPECTED_ACTION_DIM = 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_demo(demo, *, expected_action_dim: int) -> dict[str, Any]:
    if "actions" not in demo or len(demo["actions"].shape) != 2:
        raise ValueError("source demonstration must contain rank-2 actions")
    samples, action_dim = (int(value) for value in demo["actions"].shape)
    if action_dim != int(expected_action_dim):
        raise ValueError(
            f"expected action dimension {expected_action_dim}, got {action_dim}"
        )
    if samples < 1:
        raise ValueError("source demonstration contains no samples")
    if "states" not in demo or int(demo["states"].shape[0]) != samples:
        raise ValueError("source state/action length mismatch")
    if "obs" not in demo:
        raise ValueError("source demonstration contains no obs group")
    observation_keys = sorted(str(key) for key in demo["obs"].keys())
    if not observation_keys:
        raise ValueError("source demonstration contains no observations")
    for key in observation_keys:
        if int(demo["obs"][key].shape[0]) != samples:
            raise ValueError(f"source observation/action length mismatch for {key}")
    return {
        "samples": samples,
        "action_dim": action_dim,
        "state_dim": int(demo["states"].shape[1]),
        "observation_keys": observation_keys,
    }


def _registered_runs(split_config: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    raw_splits = split_config.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("split config must contain a splits mapping")
    splits = {
        name: [str(run) for run in raw_splits.get(name, [])]
        for name in MODEL_SPLITS
    }
    excluded = [str(run) for run in raw_splits.get(EXCLUDED_SPLIT, [])]
    all_runs = [run for name in MODEL_SPLITS for run in splits[name]] + excluded
    duplicates = sorted({run for run in all_runs if all_runs.count(run) > 1})
    if duplicates:
        raise ValueError(f"runs appear in multiple splits: {duplicates}")
    if not splits["train"]:
        raise ValueError("train split must not be empty")
    unknown = sorted(set(raw_splits) - {*MODEL_SPLITS, EXCLUDED_SPLIT})
    if unknown:
        raise ValueError(f"unknown split names: {unknown}")
    return splits, excluded


def merge_registered_demos(
    dataset_root: str | Path,
    split_config: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    import h5py

    dataset_root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    splits, excluded = _registered_runs(split_config)
    source_records = []
    split_demo_names: dict[str, list[str]] = {name: [] for name in MODEL_SPLITS}
    expected_observation_keys: list[str] | None = None
    expected_state_dim: int | None = None
    env_args: Any = None
    total_samples = 0
    demo_index = 0

    try:
        with h5py.File(temporary, "w") as destination:
            destination_data = destination.create_group("data")
            for split_name in MODEL_SPLITS:
                for run_name in splits[split_name]:
                    source_path = dataset_root / run_name / "grasp-demo.hdf5"
                    if not source_path.is_file():
                        raise FileNotFoundError(source_path)
                    source_digest = sha256_file(source_path)
                    with h5py.File(source_path, "r") as source:
                        if "data/demo_0" not in source:
                            raise ValueError(f"{source_path} has no data/demo_0")
                        source_data = source["data"]
                        source_demo = source_data["demo_0"]
                        details = _validate_source_demo(
                            source_demo,
                            expected_action_dim=EXPECTED_ACTION_DIM,
                        )
                        if expected_observation_keys is None:
                            expected_observation_keys = details["observation_keys"]
                            expected_state_dim = details["state_dim"]
                            env_args = source_data.attrs.get("env_args")
                        elif details["observation_keys"] != expected_observation_keys:
                            raise ValueError(
                                f"observation keys differ for {run_name}: "
                                f"{details['observation_keys']}"
                            )
                        elif details["state_dim"] != expected_state_dim:
                            raise ValueError(
                                f"state dimension differs for {run_name}: "
                                f"{details['state_dim']}"
                            )

                        demo_name = f"demo_{demo_index}"
                        source.copy(source_demo, destination_data, name=demo_name)
                        copied = destination_data[demo_name]
                        copied.attrs["source_run"] = run_name
                        copied.attrs["source_path"] = str(source_path.resolve())
                        copied.attrs["source_sha256"] = source_digest
                        copied.attrs["split"] = split_name
                        copied.attrs["num_samples"] = details["samples"]
                        split_demo_names[split_name].append(demo_name)
                        source_records.append(
                            {
                                "demo": demo_name,
                                "run": run_name,
                                "split": split_name,
                                "path": str(source_path.resolve()),
                                "sha256": source_digest,
                                **details,
                            }
                        )
                        total_samples += details["samples"]
                        demo_index += 1

            destination_data.attrs["total"] = total_samples
            destination_data.attrs["num_demos"] = demo_index
            if env_args is not None:
                destination_data.attrs["env_args"] = env_args
            merge_metadata = {
                "dataset_root": str(dataset_root),
                "splits": splits,
                "excluded_runs": excluded,
                "source_records": source_records,
            }
            destination_data.attrs["merge_metadata"] = json.dumps(
                merge_metadata,
                ensure_ascii=True,
                sort_keys=True,
            )
            masks = destination.create_group("mask")
            for split_name in MODEL_SPLITS:
                masks.create_dataset(
                    split_name,
                    data=np.asarray(split_demo_names[split_name], dtype="S"),
                )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "demo_count": demo_index,
        "total_samples": total_samples,
        "action_dim": EXPECTED_ACTION_DIM,
        "state_dim": expected_state_dim,
        "observation_keys": expected_observation_keys,
        "split_demo_names": split_demo_names,
        "split_counts": {
            name: len(split_demo_names[name]) for name in MODEL_SPLITS
        },
        "excluded_runs": excluded,
        "sources": source_records,
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
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    split_config = json.loads(args.split.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root or split_config.get("dataset_root")
    if dataset_root is None:
        parser.error("--dataset-root or dataset_root in --split is required")
    summary = merge_registered_demos(dataset_root, split_config, args.output)
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
