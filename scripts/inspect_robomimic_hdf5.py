#!/usr/bin/env python3
"""Inspect robomimic HDF5 metadata without loading trajectory arrays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
JCIIOT_OBSERVATION_KEYS = {
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
    "robot0_robotview_image",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decoded_attribute(value: Any) -> Any:
    safe = _json_safe(value)
    if not isinstance(safe, str):
        return safe
    try:
        return json.loads(safe)
    except json.JSONDecodeError:
        return safe


def _attributes(group: h5py.Group) -> dict[str, Any]:
    return {
        str(key): _decoded_attribute(group.attrs[key])
        for key in sorted(group.attrs.keys())
    }


def _dataset_metadata(dataset: h5py.Dataset) -> dict[str, Any]:
    return {
        "shape": [int(size) for size in dataset.shape],
        "dtype": str(dataset.dtype),
        "compression": dataset.compression,
    }


def is_git_lfs_pointer(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 4096:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_HEADER)) == LFS_HEADER


def _lfs_pointer_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="ascii")
    size_match = re.search(r"^size\s+(\d+)\s*$", text, flags=re.MULTILINE)
    oid_match = re.search(r"^oid\s+sha256:([0-9a-f]{64})\s*$", text, flags=re.MULTILINE)
    return {
        "path": str(path.resolve()),
        "file_size": int(path.stat().st_size),
        "materialized": False,
        "lfs_oid_sha256": oid_match.group(1) if oid_match else None,
        "lfs_size": int(size_match.group(1)) if size_match else None,
        "compatibility": {
            "classification": "lfs-pointer",
            "reasons": ["file is a Git LFS pointer, not an HDF5 payload"],
        },
    }


def classify_compatibility(summary: dict[str, Any]) -> dict[str, Any]:
    env_args = summary.get("env_args")
    env_name = env_args.get("env_name", "") if isinstance(env_args, dict) else ""
    action_dim = summary.get("action_dim")
    observation_keys = set(summary.get("observation_keys", []))
    missing = sorted(JCIIOT_OBSERVATION_KEYS - observation_keys)
    reasons = []
    if not str(env_name).startswith("FactorySorting"):
        reasons.append(f"environment is not FactorySorting: {env_name or 'unknown'}")
    if action_dim != 20:
        reasons.append(f"action dimension is {action_dim}, expected 20")
    if missing:
        reasons.append("missing JCIIOT observation keys: " + ", ".join(missing))

    if not reasons:
        classification = "task-compatible"
    elif str(env_name).startswith("FactorySorting") or action_dim == 20:
        classification = "partially-reusable"
    else:
        classification = "format-only"
    return {
        "classification": classification,
        "reasons": reasons,
        "missing_observation_keys": missing,
    }


def inspect_hdf5(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if is_git_lfs_pointer(path):
        return _lfs_pointer_summary(path)

    with h5py.File(path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError("HDF5 file has no robomimic data group")
        data = handle["data"]
        demo_names = sorted(
            name
            for name in data.keys()
            if name.startswith("demo_") and isinstance(data[name], h5py.Group)
        )
        demos = []
        action_dims = set()
        observation_keys = set()
        total_samples = 0
        for demo_name in demo_names:
            demo = data[demo_name]
            if "actions" not in demo or not isinstance(demo["actions"], h5py.Dataset):
                raise ValueError(f"{demo_name} has no actions dataset")
            actions = demo["actions"]
            if len(actions.shape) != 2:
                raise ValueError(f"{demo_name} actions must be rank 2")
            samples, action_dim = (int(actions.shape[0]), int(actions.shape[1]))
            total_samples += samples
            action_dims.add(action_dim)
            obs_metadata = {}
            if "obs" in demo and isinstance(demo["obs"], h5py.Group):
                for key in sorted(demo["obs"].keys()):
                    dataset = demo["obs"][key]
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    observation_keys.add(str(key))
                    obs_metadata[str(key)] = _dataset_metadata(dataset)
            demos.append(
                {
                    "name": demo_name,
                    "samples": samples,
                    "attributes": _attributes(demo),
                    "actions": _dataset_metadata(actions),
                    "states": (
                        _dataset_metadata(demo["states"])
                        if "states" in demo and isinstance(demo["states"], h5py.Dataset)
                        else None
                    ),
                    "observations": obs_metadata,
                }
            )

        if len(action_dims) > 1:
            raise ValueError(
                "inconsistent action dimensions: "
                + ", ".join(str(value) for value in sorted(action_dims))
            )

        root_attributes = _attributes(handle)
        data_attributes = _attributes(data)
        env_args = data_attributes.get("env_args", root_attributes.get("env_args"))
        env_info = data_attributes.get("env_info", root_attributes.get("env_info"))
        masks = {}
        if "mask" in handle and isinstance(handle["mask"], h5py.Group):
            for key in sorted(handle["mask"].keys()):
                dataset = handle["mask"][key]
                if isinstance(dataset, h5py.Dataset):
                    masks[str(key)] = _dataset_metadata(dataset)

        summary = {
            "path": str(path.resolve()),
            "file_size": int(path.stat().st_size),
            "materialized": True,
            "root_attributes": root_attributes,
            "data_attributes": data_attributes,
            "env_args": env_args,
            "env_info": env_info,
            "demo_count": len(demos),
            "total_samples": total_samples,
            "action_dim": next(iter(action_dims)) if action_dims else None,
            "observation_keys": sorted(observation_keys),
            "demos": demos,
            "masks": masks,
        }
    summary["compatibility"] = classify_compatibility(summary)
    return summary


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = inspect_hdf5(args.dataset)
    if args.output is not None:
        write_json_atomic(args.output, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if summary.get("materialized") else 2


if __name__ == "__main__":
    raise SystemExit(main())
