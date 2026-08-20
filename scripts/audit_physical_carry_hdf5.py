#!/usr/bin/env python3
"""Audit JCIIOT HDF5 trajectories before physical-carry training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_EVENTS = (
    "grasp_start",
    "grasp_end",
    "transport_start",
    "transport_end",
    "place_end",
)
REQUIRED_INTEGRITY_FIELDS = (
    "attachment_calls",
    "object_pose_writes",
    "collision_frames",
    "min_lift_m",
    "true_object_translation_m",
    "continuous_bilateral_contact",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _json_value(value.item())
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata(group: h5py.Group, name: str) -> Any:
    if name in group.attrs:
        return _json_value(group.attrs[name])
    if name in group and isinstance(group[name], h5py.Dataset):
        return _json_value(group[name][()])
    return None


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _all_finite(dataset: h5py.Dataset, *, rows_per_chunk: int = 4096) -> bool:
    if dataset.dtype.kind not in "biufc":
        return False
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]))
    for start in range(0, int(dataset.shape[0]), rows_per_chunk):
        block = dataset[start : start + rows_per_chunk]
        if not bool(np.isfinite(block).all()):
            return False
    return True


def _events(demo: h5py.Group) -> list[str] | None:
    raw = _metadata(demo, "events")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    return [str(item) for item in raw]


def _ordered_events(events: list[str] | None) -> bool:
    if events is None:
        return False
    cursor = -1
    for expected in EXPECTED_EVENTS:
        try:
            cursor = events.index(expected, cursor + 1)
        except ValueError:
            return False
    return True


def _audit_demo(
    name: str,
    demo: h5py.Group,
    *,
    action_dim: int,
    state_dim: int,
    minimum_lift_m: float,
    minimum_translation_m: float,
) -> dict[str, Any]:
    failures: list[str] = []
    rejection_failures: list[str] = []

    actions = demo.get("actions")
    if not isinstance(actions, h5py.Dataset) or actions.ndim != 2:
        failures.append("actions_shape")
        rejection_failures.append("actions_shape")
        samples = None
        found_action_dim = None
    else:
        samples = int(actions.shape[0])
        found_action_dim = int(actions.shape[1])
        if samples <= 0:
            failures.append("empty_actions")
            rejection_failures.append("empty_actions")
        if found_action_dim != action_dim:
            failure = f"action_dim:{found_action_dim}"
            failures.append(failure)
            rejection_failures.append(failure)
        if not _all_finite(actions):
            failures.append("actions_nonfinite")
            rejection_failures.append("actions_nonfinite")

    obs = demo.get("obs")
    state = obs.get("state") if isinstance(obs, h5py.Group) else None
    if not isinstance(state, h5py.Dataset) or state.ndim != 2:
        failures.append("state_shape")
        rejection_failures.append("state_shape")
        found_state_dim = None
    else:
        found_state_dim = int(state.shape[1])
        if samples is not None and int(state.shape[0]) != samples:
            failures.append("state_length")
            rejection_failures.append("state_length")
        if found_state_dim != state_dim:
            failure = f"state_dim:{found_state_dim}"
            failures.append(failure)
            rejection_failures.append(failure)
        if not _all_finite(state):
            failures.append("state_nonfinite")
            rejection_failures.append("state_nonfinite")

    timestamps = demo.get("timestamps")
    if not isinstance(timestamps, h5py.Dataset) or timestamps.ndim != 1:
        failures.append("timestamps_shape")
        rejection_failures.append("timestamps_shape")
    else:
        values = np.asarray(timestamps[()])
        if samples is not None and len(values) != samples:
            failures.append("timestamps_length")
            rejection_failures.append("timestamps_length")
        if not bool(np.isfinite(values).all()):
            failures.append("timestamps_nonfinite")
            rejection_failures.append("timestamps_nonfinite")
        elif len(values) > 1 and not bool(np.all(np.diff(values) > 0.0)):
            failures.append("timestamps_not_strictly_increasing")
            rejection_failures.append("timestamps_not_strictly_increasing")

    if not _ordered_events(_events(demo)):
        failures.append("event_order")
        rejection_failures.append("event_order")

    seed = _metadata(demo, "seed")
    for field in ("seed", "object_name", "task_level"):
        if _metadata(demo, field) is None:
            failure = f"missing:{field}"
            failures.append(failure)
            rejection_failures.append(failure)

    integrity = demo.get("integrity")
    integrity_values: dict[str, Any] = {}
    if not isinstance(integrity, h5py.Group):
        failures.append("missing:integrity")
        rejection_failures.append("missing:integrity")
    else:
        for field in REQUIRED_INTEGRITY_FIELDS:
            value = _metadata(integrity, field)
            integrity_values[field] = value
            if value is None:
                failure = f"missing:{field}"
                failures.append(failure)
                rejection_failures.append(failure)

    for field in ("attachment_calls", "object_pose_writes", "collision_frames"):
        if field not in integrity_values or integrity_values[field] is None:
            continue
        value = _finite_scalar(integrity_values[field])
        if value is None or value != 0.0:
            failures.append(field)
            rejection_failures.append(field)

    lift = _finite_scalar(integrity_values.get("min_lift_m"))
    if lift is None or lift < minimum_lift_m:
        failures.append("min_lift_m")
    translation = _finite_scalar(
        integrity_values.get("true_object_translation_m")
    )
    if translation is None or translation < minimum_translation_m:
        failures.append("true_object_translation_m")
    if integrity_values.get("continuous_bilateral_contact") is not True:
        failures.append("continuous_bilateral_contact")

    failures = list(dict.fromkeys(failures))
    rejection_failures = list(dict.fromkeys(rejection_failures))
    if not failures:
        classification = "transport_success"
    elif rejection_failures:
        classification = "rejected"
    else:
        classification = "recovery"
    return {
        "name": name,
        "seed": _json_value(seed),
        "samples": samples,
        "action_dim": found_action_dim,
        "state_dim": found_state_dim,
        "classification": classification,
        "eligible": classification == "transport_success",
        "failures": failures,
        "integrity": integrity_values,
    }


def _decode_names(dataset: h5py.Dataset) -> list[str]:
    raw = np.asarray(dataset[()]).reshape(-1).tolist()
    return [str(_json_value(item)) for item in raw]


def _audit_splits(handle: h5py.File, demo_names: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    failures: list[str] = []
    split_names = {
        "train": ("train",),
        "validation": ("validation", "valid"),
        "heldout": ("heldout", "test"),
    }
    mask = handle.get("mask")
    splits: dict[str, list[str]] = {}
    assigned: dict[str, str] = {}
    for canonical, aliases in split_names.items():
        dataset = None
        if isinstance(mask, h5py.Group):
            for alias in aliases:
                if alias in mask and isinstance(mask[alias], h5py.Dataset):
                    dataset = mask[alias]
                    break
        if dataset is None:
            failures.append(f"missing_split:{canonical}")
            splits[canonical] = []
            continue
        names = _decode_names(dataset)
        splits[canonical] = names
        for name in names:
            if name not in demo_names:
                failures.append(f"unknown_demo:{name}")
            if name in assigned:
                failures.append(f"split_leakage:{name}")
            else:
                assigned[name] = canonical
    for name in sorted(demo_names - set(assigned)):
        failures.append(f"unassigned_demo:{name}")
    return splits, list(dict.fromkeys(failures))


def audit_hdf5(
    path: str | Path,
    *,
    expected_action_dim: int = 20,
    expected_state_dim: int = 87,
    minimum_lift_m: float = 0.13,
    minimum_translation_m: float = 0.50,
) -> dict[str, Any]:
    """Return a fail-closed manifest for one task-native physical dataset."""
    path = Path(path)
    failures: list[str] = []
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError("HDF5 file has no data group")
        demo_names = sorted(
            name
            for name in data.keys()
            if name.startswith("demo_") and isinstance(data[name], h5py.Group)
        )
        if not demo_names:
            failures.append("no_demos")
        demos = [
            _audit_demo(
                name,
                data[name],
                action_dim=expected_action_dim,
                state_dim=expected_state_dim,
                minimum_lift_m=minimum_lift_m,
                minimum_translation_m=minimum_translation_m,
            )
            for name in demo_names
        ]

        env_args = _metadata(data, "env_args")
        if isinstance(env_args, str):
            try:
                env_args = json.loads(env_args)
            except json.JSONDecodeError:
                env_args = None
        env_name = env_args.get("env_name") if isinstance(env_args, dict) else None
        env_kwargs = env_args.get("env_kwargs", {}) if isinstance(env_args, dict) else {}
        robots = env_kwargs.get("robots", []) if isinstance(env_kwargs, dict) else []
        if not str(env_name).startswith("FactorySorting"):
            failures.append("environment_not_factory_sorting")
        if "Tiago" not in robots:
            failures.append("robot_not_tiago")

        seen_seeds: set[Any] = set()
        for demo in demos:
            seed = demo["seed"]
            try:
                duplicate = seed in seen_seeds
            except TypeError:
                failures.append(f"invalid_seed:{demo['name']}")
                continue
            if duplicate:
                failures.append(f"duplicate_seed:{seed}")
            seen_seeds.add(seed)

        splits, split_failures = _audit_splits(handle, set(demo_names))
        failures.extend(split_failures)

    found_action_dims = {demo["action_dim"] for demo in demos if demo["action_dim"] is not None}
    found_state_dims = {demo["state_dim"] for demo in demos if demo["state_dim"] is not None}
    counts = {
        classification: sum(
            demo["classification"] == classification for demo in demos
        )
        for classification in ("transport_success", "recovery", "rejected")
    }
    failures = list(dict.fromkeys(failures))
    eligible_demo_count = counts["transport_success"]
    eligible = not failures and eligible_demo_count == len(demos) and bool(demos)
    return {
        "path": str(path.resolve()),
        "file_size": int(path.stat().st_size),
        "eligible": eligible,
        "failures": failures,
        "demo_count": len(demos),
        "eligible_demo_count": eligible_demo_count,
        "classification_counts": counts,
        "action_dim": next(iter(found_action_dims)) if len(found_action_dims) == 1 else None,
        "state_dim": next(iter(found_state_dims)) if len(found_state_dims) == 1 else None,
        "splits": splits,
        "thresholds": {
            "action_dim": expected_action_dim,
            "state_dim": expected_state_dim,
            "minimum_lift_m": minimum_lift_m,
            "minimum_translation_m": minimum_translation_m,
        },
        "demos": demos,
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
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = audit_hdf5(args.dataset)
    if args.output is not None:
        write_json_atomic(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if manifest["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
