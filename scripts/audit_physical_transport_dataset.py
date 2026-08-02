#!/usr/bin/env python3
"""Audit attachment-free physical transport evidence for dataset admission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


INTEGRITY_COUNTERS = (
    "attachment_calls",
    "attachment_activations",
    "object_pose_writes",
    "robot_state_writes",
    "legacy_teleport_activations",
    "collision_frames",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _planar_translation(record: Mapping[str, Any]) -> float | None:
    probe = record.get("full_physical_probe")
    if not isinstance(probe, Mapping):
        return None
    start = probe.get("start_object_position")
    final = probe.get("final_object_position")
    if not isinstance(start, (list, tuple)) or len(start) < 2:
        return None
    if not isinstance(final, (list, tuple)) or len(final) < 2:
        return None
    start_xy = [_finite_number(value) for value in start[:2]]
    final_xy = [_finite_number(value) for value in final[:2]]
    if any(value is None for value in (*start_xy, *final_xy)):
        return None
    return math.hypot(
        float(final_xy[0]) - float(start_xy[0]),
        float(final_xy[1]) - float(start_xy[1]),
    )


def audit_record(
    record: Mapping[str, Any],
    *,
    minimum_object_translation_m: float = 0.50,
    minimum_object_lift_m: float = 0.13,
    maximum_object_gripper_drift_m: float = 0.05,
) -> dict[str, Any]:
    """Classify one record using measured object motion and integrity evidence."""
    failures: list[str] = []
    rejection_failures: list[str] = []

    required_fields = (
        "physical_grasp",
        "continuous_bilateral_contact",
        "dropped",
        "minimum_object_lift_m",
        "max_object_gripper_drift_m",
        *INTEGRITY_COUNTERS,
        "infrastructure_error",
    )
    for field in required_fields:
        if field not in record:
            failure = f"missing:{field}"
            failures.append(failure)
            rejection_failures.append(failure)

    object_translation_m = _planar_translation(record)
    if object_translation_m is None:
        failures.append("object_positions")
        rejection_failures.append("object_positions")

    for field in INTEGRITY_COUNTERS:
        if field not in record:
            continue
        count = _finite_number(record[field])
        if count is None or count != 0.0:
            failures.append(field)
            rejection_failures.append(field)

    if "infrastructure_error" in record and record.get("infrastructure_error"):
        failures.append("infrastructure_error")
        rejection_failures.append("infrastructure_error")

    if record.get("physical_grasp") is not True:
        failures.append("physical_grasp")
    if record.get("continuous_bilateral_contact") is not True:
        failures.append("continuous_bilateral_contact")
    if record.get("dropped") is not False:
        failures.append("dropped")

    lift_m = _finite_number(record.get("minimum_object_lift_m"))
    if lift_m is None or lift_m < float(minimum_object_lift_m):
        failures.append("minimum_object_lift_m")

    drift_m = _finite_number(record.get("max_object_gripper_drift_m"))
    if (
        drift_m is None
        or drift_m > float(maximum_object_gripper_drift_m)
    ):
        failures.append("max_object_gripper_drift_m")

    if (
        object_translation_m is None
        or object_translation_m < float(minimum_object_translation_m)
    ):
        failures.append("object_translation_m")

    failures = list(dict.fromkeys(failures))
    rejection_failures = list(dict.fromkeys(rejection_failures))
    eligible = not failures
    if eligible:
        classification = "transport_success"
    elif rejection_failures:
        classification = "rejected"
    else:
        classification = "recovery"
    return {
        "classification": classification,
        "eligible": eligible,
        "failures": failures,
        "metrics": {
            "object_translation_m": object_translation_m,
            "minimum_object_lift_m": lift_m,
            "max_object_gripper_drift_m": drift_m,
        },
        "thresholds": {
            "minimum_object_translation_m": float(
                minimum_object_translation_m
            ),
            "minimum_object_lift_m": float(minimum_object_lift_m),
            "maximum_object_gripper_drift_m": float(
                maximum_object_gripper_drift_m
            ),
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_files(paths: list[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    first_path_by_digest: dict[str, str] = {}
    for raw_path in sorted((Path(path).resolve() for path in paths), key=str):
        source_path = str(raw_path)
        source_digest = sha256_file(raw_path)
        try:
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            result = {
                "classification": "rejected",
                "eligible": False,
                "failures": ["invalid_json"],
                "metrics": {},
                "thresholds": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            if not isinstance(raw_record, Mapping):
                result = {
                    "classification": "rejected",
                    "eligible": False,
                    "failures": ["invalid_record"],
                    "metrics": {},
                    "thresholds": {},
                }
            else:
                result = audit_record(raw_record)

        duplicate_of = first_path_by_digest.get(source_digest)
        if duplicate_of is None:
            first_path_by_digest[source_digest] = source_path
        else:
            result["classification"] = "rejected"
            result["eligible"] = False
            result["failures"] = list(
                dict.fromkeys([*result["failures"], "duplicate_content"])
            )
            result["duplicate_of"] = duplicate_of
        records.append(
            {
                "source_path": source_path,
                "source_sha256": source_digest,
                **result,
            }
        )

    classification_counts = {
        name: sum(row["classification"] == name for row in records)
        for name in ("recovery", "rejected", "transport_success")
    }
    return {
        "record_count": len(records),
        "eligible_count": classification_counts["transport_success"],
        "classification_counts": classification_counts,
        "unique_content_count": len(first_path_by_digest),
        "records": records,
    }


def _expand_inputs(inputs: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in path.glob("*.json")
                if "trajectory" not in candidate.name.lower()
            )
        else:
            paths.append(path)
    return paths


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_tsv_atomic(path: str | Path, ledger: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    fieldnames = (
        "source_path",
        "classification",
        "eligible",
        "object_translation_m",
        "minimum_object_lift_m",
        "max_object_gripper_drift_m",
        "failures",
        "duplicate_of",
        "source_sha256",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in ledger["records"]:
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "source_path": record["source_path"],
                    "source_sha256": record["source_sha256"],
                    "classification": record["classification"],
                    "eligible": str(bool(record["eligible"])).lower(),
                    "object_translation_m": metrics.get(
                        "object_translation_m"
                    ),
                    "minimum_object_lift_m": metrics.get(
                        "minimum_object_lift_m"
                    ),
                    "max_object_gripper_drift_m": metrics.get(
                        "max_object_gripper_drift_m"
                    ),
                    "failures": ",".join(record.get("failures", [])),
                    "duplicate_of": record.get("duplicate_of", ""),
                }
            )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ledger = audit_files(_expand_inputs(args.inputs))
    write_json_atomic(args.json_output, ledger)
    write_tsv_atomic(args.tsv_output, ledger)
    print(
        json.dumps(
            {
                "record_count": ledger["record_count"],
                "eligible_count": ledger["eligible_count"],
                "classification_counts": ledger["classification_counts"],
                "json_output": str(args.json_output.resolve()),
                "tsv_output": str(args.tsv_output.resolve()),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
