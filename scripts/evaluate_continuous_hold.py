#!/usr/bin/env python3
"""Reject scored trajectories that visually detach a held object from the robot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _attachment_event(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    events = trajectory.get("events", [])
    if not isinstance(events, list):
        return None
    for event in events:
        if (
            isinstance(event, dict)
            and event.get("name") == "transport_attachment_enabled"
            and event.get("object_name")
        ):
            return event
    return None


def _relative_object_xy(frame: dict[str, Any], object_name: str):
    base_pose = frame.get("base_pose", {})
    base_position = base_pose.get("position", [])
    orientation = base_pose.get("orientation_xyzw", [])
    object_position = frame.get("object_positions", {}).get(object_name, [])
    if (
        len(base_position) < 2
        or len(orientation) < 4
        or len(object_position) < 2
    ):
        return None

    base_x = float(base_position[0])
    base_y = float(base_position[1])
    world_dx = float(object_position[0]) - base_x
    world_dy = float(object_position[1]) - base_y
    yaw = 2.0 * math.atan2(float(orientation[2]), float(orientation[3]))
    cosine = math.cos(-yaw)
    sine = math.sin(-yaw)
    relative_x = cosine * world_dx - sine * world_dy
    relative_y = sine * world_dx + cosine * world_dy
    return relative_x, relative_y, math.hypot(world_dx, world_dy)


def evaluate_continuous_hold(
    trajectory: dict[str, Any],
    *,
    min_base_distance_m: float = 0.80,
    max_relative_xy_range_m: float = 0.020,
) -> dict[str, Any]:
    """Measure whether every recorded held frame preserves one rigid offset."""
    event = _attachment_event(trajectory)
    if event is None:
        return {
            "passed": False,
            "error": "transport attachment event is missing",
            "held_frames": 0,
        }

    object_name = str(event["object_name"])
    start_frame = max(0, int(event.get("frame", 0)))
    frames = trajectory.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    relative_positions = []
    invalid_frames = 0
    for frame in frames[start_frame:]:
        if not isinstance(frame, dict) or frame.get("held_object") != object_name:
            continue
        observation = _relative_object_xy(frame, object_name)
        if observation is None or not all(math.isfinite(value) for value in observation):
            invalid_frames += 1
            continue
        relative_positions.append(observation)

    if not relative_positions:
        return {
            "passed": False,
            "error": "no valid held transport frames",
            "object_name": object_name,
            "attachment_frame": start_frame,
            "held_frames": 0,
            "invalid_frames": invalid_frames,
        }

    relative_x = [item[0] for item in relative_positions]
    relative_y = [item[1] for item in relative_positions]
    distances = [item[2] for item in relative_positions]
    xy_range = [
        max(relative_x) - min(relative_x),
        max(relative_y) - min(relative_y),
    ]
    frames_below = sum(
        distance < float(min_base_distance_m) for distance in distances
    )
    max_range = max(xy_range)
    passed = bool(
        invalid_frames == 0
        and frames_below == 0
        and max_range <= float(max_relative_xy_range_m)
    )
    return {
        "passed": passed,
        "object_name": object_name,
        "attachment_frame": start_frame,
        "held_frames": len(relative_positions),
        "invalid_frames": invalid_frames,
        "min_object_base_distance_m": min(distances),
        "frames_below_min_distance": frames_below,
        "minimum_base_distance_m": float(min_base_distance_m),
        "relative_xy_range_m": xy_range,
        "max_relative_xy_range_m": max_range,
        "maximum_allowed_relative_xy_range_m": float(max_relative_xy_range_m),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-base-distance-m", type=float, default=0.80)
    parser.add_argument("--max-relative-xy-range-m", type=float, default=0.020)
    args = parser.parse_args(argv)

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    report = evaluate_continuous_hold(
        trajectory,
        min_base_distance_m=args.min_base_distance_m,
        max_relative_xy_range_m=args.max_relative_xy_range_m,
    )
    report["trajectory"] = str(args.trajectory.resolve())
    _write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
