#!/usr/bin/env python3
"""Reset and step each official FactorySorting scene without policy weights."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCENE_SPECS = (
    (
        "FactorySorting1_3FO3ERFHISEM",
        "robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem",
        "FactorySorting1_3FO3ERFHISEM",
    ),
    (
        "FactorySorting3_3FO3ERRPH7X9",
        "robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9",
        "FactorySorting3_3FO3ERRPH7X9",
    ),
    (
        "FactorySorting5_3FO3ERTPXEUT",
        "robosuite.environments.factory_sorting.factory_sorting_5_3fo3ertpxeut",
        "FactorySorting5_3FO3ERTPXEUT",
    ),
    (
        "FactorySorting7_3FO3ERFKY9RN",
        "robosuite.environments.factory_sorting.factory_sorting_7_3fo3erfky9rn",
        "FactorySorting7_3FO3ERFKY9RN",
    ),
    (
        "FactorySorting9_3FO3ERT2C5FP",
        "robosuite.environments.factory_sorting.factory_sorting_9_3fo3ert2c5fp",
        "FactorySorting9_3FO3ERT2C5FP",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_scene(
    scene_name: str,
    env_factory: Callable[..., Any],
    zeros_like: Callable[[Any], Any],
    *,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run one scene and return a serializable diagnostic record."""
    started = time.perf_counter()
    env = None
    stage = "construct"
    record: dict[str, Any] = {
        "scene": scene_name,
        "success": False,
        "stage": stage,
        "steps_requested": steps,
        "steps_completed": 0,
    }
    try:
        env = env_factory(
            robots="Tiago",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            ignore_done=True,
            control_freq=20,
            seed=seed,
        )
        stage = "reset"
        env.reset()

        stage = "action_spec"
        action_low, _ = env.action_spec
        action = zeros_like(action_low)
        action_shape = getattr(action, "shape", None)
        if action_shape is None:
            action_shape = (len(action),)

        stage = "step"
        for _ in range(steps):
            env.step(action)
            record["steps_completed"] += 1

        stage = "inspect_model"
        model = env.sim.model
        record.update(
            {
                "success": True,
                "stage": "complete",
                "action_shape": [int(value) for value in action_shape],
                "model": {
                    "nq": int(model.nq),
                    "nv": int(model.nv),
                    "ngeom": int(model.ngeom),
                    "ncam": int(model.ncam),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve any scene error
        record.update(
            {
                "success": False,
                "stage": stage,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - report close failures too
                if record["success"]:
                    record.update(
                        {
                            "success": False,
                            "stage": "close",
                            "exception_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                else:
                    record["close_error"] = f"{type(exc).__name__}: {exc}"
        record["duration_s"] = round(time.perf_counter() - started, 6)
    return record


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_scenes = [item["scene"] for item in results if not item["success"]]
    return {
        "success": not failed_scenes,
        "passed": len(results) - len(failed_scenes),
        "failed": len(failed_scenes),
        "failed_scenes": failed_scenes,
    }


def _official_commit(official_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_factories(official_root: Path) -> dict[str, Callable[..., Any]]:
    import importlib

    robosuite_root = official_root / "JCIIOT" / "robosuite"
    if not robosuite_root.is_dir():
        raise FileNotFoundError(f"official robosuite checkout not found: {robosuite_root}")
    sys.path.insert(0, str(robosuite_root))

    factories: dict[str, Callable[..., Any]] = {}
    for scene_name, module_name, class_name in SCENE_SPECS:
        module = importlib.import_module(module_name)
        factories[scene_name] = getattr(module, class_name)
    return factories


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run weight-free reset/step smoke checks for official scenes."
    )
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scene",
        action="append",
        choices=[item[0] for item in SCENE_SPECS],
        help="Run only the selected scene; repeat to select multiple scenes.",
    )
    args = parser.parse_args(argv)
    official_root = args.official_root.resolve()
    started_at = _utc_now()

    report: dict[str, Any] = {
        "official_root": str(official_root),
        "official_commit": None,
        "python": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "steps": args.steps,
        "started_at": started_at,
        "finished_at": None,
        "results": [],
        "summary": None,
    }
    try:
        import numpy as np

        report["official_commit"] = _official_commit(official_root)
        factories = _load_factories(official_root)
        selected = set(args.scene or factories)
        report["results"] = [
            run_scene(
                scene_name,
                factories[scene_name],
                np.zeros_like,
                steps=args.steps,
                seed=args.seed,
            )
            for scene_name in factories
            if scene_name in selected
        ]
        report["summary"] = build_summary(report["results"])
    except Exception as exc:  # noqa: BLE001 - preserve bootstrap failures in JSON
        report["bootstrap_error"] = {
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["summary"] = {
            "success": False,
            "passed": 0,
            "failed": 0,
            "failed_scenes": [],
        }
    report["finished_at"] = _utc_now()
    _write_report(args.output, report)
    return 0 if report["summary"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
