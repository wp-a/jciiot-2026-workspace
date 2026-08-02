#!/usr/bin/env python3
"""Run a frozen L1 research probe with fail-closed attachment guards."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType


RAW_PHYSICAL_GRASP_SENTINEL = "attachment_free_inchworm"


def load_competition_transport_module() -> ModuleType:
    return importlib.import_module("robot_agent.skills.competition_transport")


def install_attachment_free_guards(
    runner: ModuleType,
    *,
    inchworm_reseat_steps: int | None = None,
    transport_module_loader=load_competition_transport_module,
) -> None:
    """Force raw grasping and reject hidden transport-state mutation."""
    if inchworm_reseat_steps is not None and inchworm_reseat_steps < 0:
        raise ValueError("inchworm reseat steps must be non-negative")
    execute_probe_grasp = runner._execute_probe_grasp

    @wraps(execute_probe_grasp)
    def raw_probe_grasp(*args, full_physical_stage=None, **kwargs):
        return execute_probe_grasp(
            *args,
            full_physical_stage=(
                RAW_PHYSICAL_GRASP_SENTINEL
                if full_physical_stage is None
                else full_physical_stage
            ),
            **kwargs,
        )

    inchworm_probe = runner._end_grasp_inchworm_probe

    @wraps(inchworm_probe)
    def guarded_inchworm_probe(*args, **kwargs):
        if inchworm_reseat_steps is None:
            result = dict(inchworm_probe(*args, **kwargs))
        else:
            transport_module = transport_module_loader()
            original_config = transport_module.InchwormCarryConfig

            def overridden_config(*config_args, **config_kwargs):
                config_kwargs["reseat_steps"] = inchworm_reseat_steps
                return original_config(*config_args, **config_kwargs)

            transport_module.InchwormCarryConfig = overridden_config
            try:
                result = dict(inchworm_probe(*args, **kwargs))
            finally:
                transport_module.InchwormCarryConfig = original_config
        attachment_activations = int(result.get("attachment_activations", 0))
        object_pose_writes = int(result.get("object_pose_writes", 0))
        attachment_active = bool(
            result.get("transport_attachment_active_before", False)
            or result.get("transport_attachment_active_after", False)
        )
        if attachment_activations or attachment_active or object_pose_writes:
            raise RuntimeError(
                "transport attachment or object-pose mutation detected during "
                "attachment-free inchworm probe"
            )
        return result

    runner._execute_probe_grasp = raw_probe_grasp
    runner._end_grasp_inchworm_probe = guarded_inchworm_probe


def load_runner(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("frozen_l1_probe_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--inchworm-reseat-steps", type=int)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    wrapper_args, runner_argv = parse_wrapper_args(argv)
    runner = load_runner(wrapper_args.runner)
    install_attachment_free_guards(
        runner,
        inchworm_reseat_steps=wrapper_args.inchworm_reseat_steps,
    )
    result = runner.run_probe(runner.parse_args(runner_argv))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
