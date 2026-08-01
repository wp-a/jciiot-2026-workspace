#!/usr/bin/env python3
"""Run official robomimic training without per-epoch resume artifacts."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path


RESUME_ARTIFACT_NAMES = frozenset({"last.pth", "last_bak.pth"})


def is_resume_artifact(path: str | Path) -> bool:
    return Path(path).name in RESUME_ARTIFACT_NAMES


@contextmanager
def suppress_resume_artifact_writes(train_module):
    original_save_model = train_module.TrainUtils.save_model
    original_copyfile = train_module.shutil.copyfile

    def save_model(*args, **kwargs):
        checkpoint_path = kwargs.get("ckpt_path")
        if checkpoint_path is not None and is_resume_artifact(checkpoint_path):
            print(f"[io-bounded] skipped resume checkpoint: {checkpoint_path}")
            return None
        return original_save_model(*args, **kwargs)

    def copyfile(source, destination, *args, **kwargs):
        if is_resume_artifact(source) or is_resume_artifact(destination):
            print(f"[io-bounded] skipped resume backup: {destination}")
            return destination
        return original_copyfile(source, destination, *args, **kwargs)

    train_module.TrainUtils.save_model = save_model
    train_module.shutil.copyfile = copyfile
    try:
        yield
    finally:
        train_module.TrainUtils.save_model = original_save_model
        train_module.shutil.copyfile = original_copyfile


def train_from_config(config_path: str | Path) -> None:
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.config import config_factory
    from robomimic.scripts import train as train_module

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        external_config = json.load(handle)
    config = config_factory(external_config["algo_name"])
    with config.values_unlocked():
        config.update(external_config)
    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    with suppress_resume_artifact_writes(train_module):
        train_module.train(config, device=device, resume=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    train_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
