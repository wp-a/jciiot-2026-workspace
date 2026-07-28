#!/usr/bin/env python3
"""Deterministic perturbation specifications for research-only evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TierLimits:
    object_xy_m: float
    object_yaw_rad: float
    base_xy_m: float
    base_yaw_rad: float
    mass_fraction: float
    friction_fraction: float


@dataclass(frozen=True)
class PerturbationSample:
    tier: str
    seed: int
    task_index: int
    object_name: str
    object_dx_m: float
    object_dy_m: float
    object_dyaw_rad: float
    base_dx_m: float
    base_dy_m: float
    base_dyaw_rad: float
    mass_scale: float
    friction_scale: float
    generator_digest: str

    def numeric_values(self) -> tuple[float, ...]:
        return (
            self.object_dx_m,
            self.object_dy_m,
            self.object_dyaw_rad,
            self.base_dx_m,
            self.base_dy_m,
            self.base_dyaw_rad,
            self.mass_scale,
            self.friction_scale,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_TIERS = {
    "nominal": TierLimits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "small": TierLimits(
        object_xy_m=0.02,
        object_yaw_rad=math.radians(5.0),
        base_xy_m=0.01,
        base_yaw_rad=math.radians(2.0),
        mass_fraction=0.0,
        friction_fraction=0.0,
    ),
    "medium": TierLimits(
        object_xy_m=0.04,
        object_yaw_rad=math.radians(10.0),
        base_xy_m=0.03,
        base_yaw_rad=math.radians(5.0),
        mass_fraction=0.10,
        friction_fraction=0.10,
    ),
    "stress": TierLimits(
        object_xy_m=0.06,
        object_yaw_rad=math.radians(15.0),
        base_xy_m=0.05,
        base_yaw_rad=math.radians(8.0),
        mass_fraction=0.20,
        friction_fraction=0.20,
    ),
}


def tier_limits(tier: str) -> TierLimits:
    key = str(tier).strip().lower()
    try:
        return _TIERS[key]
    except KeyError as exc:
        raise ValueError(f"unknown perturbation tier: {tier}") from exc


def _uniform(rng: random.Random, limit: float) -> float:
    if float(limit) == 0.0:
        return 0.0
    return rng.uniform(-float(limit), float(limit))


def sample_perturbation(
    *,
    tier: str,
    seed: int,
    task_index: int,
    object_name: str,
) -> PerturbationSample:
    key = str(tier).strip().lower()
    limits = tier_limits(key)
    identity = json.dumps(
        [key, int(seed), int(task_index), str(object_name)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(identity).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(digest[:16]), "big"))

    return PerturbationSample(
        tier=key,
        seed=int(seed),
        task_index=int(task_index),
        object_name=str(object_name),
        object_dx_m=_uniform(rng, limits.object_xy_m),
        object_dy_m=_uniform(rng, limits.object_xy_m),
        object_dyaw_rad=_uniform(rng, limits.object_yaw_rad),
        base_dx_m=_uniform(rng, limits.base_xy_m),
        base_dy_m=_uniform(rng, limits.base_xy_m),
        base_dyaw_rad=_uniform(rng, limits.base_yaw_rad),
        mass_scale=1.0 + _uniform(rng, limits.mass_fraction),
        friction_scale=1.0 + _uniform(rng, limits.friction_fraction),
        generator_digest=digest,
    )
