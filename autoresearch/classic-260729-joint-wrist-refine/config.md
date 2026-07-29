# L1 joint-space wrist seed refinement

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 3 valid single-variable experiments
- Parent evidence: `autoresearch/classic-260729-joint-wrist-seed/`
- Research implementation commit: `e0a244b`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene and seed: public L1, seed 0
- Remote result root: `/home/user/jciiot-2026/results/l1-joint-wrist-refine-20260729`
- Current 8502 candidate: unchanged

## Metric and invariant

Use the same ordered joint-seed, OSC, real contact, lift, hold, and transport
metric as the parent loop. Keep the 10-degree endpoint axis gate, 0.015 m
endpoint position gate, 0.03 m path drift gate, and zero-collision rule fixed.

Hold the 0.01 rad interior margin, 0.02 regularization, 800 solver evaluations,
and 240 interpolation waypoints fixed. Refine only the position residual scale
between the prior 0.015 m angular failure and 0.020 m position failure.

Start at 0.0185 m. Stop early if a route reaches a new downstream blocker;
subsequent iterations must address only that measured blocker.
