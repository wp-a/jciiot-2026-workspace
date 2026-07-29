# L1 joint-space wrist seed

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 4 valid single-variable experiments
- Workspace branch: `robust-hybrid-20260728`
- Research implementation commit: `e0a244b`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene: public L1 `FactorySorting1_3FO3ERFHISEM`
- Seed: `0`
- Runtime: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`
- Rendering: `MUJOCO_GL=egl`, `CUDA_VISIBLE_DEVICES=2`
- Scheduling: sequential, one worker
- Remote result root: `/home/user/jciiot-2026/results/l1-joint-wrist-seed-20260729`
- Current 8502 candidate: unchanged until two complete clean-process passes

## Metric

Rank each valid experiment lexicographically:

1. no infrastructure error, object-pose write, attachment call, or official
   collision;
2. joint seed accepted with both endpoint axis errors at most 10 degrees,
   endpoint position error at most 0.015 m, and path drift at most 0.03 m;
3. OSC alignment accepted with both errors at most 5 degrees for five
   consecutive steps;
4. bilateral physical object contact and measured lift at least 0.13 m;
5. bilateral contact retained for at least 20 hold steps and real transport.

An optimizer success, endpoint success, or orientation-only success is not a
competition score and cannot satisfy the final metric.

## Default variant

- 12-joint simultaneous bounded least squares
- official limits moved inward by 0.03 rad
- 800 maximum function evaluations
- 240 simultaneous interpolation waypoints
- 0.01 m position residual scale
- `sin(5 degrees)` closure-axis residual scale
- 0.02 normalized joint-displacement regularization
- existing OSC final correction: 0.02 action, 2600 maximum steps
- no object qpos writes and no transport attachments

## Stop rules

- Stop immediately on a collision or shortcut violation and retain evidence.
- Change only one joint-seed variable per subsequent valid iteration.
- Do not reduce collision sampling to hide an invalid path.
- After one complete pass, repeat in a clean process before promotion.
- Do not modify protected competition files or the running 8502 service.
