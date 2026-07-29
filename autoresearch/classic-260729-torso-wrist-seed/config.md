# L1 torso-redundant wrist seed

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 3 valid single-variable experiments
- Research implementation commit: `8e6f526`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene and seed: public L1, seed 0
- Remote result root: `/home/user/jciiot-2026/results/l1-torso-wrist-seed-20260729`
- Current 8502 candidate: unchanged

## Retained arm-only incumbent

- 24 continuation nodes and 240 collision-checked waypoints
- 0.0185 m position residual scale
- 0.01 rad arm interior margin
- 0.02 normalized joint regularization
- accepted seed: 6.63 / 9.93 degrees, 13.324 mm drift, zero collision
- OSC timeout: 5.98 / 9.20 degrees, zero collision

## First single-variable experiment

Enable `robot0_torso_lift_joint` as the thirteenth IK variable with a 0.005 m
interior margin. Keep every arm, continuation, OSC, contact, and safety value
fixed. Refresh the torso hold target only after a successful seed.

Rank by the unchanged ordered metric: zero violations/collisions, accepted
joint seed, accepted 5-degree OSC alignment, bilateral physical contact,
0.13 m lift, 20-step hold, and real transport. Stop and retain evidence on any
collision or shortcut violation.
