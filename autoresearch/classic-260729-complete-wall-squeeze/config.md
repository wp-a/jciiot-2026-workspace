# L1 complete wall squeeze gate

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 4 valid single-variable experiments
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene and seed: public L1, seed 0
- Remote result root: `/home/user/jciiot-2026/results/l1-complete-wall-squeeze-20260729`
- Current 8502 candidate: unchanged

## Incumbent

Retain the arm-only 24-node wrist seed, 10-degree runtime orientation entry,
0.10 m collision-checked base advance, zero center shift, 0.10 m outward
clearance, and 0.025 m inward squeeze. The first experiment changes only the
squeeze completion rule: object contact no longer terminates the open-gripper
squeeze after one frame.

## Ordered metric

1. zero infrastructure errors, official collision frames, object-pose writes,
   and attachment calls;
2. completed bounded squeeze with contact geometry progressing from knuckles
   toward fingerpads;
3. three consecutive bilateral official grasp frames;
4. measured lift of at least 0.13 m;
5. at least 20 closed-gripper hold steps with bilateral official grasp;
6. accepted physical gate.

No official score or 8502 promotion is allowed from this diagnostic loop alone.
