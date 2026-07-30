# L1 Posture-Locked Physical Carry

- Mode: classic metric loop
- Date: 2026-07-30 (Asia/Shanghai)
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-inchworm-eb48310`
- Runtime: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`
- Scene and seed: public L1, seed `0`
- Remote result root: `/home/user/jciiot-2026/results/l1-posture-carry-20260730`
- Local evidence: `artifacts/l1-posture-carry-20260730/`
- Winning external runner: `/home/user/jciiot-2026/tools/posture-carry-8ade96f/run_l1_cradle_gate.py`
- Winning runner SHA-256: `e5bb55bc53a5cfa9eef8dba84f0705b4bdc68bb53981c48e6066a48b90da8abd`
- Research implementation commit: `8ade96f`
- Official 8502 process: PID `1769287`, unchanged throughout this loop

## Hard Metric

A trial passes only when all conditions hold:

- physical scripted grasp succeeds;
- projected object progress is at least `0.08 m`;
- lateral object drift is at most `0.03 m`;
- object-to-gripper planar drift is at most `0.03 m`;
- final object height is at least `0.10 m` above the source table;
- both arms retain terminal physical object contact;
- collision frames are zero;
- transport attachment activations are zero;
- legacy held-crate teleport activations are zero;
- task-object pose writes are zero;
- no infrastructure error occurs.

The requested travel direction is resolved online from the current base and
object positions. No fixed world coordinate is required.

## Winning Control

The retained controller combines:

1. the candidate's bilateral scripted physical grasp;
2. continuous gripper-close actuator commands;
3. a `0.04 m/s` base-speed limit and `0.005 m/s` command slew limit;
4. base-relative arm, torso, and head posture locking;
5. online contact, height, object progress, drift, collision, attachment,
   legacy teleport, and object-pose-write gates.

Only robot joints are posture-locked. Gripper joints remain actuator driven,
and the task object's free joint is never written.

## Representative Command

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /home/user/jciiot-2026/envs/official-pinned-eval/bin/python \
  /home/user/jciiot-2026/tools/posture-carry-8ade96f/run_l1_cradle_gate.py \
  --candidate-root /home/user/jciiot-2026/candidates/robust-l1-inchworm-eb48310 \
  --expected-official-commit 0dcdddf18a9e694569aa1433cdfc04eb097fed78 \
  --output /home/user/jciiot-2026/results/l1-posture-carry-20260730/trial-0p10-actuated-lock-v0p04-seed0.json \
  --trajectory /home/user/jciiot-2026/results/l1-posture-carry-20260730/trial-0p10-actuated-lock-v0p04-seed0-trajectory.json \
  --seed 0 \
  --posture-locked-carry-distance-m 0.10 \
  --posture-locked-carry-max-linear-m-s 0.04 \
  --posture-locked-carry-actuated-gripper-hold \
  --posture-locked-carry-posture-lock-robot-joints
```

