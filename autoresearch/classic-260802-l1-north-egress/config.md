# L1 North-Egress Physical Carry

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Diagnosis

The best clean simultaneous physical carry moved the L1 tote `0.265401 m`
toward world `-x`, but a longer run brought the torso proxy into
`production_line_5`. The official semantic map places that line at center
`(6.152, -0.946)` with half-size `(0.350, 6.8015)`, so its north edge is near
`y=5.8555`. The base starts near `(8.000015, 4.599998)`.

Reverse `+x` egress avoided collision but opened the grasp and failed at only
`0.123148 m`. Before developing a new structural grasp, this experiment tests
the only short collision-free detour that can lead around the north end.

## Falsifiable Hypothesis

The unchanged actuator-only physical hold can carry the tote at least
`0.50 m` along world `+y` while preserving lift and bilateral contact with zero
collision or integrity violations.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner:
  `/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py`;
- runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- public L1 near tote, seed `0`;
- raw scripted physical grasp selected by `--full-physical-stage route`;
- actuator-only physical transport;
- base speed cap `0.04 m/s`;
- original `0.03 m` internal drift guard;
- arm planar feedforward `0.0`;
- inward feedforward `0.0`;
- planar recovery disabled;
- yaw held fixed;
- running 8502/8503 services unchanged.

## Single Variable

Request one straight base waypoint at approximately `(8.000015, 5.099998)`,
changing the transport direction to world `+y` while keeping requested distance
`0.50 m`.

## Keep / Discard

Keep only if the independent physical-data auditor verifies:

- true object translation greater than the clean incumbent `0.265401 m`;
- at least `0.13 m` minimum lift;
- continuous bilateral contact and no drop;
- object-to-gripper drift at most `0.05 m`;
- zero collision frames;
- zero attachment calls/activations, legacy teleports, object-pose writes, and
  robot-state writes.

Full structural success requires at least `0.50 m` true object translation.
Any failure remains recovery or rejected evidence and cannot enter successful
transport training data.

## Observed Result

The hypothesis was rejected. The controller stopped after 19 steps on
`planar_grasp_drift`, with `0.031000 m` base translation but only `0.003423 m`
true object translation. Minimum lift was `0.195191 m`; bilateral contact was
retained and all collision/integrity counters were zero. The independent
auditor classified the run as recovery, not transport success.

- result SHA-256:
  `5f4318184fa79f4b0d0d00feb141e978349151cc953d20e75b68ccf4d839bfad`;
- trajectory SHA-256:
  `9f61f58d0ca0bab0df4c30e3b93ed52054b274914f9b1e8721243ebfea445fb0`;
- audit ledger SHA-256:
  `44ab265ec08c136f7e7aa2bed29d8b6d277d81d26821114d767c3face58db5f5`;
- audit TSV SHA-256:
  `a5369b047e83ba99eaf37827d716c0cba52a15254ab05e091f3669364d09fb52`.
