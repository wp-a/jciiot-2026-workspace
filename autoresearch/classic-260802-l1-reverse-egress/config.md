# L1 Reverse-Egress Physical Carry

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Goal

Test whether reversing away from the L1 source production line produces the
first strict attachment-free physical transport with at least `0.50 m` true
object displacement.

## Immutable Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate `competition_grasp.py` SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate `competition_transport.py` SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner:
  `/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py`;
- runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- auditor:
  `/home/user/jciiot-2026/tools/physical-data-gate-944bcc7/audit_physical_transport_dataset.py`;
- auditor SHA-256:
  `944bcc7a040ee2bf198b29a7a637018844bdc75fe9d37c3112df7b68f047b79e`;
- public L1 scene, seed `0`, near container;
- isolated result root:
  `/home/user/jciiot-2026/results/l1-reverse-egress-20260802/`.

## Single Variable

Replace the previous object-facing waypoint near `(7.500, 4.610)` with the
explicit reverse-egress waypoint `(8.500015, 4.599998)`. This changes route
direction only.

Frozen settings:

- actuator-only physical transport;
- maximum base speed `0.04 m/s`;
- original runner drift guard `0.03 m`;
- arm feedforward `0.0`;
- inward feedforward `0.0`;
- planar recovery disabled;
- no heading alignment;
- no candidate, scene, grasp, lift, or service change.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/user/jciiot-2026/envs/official-pinned-eval/bin/python \
/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py \
  --candidate-root /home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3 \
  --expected-official-commit 0dcdddf18a9e694569aa1433cdfc04eb097fed78 \
  --output /home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50.json \
  --trajectory /home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50-trajectory.json \
  --seed 0 \
  --full-physical-stage route \
  --full-physical-waypoint 8.500015 4.599998 \
  --full-physical-actuator-only \
  --posture-locked-carry-max-linear-m-s 0.04
```

## Keep / Discard Metric

Keep only if the independent auditor classifies `transport_success` with:

- object translation at least `0.50 m`;
- minimum lift at least `0.13 m`;
- continuous bilateral contact and no drop;
- object-to-gripper drift at most `0.05 m`;
- zero collisions;
- zero attachment, teleport, object-pose write, and robot-state write evidence;
- no infrastructure error.

Any failure is preserved under its measured class. The final gate is not
relaxed, and no failed run is relabeled as successful training data.

## Audit Tool Formatting Note

The preregistered auditor produced the same classification but used the older
CRLF/trailing-empty-field TSV format. The already-tested canonical-output
revision was copied to the isolated path
`/home/user/jciiot-2026/tools/physical-data-gate-6ce0a9b/` and used for the
committed ledger. Its SHA-256 is
`6ce0a9b350ad94521a9313f30ae94bedc27700599e9a4bd64ad399f3a467d1a8`.
The only relevant change is TSV line/column formatting; audit thresholds and
classification semantics are unchanged. Both revisions classified the result
as `recovery`, and both ledger JSON files have SHA-256
`0a72a3292d0ca8e4948b3c063fe03fa4c8e183b48b62b1aa0943badc49a79fd3`.
