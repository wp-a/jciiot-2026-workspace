# L1 Single-Arm Bottom-Insertion Transition

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Diagnosis

The valid bilateral transition trial proved that both closed grippers simply
drag the tote down along its same-side wall contacts. End-effector target error
was small, but the tote descended `0.219540 m` and never contacted a palm,
wrist, hand, distal arm, or bottom-facing finger surface. The hand centers
remained roughly `0.28 m` above the tote bottom.

At the successful raw-grasp state, the right fingerpad is approximately
`0.141 m` above the official bottom collision geom and about `0.029 m` outside
its planar footprint. A moving arm must release side friction, descend relative
to the load, and move toward the tote center. The other arm must preserve the
load during that transition.

## Falsifiable Hypothesis

After the current candidate's verified raw scripted physical grasp and a
`0.10 m` bilateral clearance raise, the left closed gripper can preserve the
load while the right gripper opens and moves in one combined action:

- `0.18 m` downward;
- `0.08 m` horizontally toward the measured tote center;
- `0.02 m` laterally toward the two-arm midpoint.

The right gripper will enter beneath the official tote-bottom geom without
collision or height loss. A subsequent `0.05 m` right-arm raise will maintain
bottom-geom contact and physically raise the tote by at least `0.02 m` while
the left gripper remains closed.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner SHA-256 before instrumentation:
  `d894aab484c015fd0859abe96d13c32e5dd97c23b0392f8ec359df67677393e6`;
- public L1 near tote, seed `0` only;
- current raw scripted physical grasp selected by `--full-physical-stage`;
- moving arm: right; stationary arm: left;
- stationary gripper closed throughout; moving gripper open during insertion;
- no route transport or second-arm transition in this experiment;
- running 8502/8503 services unchanged.

## Keep / Discard

Keep only if all are independently verified:

- initial physical grasp and bilateral clearance raise succeed;
- stationary left object contact is continuous during insertion and proof raise;
- tote stays at least `0.10 m` above its original table reference;
- moving right robot geom contacts the official object geom whose name contains
  `col_bottom` for at least 3 consecutive frames;
- the proof raise moves the tote upward by at least `0.02 m`;
- zero judge collision frames;
- zero attachment calls/activations, legacy teleports, object-pose writes, and
  robot-state writes.

Any stationary-contact loss, height loss, side-wall-only contact, or integrity
violation rejects the route immediately. No additional seed or parameter sweep
is authorized by this registration.
