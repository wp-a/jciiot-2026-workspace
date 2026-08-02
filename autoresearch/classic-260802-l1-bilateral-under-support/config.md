# L1 Bilateral Under-Bottom Support Transition

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Diagnosis

The current attachment-free side-wall pinch can physically grasp and lift the
L1 tote, but clean transport is structurally limited. The best straight carry
moved the object `0.265401 m`; motion along the pinch axis moved it only
`0.003423 m`. Repeated base reset also rolled and dropped the load.

The official MuJoCo tote collision model has solid walls and no collision hole
at the visual handles. A table-level undercut is blocked by the nearby static
support. After the existing bilateral pinch lifts the tote, however, the bottom
becomes physically accessible. Earlier one-arm transitions failed because the
stationary side could not hold the load alone. The untested structural change
is a simultaneous two-arm transition below the already elevated bottom.

## Falsifiable Hypothesis

After the verified center grasp and hold, both closed grippers can raise the
tote an additional `0.10 m`, descend simultaneously by `0.16 m`, inset toward
the arm midpoint by `0.03 m`, and then raise by `0.06 m`. This will create
bilateral non-finger support and physically lift the tote during the final
raise without attachment, direct state writes, or collision.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- base candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- local experiment base commit:
  `1ccd561e3356f630dbd1a263c1e8bd6fb9b9c00d`;
- grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- transport SHA-256 before this experiment:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner SHA-256 before this experiment:
  `d894aab484c015fd0859abe96d13c32e5dd97c23b0392f8ec359df67677393e6`;
- public L1 near tote, seed `0` only;
- existing direct scripted center physical grasp and 20-step closed hold;
- grippers remain physically closed throughout the transition;
- running 8502/8503 services remain unchanged.

## Single Structural Change

Add one diagnostic stage after the existing closed hold and before any base
transport:

1. raise both arms `0.10 m` while requiring bilateral grasp and safe height;
2. move both arms down `0.16 m` and inward `0.03 m` simultaneously;
3. raise both arms `0.06 m` with closed grippers;
4. measure true object motion and non-finger support contacts from MuJoCo.

No long route, parameter sweep, additional seed, or learning run is authorized
by this experiment.

## Keep / Discard

Keep only if all of the following are independently verified:

- existing center grasp, lift, and 20-step hold succeed physically;
- final two-arm raise moves the object upward by at least `0.02 m`;
- at least 5 consecutive final-raise frames have allowed non-finger support on
  both arms;
- terminal object height remains at least `0.10 m` above table height;
- zero collision frames;
- zero attachment calls/activations, legacy teleports, object-pose writes, and
  robot-state writes.

Failure to enter the bottom region, finger-only contact, unsupported motion, or
any safety/integrity violation rejects the route. A failed run is diagnostic
evidence only and cannot enter successful transport training data.
