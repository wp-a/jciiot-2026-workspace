# L1 center-grasp transport conclusion

## Status

No experiment passed the 0.20 m physical-transport gate. No result in this
series is an official score claim and the 8502 service was not changed.

## Established evidence

- The frozen center grasp remained repeatable: 20 bilateral hold steps, zero
  official collision frames, zero object-pose writes, and zero attachments.
- Per-step arm lead plus base motion moved the base 0.190 m but moved the object
  only 0.027 m. The earlier transport-success boolean was therefore a false
  positive based on contact alone.
- The added object-to-gripper drift gate rejected that condition after about
  0.04 m relative motion even though both official grasp booleans remained true.
- Reducing base speed from 0.040 to 0.005 m/s improved the object/base movement
  ratio from about 0.14 to 0.27, but required 150 physics steps and let the
  object descend about 0.095 m. This cannot scale to the approximately 14 m L1
  route.
- A closed-gripper, base-stationary 0.08 m arm translation physically moved the
  object 0.107759 m with zero collision. It failed only because object body
  height fell below the declared 0.10 m margin during the stroke.

## Decision

Stop base-drag and speed sweeps. Preserve the center grasp and use a quasi-static
inchworm transport controller: arms move the object through a bounded stroke,
height is recovered, then the base advances while the arm controller holds the
grippers at fixed world poses. Promotion requires measured object motion in the
travel direction, bounded planar drift, bilateral contact, height margin, and
zero collision/write/attachment at every macro-step.

Compact iteration 4 and 5 records are stored under
`artifacts/l1-center-grasp-transport-20260729/`; the remote root is recorded in
`config.md`.
