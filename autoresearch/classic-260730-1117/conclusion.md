# L1 Posture-Locked Carry Conclusion

## Status

The attachment-free physical-carry gate passed twice at `0.10 m` with exactly
matching physical metrics, then passed at `0.25 m`. This is not yet an official
L1 score claim: no obstacle route, placement, or official scorer execution was
performed in this loop.

## Established Evidence

- The candidate's scripted grasp repeatedly lifted the L1 container by
  `0.204334 m` with bilateral contact.
- Official `follow_path` alone dropped the container because it applies an idle
  action during navigation. Reducing speed from `0.70` to `0.04 m/s` did not
  fix the drop.
- A 40-step static close-action control retained four fingertip/fingerpad
  contacts on each arm, proving that continuous grip actuation is necessary.
- Continuous grip actuation without posture locking retained the object but
  moved it only `0.067252 m` for a `0.10 m` base request.
- Combining continuous grip actuation with base-relative robot posture locking
  moved the object `0.107038 m` in both clean `0.10 m` runs and `0.253854 m` in
  the `0.25 m` run.
- The successful `0.25 m` trial ended `0.191040 m` above the source table with
  bilateral contact, `0.000499 m` lateral drift, `0.003868 m` grasp drift, and
  zero collision, attachment, legacy teleport, or object-pose-write evidence.

The static-hold record is marked diagnostic-only because the older cradle gate
counts non-finger support contacts. Its per-step observations nevertheless show
continuous bilateral fingertip and fingerpad contact, zero collision, and no
drop.

## Decision

Keep research commit `8ade96f` and reject the three prior carry variants. The
next experiment must build a collision-checked L1 route from semantic-map and
proxy geometry, execute it in bounded segments, and stop immediately on contact
loss, height loss, excessive drift, or collision. Do not promote the controller
into the formal candidate or claim `10/10` until placement and the official
scorer pass.

