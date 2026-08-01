# L4 target-margin conclusion

## Status

Candidate `l4-target-margin-cc1b5b3` passed the pre-registered promotion gate
in the pinned official simulator. It is promoted for the five-task prediction
package.

## Evidence

- Nominal seed `20260810`: 25/25, zero collision frames, one verified physical
  bilateral grasp and lift, workflow success, 0.302334 m final target distance,
  and 107.491314 s elapsed time.
- Small-perturbation seed `20260802`: 25/25, zero collision frames, one verified
  physical bilateral grasp and lift, workflow success, 0.315929 m final target
  distance, and 103.366909 s elapsed time.
- The final 100 trajectory frames kept object height between 1.356649 m and
  1.387313 m nominally, and between 1.356843 m and 1.386075 m under the small
  perturbation. The failed predecessor fell to 0.124973 m during alignment.

## Decision

Retain the verified source grasp and organizer-provided transport attachment.
At the unregistered L4 output, calculate a target yaw from measured base,
object, and target coordinates, then delegate the bounded turn to the official
attachment-aware turn helper. Do not use the failed custom direct-yaw path.

Evidence is stored in
`artifacts/l4-target-margin-cc1b5b3-nominal/` and
`artifacts/l4-target-margin-cc1b5b3-small/`.

## Deliverable

The submission archive is
`/Users/wangpeng/jciiot-2026-deliverables/JCIIOT2026_validation_predictions_20260801_L4-optimized.zip`
with SHA-256
`aa442cbf79bee4d16f5b0f392a648313880a5dbab5481687f2b56ef27136fc65`.
It contains exactly five scene-named JSON files at the ZIP root. L1, L2, L3,
and L5 are byte-identical to the prior verified package; only L4 is replaced
by the promoted nominal trajectory.
