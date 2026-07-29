# L1 Table-Edge Undercut Design

## Goal

Test whether the L1 container can be lifted through real open-gripper support,
without first pinching it. The first gate ends after one hand reaches the
exposed underside and sustains measured non-finger contact. It does not attempt
transport or claim an official score.

## Geometric Evidence

The `input_5` support reaches world `y=4.688`. The initialized container center
is approximately `y=4.620` and its half-depth is `0.200 m`, so its outer edge is
approximately `y=4.820`. About `0.132 m` of the bottom therefore overhangs the
table. The XML bottom is centered `0.116 m` below the object body and is
`0.018 m` thick.

This exposed strip lets a hand descend outside both the table and container,
move inward below the bottom, and raise into contact. It avoids the failed
assumption that one side of a pinch can carry the box while the other hand
transitions underneath.

## Probe Sequence

1. Navigate through the unmodified official source move, but do not call the
   grasp workflow.
2. Keep the right gripper open and move it above and beyond the world `+y`
   container edge.
3. Descend outside the container to a target below its collision bottom.
4. Move inward to a target that remains outside the measured table edge but
   lies inside the exposed container footprint.
5. Raise slowly until allowed right-hand, wrist, or distal-arm collision
   geometry contacts the object.
6. Require contact for consecutive steps while the object rises physically.

The other arm stays at its captured joint target. The experiment changes no
object state directly and creates no attachment.

## Hard Gates

- Judge collision frames: `0`.
- Object-pose writes and attachment calls: `0`.
- Gripper command remains open for the whole probe.
- No object contact is allowed during the outside and descent stages.
- The undercut target stays outside the configured table edge.
- Lift acceptance requires non-finger support contact and measured positive
  object-height change for consecutive steps.
- The runner records every stage, target, measured end-effector pose, object
  pose, contact geometry, and failure reason.

## First Experiment

Use the measured L1 geometry with a conservative outside clearance. Change only
the upward contact distance after the no-contact path is shown collision-free.
If the descent or inset collides, inspect the exact collision pair before
changing a target. If the hand reaches the target without support, inspect its
collision geometry orientation before attempting a larger inset.

## Submission Boundary

The first implementation is external diagnostic instrumentation under
`scripts/`. It is not part of the scored submission. Only a physically verified
controller may later be integrated under the allowed `skills/` or `workflows/`
directories.
