# Inchworm physical transport design

## Goal and invariant

Extend the repeatable center opposed-wall grasp into long-distance physical
transport without task-object pose writes or attachments. The controller must
use only robot actions and the existing bounded direct-base mechanism.

## Evidence behind the design

Base-led dragging was rejected by six measured experiments. In contrast, a
base-stationary 0.08 m closed-gripper arm stroke moved the L1 container
0.107759 m through MuJoCo contact with zero collision. The new controller uses
that effective motion as its transport phase instead of treating it as a grasp
adjustment.

## Macro-step

1. Capture base, object, and both gripper world poses.
2. Move both closed grippers a bounded distance in the desired travel direction
   while adding vertical compensation.
3. Require bilateral official grasp, positive projected object progress,
   bounded lateral drift, minimum height, and zero collision.
4. Advance the base by the measured arm stroke while commanding the arms to
   hold their captured world poses.
5. Require the reset phase to preserve object pose, bilateral grasp, height,
   and zero collision.
6. Repeat until the measured object, not the base, reaches the waypoint.

The first experiment tests only the arm-stroke phase over 0.08 m. Base reset is
added only after the stroke preserves height and produces positive projected
object progress.

## Gates

- object progress along the requested direction is positive and finite;
- planar orthogonal drift and object-to-gripper drift stay bounded;
- object body height remains at least 0.10 m above the source-table reference;
- bilateral official grasp persists;
- collision, object-pose writes, and attachment calls remain zero.

Passing a short stroke is not a score claim. The sequence must next pass 0.20 m,
1.05 m, full L1 placement, and fresh official 8502 scoring.
