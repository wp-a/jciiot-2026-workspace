# JCIIOT L1 torso-redundant wrist seed design

Date: 2026-07-29 (Asia/Shanghai)

Status: scheme-1 extension motivated by the accepted 24-node arm-only seed.

## Objective

Optionally add Tiago's official torso lift joint to the simultaneous wrist-seed
IK so the robot can trade torso height against arm posture while holding both
grip sites near their high-clearance positions. The goal is to leave enough arm
joint range for the existing OSC stage to finish the 5-degree alignment gate.

## Evidence

The arm-only continuation seed passed all 24 nodes and 240 collision-checked
waypoints with 13.324 mm maximum drift and zero collision. It ended at 6.63 and
9.93 degrees. After 2600 OSC steps it remained at 5.98 and 9.20 degrees.

The trajectory ended with `robot0_torso_lift_joint` at approximately 0.349744 m
against the official 0.35 m limit, while several arm joints also approached
their limits. The official Tiago model defines one torso slide joint with range
0 to 0.35 m. This provides a legal redundant degree of freedom; head joints do
not affect either arm kinematic chain and are excluded.

## Runtime design

The feature is opt-in. With it disabled, the accepted 12-arm-joint continuation
behavior is unchanged. With it enabled:

- append `robot0_torso_lift_joint` after the 12 arm joints;
- use official model bounds with a separate 0.005 m torso interior margin;
- normalize torso regularization by its official 0.35 m range;
- solve the same two grip-site position and closure-axis residuals over all 13
  variables at every continuation node;
- replay torso and arm values together through the unchanged 240-waypoint
  position-drift and official-collision gate;
- restore all 13 joints atomically on every failure.

The existing `synchronize_controller_goals()` already updates torso control.
After a successful seed, the runner must also replace the outer torso hold
target with the newly measured torso joint target. Otherwise the subsequent
OSC action builder would command the old near-limit height and undo the seed.

## Safety and scope

Only robot qpos is assigned. Object qpos, attachments, protected official code,
and the running 8502 candidate remain untouched. Endpoint position stays at
most 15 mm, path drift at most 30 mm, seed axis error at most 10 degrees, OSC
error at most 5 degrees for five consecutive steps, and official collision
count zero.

The first experiment changes only torso inclusion while retaining 24 nodes,
0.0185 m position scale, arm margin 0.01 rad, and every downstream parameter.
Torso inclusion is retained only if it improves the ordered gate without a
collision or shortcut violation.
