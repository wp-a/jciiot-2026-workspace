# JCIIOT L1 pregrasp base-advance design

Date: 2026-07-29 (Asia/Shanghai)

Status: scheme-1 reachability extension based on the first 10-degree center
regrasp experiment.

## Root cause

The accepted arm-only seed, 10-degree runtime orientation entry, and outward
wall retreat all passed with zero collision. During the subsequent 0.24 m
center translation, the right grip site reached within about 3 mm of its target
but the left remained about 48 mm short after 140 steps. Its first arm joint was
again near the upper limit. More OSC steps would repeat the same saturated
command.

The base was near x=8.000 while the tabled object was near x=7.060, facing the
object. Advancing the unloaded mobile base by 0.10 m reduces the required arm
extension without touching the object.

## Selected behavior

Add an opt-in high-clearance base-advance stage after orientation entry and
before outward wall retreat. Compute the planar direction from the current base
to the physical object, convert a bounded world velocity to the current base
frame, and use the existing `OfficialPhysicalCarryDriver.step()` direct-base
boundary.

The stage:

- keeps both grippers open;
- sends zero arm deltas so the high-clearance end effectors move with the base;
- preserves the current torso and head hold targets;
- limits speed to 0.04 m/s and the final step to the remaining distance;
- records every simulator step, base pose, EEF positions, object position,
  contacts, and official collision flag;
- stops immediately on official collision or any robot-object contact;
- succeeds only after measured base translation reaches the requested distance.

The first experiment requests 0.10 m. The default is 0 m, preserving all prior
results.

## Compliance and promotion

The stage uses only robot base motion and official collision feedback. It does
not write object qpos, attach objects, modify protected files, or change 8502.
The real-gripper close stage remains mandatory before lift. A base-advance pass
is only reachability evidence; bilateral wall contact, three-step closed-grasp
contact, 0.13 m lift, and 20-step hold are still required.
