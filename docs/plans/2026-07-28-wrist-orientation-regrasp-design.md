# JCIIOT L1 wrist-orientation regrasp design

Date: 2026-07-28 (Asia/Shanghai)

Status: approved by the user on 2026-07-28 after the fixed-wrist gate failed.

## Objective

Determine whether Tiago can physically reorient both Robotiq 85 grippers in a
collision-free high-clearance pose so their closure axes align with the L1
container's opposed long-wall normals. If alignment is reachable, use it for a
table-assisted center regrasp and require a measured 0.13 m lift before any
transport work starts.

This is a research-gate change. It does not replace the current 8502 candidate
and does not enter the scored submission path until the physical gate passes
twice from clean processes.

## Considered approaches

### Fixed incremental rotation

Apply a hand-chosen roll or pitch action for a fixed number of steps. This is
small but fragile: the OSC input is expressed in the controller base frame,
the two wrists start with different orientations, and a visually plausible
axis can rotate the closure direction the wrong way.

### Precomputed joint pose

Solve and store one upper-body joint configuration. This is deterministic but
overfits the public L1 reset and hides whether the actual end-effector closure
axis reached the desired direction. It also creates a larger joint-space
collision and reachability search before the task-space hypothesis is proven.

### Closed-loop task-space axis alignment

Read each `grip_site` rotation from MuJoCo. The Robotiq XML shows that the
finger separation direction corresponds to the grip site's local X axis. For
each arm, compute the minimum rotation that maps this undirected axis to the
container wall-normal axis, transform the rotation into the OSC controller's
base reference frame, and recompute the bounded command after every step.

This is the selected approach because it minimizes rotation, handles both arms
independently, and exposes a measurable residual angle instead of relying on a
step count.

## Components

### Pure orientation geometry

Add pure NumPy helpers to the research runner for:

- selecting the sign of an undirected target axis that gives the smaller
  rotation;
- constructing the minimum rotation between normalized source and target
  axes, including parallel and antiparallel cases;
- converting a rotation matrix to an axis-angle vector;
- normalizing a controller action with the OSC orientation output scale;
- calculating closure-axis angular error with sign ambiguity removed.

These functions are unit tested without importing MuJoCo or robosuite.

### Runtime orientation observation

For each arm, read `sim.data.site_xmat[robot.eef_site_id[arm]]`. The first
matrix column is recorded as the current closure axis. Record the full site
orientation, closure axis, target axis, angle error, and end-effector position
on every orientation-control step.

### High-clearance controller stage

Insert `align_closure_axes` only after the object has physically settled on the
table and both open grippers have risen to the existing 0.18 m clearance pose.
The controller holds each end-effector position while applying a bounded
orientation delta. Torso and head retain their existing hold targets.

The stage succeeds only when both closure-axis errors are at most 5 degrees
for five consecutive steps, end-effector position drift is at most 30 mm, and
no official judge collision occurs. It stops immediately on collision,
non-finite geometry, excessive drift, or timeout.

### Regrasp continuation

After alignment, retain the existing high-clearance wall approach. The first
experiment uses the current 0.10 m wall clearance and 0.24 m center shift. The
grippers close only after bilateral wall contact. A result is promoted only if
the box rises at least 0.13 m and retains bilateral physical contact during a
20-step hold.

## Data flow

1. Official physical grasp lifts the near L1 container.
2. The controller lowers and releases it onto the real table.
3. Both open grippers rise to the no-contact clearance pose.
4. MuJoCo site matrices produce current closure axes.
5. Minimum-rotation geometry produces bounded OSC orientation actions.
6. The stage records angle error, position drift, and collision state.
7. On verified alignment, the controller approaches opposed walls, closes,
   and attempts the measured lift.
8. The hard gate writes an atomic JSON record and retains the original remote
   trajectory regardless of pass or fail.

## Failure handling

- Rotation math rejects zero, non-finite, or malformed axes.
- Controller scaling rejects non-finite or non-positive orientation ranges.
- Any collision terminates the stage and rejects the run.
- Alignment timeout is a valid physical failure, not an infrastructure error.
- A simulator or code exception remains an infrastructure error and cannot be
  used in the physical denominator.
- Failure to lift ends the wrist-orientation route after the bounded variants;
  the next route is the separately approved table-supported push/drag gate.

## Verification and promotion

Automated verification requires the focused unit tests, the complete local
test suite, scored-path audit, and workspace check. Server promotion requires:

- official source commit `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- seed 0 first, followed by a clean-process repeat;
- closure-axis error at most 5 degrees on both arms;
- zero official collision frames, attachment calls, and object-pose writes;
- at least 0.13 m measured regrasp lift and a 20-step bilateral contact hold.

Only after two clean passes may this pose enter contact-preserving transport.
