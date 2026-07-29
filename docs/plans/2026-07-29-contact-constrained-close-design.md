# Contact-Constrained Close Design

## Problem and evidence

The high-precenter sequence moves both open grippers over distinct long walls
without contact, then descends. The generic end-effector target gate remains
false because the left arm is contact-limited about 5 mm outside the common
12 mm tolerance. As a result, closure is never attempted.

The detailed MuJoCo geometry snapshot proves that the failed endpoint is
nevertheless a valid physical close-ready geometry in the wall-normal
direction:

- right fingerpad centers project to `y=4.746/4.875`, bracketing the back wall
  at `y=4.809`;
- left fingerpad centers project to `y=4.363/4.491`, bracketing the front wall
  at `y=4.423`;
- the two grippers are assigned to different opposed walls;
- the trajectory contains zero official judge-collision frames.

This is not yet a grasp. It is evidence that each wall is physically between
the corresponding open fingers and that trying closure is justified.

## Alternatives

1. **Contact-constrained close readiness** (selected): accept either the normal
   pose target or a strict two-wall fingerpad-bracket condition, then attempt
   the unchanged physical close.
2. Increase the global approach tolerance to 20 mm. This hides the physical
   reason for accepting the pose and could affect unrelated stages.
3. Continue lowering or increase the action horizon. The rigid knuckle/rim
   contact has already persisted for 220 steps, so this adds force without
   improving geometry.

## Components

`fingerpad_world_positions` reads the two official important fingerpad geoms
for each gripper. `opposed_object_wall_centers` reads object geom centers and
selects the minimum and maximum projections along the measured separation
axis. Neither helper writes simulator state.

`fingerpad_bracket_evidence` is pure. It validates finite shapes, normalizes the
axis, projects the two wall centers and each pair of fingerpads, assigns each
arm to its nearest wall, and returns structured evidence. Readiness requires:

1. each wall projection lies between that arm's two fingerpad projections;
2. the two arms are assigned to distinct wall centers;
3. all values are finite.

If `approach_center_walls` reaches the normal pose target, behavior is
unchanged. If it is contact-limited without an official collision, the bracket
evidence is evaluated. A positive result changes only the stage completion
mode to `fingerpad_bracket` and permits `close_center_grasp` to run. It never
sets `physical_grasp`, `lift_m`, support, hold, or score fields.

## Hard gates

The close stage still requires three consecutive bilateral official
`grasp_status` frames. Lift still requires measured object height at least
0.13 m above the table reference. Hold still requires at least 20 steps with
closed grippers. Any official collision, object-pose write, attachment call,
contact loss, or drop rejects the run.

The feature remains in the research runner until two clean physical passes and
later integration into only competition-allowed submission paths.
