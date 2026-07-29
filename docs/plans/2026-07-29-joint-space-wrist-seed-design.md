# JCIIOT L1 joint-space wrist seed design

Date: 2026-07-29 (Asia/Shanghai)

Status: approved by the user on 2026-07-29 as route 1.

## Objective

Move both Tiago arms from the existing collision-free high-clearance pose to a
new robot-only joint configuration that keeps both grip sites nearly fixed but
places each Robotiq 85 closure axis near the L1 container wall-normal axis.
The existing bounded OSC stage then performs only the final correction before
the physical center regrasp, lift, hold, and transport gates continue.

This is an isolated research-gate change. It does not replace the current 8502
candidate and does not enter the scored submission path until the complete
physical route passes twice in clean official-runtime processes.

## Evidence motivating the change

Three extended OSC experiments remained collision-free and below the 0.03 m
grip-site drift limit, but none reached the 5-degree closure-axis gate. The
best terminal errors were 9.68 and 15.22 degrees. Both arm joint-1 values then
reached the official 1.570796 rad upper bound, and right arm joint 6 reached
its -1.413717 rad lower bound. More OSC iterations or gain tuning cannot move
through those joint limits, so the starting joint configuration must change.

## Considered implementations

### Independent per-arm IK

Solve six joints for one arm at a time while the other arm remains fixed. This
is simple, but independently acceptable endpoints can conflict when combined,
and it cannot trade small motion between arms to preserve their shared
clearance. It is retained only as a possible diagnostic fallback.

### Simultaneous 12-joint IK seed

Solve both six-joint arms in one bounded least-squares problem. The residual
contains both grip-site position errors, both directed closure-axis errors,
and a small normalized displacement penalty from the high-clearance start.
This is the selected implementation because both arms are evaluated in the
same simulator state and the resulting endpoint is coherent before path
validation.

### Stored hand-tuned joint pose

Record a pose found for the public L1 reset. This is deterministic but brittle
to scene and reset changes, provides no explicit grip-site constraint, and
would be hard to justify as an adaptive technical contribution. It is not
selected.

## Solver model

The optimization variable is the ordered vector of 12 official arm joints,
six right followed by six left. Each joint uses the model's official lower and
upper limit, moved inward by a small configurable angular margin. Invalid or
empty interior bounds reject the solve.

For each arm, the target closure-axis sign is selected once at the start so
the target is the closer of the two equivalent wall-normal directions. Keeping
this directed target fixed removes the discontinuity caused by changing signs
inside the optimizer.

The residual concatenates:

1. grip-site XYZ error divided by a position scale;
2. grip-site local-X closure-axis vector error divided by an angular scale;
3. joint displacement from the start, normalized by joint range and multiplied
   by a small regularization weight.

Every residual evaluation changes only robot arm qpos, calls `sim.forward()`,
and never steps physics or writes object qpos. A `finally` block restores the
complete starting arm vector and forwards the simulator after every solve,
including solver exceptions.

## Endpoint and path gates

The solver result is only a proposal. Before any recorded motion, the runner
sets the proposed robot joints, forwards the simulator, and verifies finite
state, both closure-axis errors, both grip-site position errors, official
joint-bound margins, and official judge collisions. It then restores the start
state.

An accepted proposal is applied by simultaneous linear interpolation of all
12 arm joints. Every waypoint is forwarded and checked using the official
`_navigation_collisions` boundary. Both grip-site positions are compared with
their original high-clearance positions at every waypoint; the maximum drift
must remain at most 0.03 m. Frames are recorded only for this validated robot
path.

Any non-finite state, bound violation, collision, drift excess, endpoint error,
or exception restores all 12 starting joints, forwards the simulator, and
synchronizes the OSC controller goals. Collision details and the failed
waypoint remain in the evidence record.

## OSC continuation

After a successful joint seed, controller goals are synchronized at the new
pose. The existing closed-loop OSC stage re-reads the grip-site positions and
rotations, then requires both closure-axis errors to remain at most 5 degrees
for five consecutive steps. Its existing collision and 0.03 m drift gates
remain active.

Joint-seed success is not physical-task success. It only permits the existing
table-assisted center approach, bilateral contact check, real gripper closure,
0.13 m measured lift, 20-step bilateral hold, and contact-preserving transport
stages to run.

## Evidence record

Each run records:

- official source commit, research commit, scene, seed, runtime and command;
- joint names, start joints, proposed joints, and final bound margins;
- solver status, evaluations, cost, exception, and residual components;
- initial and endpoint closure-axis errors and grip-site position errors;
- interpolated waypoint count, maximum grip-site drift, collision frames, and
  collision geom pairs;
- rollback status and controller synchronization status;
- all existing OSC, contact, lift, hold, shortcut-audit, and score evidence.

Records are written atomically. Large trajectories stay on the server and are
referenced locally by path and SHA-256 rather than committed.

## Scope and competition compliance

The first implementation changes only the research runner, its tests, and
traceable research documents. It does not modify any protected competition
file. Promotion, if earned, may change only
`src/robot_agent/skills/`, `src/robot_agent/workflows/`, and
`knowledge/robot_params.json`.

The route never writes object qpos, never creates a transport attachment, and
never bypasses MuJoCo contact dynamics. Direct robot-joint qpos interpolation
follows the legal pattern already present in the allowed competition skill and
is guarded at every waypoint by the official collision checker.

## Verification and promotion

Local verification requires focused helper and gate tests, the complete test
suite, syntax compilation, scored-path audit, workspace check, and a clean
diff check. Server verification uses the pinned official environment and one
GPU worker at a time.

The research route is discarded if it fails safety, convergence, regrasp,
lift, or hold. It is promoted to the allowed competition skill only after one
complete pass and one clean-process repeat with zero official collisions, zero
object-pose writes, zero attachment calls, and the original trajectory
retained for audit. The current 8502 candidate remains unchanged until then.
