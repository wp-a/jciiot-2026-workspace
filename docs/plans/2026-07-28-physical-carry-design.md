# Physical Carry And Placement Design

## Objective

Replace the current hybrid transport path with a fully physical manipulation
path. The robot must grasp with both grippers, lift through MuJoCo contacts,
carry through controller actions, lower onto the destination, and release by
opening the grippers. The scored path must never write a task object's pose or
activate the official transport attachment.

## Competition Boundaries

Submission changes are limited to:

- `JCIIOT/src/robot_agent/skills/`
- `JCIIOT/src/robot_agent/workflows/`
- `JCIIOT/knowledge/robot_params.json`

The implementation will not modify `core/`, `environments/`, `app.py`, or
`knowledge/task_config.json`. Tests, plans, experiment ledgers, and local
evidence remain outside the submission overlay.

## Root Cause

The incumbent performs a physical two-gripper close and lift, then calls
`capture_transport_attachment`. Direct navigation changes the base pose and
the attachment helper synchronizes the object's free joint. Final placement
also changes attachment-relative translation and rotation before release.
Consequently, the public scorer can award full points while the robot-view
replay shows an object moving without physical support.

The official UI constructs the backend with `drive_mode="direct"`. A remote
probe on the locked simulator showed why: 100 consecutive full-scale base
actions moved the Tiago only about 5.6 mm, while a physically grasped L1 box
slipped by 25 mm after 83 lower-amplitude steps. Controller-only travel across
a factory scene would therefore require roughly one hundred thousand steps and
cannot retain the official grasp.

The carry path consequently uses the same incremental direct-base convention
as the unmodified official UI, but never applies it to a task object. The base
joint is advanced by a bounded world step, then normal arm and gripper actions
advance MuJoCo exactly once. The box remains a free body and can move only
through bilateral gripper contact and simulated forces.

## Selected Architecture

### Grasp

Keep the existing OSC approach, bilateral contact confirmation, and physical
lift. Remove transport attachment capture and transport stow interpolation.
A successful grasp ends with both grippers closed, both contacts present, and
the object above its pre-grasp height.

### Transport

Add `competition_transport.py` under the allowed skills directory. It accepts
an A* path, the held object name, and a small explicit configuration.

At every control step it will:

1. Read the current base pose and next path waypoint.
2. Convert the desired world-frame planar velocity into the base frame and a
   bounded direct-base step using the official 20 Hz convention.
3. Hold the grasp yaw and apply a bounded upward OSC correction from measured
   object-height error, resisting gravity-driven slip.
4. Send the OSC corrections, closed-gripper actions, absolute hold targets for
   torso and head, and a zero wheel command after the base step.
5. Advance physics exactly once with `env.step(action)` so the free object is
   carried only by contact.
6. Record the trajectory and verify collision state, bilateral grasp contact,
   and minimum object lift.

Any lost gripper contact, collision, excessive object drop, missing controller,
or step-budget exhaustion fails transport immediately. There is no kinematic
fallback.

The base remains holonomic and preserves the grasp orientation. This avoids
large rotations of the held object and reduces inertial load. Step size,
acceleration, and vertical compensation remain bounded and are tuned only from
saved L1 trajectories.

### Placement

At the destination, the base stops while both grippers remain closed. The
controller lowers both end effectors together in small world-Z increments.
Placement readiness requires physical evidence: the object has descended from
its carry height and either reaches the expected support height or stops moving
downward for several commanded steps while remaining bilaterally grasped.

The controller then opens both grippers through normal actions and advances
physics for a bounded settling window. Success requires no collision, no
remaining bilateral grasp, and final planar distance below the official
0.8-metre arrival threshold. L5 slot offsets remain deterministic but must be
reached by base and arm actions, not by object-pose edits.

## Data Flow

The existing workflow still plans source and destination routes from the
semantic map. Empty-handed navigation may continue to use the official direct
backend. Once grasp succeeds, carrying navigation is routed exclusively to the
new physical controller. Placement consumes the physical hold state produced
by grasp and transport; it never reads or mutates attachment state.

Trajectory events will mark physical transport start/end, contact-loss or
collision failures, physical descent, release, and settle outcome. These
events support later report generation without changing the official scorer.

## Tests And Acceptance

Local tests must prove:

- the submission contains no transport-attachment import or call;
- the submission contains no write to object pose or object free-joint qpos;
- the grasp sequence does not attach or stow after lift;
- physical actions include base, both arms, both grippers, torso, and head;
- contact loss and collision fail immediately;
- placement opens the grippers only after a physical descent condition;
- workflow carrying routes through the physical controller;
- all overlay paths remain within the official allowlist.

Remote acceptance is staged:

1. Run L1 through the candidate experiment entrypoint and inspect contact,
   object-height, collision, and score evidence.
2. Render both birdview and `robot0_robotview`; reject any result that visually
   shows unsupported object motion.
3. Run L1 through the unmodified 8502 `app.py` page by clicking `Execute`.
4. Only after L1 passes, run L2-L4 and then the three-object L5 task.
5. Preserve original scored trajectories, scores, logs, source hashes, and
   multi-view GIFs in a timestamped local evidence directory.

Public-scene scores will be reported only as public fixed-scene results, not as
hidden-test or BienData scores.
