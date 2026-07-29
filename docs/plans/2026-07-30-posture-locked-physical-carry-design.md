# Posture-Locked Physical Carry Design

## Context

The official L1 grasp produces bilateral robot-object contact and lifts the
near container by about 0.204 m without attachment calls or task-object pose
writes. Ordinary base control then compensates the end effectors in world
coordinates, so the base moves while the container remains behind or slips.

The open-fork undercut route is rejected. The official L1 scene creates an
invisible, static world geom named `line_5_container_h01_near_support` below
the container. Its half-size is `0.340 x 0.240 x 0.010 m`, wider than the
container bottom half-size of approximately `0.300 x 0.200 m`. A fork cannot
enter below the load without penetrating this static support.

## Decision

Preserve the official physical grasp and move the robot with the official
upper-body posture-locking navigation mechanism. The object must move only
through MuJoCo contact dynamics. This route is tested first over 0.10 m and is
not integrated into the candidate until it passes every hard gate.

Table-supported pushing remains a recovery method. Policy training remains
deferred until the deterministic controller yields successful Tiago
trajectories.

## Experiment Architecture

Add an opt-in posture-locked carry mode to the external L1 research runner:

1. Load the immutable official L1 candidate and navigate to `input_5`.
2. Execute the official bilateral grasp and verify its reported lift and
   physical contacts.
3. Capture base, object, end-effector, and object-to-gripper transforms.
4. Build one 0.10 m base waypoint along a requested world direction.
5. Call the official posture-locked `backend.follow_path` implementation.
6. Record object progress, lift, bilateral contact, grasp-frame drift,
   collision state, attachment calls, and object-pose writes.
7. Reject the run on any hard-gate failure and preserve the full trajectory.

The research runner may import existing functions from the protected core but
must not modify protected files. The formal candidate may change only
`src/robot_agent/skills/`, `src/robot_agent/workflows/`, and
`knowledge/robot_params.json`.

## Short-Distance Gates

The 0.10 m trial passes only when all conditions hold:

- projected object progress is at least 0.08 m;
- lateral object drift is at most 0.03 m;
- object-to-gripper planar drift is at most 0.03 m;
- final object height remains at least 0.10 m above its source-table height;
- bilateral physical object contact is observed at the terminal state;
- official collision frames are zero;
- attachment calls and task-object pose writes are zero;
- no infrastructure error occurs.

The official scorer is not invoked and no score is claimed at this stage.

## Scale-Up Sequence

After the 0.10 m gate passes:

1. repeat at 0.25 m with the same direction and gates;
2. execute a two-segment path around the nearest production-line proxy;
3. plan a complete collision-free path from `input_5` to `output_4` using the
   official semantic map and measured grasp footprint;
4. release physically at the target and require final planar distance below
   0.8 m;
5. run the unmodified official app scorer and require L1 `10/10`;
6. repeat across seeds and bounded pose/controller perturbations before
   generalizing the controller to L2-L5.

## Failure Handling

- If the object moves less than the base, classify the run as grasp slip.
- If height falls below the threshold, stop before another waypoint.
- If bilateral contact is lost, stop and preserve the first failing frame.
- If a collision is reported, stop immediately and record collision pairs.
- If attachment or object-pose writes are observed, invalidate the run rather
  than treating it as a physical result.

## Verification

Pure gate and target-construction logic is test-driven. Runtime verification
uses the official pinned environment, a full trajectory, compact JSON evidence,
and hash equality between the local and remote research runner. Candidate
integration additionally requires the allowlist audit and the unmodified 8502
official scoring path.
