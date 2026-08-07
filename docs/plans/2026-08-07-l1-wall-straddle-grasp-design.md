# L1 Wall-Straddle Grasp and Physical Carry Design

Date: 2026-08-07 (Asia/Shanghai)

## Goal

Replace the L1 same-side friction pinch with a wall-straddling grasp that gives
each Tiago gripper direct actuator-controlled clamping on one thin container
wall. Prove the new contact topology with a short physical lift and hold before
attempting a 0.50 m attachment-free carry.

This is a research gate, not a score claim. The existing floor-push incumbent
and the current L4 work remain unchanged until the new route passes every gate.

## Evidence Boundary

The official L1 collision model represents the container with a bottom plate
and four solid walls. Each wall has a half-thickness of `0.007 m`, so the full
collision thickness is approximately `0.014 m`. The visual side handles are not
collision openings and cannot be used as physical insertion points.

Prior experiments close the following branches:

- outside-to-outside arm squeezing can lift statically but slips during carry;
- the fixed source support is larger than the movable bottom and blocks a real
  table-edge undercut;
- one-arm airborne regrasp cannot resist load torque;
- simultaneous closed-gripper descent drags the tote down its side walls;
- wheel-action navigation is ineffective in the locked simulator.

The untested structural change is to place one finger inside and one finger
outside a container wall, then close that individual gripper around the wall.
The left and right grippers clamp opposite walls. This changes the normal force
source from opposed arm motion to the two gripper actuators.

## Alternatives

1. **Wall-straddle grasp (selected):** lowest-cost new contact topology and a
   direct test of actuator-controlled wall clamping.
2. **Supported tilt and bottom regrasp:** mechanically promising but requires a
   longer pre-positioning and tilt sequence with higher collision risk.
3. **Residual control on the current side pinch:** smallest code change, but it
   preserves the topology that has already plateaued at short translation.

## Architecture

### Geometry planner

The planner derives the container pose and collision wall extents from the live
scene. It chooses the accessible opposite wall pair and returns four staged
targets per arm:

1. open-gripper clearance above the wall;
2. vertical descent with the wall centered between the two fingers;
3. bounded close while preserving end-effector pose;
4. symmetric proof lift.

No target may be accepted from the visual mesh alone. Geometry checks use the
official collision wall and current object transform.

### Contact gate

The gate classifies finger-to-object contacts by arm and finger side. A valid
wall clamp requires both fingers of each participating gripper to contact the
same object wall with opposing contact normals for consecutive frames. A plain
outside fingerpad contact does not pass.

### Controller

The existing OSC action interface executes bounded Cartesian stages. During
close, lift, hold, and carry, the controller continuously checks:

- wall-clamp contact continuity;
- object height relative to the pre-lift pose;
- object-to-gripper planar drift;
- judge collision state;
- integrity counters for attachment and state writes.

The first implementation contains no learned policy. Learning begins only
after a valid physical success trajectory exists.

### Experiment runner

The runner operates on L1, seed 0, and uses two sequential gates:

- `G1`: wall-straddle acquisition, `0.05 m` lift, and 100-step hold;
- `G2`: unchanged `G1` grasp followed by `0.50 m` straight physical carry.

`G2` is forbidden until `G1` passes. Full-route scoring is out of scope for the
first experiment.

## Safety and Integrity Gates

A result is accepted only when all fields are independently measured:

- attachment calls and activations: `0`;
- task-object pose writes: `0`;
- legacy teleport activations: `0`;
- judge collision frames: `0`;
- per-arm opposing dual-finger wall contact: true;
- proof lift: at least `0.05 m`;
- hold duration: at least 100 control steps;
- maximum object-to-gripper planar drift: at most `0.02 m`;
- carry translation for `G2`: at least `0.50 m`.

The runner stops immediately on collision, unilateral clamp loss, height loss,
excessive drift, integrity violation, or timeout. Failed trajectories remain
diagnostic evidence and never enter successful training data.

## Allowed Change Surface

Competition code changes are limited to:

- `submission/JCIIOT/src/robot_agent/skills/`;
- `submission/JCIIOT/src/robot_agent/workflows/`;
- `submission/JCIIOT/knowledge/robot_params.json` only if a measured parameter
  must be promoted after the research gate.

The protected core, environments, `app.py`, and `task_config.json` remain
unchanged. Research-only runners and audits live outside the submission overlay.

## Testing

Unit tests are written before implementation for:

- selecting an accessible opposite wall pair from live collision geometry;
- producing straddle targets with the wall between open fingers;
- rejecting same-side or single-finger contact;
- accepting consecutive opposing dual-finger contact;
- fail-closed behavior on drift, height loss, collision, or integrity evidence;
- preventing `G2` execution before `G1` passes.

The server experiment records a manifest, compact result, full trajectory,
event ledger, hashes, and a first-person GIF only after a successful gate.

## Decision Rule

Keep the route only if `G1` passes without relaxing any integrity or contact
threshold. If `G1` fails because the gripper cannot geometrically straddle the
wall, stop this branch and move to supported tilt; do not sweep gains, seeds, or
timeouts against an invalid contact topology. If `G1` passes, freeze its grasp
parameters and test only the `0.50 m` carry in `G2`.
