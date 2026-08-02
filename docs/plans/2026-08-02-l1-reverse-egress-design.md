# L1 Reverse-Egress Physical Carry Design

## Context

The best clean L1 physical route moved the object `0.265401 m` with continuous
bilateral contact, no collision, no attachment, and no state writes. Raising
the internal grasp-drift guard increased motion to `0.275275 m` but produced
two collision frames between `robot0_torso_fixed_collision_box_1` and
`scene_aabb_proxy_production_line_5`.

The failed route moved the base from approximately `(8.000, 4.600)` toward the
grasped object and production line, ending at `(7.669, 4.607)`. Its requested
waypoint was `(7.500, 4.610)`. This establishes that the route direction, not
the drift guard, drove the robot into the source station.

## Decision

Test a straight reverse-egress segment along world `+x`. The base starts near
`(8.000, 4.600)` and the first requested waypoint is approximately
`(8.500, 4.600)`. This pulls the physically grasped container away from the
production line instead of pushing it deeper into the station.

This is a one-variable experiment. Keep the incumbent candidate, public L1
scene, seed 0, actuator-only physical carry, `0.04 m/s` base speed, zero arm
feedforward, zero inward feedforward, disabled planar recovery, and the
original `0.03 m` internal grasp-drift guard.

## Alternatives Rejected For This Iteration

- Lateral peel before retreat adds a new shear direction and does not isolate
  the diagnosed route-direction error.
- In-place heading alignment introduces pivot compensation and rotation before
  safe source egress is established.
- Relaxing the drift guard already improved distance at the cost of collision
  and remains rejected.

## Evidence Contract

The run is a physical-transport success only if all conditions hold:

- measured planar object translation is at least `0.50 m`;
- minimum object lift is at least `0.13 m`;
- bilateral object contact is continuous and the object is not dropped;
- maximum object-to-gripper drift is at most `0.05 m`;
- collision frames are zero;
- attachment calls and activations are zero;
- legacy teleport, object-pose writes, and robot-state writes are zero;
- no infrastructure error occurs.

The result must be classified by the fail-closed physical-data auditor. Runner
`accepted` and base translation are not success evidence.

## Iteration Ladder

1. Run one seed-0 `+x`, `0.50 m` diagnostic.
2. If it passes, extend the same direction to the official `>1.0 m` source
   departure threshold without changing the controller.
3. If `0.50 m` fails, inspect the first physical boundary and change only the
   parameter responsible for that boundary.
4. After a clean source exit, construct an explicit, collision-checked route
   through the free corridor toward `output_4` at `(-0.166, -7.290)`.
5. Attempt physical placement and run the unchanged official scorer only after
   route integrity passes.

## Data Policy

A passing trajectory enters the `transport_success` stratum. A clean physical
contact or drift failure may enter the separate recovery stratum. Collision,
attachment, state mutation, or infrastructure failures are rejected. Dataset
scale begins only after the first complete successful transport exists.

## Service Boundary

Experiments use isolated server tool and result directories. The running 8502
and 8503 services and the current submission candidate are not restarted or
modified during this diagnostic.
