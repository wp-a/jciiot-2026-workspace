# Center-Grasp Physical Transport Design

## Scope and invariant

The repeatable L1 center opposed-wall grasp is frozen. Two independent runs
already proved official bilateral grasp through close, 0.134882 m lift, and a
20-step hold with zero collision, object writes, attachments, or drop.

This phase adds transport only. It remains in the research runner and does not
modify the official candidate, submission paths, or current 8502 service.

## Options considered

1. **Reuse `run_physical_transport`** (selected). It uses controller-stepped
   paired arm/base phases and checks official bilateral grasp, object height,
   and collision after every substep. It is already tested in the allowed
   transport skill and gives the strongest comparison with earlier cantilever
   carry failures.
2. Move the base while holding arm joints. Earlier cantilever experiments lost
   contact after about 0.1 m because moving-base controller goals and world
   gripper positions diverged.
3. Write a new object-relative MPC. That may become useful later, but is not
   justified before testing the existing physical helper with the much stronger
   center grasp.

## Runtime sequence

After `hold_center_grasp` completes 20 bilateral official grasp steps:

1. read current base pose, object position, gripper positions, and grasp status;
2. define a straight carry direction from the base toward the grasped object;
3. set one base waypoint at the requested distance along that direction;
4. call `run_physical_transport` with closed grippers, current yaw, a minimum
   object height 0.10 m above the table reference, 0.01 m waypoint tolerance,
   conservative 0.04 m/s linear speed, 0.01 m/s command slew, and sufficient
   step budget;
5. append the helper's result as `transport_center_grasp` evidence;
6. reject immediately on contact loss, object height loss, official collision,
   timeout, or failed height recovery.

The first experiment requests 0.20 m. If it preserves every hard gate, only the
distance changes to 0.50 m, then 1.05 m. No speed and distance change may occur
in the same experiment.

## Honest route-specific gate

The older cradle gate requires non-finger robot-link support and therefore
cannot represent a verified gripper carry. A separate diagnostic gate requires:

- repeatable physical grasp and at least 0.13 m measured lift;
- at least 20 consecutive bilateral official grasp hold steps;
- `run_physical_transport` success with no contact loss;
- measured object planar translation greater than 1.00 m;
- zero official collision frames, object-pose writes, and attachments;
- no drop and no infrastructure error.

The gate uses object motion, not base motion. Passing it proves the scoring
rule's "successful leave" condition on the diagnostic trajectory, but not the
official destination condition or official score. Those require later
integration and the unchanged scorer.

## Verification

Pure tests cover carry direction and route-specific failure conditions. Source
sequence tests prove transport runs only after the 20-step closed hold. Existing
tests continue to prove bracket readiness cannot set grasp success. The full
suite, scored-path audit, workspace checks, immutable official commit, runner
hash, and unchanged 8502 PID are mandatory before every server series.
