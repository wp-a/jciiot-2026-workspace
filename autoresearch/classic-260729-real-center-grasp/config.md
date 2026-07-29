# L1 real center-grasp gate

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 3 valid single-variable experiments
- Research implementation commit: `db3425f`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene and seed: public L1, seed 0
- Remote result root: `/home/user/jciiot-2026/results/l1-real-center-grasp-20260729`
- Current 8502 candidate: unchanged

## Incumbent and single change

Use the accepted arm-only 24-node, 240-waypoint seed with 0.0185 m position
scale and 0.01 rad arm margin. Change only the orientation-stage entry
tolerance from 5 to 10 degrees so the already accepted seed can proceed to the
physical center-grasp stages.

This tolerance is not an official score claim. The existing report-level
5-degree orientation gate remains visible and is expected to fail. Retention is
decided only by downstream physical evidence:

1. zero official collisions, object writes, and attachment calls;
2. opposed wall contact with both arms;
3. gradual physical gripper closure and three consecutive bilateral grasp
   contact steps;
4. measured object lift of at least 0.13 m;
5. at least 20 closed-gripper hold steps with bilateral contact.

No lift is attempted unless the explicit `close_center_grasp` stage succeeds.

## Iteration ledger

- Iteration 1 raised only the runtime orientation entry tolerance to 10 degrees.
  It passed the collision-free wrist seed and orientation stages but stopped at
  `translate_to_center` because the left arm remained about 48 mm short.
- Iteration 2 added only 0.10 m of collision-checked base advance. It completed
  the translation and reached opposed walls with zero official collision, but
  the contacts were `robot0_arm_5*_collision`, not bilateral fingerpad grasps.
  Closing for 80 steps therefore failed the three-frame physical grasp gate.
- Iteration 3 keeps the 0.10 m base advance and changes only
  `regrasp_center_shift_m` from 0.24 m to 0.0 m. This tests whether descending
  over the near wall avoids premature forearm contact and lets the fingerpads
  establish a real grasp.
