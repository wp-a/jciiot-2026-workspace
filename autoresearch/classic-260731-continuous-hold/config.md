# Continuous hold transport iteration

- Date: 2026-07-31
- Mode: classic
- Objective: keep the task-1 tote between both grippers throughout official
  attachment transport, with no legacy teleport to the mobile-base center.
- Observed failure: after frame 741, the 10/10 trajectory repeatedly recorded
  the tote at the base XY while both arm joint vectors remained fixed.
- Root cause: the workflow enabled both the official relative transport
  attachment and the backend's legacy `_held_crate_name` path. The latter calls
  `_update_held_crate_position()` from the navigation frame callback and
  overwrites the tote freejoint at the base center.
- Controlled change: keep the legacy held-crate handle disabled during
  navigation and set it only immediately before `place_object_physics()`.
- Primary runtime metrics:
  - official task-1 score is `10/10`;
  - collision frames are `0`;
  - no held transport frame has object-to-base planar distance below `0.80 m`;
  - object-to-base relative XY range before lowering is at most `0.02 m`;
  - first-person replay shows the tote continuously between the grippers.
- Safety invariants: protected files unchanged; no submission-owned object qpos
  write; physical bilateral grasp and lift occur before attachment; one attempt
  per official seed.

