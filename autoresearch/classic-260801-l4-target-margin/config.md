# L4 target-margin optimization

- Date: 2026-08-01
- Mode: classic
- Objective: retain the strict L4 acceptance gate while reducing the nominal
  final target distance from 0.747841 m to less than 0.40 m.
- Baseline: candidate `final-100-l3-release-hold-20260801`, task 4 seed
  `20260810`, score 25/25, zero collision frames, one verified bilateral grasp
  and lift, final target distance 0.747841 m.
- Root cause hypothesis: collision-aware navigation stops the base at
  `[5.8, -8.2]`, but preserves the source grasp yaw. The attached container is
  therefore tangent to the target boundary. Rotating the stationary base so
  the measured base-relative container offset points at the target center has
  a predicted closest distance of approximately 0.302 m.
- Controlled change: L4 only; calculate the alignment yaw from measured base,
  object and target coordinates, rotate in place through the existing bounded
  collision-checked yaw controller while invoking the official attachment sync
  at each turn step, then apply the existing read-only target verification.
- Promotion gate: 25/25, zero collision frames, one verified grasp and lift,
  workflow success, final distance below 0.40 m in nominal and below 0.60 m in
  the deterministic small perturbation run.
- Safety boundary: no direct object qpos write, no attachment relative-state
  mutation, no protected-file modification, and no change to L1, L2, L3 or L5
  routing. Object following during direct base motion is delegated only to the
  official `sync_transport_attachment` helper used by official navigation.
