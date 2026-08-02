# Next Physical Carry Experiment

Date: 2026-08-02 (Asia/Shanghai)

## Mode

Classic, one-variable, one-seed diagnostic. The live 8502/8503 services and the
submission candidate remain unchanged.

## Incumbent

- Record: `s0-actuator-g0p35-vertical-0p50-seed0.json`
- Candidate provenance was not stored in the incumbent result. The initial
  assumption that it was `robust-l1-inchworm-eb48310` was disproved before
  transport because that candidate does not export
  `PostureLockedPhysicalCarryDriver`.
- The only server candidate containing the runner's required
  `base_motion_arm_feedforward_gain` interface is
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`. Its seed-0
  post-grasp object state, base state, gripper states, and generated waypoint
  exactly reproduce the incumbent record, so it was used for the valid run.
- Source runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`
- True object translation: `0.26540101992172577 m`
- Minimum lift: `0.19783002518930526 m`
- Maximum object-to-gripper drift: `0.03153650001699935 m`
- Continuous bilateral contact, no drop, zero collision, and zero integrity
  violations.

## Falsifiable Hypothesis

The actuator-only controller stopped early because its internal
`max_object_gripper_drift_m` protection was `0.03 m`, while the load remained
lifted and in continuous bilateral contact. Raising only that internal guard to
`0.04 m` will improve true planar object translation beyond `0.265401 m`
without collision, drop, attachment, object-pose write, or robot-state write.

## Frozen Parameters

- official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- public L1, seed `0`, near container;
- straight requested base segment: `0.50 m`;
- actuator-only physical transport;
- maximum base speed: `0.04 m/s`;
- arm and inward feedforward: `0.0`;
- planar recovery: disabled;
- grasp, lift, candidate code, scene, and every other runner parameter unchanged.

The candidate source hashes used for the valid run were:

- `competition_grasp.py`:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- `competition_transport.py`:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`.

The patched runner SHA-256 was
`a19ee5322c734c1566cda64dcf9863dd62e583e54d49fdff93b955102f145d61`.
Its only source change from the incumbent runner was the internal full-route
object-to-gripper drift guard from `0.03 m` to `0.04 m`.

## Keep / Discard Metric

Keep the change only if true object translation is greater than `0.265401 m`
and all strict integrity and safety fields remain clean. Full dataset success
still requires at least `0.50 m` object translation, `0.13 m` minimum lift,
continuous bilateral contact, at most `0.05 m` drift, and zero integrity or
safety violation. The final gate is not relaxed.

## Infrastructure Exclusions

Two launches were excluded before hypothesis evaluation:

1. `robust-l1-inchworm-eb48310` lacked
   `PostureLockedPhysicalCarryDriver`. Excluded result:
   `/home/user/jciiot-2026/results/physical-carry-drift04-20260802/seed0-drift04.json`,
   SHA-256
   `16521119e45b9b516a6c934426f781e360a039cf7eaf2074d95e595759e8f3e0`.
2. `robust-l1-fullroute-43ba091` had an older `PhysicalCarryConfig` without
   `base_motion_arm_feedforward_gain`. Excluded result:
   `/home/user/jciiot-2026/results/physical-carry-drift04-20260802/seed0-drift04-compatible.json`,
   SHA-256
   `35703ff88e1788052152b2a93424dc8078b0e403db3e167bfaf9812c0109fb7b`.

Both runs failed before base transport and are not controller or dataset
samples.
