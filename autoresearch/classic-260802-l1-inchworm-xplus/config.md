# L1 Current-Controller Inchworm +X Experiment

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Architecture Reset

Three local modifications to simultaneous base-drag did not solve L1:

- reverse egress removed collision but lost right contact after 57 steps;
- planar inward hold lost right contact after 54 steps;
- bounded vertical hold retained bilateral contact but hit the `0.03 m` planar
  drift guard after 54 steps.

In the last run, the grippers advanced approximately `0.105-0.109 m` while the
object advanced only approximately `0.084 m`. This establishes longitudinal
slip in a side-wall grasp and ends further squeeze/lift/threshold tuning of
simultaneous base-drag.

## Existing Physical Evidence

The actuator-only inchworm controller separates motion into stationary-base
dual-arm strokes and compensated base resets. Historical `+x` end-grasp runs
on the same L1 scene established:

- one `0.08 m` stroke plus one `0.06 m` reset: `0.079330 m` measured object
  progress, bilateral terminal contact, zero collision and zero integrity
  violations;
- two strokes with only the first reset: `0.149215 m` measured object progress,
  bilateral terminal contact, zero collision and zero integrity violations;
- the older long run failed after its second reset with unilateral contact.

The current frozen candidate adds planar reset compensation gain `4.0` and four
physical reseat steps of `0.002 m` after each completed reset. This experiment
tests that current architecture on `+x`; it does not tune those defaults.

## Falsifiable Hypothesis

The current compensated-reset and reseat implementation will prevent the
second-reset contact loss seen in the older `+x` controller and complete at
least `0.50 m` measured physical object progress while maintaining the load,
zero collision, and zero attachment/teleport/state writes.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner:
  `/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py`;
- runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- canonical auditor SHA-256:
  `6ce0a9b350ad94521a9313f30ae94bedc27700599e9a4bd64ad399f3a467d1a8`;
- public L1, seed `0`, near container;
- travel direction: world `+x`;
- travel distance: `0.50 m`;
- stroke distance: `0.08 m`;
- stroke vertical feedforward: `0.015 m`;
- stroke height gain: `0.75`;
- reset distance: `0.06 m`;
- reset maximum linear speed: `0.04 m/s`;
- reset compensation gain: `4.0`;
- reset inward feedforward: `0.0`;
- reseat: 4 steps at `0.002 m`;
- maximum lateral drift: `0.03 m`;
- running 8502/8503 services unchanged.

## Single Structural Intervention

Relative to the rejected simultaneous base-drag architecture:

```text
--end-grasp-inchworm-distance-m 0.50
--end-grasp-inchworm-world-direction-x 1.0
--end-grasp-inchworm-world-direction-y 0.0
```

All inchworm controller parameters remain at the frozen candidate defaults.

## Keep / Discard

Keep only if measured `hold_probe.object_progress_m` exceeds the best clean
inchworm result (`0.149215 m`), minimum object lift remains at least `0.13 m`,
terminal contact is bilateral, and all collision/integrity counters are zero.
Full structural success requires at least `0.50 m` measured object progress.
The outer compact record and trajectory will then be converted to the canonical
full-physical audit schema before any training-data or score claim. Otherwise
record the exact failing macro and revise the reset architecture rather than
sweeping unrelated grasp parameters.
