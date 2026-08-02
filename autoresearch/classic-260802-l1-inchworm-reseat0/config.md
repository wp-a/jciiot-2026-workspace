# L1 Inchworm Reseat-Zero Experiment

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Diagnosis

The attachment-free current controller completed both 30-step compensated base
resets with bilateral contact. In cycle 2, the first following center-directed
reseat step caused both contacts to be lost. Effective object progress stopped
at `0.128528 m`. The reseat helper moves each gripper toward the object center
in full XY, so it can introduce a longitudinal correction after a +X stroke.

## Falsifiable Hypothesis

Disabling the four post-reset reseat steps will preserve bilateral contact after
cycle 2 and exceed the best historical clean inchworm progress of `0.149215 m`.
All transport motion must remain actuator-only, collision-free, and free of
attachment, teleport, or object-pose writes.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- frozen runner:
  `/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py`;
- frozen runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- attachment-free parameter wrapper SHA-256:
  `226e5c5cef40541b1b06846c7f287b9aca66839cb431c83c7653d03e98adf3fa`;
- public L1, seed `0`, near container;
- travel direction: world `+x`;
- requested controller progress: `0.50 m`;
- arm stroke: `0.08 m`;
- vertical feedforward: `0.015 m`;
- height gain: `0.75`;
- base reset: `0.06 m` at maximum `0.04 m/s`;
- reset compensation gain: `4.0`;
- reset inward feedforward: `0.0`;
- maximum lateral drift: `0.03 m`;
- running 8502/8503 services unchanged.

## Single Intervention

```text
reseat_steps: 4 -> 0
```

The attachment-free research wrapper will inject this existing configuration
field before the frozen runner creates `InchwormCarryConfig`. It may not change
the candidate source, runner, simulation state, action stream, or integrity
guards.

## Keep / Discard

Keep only if all conditions hold:

- `hold_probe.object_progress_m > 0.149215`;
- terminal right and left contacts are both true;
- controller minimum object lift remains at least `0.13 m` above the support
  reference;
- collision frames, attachment activation/active flags, object-pose writes,
  teleport/state-write counters are all zero.

Full structural success additionally requires at least `0.50 m` measured
object progress. Otherwise discard the intervention, record the exact failing
macro, and do not add the run to successful transport training data.
