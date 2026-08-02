# L1 Reverse-Egress Vertical-Hold Experiment

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Incumbent

- evidence: `autoresearch/classic-260802-l1-reverse-egress/`;
- explicit waypoint: `(8.500015, 4.599998)`;
- true object translation: `0.12314848253481103 m`;
- controller steps: `57`;
- controller object-height loss: approximately `0.073781 m`;
- terminal failure: right-contact loss;
- collision and integrity violations: zero.

The rejected inward-hold ablation lost the same right contact after 54 steps
and about `0.069053 m` of controller-observed height loss. This isolates
vertical load support, rather than additional planar squeeze, as the next
causal variable.

## Working Pattern

On the same candidate, grasp, seed, and actuator-only controller, the frozen
vertical profile `(feedforward=0.0004, gain=0.8, max_delta=0.003)` completed a
forward `0.25 m` probe in 256 steps with bilateral terminal contact, only
`0.002575 m` controller-observed height loss, zero collision, zero attachment,
and zero state writes. The canonical auditor retained that result as recovery
only because true object translation was `0.196812 m`, below its `0.50 m`
dataset threshold. Source result SHA-256:
`1fcc4deb67ce23408ffb76fb622c829eefee562540aacda8ccda367a9ac5aeb9`.

## Falsifiable Hypothesis

Applying that previously demonstrated bounded vertical-support profile to the
zero-collision world `+x` reverse-egress route will prevent gravity-driven
height loss, retain both physical contacts, and complete the requested
`0.50 m` base route without attachment, collision, teleport, or state writes.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- research runner:
  `/home/user/jciiot-2026/tools/full-physical-l1-332a60b/run_l1_cradle_gate.py`;
- research runner SHA-256:
  `d894aab484c015fd0859abe96d13c32e5dd97c23b0392f8ec359df67677393e6`;
- canonical auditor SHA-256:
  `6ce0a9b350ad94521a9313f30ae94bedc27700599e9a4bd64ad399f3a467d1a8`;
- public L1, seed `0`, near container;
- actuator-only route, speed `0.04 m/s`;
- explicit waypoint `(8.500015, 4.599998)`;
- original `0.03 m` internal drift guard;
- arm-motion feedforward `0.0`;
- planar inward feedforward `0.0`;
- planar recovery disabled;
- heading alignment disabled;
- running 8502/8503 services unchanged.

The research runner differs from the preceding frozen runner only by validating
and forwarding the already-existing vertical-hold parameters through its probe
functions and CLI. With all three parameters at zero, behavior is unchanged.

## Single Controller Intervention

```text
--physical-carry-vertical-feedforward-m 0.0004
--physical-carry-vertical-gain 0.8
--physical-carry-max-vertical-delta-m 0.003
```

These three values form one indivisible vertical-support profile already tested
in the working-pattern probe; no sweep will be performed.

## Keep / Discard

Retain as a better recovery controller only if it exceeds the incumbent true
object translation, preserves at least `0.13 m` minimum lift and bilateral
terminal contact, and every collision/integrity counter remains zero. Admit as
`transport_success` only if the independent auditor measures at least `0.50 m`
true object translation and all strict gates pass. Otherwise discard the
profile for this reverse-egress posture and return to root-cause analysis.
