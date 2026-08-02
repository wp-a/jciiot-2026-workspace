# L1 Reverse-Egress Inward-Hold Experiment

[autoresearch] mode: classic

Date: 2026-08-02 (Asia/Shanghai)

## Incumbent

- evidence: `autoresearch/classic-260802-l1-reverse-egress/`;
- explicit waypoint: `(8.500015, 4.599998)`;
- true object translation: `0.12314848253481103 m`;
- base translation: `0.10701284662200282 m`;
- failure stage: terminal right-contact loss followed by object settling;
- minimum lift: `0.026332585685933996 m`;
- collision and integrity violations: zero.

## Falsifiable Hypothesis

Changing only `planar_hold_inward_feedforward` from `0.0` to `0.0005 m` per
control step will keep both grippers seated against the container during `+x`
reverse egress. It should improve true object translation beyond
`0.1231484825 m`, preserve minimum lift at or above `0.13 m`, retain bilateral
contact, and introduce no collision or integrity regression.

## Frozen Inputs

- official commit:
  `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate:
  `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate grasp SHA-256:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate transport SHA-256:
  `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner SHA-256:
  `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- canonical auditor SHA-256:
  `6ce0a9b350ad94521a9313f30ae94bedc27700599e9a4bd64ad399f3a467d1a8`;
- public L1, seed `0`, near container;
- actuator-only route, speed `0.04 m/s`;
- explicit waypoint `(8.500015, 4.599998)`;
- original `0.03 m` internal drift guard;
- arm-motion feedforward `0.0`;
- planar recovery disabled;
- heading alignment disabled;
- running 8502/8503 services unchanged.

## Single Change

```text
--physical-carry-inward-feedforward-m 0.0005
```

## Keep / Discard

Retain as a better recovery controller only if true object translation exceeds
the incumbent, minimum lift is at least `0.13 m`, bilateral terminal contact is
restored, and every collision/integrity counter stays zero. Admit as
`transport_success` only if the independent auditor additionally measures at
least `0.50 m` true object translation and all strict gates pass.
