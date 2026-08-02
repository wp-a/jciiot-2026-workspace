# L1 Reverse-Egress Vertical-Hold Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Discard vertical support as a sufficient fix for simultaneous base-drag on the
world `+x` route. Keep the bounded profile as a useful component for future arm
strokes, but do not promote this controller result as transport success. The
independent auditor classified it as `recovery`.

## Measured Result

- requested base segment: `0.49999983686408217 m`;
- measured base translation: `0.10101173267935715 m`;
- true audited object translation: `0.1242823835325652 m`;
- controller object-height loss: `0.0522483553400284 m`;
- minimum audited object lift after settling: `0.03615180894453629 m`;
- maximum object-to-gripper drift: `0.030096298069455695 m`;
- controller steps: `54`;
- terminal contacts: left `true`, right `true`;
- controller failure stage: `planar_grasp_drift`;
- collision frames: `0`;
- attachment calls / activations: `0 / 0`;
- legacy teleport activations: `0`;
- object-pose / robot-state writes: `0 / 0`;
- infrastructure error: none;
- auditor classification: `recovery`.

Vertical support changed the terminal condition from unilateral contact loss to
the fail-closed `0.03 m` planar-drift guard and preserved both contact booleans.
It nevertheless stopped after the same 54 control steps and about `0.101 m` of
base translation as the rejected inward-hold run. At controller termination the
two grippers had advanced about `0.105-0.109 m` in world x while the object had
advanced only about `0.084 m`. The box was therefore sliding longitudinally
inside a bilateral side-wall grasp.

This disproves the assumption that gravity-driven settling was the primary
failure. The limiting mechanism is simultaneous base/arm motion coupled through
a friction-only side grasp. Further squeeze, lift, or drift-threshold sweeps
would treat symptoms and are stopped.

## Evidence

- compact result SHA-256:
  `a2bc9589b9531a54a8157e71112760ac7b7a91779fccbca0d37aa3e6f6c6214f`;
- remote trajectory:
  `/home/user/jciiot-2026/results/l1-reverse-egress-vertical-20260802/seed0-xplus-vertical-p0p0004-k0p8-max0p003-0p50-trajectory.json`;
- remote trajectory SHA-256:
  `0978d8af70b9a1343d0292276cbfc704c4c0ea200bd8bc6c112202e00205bf1f`;
- ledger SHA-256:
  `70a5971bafb1502a135de5fde6d5b07d56c8cbaec8666d436cde57455a448398`;
- canonical TSV SHA-256:
  `340c034c67edb63ac4c9b8e299d2a5058bb7de480299876182502f671d5ec1d3`.

## Architecture Decision

Stop simultaneous base-drag for L1. The next structural experiment uses the
existing actuator-only inchworm controller: stationary-base dual-arm strokes
move the object, then compensated bounded base resets preserve the grippers in
world space, followed by physical reseating. Historical `+x` evidence already
shows one clean `0.079 m` macro and a clean two-stroke `0.149 m` transfer; the
current candidate additionally uses reset compensation gain `4.0` and four
physical reseat steps. The next run tests this current controller unchanged on
the `+x` direction with the same strict integrity gates.
