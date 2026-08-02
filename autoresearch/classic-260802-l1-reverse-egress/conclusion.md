# L1 Reverse-Egress Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Keep the `+x` reverse-egress direction as the safe route direction, but do not
promote this controller result as transport success. The independent auditor
classified it as `recovery`.

## Measured Result

- requested base segment: `0.49999983686408217 m`;
- measured base translation: `0.10701284662200282 m`;
- true audited object translation: `0.12314848253481103 m`;
- minimum object lift: `0.026332585685933996 m`;
- maximum object-to-gripper drift: `0.029638127430340718 m`;
- collision frames: `0`;
- attachment calls / activations: `0 / 0`;
- legacy teleport activations: `0`;
- object-pose / robot-state writes: `0 / 0`;
- infrastructure error: none;
- auditor classification: `recovery`.

The transport controller stopped after 57 steps with failure stage `contact`.
Its terminal contact state was left `true`, right `false`. The object then
settled from the controller-observed `z=1.302184 m` to `z=1.151811 m`, below
the `0.13 m` lift gate. The route removed the production-line collision but
did not preserve the bilateral load-bearing grasp.

The top-level `continuous_bilateral_contact=true` field is not sufficient to
override the controller's terminal unilateral contact, `dropped=true`, and
measured minimum lift. The fail-closed auditor correctly did not admit the
episode as success.

## Evidence

- compact result SHA-256:
  `393e8d420b3f8896e279e648574d28fe16159a849b7f39257646aca41fe7253c`;
- remote trajectory:
  `/home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50-trajectory.json`;
- remote trajectory SHA-256:
  `9afd9ac49f7fce50273ba311da84d9ca503300e98cf24f8c5aa52fcc88066708`;
- ledger SHA-256:
  `0a72a3292d0ca8e4948b3c063fe03fa4c8e183b48b62b1aa0943badc49a79fd3`;
- canonical TSV SHA-256:
  `363675a94c965baaeaf09c11adb9982c830549115d806872920f1c3d8dbb9262`.

## Next Falsifiable Hypothesis

The reverse motion opens the asymmetric bilateral grasp, losing the right
contact before reaching the waypoint. Change only
`planar_hold_inward_feedforward` from `0.0` to `0.0005 m` per control step.
This commands each gripper toward the measured object center while preserving
the `+x` route, speed, `0.03 m` internal drift guard, disabled recovery, and all
integrity gates.

Keep the second result only if it increases true object translation, retains
bilateral contact and lift, and introduces no collision or integrity
regression. Full success still requires at least `0.50 m` true object motion.
