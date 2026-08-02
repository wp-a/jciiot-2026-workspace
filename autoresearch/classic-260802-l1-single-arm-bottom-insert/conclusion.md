# L1 Single-Arm Bottom-Insertion Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Discard the one-arm release and bottom-insertion transition. Do not tune its
distances or action gain, do not merge the experimental runner into the stable
candidate, and do not treat its trajectory as successful training data.

## Result

The current raw scripted physical grasp succeeded, followed by a successful
`0.10 m` bilateral clearance raise. During the registered right-arm insertion:

- the stationary left arm retained object contact in every one of 180 frames;
- the right arm reached its combined descent/forward/inset target;
- the right arm never contacted the official `col_bottom` geom;
- all final right contacts remained on `col_right`;
- no proof raise was attempted;
- judge collision frames: `0`;
- attachment calls: `0`;
- object-pose writes: `0`;
- infrastructure error: none.

The tote moved from `[7.050356, 4.613673, 1.451829]` to
`[7.081574, 4.432718, 1.233018]` during insertion: approximately `+0.031218 m`
in x, `-0.180955 m` in y, and `-0.218811 m` in z. Final lift over the original
table reference was only `0.107540 m`.

## Root Cause

The left gripper preserves contact but cannot preserve the tote pose or resist
the torque produced by releasing the other same-side contact. The tote swings
and slides around the stationary contact as the right arm moves. Because the
load moves down and laterally with the transition, the right fingertips remain
on the side wall even after reaching their Cartesian target.

This falsifies the missing-force-closure approach, not the target speed. A
single same-side gripper is not a stable temporary fixture for converting the
other arm to bottom support. Further distance, action-gain, or seed sweeps would
repeat the same contact topology and are not justified.

The next route must first place the tote onto a passive physical support at a
new pose with exposed underside, or use an opposed/force-closure grasp that can
hold pose while one arm regrasps. The current official tote has solid collision
walls and no usable handle hole, so visual handle insertion remains invalid.

## Evidence

- result SHA-256:
  `52edc78be87365af2b06275101cfbf08980f1fb3e7679e094a0320f1e4fec39d`;
- trajectory SHA-256:
  `3bb78e88eadffc4f9694106f166da49a29a4e7a1c78eeeb4979cc11c8bff5208`;
- audit ledger SHA-256:
  `ede02665e2f9a7cc70e7692973d86969da31af845a2c40b6777c4058f1a46472`;
- audit TSV SHA-256:
  `4c1dd741a1088bfef1066df812876178e1f63310b9fff7cc21fc9f3f32550162`;
- independent classification: `rejected`.

All artifacts are stored in `artifacts/`. The frozen candidate and live
8502/8503 services were not modified.
