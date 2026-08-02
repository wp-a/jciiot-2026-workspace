# Physical Carry Drift-Guard Experiment Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Discard the `0.03 m` to `0.04 m` internal drift-guard change.

## Valid Result

- true object translation: `0.27527476256830846 m`;
- incumbent true object translation: `0.26540101992172577 m`;
- absolute translation gain: `0.00987374264658269 m`;
- minimum object lift: `0.19442237832863896 m`;
- maximum object-to-gripper drift: `0.030449263360874522 m`;
- continuous bilateral contact: yes;
- dropped: no;
- attachment calls / activations: `0 / 0`;
- object-pose / robot-state writes: `0 / 0`;
- collision frames: `2`;
- strict data classification: `rejected`.

The load remained physically grasped, but the run stopped when
`robot0_torso_fixed_collision_box_1` contacted
`scene_aabb_proxy_production_line_5` after `0.3308901012596204 m` of base
translation. The change improved object progress by about `9.9 mm` but
regressed safety, so it is not eligible for training or submission evidence.

## Evidence

- result SHA-256:
  `99504d5894a9ca53351b8495c18e10896d8f2f08f5d79c9118af9e9dab37412b`;
- remote trajectory:
  `/home/user/jciiot-2026/results/physical-carry-drift04-20260802/seed0-drift04-interface-match-trajectory.json`;
- remote trajectory SHA-256:
  `bb9c679c1a45803d38425d15810355cb55bd9279757d9989c9be76679dfe6b7d`;
- audit ledger SHA-256:
  `f7606935cbfc063f4b6e4cfa1d7a042840a5e8bbbacb554d4dabdc5a0102b709`.

## Next Hypothesis

The current limiting factor is no longer the `0.03 m` drift guard. The next
experiment should change path geometry or the base-relative carry posture to
keep the torso outside the production-line collision proxy while preserving
the demonstrated bilateral hold. It must retain the original drift guard and
the unchanged strict data gate.
