# Physical Transport Data Gate Configuration

Date: 2026-08-02 (Asia/Shanghai)

## Scope

- Source records: `/home/user/jciiot-2026/results/full-physical-l1-20260802/*.json`
- Full trajectory JSON files are excluded by filename.
- Source records are read-only.
- The live Streamlit processes on ports 8502 and 8503 are unchanged.
- No submission overlay, protected official file, scene, or scorer is modified.

## Auditor

- Local and remote SHA-256:
  `944bcc7a040ee2bf198b29a7a637018844bdc75fe9d37c3112df7b68f047b79e`
- Remote tool:
  `/home/user/jciiot-2026/tools/physical-data-gate-944bcc7/audit_physical_transport_dataset.py`
- Runtime:
  `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`

## Fixed Acceptance Thresholds

- planar object translation: at least `0.50 m`, derived from the transport
  probe's start and final object positions;
- minimum object lift during transport: at least `0.13 m`;
- maximum object-to-gripper planar drift: at most `0.05 m`;
- verified physical grasp and continuous bilateral contact;
- no drop, collision, attachment call or activation, task-object pose write,
  robot-state write, legacy teleport, or infrastructure error;
- all required evidence fields must be present.

An old runner's `accepted`, `transport_success`, or base-translation value is
not used as evidence of object translation.

## Outputs

- Remote ledger SHA-256:
  `72fb002d5356cbca168e69c64180106841df928e5427eda110ba8415a4ec834e`
- Remote TSV SHA-256:
  `c7c47e58bedfc6fbf75a5a3c1a52934ea7fe87b7577d545d3e4d16c27646f5b9`
- Local copies: `ledger.json` and `results.tsv` in this directory.
