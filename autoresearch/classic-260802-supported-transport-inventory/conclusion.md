# Supported-Transport Evidence Inventory

Date: 2026-08-02 (Asia/Shanghai)

## Decision

The L1 floor-push route is the current practical incumbent. Two archived runs,
`iter33` and `iter34`, each receive `10/10` from the unmodified official
`app._score_steps(0)` at commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`. No new simulation was required to
establish this result.

This result has a strict claim boundary:

- the container moves through MuJoCo contact, not attachment or object-pose
  writes;
- the route performs a real bilateral grasp before placing the container on
  the floor and physically pushing it;
- the robot base uses the official-style direct-qpos navigation abstraction;
- it is therefore an attachment-free physical-object solution, not a complete
  wheel-actuator dynamics result;
- the two runs are identical despite different seeds, so they establish
  deterministic replay, not geometry or dynamics robustness.

## Verified L1 Result

Both `iter33` and `iter34` contain 14,299 frames and the same terminal state:

- verified `grasp_end success=true` at the configured L1 source;
- object start: `[7.059, 4.619, 1.1255]`;
- object end: `[-0.18835, -8.03784, 0.12487]`;
- official source maximum-axis displacement: `11.975867 m`;
- official target-center distance: `0.748201 m`;
- measured physical floor-contact push steps: `6,985`;
- collision frames: `0`;
- attachment calls: `0`;
- task-object pose writes: `0`;
- elapsed time: `608.30 s` and `606.68 s`.

The event sequence is physical grasp, two bounded inchworm extraction/setdown
macros, arm retraction, floor contact acquisition, long corridor push, safe
reposition, and two final push segments. The last event reports
`floor_corridor_push_end success=true` with target distance `0.748201 m`.

The unmodified official scorer returned the following for each full trajectory:

- `5/5`: grasp success and source displacement;
- `5/5`: target distance `0.75 m < 0.80 m`;
- collision penalty: none;
- total: `10/10`.

## Integrity Boundary

The current AST scored-path audit reports zero hard attachment/object-state
violations. It also reports 24 warnings: 11 private-backend imports and 13
direct robot-qpos writes. These warnings are material. The physical-object
claim is valid, while a full robot-dynamics claim is not.

The maximum inter-record object displacement is `0.145547 m` around the
physical setdown/fall transition. The complete first-person replay shows the
container settling to floor height and then remaining in contact during the
push; it does not show the object flying or remaining rigidly attached to a
gripper.

## Route Inventory

The complete evidence inventory is in `results.tsv`. It closes the following
branches without another run:

- wheel-action navigation: structurally insufficient;
- same-side long carry and base drag: longitudinal slip/drop;
- center-grasp base drag: object/base decoupling;
- repeated inchworm carry: reset-induced roll/drop or collision;
- table-edge undercut: blocked by the larger static support geom;
- sequential or bilateral bottom conversion: side contact drags the tote;
- floor push with direct-base navigation: L1 objective success twice.

## Next Gate

Do not tune the floor-push controller again. The next task is integration, not
physics search: port the frozen `iter33` route into the official
`RobotAgent.run()` path using only allowed `skills/` and `workflows/` files,
then run one L1 official-entrypoint verification. Only after that passes should
the route be generalized to L2.

## Evidence Locations

Compact results, logs, the frozen allowed-code overlay, and AST audits are
stored in `artifacts/`. Large trajectories and GIFs are kept outside Git at
`/Users/wangpeng/jciiot-2026-assets/physical-floor-push-20260730/`; exact hashes
are recorded in `asset-manifest.md`.
