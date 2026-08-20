# Decision 0005: Strict Physical Carry Only

Date: 2026-08-14

## Decision

The official submission entrypoint must use `transport_mode="physical_carry"` for L1-L5. `l1_floor_push`, `transport_attachment`, direct task-object pose writes, and equivalent simulator-follow shortcuts are forbidden for the final route.

## Evidence

- The prior public-scored L1 floor route reached 10/10 but used floor contact after setdown and direct base-qpos navigation.
- The prior five-level 100/100 fixed-scene baseline used official attachment for L2-L5.
- The locked L1 collision model has no physical handle collision channel; its visual mesh has collision disabled.
- Forty-one physical transport diagnostics contain no complete 0.50 m continuous suspended-carry success.
- BC-RNN achieved 0/5 and same-data Diffusion Policy 2/5 closed-loop grasp/lift success, so increasing epochs cannot replace a missing physical transport teacher.

## Consequences

- `run_official_task` now passes only `physical_carry`.
- The driver constructor rejects shortcut modes before importing simulation dependencies.
- Historical floor-push and attachment helpers remain in the workspace only for evidence replay and are unreachable from the official entrypoint; they are not final-submission behavior.
- The next accepted result must include bilateral contact, minimum lift, continuous hold, zero collision, zero attachment, zero object-pose writes, and real object displacement.

## Reversal condition

This decision may be revisited only if the competition organizer publishes a rule clarification explicitly permitting a shortcut. A local public score or a self-reported JSON is not sufficient.
