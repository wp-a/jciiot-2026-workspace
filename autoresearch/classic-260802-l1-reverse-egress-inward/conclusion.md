# L1 Reverse-Egress Inward-Hold Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Discard the `0.0005 m` per-step inward hold feedforward. Preserve the world
`+x` reverse-egress direction and the original zero-inward controller as the
diagnostic incumbent. The independent auditor classified this run as
`recovery`, not transport success.

## Measured Result

- requested base segment: `0.49999983686408217 m`;
- measured base translation: `0.10100155346202844 m`;
- true audited object translation: `0.11692233164695394 m`;
- minimum object lift: `0.028646401283812795 m`;
- maximum object-to-gripper drift: `0.0281353058683195 m`;
- controller steps: `54`;
- terminal contacts: left `true`, right `false`;
- collision frames: `0`;
- attachment calls / activations: `0 / 0`;
- legacy teleport activations: `0`;
- object-pose / robot-state writes: `0 / 0`;
- infrastructure error: none;
- auditor classification: `recovery`.

Against the zero-inward incumbent, true object translation decreased by
`0.00622615088785709 m` and contact failed three control steps earlier. The
maximum planar drift improved by only `0.001502821562021218 m`; this did not
restore the right gripper or retain the load. The controller observed the
object descend from `z=1.375965 m` to `z=1.306912 m` before contact failure,
after which it settled to `z=1.154751 m`. This falsifies the hypothesis that
insufficient planar inward seating was the primary cause.

The top-level `continuous_bilateral_contact=true` field reflects sampled
recorder frames and is not authoritative after the controller reports terminal
right-contact loss. The fail-closed auditor correctly rejects that apparent
positive signal.

## Evidence

- compact result SHA-256:
  `fc12ac28603fe3a5d69aee6f7f71411494cfdbfcac7112b56dc647e725673488`;
- remote trajectory:
  `/home/user/jciiot-2026/results/l1-reverse-egress-inward-20260802/seed0-xplus-inward0p0005-0p50-trajectory.json`;
- remote trajectory SHA-256:
  `ac6d6f989c7f5d13fc098d50043b909ff5cf8347482c2516607f3314d0825961`;
- ledger SHA-256:
  `6d7da0b606241c75f4b3200bf59d9bbbbad924fc0508142ee689068790db04e3`;
- canonical TSV SHA-256:
  `f788e8b18e83e291e7115f11c17f71d825994b5e5f1e1005f906caca108df45b`.

## Root-Cause Direction

The controller loses about `0.069 m` of object height before terminal contact
failure while planar drift remains below `0.03 m`. In actuator-only mode the
runner explicitly disables discrete height recovery, and its
`PhysicalCarryConfig` leaves continuous vertical hold feedforward, gain, and
maximum delta at zero. The next experiment must therefore test bounded
actuator-only vertical support, not another planar squeeze change. That test
will remain a single pre-registered controller intervention with the route,
speed, collision gate, drift guard, and all integrity gates frozen.
