# L1 Pure-Actuator Exposed-Edge Setdown Conclusion

## Decision

Rejected. The registered setdown gate was not reached, and no score is
claimed. Do not sweep base-action magnitude, step budget, or seeds for this
route.

## Corrected Run

The corrected run used the raw attachment-free scripted grasp and sent the
base, arms, torso, head, and grippers through a single composite
`env.step(action)`. It recorded zero collision frames, attachment calls or
activations, legacy teleport activations, object-pose writes, and robot-state
writes.

The grasp initially passed, but the commanded `0.25 m` world-`-x` base move
timed out after 635 action steps:

- base translation: `0.002160790 m`;
- raw object-body translation: `0.100385737 m`, in the wrong `+x` slip/fall
  direction rather than the requested `-x` direction;
- minimum object lift during transport: `0.000016264 m`;
- maximum object-gripper planar drift: `0.015390389 m`;
- continuous sampled bilateral contact: `true`;
- collision frames: `0`;
- physical placement attempted: `false`;
- stable passive-support frames: `0`;
- directed exposed bottom: `0.0 m`.

The object fell back to the fixed source support while the mobile base remained
effectively stationary. The geometric exposed-edge idea was therefore not
tested because its prerequisite actuator-only base translation failed.

## Prior-Evidence Reconciliation

This result reproduces evidence already archived in
`autoresearch/classic-260728-1443`: the official Tiago wheel controller moved
only about `5.6 mm` in 100 full-scale actions, so action-only cross-scene
navigation is not viable in the locked environment. The later
`0.265401 m` contact-transport record used direct base-qpos stepping. It did
not write object pose or attach the object, but it is not a pure robot-dynamics
trajectory.

Running this corrected probe was redundant because the earlier controller
limit was not represented in the active research-state index. The central
workspace state must now treat the following as hard route boundaries:

1. no more wheel-action amplitude or seed sweeps;
2. direct-base navigation must be labelled as simulator-level robot-state
   navigation, never pure dynamics;
3. object attachment and object-pose writes remain forbidden for a physical
   submission claim;
4. future physical transport work must start from the supported-push evidence
   inventory or an organizer-confirmed public navigation abstraction.

## Setup Exclusion

The first launch passed `full_physical_stage=None`, causing the default grasp
wrapper to activate transport attachment. The audit stopped before base or
object motion. It is retained only as a setup-error artifact and is not used
to evaluate the registered physics hypothesis.

## Evidence

- corrected result SHA-256:
  `8a36cc58e1c77b051b1594acf2748fc35f5451e4d22787f1f27d05f2ecab3eeb`;
- corrected trajectory SHA-256:
  `938b6cc7c38d15002474446755a8c2e6112db719924876bbebd0eccba880561e`;
- setup-exclusion result SHA-256:
  `873e79b4bdd8d982ce3848f0d58d4502a96eb52066f02e67e1afd5a0968c7f3d`;
- setup-exclusion trajectory SHA-256:
  `fd88a5d3f7f04bad8299f1b33847799bbfef0470cb63a87d56abbeffd4d0c435`.
