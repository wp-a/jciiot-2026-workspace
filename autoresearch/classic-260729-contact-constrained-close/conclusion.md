# L1 contact-constrained close conclusion

The center opposed-wall grasp passed its dedicated physical evidence twice in
independent simulator processes with identical seed-0 measurements:

- collision-free two-wall fingerpad bracket readiness;
- three consecutive bilateral official `_check_grasp` frames by close step 12;
- 69/69 lift steps with bilateral official grasp;
- measured lift of `0.134882407 m`;
- 20/20 closed-gripper hold steps with bilateral official grasp;
- zero official collision frames, object-pose writes, and attachment calls;
- no drop after hold.

The diagnostic record's top-level `accepted` field remains false because it is
still evaluated by the older cradle-transfer gate, which requires 20 steps of
non-finger robot-link support and at least 0.5 m base translation. Those are not
claims about the successful center pinch, and neither run moved the object to a
scoring destination. Therefore these results prove a repeatable real grasp,
not a completed L1 task or score.

The next stage freezes all grasp parameters and adds only a physical transport
phase using the existing controller-stepped `run_physical_transport` helper.
Transport must preserve bilateral official grasp and object height after every
substep. A route-specific diagnostic gate will keep gripper-contact stability
separate from cradle-link support and require measured object translation.

## Artifact hashes

- First grasp: `8bb42bacbbdb66dbfb391ccb158ea729707294cc49dfece3813e33273be75a0e`
- Independent repeat:
  `81709fdf212d8ea562656f4b1e47665e0c5638cf918b5e750af14ac2771e7390`

Compact records are under
`/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-contact-constrained-close-20260729/`;
full trajectories remain in the remote root recorded by `config.md`.
