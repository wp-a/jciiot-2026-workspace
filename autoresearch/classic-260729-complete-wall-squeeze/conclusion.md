# L1 complete wall squeeze conclusion

No run in this four-iteration loop passed the physical grasp gate. No result is
an official score claim, and the current 8502 candidate remains unchanged.

Completing the original 25 mm squeeze increased the stage from one step to ten
steps but left both gripper centers outside the wall planes. A 40 mm target
placed both centers within about 1--2 mm of the long-wall planes, still with
inner-knuckle rather than fingerpad contact. Completing the original descent
for 220 steps showed that the low-first sequence physically jammed the inner
knuckles on the top rim.

Reordering the motions to precenter at high clearance completed a collision-free
16-step lateral motion with no object contact. The subsequent descent still
finished just outside the generic end-effector target tolerance: the right arm
reached the target band while the left remained about 17 mm above its target.
The stage therefore never attempted closure.

A read-only diagnostic rerun expanded the geometry snapshot without changing
the motion. It established that, at the failed descent endpoint, each physical
wall was already between its gripper's two fingerpads:

- right fingerpads: `y=4.746/4.875`, wall: `y=4.809`;
- left fingerpads: `y=4.363/4.491`, wall: `y=4.423`.

This supports a contact-constrained readiness gate: if the generic pose target
is contact-limited but both walls are geometrically bracketed by distinct
fingerpad pairs and there is no official collision, closure may be attempted.
That readiness condition must not set grasp success. The unchanged official
bilateral grasp, 0.13 m lift, 20-step hold, collision, object-write, and
attachment gates remain authoritative.

## Artifact hashes

- Iteration 1: `4ec95b7670d500fc4d145248f361671e0adcd13e7adca28010c17c68d2b9a45d`
- Iteration 2: `f2320ff1f9645068716584a22d3e992f7cebc03095a2b085aced767c4d26c1ef`
- Iteration 3: `075cff2b6c9c3a142d1d4391e930998e075399d210d11b0ac46b90676844c5a7`
- Iteration 4: `47adae371f5404e30f0d6bba9895bdf0d93303fb70337f10ae2a11092170611e`
- Detailed-geometry diagnostic:
  `4b301b775be30dc6954504fb4364b680b20d78516801c9ac9acbe5e0ee314f6f`

Compact JSON files are stored under
`/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-complete-wall-squeeze-20260729/`.
Full trajectories remain under the remote root in `config.md`.
