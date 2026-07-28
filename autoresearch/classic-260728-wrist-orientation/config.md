# L1 high-clearance wrist-orientation experiments

- Mode: classic
- Workspace branch: `robust-hybrid-20260728`
- Workspace commit: `b961daa`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `robust-l1-cradle-20260728a`
- Scene: public L1 `FactorySorting1_3FO3ERFHISEM`
- Seed: `0`
- Maximum experiments: `4`
- Scheduling: sequential, one worker, EGL GPU 2
- Protected source changes: none
- Remote result root: `/home/user/jciiot-2026/results/l1-wrist-orientation-20260728`

## Objective

At high clearance with the L1 container physically resting on the table,
align both Robotiq grip-site closure axes with the container's opposed-wall
normal. Continue to the center regrasp only after both errors remain at or
below 5 degrees for five consecutive simulation steps.

## Hard ordering

1. Zero official judge-collision frames.
2. Both closure-axis errors at or below 5 degrees for five stable steps.
3. Maximum end-effector position drift at or below 0.03 m.
4. Bilateral physical object contact during center regrasp.
5. Physical object lift of at least 0.13 m without pose writes or attachments.

Experiments stop immediately on collision or excess position drift. A public
scene score is not treated as evidence for this research gate.

## Planned variants

1. `orientation_max_action=0.20`, conservative direction validation.
2. `orientation_max_action=0.30` only if variant 1 is safe but times out.
3. One targeted controller adjustment based on the first two traces.
4. Confirmation repeat of the best safe variant, only if warranted.
