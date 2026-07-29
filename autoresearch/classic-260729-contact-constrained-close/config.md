# L1 contact-constrained close gate

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 4 valid single-variable experiments
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Scene and seed: public L1, seed 0
- Remote root: `/home/user/jciiot-2026/results/l1-contact-constrained-close-20260729`
- Current 8502 candidate: unchanged

## Incumbent

Retain the arm-only 24-node wrist seed, 10-degree runtime entry, 0.10 m base
advance, zero center shift, high-clearance 0.040 m precenter motion, and bounded
descent. The first run changes only the approach completion rule: a
collision-free endpoint may attempt closure when two distinct wall planes are
explicitly bracketed by the official fingerpad pairs.

## Ordered metric

1. no infrastructure error, official collision, object write, or attachment;
2. two distinct walls bracketed by the two official fingerpad pairs;
3. three consecutive bilateral official grasp frames;
4. measured lift at least 0.13 m;
5. at least 20 closed-gripper hold steps with bilateral official grasp;
6. accepted physical gate.

Bracket readiness is never reported as grasp or score.
