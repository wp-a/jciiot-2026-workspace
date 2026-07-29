# L1 center-grasp physical transport

- Mode: classic
- Date: 2026-07-29 (Asia/Shanghai)
- Iteration limit: 6 valid single-variable experiments
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Frozen-grasp candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Iteration 2 candidate: frozen grasp plus transport skill from `6007631`
- Iteration 3 candidate: `/home/user/jciiot-2026/candidates/robust-l1-carry-424a08b`
- Next candidate: `/home/user/jciiot-2026/candidates/robust-l1-carry-0ca4367`
- Scene and seed: public L1, seed 0
- Remote root: `/home/user/jciiot-2026/results/l1-center-grasp-transport-20260729`
- Current 8502 candidate: unchanged

## Frozen grasp

Use the repeated arm-only 24-node wrist seed, 10-degree runtime entry, 0.10 m
pregrasp base advance, zero center shift, high-clearance 0.040 m precenter,
fingerpad-bracket readiness, gradual close, 0.13 m lift gate, and 20-step hold.

## Single-variable series

Start with 0.20 m requested straight transport. Preserve all controller
parameters while increasing only distance to 0.50 m and then 1.05 m. If the
0.20 m gate fails, keep its distance fixed and change one transport-control
variable per experiment. A distance is advanced only if the previous run
retains bilateral official grasp, bounded object-to-gripper drift, object
height, zero collision, zero writes, and zero attachments.

## Ordered metric

1. no infrastructure error, collision, object write, or attachment;
2. repeatable center grasp, lift, and 20-step hold;
3. physical transport success without contact/height loss;
4. maximize measured object planar translation;
5. exceed 1.00 m object translation;
6. minimize time only after the physical gates pass.

This series does not claim destination arrival or official score.
