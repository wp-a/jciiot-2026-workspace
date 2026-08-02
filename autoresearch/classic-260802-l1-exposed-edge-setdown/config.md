# L1 Pure-Actuator Exposed-Edge Setdown

Date: 2026-08-02 (Asia/Shanghai)

## Objective

Test whether the official raw bilateral grasp can move the L1 container far
enough along world `-x`, using only composite-controller actions, then place it
stably on its fixed passive support with a measurable strip of bottom exposed.
This is a contact-topology preparation experiment, not an official-score claim.

## Root-Cause Correction

The earlier `0.265401 m` transport used `_set_base_xy_direct` inside the
research driver's base step. It moved the object through real contact and did
not write object pose, but it was not a pure base-actuator trajectory. This
experiment therefore uses a new `PureActuatorPhysicalCarryDriver` whose base,
arms, grippers, torso, and head are all sent through one composite
`env.step(action)` call. It does not call direct base setters or restore robot
`qpos/qvel`.

## Frozen Setup

- Candidate base: `feature/l4-target-margin-20260801` at `1ccd561`.
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Scene: public L1 `FactorySorting1_3FO3ERFHISEM`, seed `0`.
- Object: `line_5_container_h01_near`.
- Grasp: unchanged raw scripted physical grasp.
- Requested base motion: `0.25 m` along world `-x`.
- Maximum base action: `0.04`.
- Arm planar, inward, and vertical feedforward: all zero.
- Planar recovery: disabled.
- Physical setdown descent: `0.001 m` per step, at most `0.30 m`.
- Release / internal settle: `40 / 80` action steps.
- Additional measured passive-support hold: `20` open-gripper action steps.
- Fixed support: `line_5_container_h01_near_support`.

Source hashes before execution:

- runner: `461a21ecb252d6ed61f68b58a52da32b52ba7dab7a35b7fb3eccdca3c4f9667e`;
- experimental transport skill:
  `bcd9ff00d3ce14284409acab9383bd68b47a08cfe149c4a6b717f9690d058f53`;
- unchanged grasp skill:
  `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`.

## Registered Gate

Keep the route only if all conditions hold:

- physical grasp and carry controller success are true;
- true object translation is at least `0.15 m`;
- final directed exposed bottom width is at least `0.10 m`;
- physical place succeeds and both grippers release;
- the object contacts the exact fixed support geom for 20 consecutive
  post-release action steps;
- post-place object motion over those 20 steps is at most `0.005 m`;
- collision frames, attachment activations, legacy teleport activations,
  object-pose writes, and robot-state restores are all zero;
- no infrastructure error occurs.

If the base action does not move the held load or any hard gate fails, archive
the result and stop. Do not tune a seed or parameter in the same iteration.

## Setup Exclusion

The first launch is excluded before hypothesis evaluation. The new mode passed
`full_physical_stage=None` into the grasp dispatcher, so the official default
wrapper activated transport attachment. The transport audit stopped before
calling the base controller: control result was absent and base/object motion
were exactly zero. A regression test now requires this mode to select the same
raw attachment-free scripted grasp as other physical stages. No registered
physics parameter changed.

Corrected runner SHA-256:
`6a33501c48f8aa57ccfc4379dbdfe45312078b0715cdff0b1125e2d97c839394`.

## Outcome

The corrected run was rejected. The base moved only `0.002160790 m` in 635
composite action steps; the object fell to its source support and moved
`0.100385737 m` in the opposite direction through slip. Placement was never
attempted. This is consistent with the earlier locked-controller result in
`autoresearch/classic-260728-1443`, which should have stopped this experiment
before launch. See `conclusion.md` and `results.tsv`; no further action-scale or
seed sweep is permitted.
