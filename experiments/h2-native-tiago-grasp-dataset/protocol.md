# H2 Protocol: Competition-Native Tiago Grasp Dataset

Date locked: 2026-08-01 (Asia/Shanghai)

Type: confirmatory data-interface and teacher-robustness experiment.

## Hypothesis

A research-only recorder can extract temporally aligned 20-dimensional Tiago
actions and competition observations from the verified scripted grasp window.
Successful object-relative teacher rollouts should remain collectable under
explicit pose perturbations, producing a valid robomimic dataset without using
the incompatible Fetch/iGibson sample.

## Frozen code and environment

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Candidate: `l4-target-margin-cc1b5b3`.
- Scored submission overlay remains unchanged during collection.
- Collection is implemented only in `scripts/` and writes large artifacts to
  `/data01/user/jciiot-2026/model-research/h2-native-tiago-grasp-dataset/`.
- Execution uses the existing full workflow and unmodified public scorer. The
  recorder is active only from `grasp_start` through `grasp_end`.
- One attempt per registered run; failed runs are retained and are not replaced.

## Pilot runs

| Run | Level | Tier | Seed | Object |
|---|---|---|---:|---|
| pilot-nominal | L1 | nominal | 20260840 | `line_5_container_h01_near` |
| pilot-small | L1 | small | 20260841 | `line_5_container_h01_near` |

Both pilots must satisfy the complete L1 acceptance gate: 10/10, one verified
bilateral grasp and lift, zero collision frames, workflow success, and final
target distance below 0.8 m.

## Dataset expansion runs

After both pilot files pass the schema gate, retain all twelve registered runs:

- nominal seeds `20260850`, `20260851`;
- small seeds `20260852` through `20260859`;
- medium seeds `20260860`, `20260861`.

Only physically successful grasp windows enter the training demonstrations, but
every failed manifest remains in the experiment results and counts against the
teacher collection success rate.

## Required observations

- `robot0_left_eef_pos`
- `robot0_left_eef_quat`
- `robot0_left_gripper_qpos`
- `robot0_right_eef_pos`
- `robot0_right_eef_quat`
- `robot0_right_gripper_qpos`
- `line_5_container_h01_near_pos`
- `line_5_container_h01_near_quat`
- `robot0_robotview_image`, rendered at 128 x 128 for future visual ablations

## Data validity gates

1. Actions have shape `[T, 20]`, finite values, and absolute maximum at most
   `1.000001`.
2. Every required observation has exactly `T` samples and finite non-image
   values. Images have shape `[T, 128, 128, 3]`, `uint8` dtype, and nonzero
   temporal or spatial variance.
3. `T >= 250`; the action sequence contains both nonzero arm commands and a
   gripper close phase.
4. The captured first state precedes its action. Dataset attributes record the
   official commit, workspace commit, task, seed, perturbation, score, event
   frame bounds, action dimension, and observation keys.
5. The HDF5 inspector reproduces the action width, demonstration count, sample
   count, observation shapes, and file SHA-256 without loading all images.
6. The scored-path audit remains at zero violations because no recorder or
   model-training code is placed under the submission overlay.

## Interpretation rule

- Passing both pilots confirms only the data interface, not policy quality.
- Expansion success rate and failure stages measure teacher coverage under
  perturbations.
- BC-RNN receives a separate locked protocol only after the dataset passes all
  H2 gates. BC-Transformer, Diffusion Policy, and ACT remain out of scope until
  the simple recurrent baseline is evaluated.
