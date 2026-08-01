# H3 Protocol: Low-Dimensional BC-RNN Grasp Policy

Date locked: 2026-08-01 (Asia/Shanghai)

Type: confirmatory offline training followed by closed-loop evaluation.

## Hypothesis

A compact recurrent policy trained on competition-native, object-relative Tiago
observations can reproduce the scripted grasp window on unseen L1 object/base
perturbations. The learned policy is a local grasp candidate only; navigation,
safety, lift verification, transport, and placement remain deterministic.

## Frozen inputs

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Recorder commit: `976186b034578898799313f3b96277039f478065`.
- Data: the 14 H2 runs documented in `../h2-native-tiago-grasp-dataset/results.md`.
- Framework: the competition-bundled `robomimic 0.5.0` and PyTorch
  `2.7.0+cu126` from the pinned server environment.
- Training and outputs remain under `/data01/user/jciiot-2026/model-research/`.
- No submission overlay code changes are allowed during H3 training.

## Leakage-resistant split

- Train: nominal seed `20260840`; small seeds `20260841`, `20260852` through
  `20260857` (8 demonstrations).
- Validation: small seed `20260858`; medium seed `20260860` (2 demonstrations).
- Held out: small seed `20260859`; medium seed `20260861` (2 demonstrations).
- Excluded duplicates: nominal seeds `20260850`, `20260851`. They are retained
  in H2 but excluded from all H3 metrics because their deterministic action
  sequence duplicates the nominal teacher.

The split is fixed by run seed, never by individual frames.

## Model and inputs

- Algorithm: deterministic robomimic BC-RNN.
- Observation dimension: 33 low-dimensional values from bilateral EEF position
  and quaternion, bilateral gripper qpos, and object position and quaternion.
- RGB is retained in H2 but excluded from this first baseline.
- Action dimension: 20 normalized Tiago controller commands.
- RNN: two-layer LSTM, hidden dimension 400, horizon and sequence length 10.
- Actor MLP: `[512, 512]`.
- Loss: action L2; action normalization disabled; observation normalization
  enabled using training demonstrations only.
- Optimizer: Adam, learning rate `1e-4`, batch size 64, seed `20260801`.
- Budget: 300 epochs, one full training and validation pass per epoch, save the
  checkpoint with lowest validation action loss. No environment rollout occurs
  inside the generic robomimic trainer.

## Baselines and gates

1. The merged HDF5 must reproduce all source hashes, shapes, metadata, and split
   masks. A one-epoch loader smoke test must finish before the full run.
2. Report constant-zero and training-mean action MSE on validation and held-out
   demonstrations.
3. The selected BC-RNN must improve held-out action MSE by at least 25% relative
   to the better constant baseline. Report whole-action and controller-group
   errors; do not select a checkpoint on the held-out set.
4. Closed-loop promotion uses five new pre-registered small-perturbation seeds:
   `20260870` through `20260874`, one attempt each. Success requires bilateral
   contact, verified lift, no collision, and no direct object-state write or
   attachment during the learned grasp window.
   Each learned window is capped at 400 controller steps and must maintain
   bilateral contact while exceeding the configured lift target minus tolerance
   for five consecutive frames. A failed learned window receives no geometric
   approach, close, contact-polish, or lift fallback.
5. Promotion to a submission experiment requires at least 4/5 closed-loop grasp
   and lift successes. A lower result keeps the deterministic teacher as primary
   and triggers a same-data Diffusion Policy comparison; it is not repaired by
   tuning on held-out seeds.

## Claims boundary

Offline imitation loss is not competition score. Even if H3 passes, the model
does not replace the scored path until a separate full-workflow experiment shows
that the safety and performance gates remain satisfied.
