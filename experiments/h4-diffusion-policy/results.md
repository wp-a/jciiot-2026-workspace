# H4 Results: Same-Data Low-Dimensional Diffusion Policy

Date completed: 2026-08-01 (Asia/Shanghai)

Decision: **rejected for submission promotion**. The validation-selected policy
passed the offline gate but achieved only 2/5 pre-registered closed-loop
grasp-and-lift successes, below the locked 4/5 promotion threshold.

## Frozen inputs

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Dataset SHA-256:
  `62f069ac17a337b338afdc370119b79c53a7cbc6623dac258d33865494770b47`.
- Split: the frozen H3 masks with 8 train, 2 validation, and 2 held-out
  demonstrations.
- Inputs: 33-dimensional low-dimensional observation, two-frame history.
- Outputs: 20-dimensional normalized Tiago action.
- Model: bundled robomimic Diffusion Policy, prediction horizon 16, action
  horizon 8, conditional UNet dimensions `[256, 512, 1024]`.

## Training and checkpoint selection

The completed `v3_io_bounded` run trained 300 epochs from 22:23:01 to 22:33:51
on 2026-08-01, approximately 10 minutes 50 seconds. This short duration is
expected for eight low-dimensional training demonstrations; it is not evidence
of broad scene coverage.

The first two formal launches were stopped before held-out evaluation because
robomimic checkpoint persistence saturated storage I/O. The final research-only
wrapper suppressed only unconditional `last.pth` and `last_bak.pth` resume
artifacts while retaining epoch 1 and every-tenth-epoch official checkpoints.
The wrapper passed an isolated two-epoch integration smoke test before restart.

Validation-only selection chose epoch 80:

- Logged validation loss: `0.015898418030701578`.
- Checkpoint size: `1059291741` bytes.
- Checkpoint SHA-256:
  `5a24744c8453d74f8e902cd8bdb3c9fb83091824052a36d7a29a4b753fda09af`.
- Remote checkpoint:
  `/data01/user/jciiot-2026/model-research/h4-diffusion-policy/training/h4_diffusion_l1_lowdim_v3_io_bounded/20260801222301/models/model_epoch_80.pth`.

## Offline gate

Three fixed diffusion sampling seeds produced finite 20-dimensional actions.
Median held-out MSE was `0.01882587564127486`, versus `0.12458633697730398`
for the training-mean constant baseline, an `84.8893%` relative improvement.
The offline gate passed. Output clipping occurred on 89 to 123 of 538 evaluated
steps, which was retained as a deployment warning rather than hidden.

Evidence: [`offline-evaluation.json`](offline-evaluation.json), SHA-256
`b81b44e6ad3e926f207d665f40f49bc6badd2fb3ce5a400587180cb45cd39944`.

## Closed-loop gate

Each row is one valid execution with the unchanged epoch-80 checkpoint, no
geometric fallback, at most 400 learned controller steps, and a five-frame
bilateral-contact plus 0.13 m lift gate.

| Seed | Contacts at end | Lift (m) | Steps | Clipped actions | Collision frames | Local public score | Gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 20260880 | left + right | 0.142616 | 294 | 70 | 0 | 10/10 | pass |
| 20260881 | left + right | -0.000466 | 400 | 39 | 0 | 0/10 | fail |
| 20260882 | neither | -0.000102 | 400 | 108 | 0 | 0/10 | fail |
| 20260883 | left + right | 0.136521 | 292 | 69 | 0 | 10/10 | pass |
| 20260884 | left + right | 0.007123 | 400 | 114 | 0 | 0/10 | fail |

The 2/5 result is a local closed-loop research outcome, not a BienData or
organizer-verified score. The two locally scored successes use the existing
workflow after the learned physical grasp window; they do not promote the model
because the grasp gate itself failed three times.

Evidence files:

- [`closed-loop-seed-20260880.json`](closed-loop-seed-20260880.json), SHA-256
  `ef2e69599e88a534d1256abd6be8031fe4135f1f49dc0afeb6ee352ad7d4419f`.
- [`closed-loop-seed-20260881.json`](closed-loop-seed-20260881.json), SHA-256
  `e2e4a27851c32012087152ee6bce30cb8dde5524a003ca47dbc45d1ebd15521d`.
- [`closed-loop-seed-20260882.json`](closed-loop-seed-20260882.json), SHA-256
  `620b31f870e026453c8095c07001b29d82e0a2d44f33435afed1025d0d9201b7`.
- [`closed-loop-seed-20260883.json`](closed-loop-seed-20260883.json), SHA-256
  `bf926c675183cb124a3ea084ee36b60015b012fe21d7a8bc376c0abc5e0b7cae`.
- [`closed-loop-seed-20260884.json`](closed-loop-seed-20260884.json), SHA-256
  `b1b0012388e5ce6052a124d50a5f61a6bba2011d3df31834c70ac7ebe12a51c3`.

## Infrastructure-invalid launches

Two seed-81/82 launches failed before policy execution because the physical GPU
and `MUJOCO_EGL_DEVICE_ID` differed. Two seed-83/84 launches failed before
policy execution because the launcher pre-created a directory guarded by
`exist_ok=False`. These four launches are retained remotely with explicit
invalid names and are not counted as trials or model failures.

## Failure interpretation and next decision

Diffusion improved over H3 BC-RNN's 0/5 result but did not solve the data
coverage problem. Two failures reached bilateral contact without producing the
required lift, and one failed before contact. This matches approach-drift and
contact-recovery gaps in the 12-demonstration dataset.

Per the pre-registered stop rule, same-data BC-Transformer, ACT, VLA, and
from-scratch reinforcement-learning escalation stop here. H5 must add
policy-state recovery trajectories, then retrain the same H4 configuration to
isolate the effect of data coverage.
