# H4 Protocol: Same-Data Low-Dimensional Diffusion Policy

Date locked: 2026-08-01 (Asia/Shanghai)

Type: confirmatory algorithm comparison after H3 BC-RNN rejection.

## Hypothesis

Predicting temporally coherent action chunks with the competition-bundled
Diffusion Policy will reduce the BC-RNN's compounding single-arm contact error on
unseen L1 pose perturbations, without changing data coverage or the robot API.

## Frozen inputs and split

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Dataset SHA-256:
  `62f069ac17a337b338afdc370119b79c53a7cbc6623dac258d33865494770b47`.
- Exact H3 masks: 8 train, 2 validation, 2 held out.
- The five H3 closed-loop seeds `20260870` through `20260874` are diagnostic
  failures only and must not enter H4 training, validation, or model selection.
- Framework: bundled robomimic 0.5.0 Diffusion Policy and PyTorch 2.7.0+cu126.
- No submission overlay changes are allowed during H4.

## Model and training

- Inputs: the same 33 low-dimensional object and bilateral robot observations
  used by H3. RGB remains excluded for a controlled algorithm comparison.
- Outputs: 20-dimensional normalized Tiago actions.
- Observation horizon: 2; prediction horizon: 16; action horizon: 8.
- Network: bundled conditional 1D UNet with dimensions `[256, 512, 1024]`,
  diffusion embedding 256, kernel size 5, and EMA enabled.
- Scheduler: DDPM, 100 training and 100 inference diffusion steps, cosine beta
  schedule, epsilon prediction, clipped samples.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-6`, cosine schedule,
  500 warmup updates.
- Training: 300 epochs, full train/validation passes, batch size 64, observation
  normalization from train only, seed `20260802`.
- Select the checkpoint with minimum validation diffusion loss. Neither held-out
  actions nor closed-loop runs may select a checkpoint.

## Gates

1. A two-epoch, three-step-per-epoch smoke run must load the exact masks and
   complete train and validation updates before the full run.
2. Offline sanity uses three fixed sampling seeds `20260820`, `20260821`, and
   `20260822`. Report median held-out action MSE, controller-group errors, action
   clipping, and non-finite outputs. The median must beat the training-mean
   constant baseline and emit finite 20-dimensional actions.
3. Closed-loop evaluation uses five new small-perturbation seeds `20260880`
   through `20260884`, one valid policy attempt each. The learned window is
   capped at 400 controller steps and receives no geometric fallback.
4. Each closed-loop success requires five consecutive frames of bilateral
   contact above the configured lift target minus tolerance, no collision, and
   no direct object-state write or attachment inside the learned grasp window.
5. Promotion requires at least 4/5 closed-loop grasp-and-lift successes. If H4
   fails this gate, same-data model escalation stops; the next experiment must
   add explicitly labeled approach-drift and single-arm-contact recovery data.

## Claims boundary

Diffusion validation loss and held-out action MSE are not competition scores.
The deterministic teacher remains the scored incumbent unless a later separate
full-workflow regression promotes a learned grasp policy.
