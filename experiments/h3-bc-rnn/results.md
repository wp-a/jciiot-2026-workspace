# H3 Results: Low-Dimensional BC-RNN Grasp Policy

Date completed: 2026-08-01 (Asia/Shanghai)

## Decision

Reject BC-RNN for submission promotion. The policy passed the pre-registered
offline gate but failed all five unseen small-perturbation closed-loop runs. The
deterministic teacher remains the primary scored implementation. H4 will compare
the bundled Diffusion Policy on the exact same data and split before collecting
additional recovery demonstrations.

## Frozen artifacts

- Dataset: `/data01/user/jciiot-2026/model-research/h3-bc-rnn/data/l1-grasp-lowdim-v1.hdf5`
- Dataset SHA-256: `62f069ac17a337b338afdc370119b79c53a7cbc6623dac258d33865494770b47`
- Split: 8 train / 2 validation / 2 held out; 2 duplicate nominal runs excluded
- Framework: robomimic 0.5.0, PyTorch 2.7.0+cu126
- Training: 300 epochs, seed 20260801, two-layer LSTM, hidden size 400
- Selected checkpoint: epoch 11, chosen only by validation loss
- Checkpoint SHA-256: `b228cf56b64d378c119e70e4ee2438bb97de736c25c73c0122000ccf09ed82e8`
- Full offline JSON: `/data01/user/jciiot-2026/model-research/h3-bc-rnn/offline-evaluation.json`

## Offline result

| Metric | Value |
|---|---:|
| Best trainer validation L2 | 0.012668 |
| Held-out BC-RNN MSE | 0.009018 |
| Held-out training-mean baseline MSE | 0.124586 |
| Held-out zero baseline MSE | 0.173140 |
| Improvement over better constant | 92.762% |
| Held-out right gripper MSE | 0.025965 |
| Held-out left gripper MSE | 0.025911 |
| Predicted action out-of-range fraction | 0.0% |

The 25% offline gate passed. Validation loss began overfitting after the early
best checkpoint, so neither the final epoch nor the held-out set was used for
checkpoint selection.

## Closed-loop result

Each run used one attempt, 400 policy steps, no geometric fallback, a 0.13 m
lift gate, and five required stable bilateral-contact frames.

| Seed | Final contacts | Lift (m) | Collision frames | Action clips | Result |
|---:|---|---:|---:|---:|---|
| 20260870 | left only | 0.060833 | 0 | 0 | fail |
| 20260871 | left only | 0.022827 | 0 | 0 | fail |
| 20260872 | right only | -0.000511 | 0 | 0 | fail |
| 20260873 | left only | 0.019380 | 0 | 0 | fail |
| 20260874 | right only | -0.000456 | 0 | 0 | fail |

Closed-loop promotion result: 0/5, below the required 4/5.

The first infrastructure launch for seed 20260874 stopped before any trajectory
or summary file was created because GPU 3 reported an uncorrectable ECC error
(volatile count 1, aggregate count 3). Its empty output directory was retained.
The table uses the only valid policy execution for that seed, run unchanged on
healthy GPU 0.

## Interpretation

Low teacher-forced action MSE masked compounding bilateral-contact error. The
policy reproduced coarse phases and stayed within action bounds, but small
closed-loop deviations caused one arm to miss the object; the RNN then lacked
recovery behavior because the successful-only dataset contains no corrective
states. More BC-RNN epochs would worsen overfitting rather than solve this
coverage gap.
