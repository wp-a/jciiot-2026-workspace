# H5 Protocol: Policy-State Recovery Data

Date locked: 2026-08-01 (Asia/Shanghai)

Type: confirmatory data intervention after H3 BC-RNN 0/5 and H4 Diffusion 2/5.

## Hypothesis

Adding teacher-labeled corrections from learner-reachable approach-drift and
asymmetric-contact states will raise the unchanged H4 Diffusion Policy from 2/5
to at least 8/10 unseen L1 grasp-and-lift successes.

## Frozen baseline

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- H4 architecture, optimizer, horizons, normalization, checkpoint schedule, and
  validation-only selection remain unchanged.
- H3/H4 closed-loop seeds `20260870` through `20260884` are diagnostic only and
  may not be used for H5 collection, training, validation, or selection.
- The deterministic controller remains the scored incumbent. No submission
  overlay changes are allowed during H5.

## Recovery collection

Collect 24 accepted correction windows on new seeds `20260900` through
`20260923`, six in each registered bucket:

1. left-biased approach drift;
2. right-biased approach drift;
3. left-only contact requiring right-arm recovery;
4. right-only contact requiring left-arm recovery.

The learner generates the pre-correction state. A deterministic teacher takes
over only after the registered trigger and labels the recovery through stable
bilateral contact and lift. Abort on collision, non-finite action, direct object
state write, or failure to reach the physical gate. Do not retry a seed and
silently replace its result; rejected attempts remain in the collection log.

Each accepted window must record the seed, bucket, trigger measurements, object
and end-effector poses, 20-dimensional actions, collision count, contact sides,
lift, source checkpoint hash, and trajectory hash.

## Split and leakage control

- Recovery train: seeds `20260900` through `20260915`, four per bucket.
- Recovery validation: seeds `20260916` through `20260919`, one per bucket.
- Recovery heldout: seeds `20260920` through `20260923`, one per bucket.
- Combine these with the frozen H3 train/validation/heldout partitions without
  moving an existing demonstration between masks.
- Split by complete seed trajectory, never by frame.

If fewer than 20 of the 24 registered attempts pass the physical acceptance
gate, stop and repair the collector before training. Do not compensate by
sampling extra unregistered easy seeds.

## Training and gates

1. Verify HDF5 schema, action width 20, observation keys, finite values, mask
   disjointness, and bucket counts.
2. Run a two-epoch smoke test and a one-window closed-loop replay.
3. Train the unchanged H4 Diffusion configuration for 300 epochs. Select only by
   validation loss at saved epochs.
4. Offline heldout actions must be finite, width 20, beat the training-mean
   baseline, and report clipping and controller-group errors.
5. Run ten new closed-loop small-perturbation seeds `20260940` through
   `20260949`, one valid attempt each, no geometric fallback.
6. Promotion requires at least 8/10 physical grasp-and-lift successes, zero
   collision frames, and no direct object-state write or attachment inside the
   learned grasp window.

## Stop decisions

- `>=8/10`: promote only to a separate full L1 regression; do not yet replace
  the five-level incumbent.
- `5-7/10`: inspect registered bucket failures and run one additional targeted
  data cycle only if a single missing recovery mode is established.
- `<5/10`: reject the learned local policy for this submission cycle and retain
  the deterministic physical controller.
- Reinforcement learning remains out of scope unless H5 leaves a small,
  repeatable residual error after imitation initialization. Any later RL must be
  a bounded local residual with collision and action barriers, not from-scratch
  whole-robot PPO/SAC.

## Claims boundary

H5 tests whether recovery-state coverage fixes closed-loop distribution shift.
It does not establish cross-scene generalization, official leaderboard score,
or a new reinforcement-learning method.
