# Quality-Gated Physical Transport Data Design

## Context

The current competition-native dataset is too small for robust 20-dimensional
bimanual closed-loop control, but scale is not the only limitation. H3 BC-RNN
reached low held-out error and failed 0/5 closed-loop trials. H4 Diffusion
Policy passed 2/5 trials. The collected grasp and recovery windows do not yet
prove a complete attachment-free 0.50 m transport, so copying them cannot teach
the missing load-bearing behavior.

The older fixed-scene full-score candidate is not a source of physical
transport demonstrations because its long-distance transport uses the official
attachment helper. It remains a diagnostic baseline only.

## Decision

Build a quality-gated data pipeline before increasing dataset size. A trajectory
is admitted as a physical transport success only when immutable runtime evidence
proves bilateral contact, lift, continuous object motion, zero collision, zero
attachment use, and zero task-object pose writes. Recovery trajectories are kept
in a separate stratum and never counted as successful transport.

Do not start another model training run until at least one complete 0.50 m
physical transport passes this gate. Do not launch reinforcement learning in
this phase.

## Architecture

The pipeline has four stages:

1. The existing isolated L1 runner produces trajectory JSON, audit metadata,
   and an HDF5 episode from actual MuJoCo actions and observations.
2. A pure validation module classifies each episode as `transport_success`,
   `recovery`, or `rejected` and emits explicit rejection reasons.
3. A manifest builder assembles leakage-safe train, validation, and held-out
   splits by run and perturbation family rather than by action window.
4. Training is unlocked only when the manifest satisfies minimum coverage and
   the transport-success gate. ACT and Diffusion Policy are then compared under
   the same split and closed-loop seeds.

All competition submission changes remain inside the official allowlist. The
data validator and experiment runners live outside the scored submission tree.

## Physical Success Contract

A `transport_success` episode must satisfy every condition below:

- action dimension is 20 and all action, state, and observation values are
  finite;
- a verified bilateral grasp event exists;
- measured object lift is at least 0.13 m;
- measured planar object translation is at least 0.50 m;
- terminal bilateral contact is present and contact loss never exceeds the
  configured short recovery allowance;
- object-to-gripper planar drift is at most 0.05 m;
- official collision-frame count is zero;
- attachment call count is zero;
- task-object pose-write count is zero;
- the run completed without an infrastructure error.

The validator fails closed when required audit fields are absent. This prevents
old attachment-based or under-instrumented demonstrations from entering the
physical-success split.

## Sampling Strategy

After the first successful trajectory, generate data in bounded batches and
recompute coverage after every batch:

- pilot: 20 successful transports across straight directions and small initial
  pose perturbations;
- scale gate: 60 additional successful transports only if the pilot has at
  least 80% success and no integrity violation;
- recovery set: 40 targeted interventions split across left-contact loss,
  right-contact loss, asymmetric lift, pitch growth, and early transport slip;
- held-out set: complete unseen perturbation families, never random windows
  from training runs.

These are engineering quotas, not a claim that a fixed number guarantees model
success. A failed batch is used to improve the controller or recovery policy;
it is not relabeled as successful data.

## Model Promotion

The first learned candidates are ACT and low-dimensional Diffusion Policy. Both
use the same observation history, normalized action space, run-level split, and
closed-loop test seeds. The deterministic controller remains the safety and
fallback layer; a learned policy may initially predict only residual arm and
gripper corrections.

Offline MSE is diagnostic only. Promotion requires at least 8/10 attachment-free
closed-loop L1 grasp-and-0.50 m transport successes, zero collision frames, and
zero integrity violations. Full-route scoring and L2-L5 transfer happen only
after this gate.

## Failure Handling

- Missing audit fields: reject the episode.
- Attachment or object-pose mutation: reject and quarantine the episode.
- Valid physical attempt with a recoverable contact failure: classify as
  `recovery` with its failure-mode label.
- Collision or infrastructure failure: reject, preserving evidence for
  diagnosis but excluding it from training by default.
- Dataset split overlap or duplicate content hash: abort manifest generation.

## Verification

The validator is implemented test-first with synthetic metadata and small HDF5
fixtures. Red tests prove that missing audit data, attachment use, short object
motion, collisions, and cross-split duplicates are rejected. Existing dataset
and competition tests must remain green.

Server experiments additionally require a reachable pinned runtime, matching
runner SHA-256, immutable raw evidence, and a completed JSON summary. The live
8502 candidate is not changed by data collection or offline training.
