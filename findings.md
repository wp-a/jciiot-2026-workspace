# Research Findings

## Research Question

Can a verified hybrid mobile-manipulation system preserve the five-level public
score while generalizing to measured pose and dynamics shifts and avoiding
private simulator-state shortcuts?

## Current Understanding

The incumbent is a useful deterministic teacher, not a proven final solution.
It obtained 100/100 from the unmodified local public scorer on the five published
fixed scenes with zero scored collisions. L2-L5 also completed 80/80 repeated
processes, but every run in a level had the same trajectory length and final
geometry. Those results establish repeatability under the same reset, not
distribution-shift robustness.

The strongest practical architecture remains hybrid. Deterministic task
compilation, navigation, safety, and physical verification fit the known
factory layout and collision penalty. Learning should be limited to contact-rich
grasp and local placement, where pose variation and closed-loop corrections can
justify a policy. Model complexity is promoted only by held-out rollout evidence.

## Key Results

- Fixed-public-scene baseline: 100/100 locally, seven required grasp events,
  and zero scored collision frames. This is not a BienData or organizer result.
- Repeated fixed geometry: L2-L5 completed 80/80 runs and 120/120 grasp events,
  but the seed did not change scene geometry.
- Official checkpoint audit: the L1 BC-Transformer archive says epoch 500 and
  saved `best_success_rate=0.0`; a pinned official direct evaluator reproduced
  0/3 identical failed resets.
- Official sample-data audit: the HDF5 contains Fetch/iGibson action width 10,
  not the Tiago action width 20 and competition observation schema.
- Competition-native teacher data: 14/14 pre-registered L1 nominal, small, and
  medium perturbation runs scored 10/10 with zero collision frames and produced
  4,065 aligned Tiago grasp samples with action width 20.
- Low-dimensional BC-RNN: epoch 11 improved held-out action MSE by 92.76% over
  the better constant baseline, but failed 0/5 unseen small-perturbation
  closed-loop grasps. Every failure ended with only one arm in contact.
- Same-data Diffusion Policy: validation selected epoch 80 after a 300-epoch,
  10-minute-50-second low-dimensional run. Median held-out MSE improved 84.89%
  over the constant baseline, but closed-loop grasp-and-lift success was only
  2/5 versus the locked 4/5 gate. All five runs had zero collision frames.

## Patterns and Insights

- More epochs are not evidence of a usable policy. Physical replay success is
  the gate between training and deployment.
- Low offline imitation error is also insufficient. Bilateral manipulation can
  fail catastrophically when a small per-arm error changes contact mode, even
  while aggregate action MSE and action bounds look good.
- Deterministic repeat seeds can create false confidence when the environment
  reset distribution is fixed. Perturbations must be applied, measured, and
  written into each manifest.
- Object-relative grasp geometry is a strong teacher representation, but its
  direct state-setting helpers are unsuitable for the final scored path.
- The official source already embeds robomimic 0.5.0, including BC-Transformer
  future-action prediction and UNet Diffusion Policy. A second training stack
  is unnecessary; the bundled same-data comparison is now complete.
- BC-RNN 0/5 and Diffusion 2/5 show that the next useful variable is recovery
  data, not more epochs or a larger sequence model on the same 12 demonstrations.

## Lessons and Constraints

- Never train on the official Fetch sample as if it were JCIIOT Tiago data.
- Never select a checkpoint by loss, epoch, or self-reported score alone.
- Keep fixed-scene, perturbation, and organizer/BienData results as separate
  metrics.
- MimicGen's object-relative subtask transformation is useful as a design, but
  its current repository license is not appropriate for direct competition-code
  reuse without review. Reimplement only the necessary idea.
- The final scored path must not assign robot/object qpos or attachment-relative
  state. Research-only perturbation setup may set initial state before recording,
  and must record the exact measured change.
- The previous `0.265401 m` best contact-transport record used direct base-qpos
  stepping. It is evidence of free-object contact transport without attachment,
  but not of complete robot dynamics. A corrected composite-action run moved
  the base only `0.002161 m` in 635 steps and was rejected.
- Wheel-action navigation was already structurally rejected by the archived
  `classic-260728-1443` full-scale test. Do not spend runs on action magnitude,
  duration, or seed sweeps. Any future direct-base result must be labelled as an
  organizer-facing navigation abstraction, not pure dynamics.

## Open Questions

- Can a supported physical push/drag topology move L1 more than `1 m` while
  preserving zero collision, attachment, object-pose writes, and an explicitly
  declared navigation boundary?
- Can supported controller actions reproduce L5 placement without private
  attachment mutation while retaining all three objects within 0.8 m?
- If H5 passes L1, how much additional recovery coverage is required before the
  same local policy can be evaluated on the other object families?

## Optimization Trajectory

The deterministic object-relative teacher passed 14/14 registered L1 grasp
data runs. BC-RNN passed its offline gate but was rejected after 0/5 closed-loop
successes; same-data Diffusion reached 2/5 and also failed promotion. Recovery
training is paused because no strict full-transport teacher exists. The next
step is a no-run inventory of supported-transport evidence, followed by one
pre-registered L1 topology experiment only if it adds information.
