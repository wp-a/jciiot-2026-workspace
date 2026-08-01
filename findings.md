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
  is unnecessary until the bundled algorithms are compared correctly.

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

## Open Questions

- Can a same-data action-sequence model avoid the BC-RNN's single-arm contact
  collapse, or is explicit recovery data required?
- Can supported controller actions reproduce L5 placement without private
  attachment mutation while retaining all three objects within 0.8 m?
- After the data interface is verified, does BC-Transformer improve held-out
  success over BC-RNN enough to justify image inference and temporal context?

## Optimization Trajectory

The deterministic object-relative teacher passed 14/14 registered L1 data
collection runs. BC-RNN passed its offline gate but was rejected after 0/5
closed-loop successes. The next registered point is a same-data Diffusion Policy
comparison; it cannot alter the split or use the five BC-RNN closed-loop seeds
for checkpoint selection.
