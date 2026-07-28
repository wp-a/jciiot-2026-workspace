# JCIIOT 2026 robust hybrid competition design

Date: 2026-07-28 (Asia/Shanghai)

Status: approved by the user through the instruction to continue with the
recommended final technical route.

## Objective

Maximize the five-level public score while preserving the evidence needed to
survive organizer reproduction and plausible hidden perturbations. Performance
claims must distinguish the unmodified fixed-public-scene scorer from
perturbation evaluation and from any BienData or organizer result.

The optimization order is lexicographic:

1. task score;
2. zero scored collisions and no abnormal acceleration;
3. held-out robustness;
4. elapsed time;
5. implementation and report novelty.

No speed improvement is kept if it reduces score, safety, or held-out success.

## Considered approaches

### A. Geometry-only hardening

Keep the deterministic controller, replace private-state writes with controller
actions, and tune explicit object-family profiles. This is the fastest route to
a compliant fixed-scene candidate, but it has limited evidence for hidden pose
or dynamics changes and a weaker innovation case.

### B. Verified hybrid mobile manipulation (selected)

Keep deterministic task compilation, navigation, safety checks, and physical
verification. Use the existing geometric controller as a teacher and recovery
reference. Train a task-specific visuomotor policy only for the contact-rich
grasp and local placement segments, then gate policy actions with collision,
contact, lift, and target-distance checks.

This route matches the small number of scenes, the 20-dimensional Tiago action
interface, and the competition's collision penalty. It also gives a credible
innovation claim through the integration of semantic validation, synthetic
trajectory variation, learned manipulation, and explicit physical evidence.

### C. End-to-end Diffusion Policy, ACT, or VLA

Train one model to own long-horizon navigation and manipulation. This has the
highest research ceiling but also the largest data, integration, and safety
risk. It is rejected as the first implementation. Diffusion Policy or ACT is an
escalation only after held-out results demonstrate that deterministic or simple
BC actions are genuinely multimodal or suffer from compounding error.

## Architecture

```text
official task + SOP + semantic map
  -> validated symbolic task compiler
  -> explicit atomic trace: move / pick_up / move / place_down
  -> clearance-aware geometric navigation
  -> object-family manipulation router
       -> geometric teacher and deterministic recovery
       -> BC-RNN or BC-Transformer grasp / local-place policy
       -> optional Diffusion Policy or ACT challenger
  -> action and collision safety gate
  -> contact + bilateral grasp + physical lift verifier
  -> physical transport and release through supported backend actions
  -> target-distance, collision, timing, and trajectory audit
```

The LLM may propose symbolic steps, but it never emits base or joint actions.
All entities and step order are validated against the official task. A
deterministic symbolic plan is the fallback while LLM service is unavailable;
this fallback must remain explicit in the trace and be confirmed with the
organizer if required.

## Compliance boundary

The final scored path may read observations exposed by the environment and may
call supported backend actions. It must not:

- set object or robot `qpos` directly;
- mutate transport-attachment relative state;
- move an object through a private synchronization helper;
- modify protected official files or the public scorer;
- report a local score as a BienData or organizer-verified score.

The current fixed-scene controller remains an offline teacher until all direct
state writes in its scored path are replaced or isolated outside the final
candidate.

## Data design

The official Fetch/iGibson HDF5 is a format reference only. New data must be
generated in the locked JCIIOT environments with the Tiago 20-dimensional
action and exact evaluation observation schema.

Each object-family dataset contains source scene, object pose, base pose,
dynamics tier, action, observation, stage, contact state, lift result, collision
state, and terminal score. Train and validation splits are grouped by
perturbation seed so transformed variants cannot leak across the split.

The initial tiers are:

| Tier | Object XY | Object yaw | Base XY | Base yaw | Dynamics |
|---|---:|---:|---:|---:|---:|
| small | +/- 2 cm | +/- 5 deg | +/- 1 cm | +/- 2 deg | nominal |
| medium | +/- 4 cm | +/- 10 deg | +/- 3 cm | +/- 5 deg | mass/friction +/- 10% |
| stress | +/- 6 cm | +/- 15 deg | +/- 5 cm | +/- 8 deg | mass/friction +/- 20% |

Data generation follows the transferable part of MimicGen: segment teacher
trajectories by atomic subtask, transform object-relative waypoints to sampled
initial states, replay through real controllers, and retain only physically
verified demonstrations. Failed attempts remain in a separate diagnostic
ledger and are never silently removed from evaluation statistics.

## Model ladder

1. Low-dimensional BC-RNN proves the action and temporal pipeline.
2. Image plus proprioception BC-Transformer, context 10, is the primary learned
   candidate because the official checkpoint and bundled robomimic use this
   interface.
3. robomimic v0.5 UNet Diffusion Policy is the first challenger if the simple
   policies fail on multimodal corrections. It supports action normalization
   and the bundled official source already contains the implementation.
4. ACT is evaluated only if action chunking is the identified missing factor.
5. A general VLA is out of scope unless the task distribution changes to
   unknown objects or instructions.

Every trained configuration uses three seeds and is selected by held-out
rollout success, collision rate, and target distance rather than training loss
or epoch number.

## Recovery and failure handling

Failures use explicit stage codes: `plan`, `move_source`, `grasp_contact`,
`lift`, `transport`, `place`, `verify`, `collision`, and `timeout`.

Recovery is bounded:

- navigation failure returns to the last collision-free waypoint;
- grasp failure opens the grippers, retreats vertically, and allows one
  re-approach with a different verified offset;
- lift failure releases and retries rather than transporting an unverified
  object;
- placement failure preserves already verified objects and chooses an
  unoccupied target slot;
- collision or abnormal action terminates the attempt and records all evidence.

No stage may declare success from a skill return value alone. Physical contact,
lift, release, and target position are checked independently.

## Evaluation protocol

The fixed official scene and perturbation harness are separate. The harness
must not modify the locked source used for public-score reproduction.

Promotion gates:

- public fixed scenes: all five levels at their local public maximum, zero
  collisions, 20 clean-process repeats per level;
- held-out grasp: at least 95% success on small and 90% on medium perturbations
  for every object family;
- full task: at least 20 runs per level and perturbation tier, with Wilson 95%
  intervals and every failure retained;
- scored path: zero object/robot direct-state writes and zero private
  attachment mutations;
- reproduction: one clean Linux environment can run the official Agent entry,
  produce trajectories, and invoke the unmodified scorer from the README;
- reporting: fixed-scene local score, perturbation score, collision rate,
  duration, and any official result are separate fields.

The first experiment is a measurement experiment, not training: add the
perturbation harness and measure the current candidate's real robustness. The
first learning experiment is one-trajectory overfit. A large data run is not
authorized until that checkpoint can replay its own grasp through the exact
evaluation interface.

## Technical sources

- JCIIOT official repository: https://github.com/JCIIOT2026/JCIIOT2026
- robomimic releases and v0.5 Diffusion Policy support:
  https://github.com/ARISE-Initiative/robomimic/releases
- MimicGen project and released data-generation design:
  https://mimicgen.github.io/
- Diffusion Policy official implementation:
  https://github.com/real-stanford/diffusion_policy
- ACT / Mobile ALOHA implementation:
  https://github.com/MarkFzp/act-plus-plus

These projects are design references. Any incorporated code must be public,
license-compatible, attributed in `THIRD_PARTY_NOTICES.md`, and isolated from
the locked official source.
