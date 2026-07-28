# JCIIOT three-layer physical manipulation design

Date: 2026-07-28 (Asia/Shanghai)

Status: approved by the user on 2026-07-28.

## Objective

Develop a reproducible five-level manipulation system that performs a real
physical grasp, stable supported transport, and physical placement without
object pose writes or transport attachment. Training is evidence-driven: no
large model run starts until the official baseline, real dataset metadata, and
one successful L1 teacher trajectory have been established.

Only the competition-permitted submission surfaces may change:

- `src/robot_agent/skills/`
- `src/robot_agent/workflows/`
- `knowledge/robot_params.json`

Protected official source and the scorer remain unchanged.

## Required gate order

### Gate 1: real HDF5 audit

Inspect the materialized HDF5 file, not its Git LFS pointer. Record:

- `data` attributes, especially `env_args` and `env_info`;
- demonstration count and per-demonstration sample counts;
- observation keys, shapes, and dtypes;
- action dimension, range, and dtype;
- state dimension and any success or mask metadata;
- whether the environment, robot, cameras, controller, and action schema match
  the locked JCIIOT evaluation environment.

The output is a machine-readable JSON summary plus a Markdown conclusion that
classifies the file as task-compatible, partially reusable, or format-only.
No training may use the file before this classification.

### Gate 2: five-level official-checkpoint grasp baseline

Run the unmodified official BC checkpoint through the official execution
interface. The exploratory baseline uses at least 10 independent resets for
each scored object. L5 evaluates all three required totes independently.

For every attempt record:

- scene, object, reset seed, checkpoint identity, and starting pose;
- bilateral fingerpad contact and `grasp_end` result;
- measured object lift and final relative object-to-gripper displacement;
- collision flag, elapsed time, and trajectory path;
- structured failure stage.

A grasp counts as successful only when it is physical, bilateral, collision
free, and reaches the official 150 mm lift target within its tolerance. Skill
return values alone are not accepted as evidence.

The checkpoint is frozen for any object family with at least 90% exploratory
success. Families below that threshold become fine-tuning candidates.

### Gate 3: first L1 physical cradle-transfer trajectory

Use the verified L1 side grasp as the start state. Build a teacher controller
that transfers load support from finger friction to real robot-link contact:

1. retain bilateral gripper contact after the 150 mm lift;
2. move both arms synchronously into a forearm or wrist cradle pose;
3. establish real box-to-robot support contact while keeping lateral caging;
4. hold the supported state for at least 20 control steps;
5. translate the base at least 0.5 m with no attachment, object pose write,
   collision, or drop;
6. retain sufficient control authority to reverse or proceed to placement.

The first trajectory is a research gate, not a score claim. It is accepted
only when the original unmodified trajectory and multi-view replay prove that
the object is supported through MuJoCo contact.

### Gate 4: BC-RNN versus Diffusion Policy decision

Do not choose from model reputation or training loss. Segment successful
teacher trajectories into side-grasp, cradle transfer, supported transport,
and placement. First prove that each learner can overfit and replay one
trajectory through the exact evaluation observation and action interface.

Use BC-RNN as the primary candidate when the successful action distribution is
locally unimodal and temporal memory is the main missing capability. Escalate
to Diffusion Policy only when repeated valid demonstrations show genuinely
multimodal corrections or BC-RNN suffers measurable compounding error.

The comparison uses identical train/validation splits and reports physical
rollout success, collision rate, maximum slip, terminal target distance,
inference latency, and checkpoint size. Training loss is diagnostic only.

## Three-layer runtime architecture

### Layer 1: verified grasp initializer

The official BC policy owns only local visual grasp execution. A deterministic
wrapper verifies bilateral contact and lift, applies one bounded recovery, and
rejects unverified grasps. Scene navigation and task compilation remain
deterministic.

### Layer 2: learned or optimized cradle transfer

A contact-rich local policy owns only the transition from side grasp to
supported caging and the inverse transition near placement. A geometric
teacher, contact checks, and joint/action bounds constrain every rollout.

### Layer 3: contact-aware supported transport

Clearance-aware path planning drives the base. A low-rate model predictive or
residual controller adjusts arm targets from object-to-wrist slip and support
contact. It terminates or recovers on contact loss, excessive slip, collision,
or abnormal acceleration. It never synchronizes the object pose to the robot.

## Evidence and promotion rules

Every experiment writes an immutable row to an autoresearch TSV ledger and
retains the corresponding trajectory, score file, configuration, and replay
paths. Failed attempts remain in the denominator.

A candidate may replace the current 8502 target only after:

- the submission-boundary audit reports zero hard violations;
- all local automated tests pass;
- an unmodified official entrypoint run produces the claimed trajectory;
- L1 completes real grasp, supported transport, and physical placement with
  zero scored collisions;
- the result has been repeated from a clean process.

Five-level promotion follows only after L1 passes. Local public scores are
always labeled local and are never represented as organizer-verified results.
