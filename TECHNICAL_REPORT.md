# JCIIOT 2026 Technical Report

## 1. Executive summary

We implement a deterministic, verifiable mobile-manipulation system for the
five official JCIIOT factory-sorting scenes. The design combines semantic task
resolution, a per-object state machine, collision-aware base navigation,
object-family geometric grasp control, physically verified transport and
stable multi-object placement. A language model is not allowed to issue base
or joint commands on the scored path.

On the published baseline commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`, the attachment-based fixed-scene
candidate scores with the unmodified public scorer
`10/10`, `15/15`, `20/20`, `25/25` and `30/30`, for `100/100` total, with
seven required successful grasp events and zero collision frames. An additional
80-process repeatability batch on L2-L5 produced 80/80 full-score runs,
120/120 required grasp events and zero collision frames. Separately, the L1
no-attachment incumbent `L1-PD-FLOOR-64797D3` scores `10/10` twice with zero
collisions, zero attachment calls and zero object-pose writes. The two result
families are deliberately kept separate: the five-level result is the verified
attachment baseline, while the no-attachment result is currently verified only
for L1. These repetitions do not perturb scene geometry, so they establish
execution repeatability rather than pose-distribution robustness. No result in
this report is a BienData score or an organizer-reproduced result.

## 2. Task and scoring constraints

Each scene requires a Tiago mobile manipulator to transport one or more named
objects from an input station to an output station. The official scorer awards
half of a scene's points when a successfully grasped object leaves the source
by more than 1 m on either planar axis, and the other half when its final planar
distance to the target-table center is below 0.8 m. L5 scores three white totes
individually. Any collision recorded in the trajectory causes a five-point
deduction; time is used only to rank equal scores.

Our local acceptance gate is stricter than the point total alone. It requires
full public-scorer score, every required successful `grasp_end` event, zero collision
frames, final target distance below 0.8 m and a successful workflow result.

## 3. System architecture

```text
Official task metadata and semantic map
  -> entity and task validation
  -> deterministic per-object state machine
  -> grasp candidate and station approach selection
  -> safe-corridor base navigation and heading alignment
  -> object-family two-arm geometric grasp
  -> bilateral contact and physical lift verification
  -> L1: physical setdown and floor contact push; L2-L5 baseline: official attachment transport
  -> target-table release and multi-object slot assignment
  -> final state, collision and event verification
  -> published trajectory recorder and unmodified public scorer
```

The official application executes
`app.py -> task_subprocess_runner.py -> RobotAgent.run()`. We add
`CompetitionTaskSkill` in an allowed directory and register it first in the
official skill library. During scored execution the official
`GATE_PLANNER=false` gate disables the live LLM planner. The skill obtains the
official zero-based task index from Agent metadata, reads the immutable
`knowledge/task_config.json`, and calls the verified workflow. Invalid task
metadata, malformed configuration, exceptions and failed physical workflows
are returned as explicit failures rather than silently falling back.

## 4. Task orchestration and recovery semantics

Each object follows:

```text
pending -> approached -> grasped -> lifted -> transported -> placed -> verified
```

The history is recorded per object. An object reaches `verified` only after the
complete physical chain succeeds. The current fixed-scene candidate uses one
attempt per object because all published-scene runs succeeded without retry; the state
machine keeps failure stage and object identity explicit for future bounded
recovery policies.

L5 is handled as three complete transport cycles, not as the official
baseline's single four-step plan. Already placed objects remain in the scene
and therefore constrain subsequent transport and release paths.

## 5. Navigation

The navigation layer uses the official occupancy grid and backend motion
interfaces. It separates global translation, station staging and final heading
alignment. Upper-row input stations use a verified upper corridor before the
final approach. Delivery targets may be inset toward the table center and any
extension is bounded by the remaining target distance.

The controller preserves the upper-body posture while rotating the base,
records every simulation frame and aborts immediately if the official backend
reports a judged collision. The L2-L5 baseline uses a stowed arm posture and
the official attachment mechanism so the payload remains synchronized with the
base without writing final object poses directly. The separate L1 incumbent
sets the object down and uses measured base-object contact to push it through
the floor corridor; it does not activate transport attachment or write the
object pose. Its navigation boundary is direct base-qpos, so it is not claimed
as complete mobile-base actuator dynamics.

## 6. Geometric grasp controller

The grasp controller uses grasp sites embedded in the official MuJoCo objects.
It computes a base pose perpendicular to the grasp-site axis, stages the robot,
orients the base, and commands both end effectors through the configured OSC
controllers. It does not teleport the object.

Success requires consecutive bilateral gripper contact and a measured object
lift relative to its pre-close height. Fine contact search is bounded; torso,
wrist and lift actions have explicit tolerances and maximum steps. Controller
goals and joint velocities are synchronized after navigation to avoid stale
state entering a grasp.

Geometry-specific behavior is isolated by object family:

- `container_h01` uses the validated closed-container profile.
- `green_tote_b01` mirrors the second arm in the world X frame.
- `blue_tote_b01` and wall-side `white_tote_b01_left` use heading-aware
  finger-pad mirroring.
- Later L5 transfers begin from a measured, collision-free torso and dual-arm
  clearance posture.

This isolation was necessary: applying the L3 heading-axis mirror globally
caused L2 to lose bilateral contact. Restoring separate profiles recovered L2
without regressing L3.

## 7. Stable placement

Single-object scenes target a safe location near the output center, release the
attachment, and allow physics to settle. L5 requires packing three
`0.600 x 0.400 m` white totes on a measured `1.683 x 0.836 m` output table.
Placing all three at the center caused later objects to push earlier objects out
of scoring range.

The final method aligns every tote to one shared world yaw and assigns target-X
offsets `-0.60`, `0.00` and `+0.60 m`. Each object first moves laterally to its
slot column and then approaches along the table direction, avoiding a path
through occupied slots. The attachment is released before a 200-frame physical
settling period. Final L5 distances are `0.60`, `0.00` and `0.60 m`, all below
the 0.8 m threshold.

## 8. SOP knowledge generation

The candidate includes an original, deterministic DOCX-to-knowledge pipeline.
It reads the five official `sop+prompt/*.docx` Open XML archives directly and
does not read the official hand-written `knowledge/sop*.md` reference files.
Task-specific Prompt text supplies material, quantity and station labels. The
result is then cross-checked against the published Erratum, immutable official
task configuration and the matching semantic map.

Embedded images are hash-addressed and analyzed offline with the public
Qwen3-VL-2B-Instruct model. Its output is restricted to five evidence fields,
uses deterministic decoding and gets at most one schema-repair attempt. Image
evidence is advisory: it cannot override the task contract, assert task success
or generate robot actions. Scored execution does not load the VLM or weights.

All five Word files generated Markdown and provenance JSON. The pipeline
analyzed 25/25 images: 20 responses passed on the first attempt and five passed
the schema-repair attempt. L2 explicitly preserves two template conflicts and
resolves them using the task Prompt and published correction. Each generated
file records source, image and raw-response hashes, validation results and any
conflict resolution.

## 9. Quantitative results

### SOP generation validation

| Measure | Result |
|---|---:|
| Original DOCX files generated | 5/5 |
| Embedded images analyzed | 25/25 |
| First-attempt schema valid | 20/25 |
| Repaired on second attempt | 5/5 |
| Task-config and semantic-map checks | 5/5 |
| Official hand-written SOP files used as input | 0 |

### Published Agent entry, local fixed-scene validation

| Level | Score | Grasps | Collision frames | Maximum target distance | Wall time |
|---|---:|---:|---:|---:|---:|
| L1 | 10/10 | 1/1 | 0 | 0.162809 m | 70.906 s |
| L2 | 15/15 | 1/1 | 0 | 0.407609 m | 65.040 s* |
| L3 | 20/20 | 1/1 | 0 | 0.333074 m | 68.378 s* |
| L4 | 25/25 | 1/1 | 0 | 0.328942 m | 101.424 s* |
| L5 | 30/30 | 3/3 | 0 | 0.600000 m | 286.648 s* |

Total: `100/100`, `7/7` grasps, zero collision frames under the local public
scorer. Asterisks mark tasks run concurrently; those wall times are not used
for ranking claims. These results were not uploaded to BienData and do not
establish performance under hidden perturbations.

### 2026-08-06 current hybrid candidate, server validation

After restoring the previously verified L1 floor-route profile, the current
candidate was materialized from the locked upstream commit and run sequentially
through all five official tasks on the server with `runner=agent`, `seed=0` and
the unmodified public scorer:

| Level | Score | Collision frames | Successful grasps | Required grasps | Frames | Wall time | Final target distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 10/10 | 0 | 2 | 1 | 14,300 | 622.193 s | 0.748171 m |
| L2 | 15/15 | 0 | 1 | 1 | 1,646 | 72.088 s | 0.420097 m |
| L3 | 20/20 | 0 | 1 | 1 | 1,744 | 75.757 s | 0.139142 m |
| L4 | 25/25 | 0 | 1 | 1 | 2,190 | 108.217 s | 0.302334 m |
| L5 | 30/30 | 0 | 3 | 3 | 6,639 | 294.743 s | 0.303742 m |

This is a fresh local acceptance result of `100/100`, `7/7` required grasps
and zero collision frames. It is a **hybrid** result: L1 uses the verified
no-attachment physical floor-contact route; L2-L5 use the official attachment
transport baseline. The result is not an organizer/Biendata score and has not
been tested under hidden perturbations.

The candidate is
`/home/user/jciiot-2026/candidates/hybrid-r3-l1-profile-5dfb364` on the
server, based on workspace commit `5dfb364-l1-profile` and official commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`. Machine-readable evidence is
archived under
`artifacts/remote-hybrid-r3-l1-profile-20260806/`, including one manifest and
one JSONL trajectory per level. The trajectory SHA-256 values are:

```text
L1 fd7f01c24e18ba2d62fc04fe236522ecf5ca7c05a7343cd7a5e6924e05664748
L2 2df8f6186d863ecf25055cba5f8d6857dccaa7baa5e2f70994e7a38fca515d62
L3 f7c4314b9b793a6c319a51e6573fa5ef3324b57c87af5c8de681b4f84e18db1d
L4 32add87c797253132a90d728b56925fb2d10eb140ac5225e731061aa4b96dbca
L5 b83f3f9031f6fdcebe85bdb0e0be23cb24cc1a193e5d7e989ecbbf6a78b4da96
```

For L1 specifically, the fresh trajectory has zero occurrences of
`attachment`, `object_pose_write`, `object qpos` or `teleport` in its runtime
event/step records. Its event ledger contains bilateral grasp/lift, physical
setdown, two inchworm transports, and three floor-base contact-push segments.
The extracted physical-contact count is 6,985 steps. This is the current
evidence for the remembered pure-physical 10/10 route; it does not make the
other four levels no-attachment.

Every manifest reports `runner.execution_mode=agent`, selected skill
`competition_task`, no planner output and every task object in `verified`
state. The protected `app.py`, `task_subprocess_runner.py` and
`knowledge/task_config.json` hashes match the official baseline.

### L1 no-attachment physical incumbent

| Evidence | Result |
|---|---:|
| Official scorer (`app._score_steps(0)`) | 10/10 |
| Independent complete trajectories | 2 |
| Frames per trajectory | 14,299 |
| Collision frames | 0 |
| Attachment calls / object-pose writes | 0 / 0 |
| Physical contact-push steps | 6,985 |
| Final target distance | 0.748201 m |

The trajectory contains bilateral physical grasp/lift events, physical
setdown, and three floor-push segments. This is a genuine no-attachment
object-motion result under MuJoCo contacts, but its direct base-qpos navigation
must be stated explicitly in any submission or paper. The complete route
registry, event audit and hashes are in
`docs/11-route-registry-and-evidence-index.md`.

### L2-L5 repeatability batch

| Level | Full-score runs | Wilson 95% interval | Grasps | Collision runs | Errors |
|---|---:|---:|---:|---:|---:|
| L2 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L3 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L4 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L5 | 20/20 | 83.8875%-100% | 60/60 | 0/20 | 0 |

The batch totals `1800/1800` local public-scorer points and `120/120` required grasp
events. Seeds did not change frame counts or final geometry, so no geometric
perturbation claim is made.

## 10. Development evidence and failure analysis

The final L5 design resulted from retained full and failed trajectories:

- Center-stacking scored 25/30 because a later tote pushed the middle tote to
  1.892 m from the target.
- Stepping contact dynamics while rotating an attached payload caused MuJoCo
  generalized-position instability.
- An early column-entry spacing pushed a previously placed tote off the table.
- The final slot and column-entry method scored 30/30 with zero collision.

These cases support the design choices more directly than reporting only the
best trajectory. However, a formal module-by-module ablation matrix remains
future work.

## 11. Novelty statement

The contribution is system and integration innovation for reliable embodied
execution in constrained industrial scenes, not a new general robot-learning
theory.

1. **Station-frame-aware bilateral grasping.** The second-arm target is mirrored
   in either a world axis or the station/robot heading axis according to object
   geometry, while a constrained inverse-kinematics path preserves physical
   reachability.
2. **Deterministic upper-body reset for repeated mobile manipulation.** Later L5
   cycles enter a measured collision-free whole-upper-body state before contact,
   removing accumulated controller history between transfers.
3. **Score-constrained stable multi-object packing.** Table and object collision
   dimensions determine shared orientation, three distinct slots and a
   column-entry sequence that avoids occupied placements.
4. **Conflict-preserving SOP evidence generation.** Task-specific Prompt facts,
   Erratum corrections, official entities and hash-addressed image evidence are
   kept separate; contradictions remain visible instead of being silently
   summarized away.
5. **Joint semantic and physical evidence ledger.** Task state advances only
   when contact, lift, transport, release and final geometry agree; LLM output
   alone cannot assert success.

Compared with the official LLM-plus-BC baseline, the scored path needs no
external checkpoint or online model call, supports all three L5 transfers, and
provides explicit physical success evidence. Compared with five unrelated
scripts, it shares one state machine, navigation contract, verification layer
and scoring protocol while isolating only geometry-dependent grasp profiles.

## 12. Third-party components

The scored path directly uses Python, NumPy, and interfaces already supplied by
the official JCIIOT baseline, including MuJoCo and robosuite. Offline SOP image
evidence uses the public Apache-2.0 Qwen3-VL-2B-Instruct model through PyTorch
and Transformers. Its weights are not bundled and are never loaded during
scored execution. No private code, private data or authorization-only tools are
used. Related repositories studied during design were not copied into the
candidate. Exact provenance, weight hash and license notes are recorded in
`config/sop-vlm-lock.json` and `THIRD_PARTY_NOTICES.md`.

## 13. Reproducibility

The code package fixes the official commit in `config/upstream-lock.json` and
provides `scripts/materialize_submission.sh`, which rejects both an incorrect
baseline commit and an overlay outside allowed paths. The official Agent runner
stores the official/workspace commits, seed, runtime, trajectory path, score,
grasp count, collision count, target distances, workflow payload and timestamps
in a machine-readable manifest.

The current route candidate is workspace commit `5dfb364-l1-profile`; the
explicit L1 extraction and floor-contact staging parameters are covered by
regression tests. A fresh materialization from the locked
upstream commit succeeds, and the scored-path audit reports `0` hard violations (the
and `0` warnings for the current submission overlay. A separate historical
whole-workspace audit recorded 28 allowed-directory backend/qpos boundary
warnings; those are retained in the evidence history and are not hidden. The corresponding reproducible
archive is `JCIIOT2026_code_and_report_20260806_hybrid_r5.zip`; the archive
SHA-256 is reported alongside the delivered file rather than embedded here,
avoiding a self-referential checksum.

The current server validation used Ubuntu 24.04.3, Python 3.11.15, MuJoCo 3.9.0
and robosuite 1.5.2. The five JSON files in the separate baseline prediction ZIP
are the exact trajectories from the clean published Agent entry validation;
the ZIP is not labeled as a final or organizer-verified result.
The SOP generator has a separate locked environment because its model stack is
not part of the simulator. Reproduction commands, 25 image records and output
checksums are included under `sop_generated/`.

## 14. Limitations and next experiments

- VLM image descriptions are advisory and may be empty or imprecise; task facts
  remain grounded in the Prompt, published Erratum, official configuration and
  semantic map.
- The 80-run batch tests process-level repeatability, not perturbations of
  object pose, base localization or dynamics.
- The public L1 BC-Transformer checkpoint records epoch 500 but a best rollout
  success rate of 0.0; direct public-evaluator replay failed 0/3 repeated
  resets in a requirements-pinned inference environment. The supplied 591 MB
  HDF5 is a 10-action Fetch/iGibson format example, not JCIIOT Tiago training
  data.
- The five-level attachment baseline and the L1 no-attachment incumbent are
  different evidence families. The current worktree has not yet produced a
  single five-level no-attachment candidate; L2-L5 physical transport remains
  research work and must not be inferred from the L1 result.
- The current L5 baseline changes private transport-attachment relative state
  for final reach. Although it does not directly write object qpos, this is a
  compliance and reviewer-perception risk and must be replaced with physical
  base/arm action before that baseline is called final.
- Disabling the planner and first-matching every task with one aggregate skill
  diverges from the training deck's standard atomic-skill story. The final
  package needs either organizer clarification or a validated atomic-skill
  execution trace.
- The official visible Streamlit Execute button was not exercised on the
  headless server because no accessible X display or Xvfb was available. The
  same unmodified `RobotAgent.run()` path was tested headlessly.
- L5 is reliable but slow. Settling and clearance-step ablations should reduce
  time only while preserving full score and zero collisions.
- A formal ablation table, explicit pose/dynamics stress test and synchronized
  five-scene video remain necessary for the final innovation submission.

## 15. Evidence index

- `experiments/2026-07-28-five-level-performance-baseline.md`
- `experiments/2026-07-28-multiseed-stability.md`
- `experiments/2026-07-28-official-agent-entrypoint.md`
- `experiments/2026-07-28-sop-qwen3vl-generation.md`
- `sop_generated/README.md`
- `sop_generated/generated_sop_manifest.json`
- `config/sop-vlm-lock.json`
- `docs/09-current-route-and-optimization-plan.md`
- `docs/11-route-registry-and-evidence-index.md`
- `THIRD_PARTY_NOTICES.md`
- `config/upstream-lock.json`
