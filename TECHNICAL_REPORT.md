# JCIIOT 2026 Technical Report

## 1. Executive summary

We implement a deterministic, verifiable mobile-manipulation system for the
five official JCIIOT factory-sorting scenes. The design combines semantic task
resolution, a per-object state machine, collision-aware base navigation,
object-family geometric grasp control, physically verified transport and
stable multi-object placement. A language model is not allowed to issue base
or joint commands on the scored path.

On official baseline commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`, the final candidate scores
`10/10`, `15/15`, `20/20`, `25/25` and `30/30`, for `100/100` total, with
seven required successful grasp events and zero collision frames. An additional
80-process repeatability batch on L2-L5 produced 80/80 full-score runs,
120/120 required grasp events and zero collision frames. These repetitions do
not perturb scene geometry, so they establish execution repeatability rather
than pose-distribution robustness.

## 2. Task and scoring constraints

Each scene requires a Tiago mobile manipulator to transport one or more named
objects from an input station to an output station. The official scorer awards
half of a scene's points when a successfully grasped object leaves the source
by more than 1 m on either planar axis, and the other half when its final planar
distance to the target-table center is below 0.8 m. L5 scores three white totes
individually. Any collision recorded in the trajectory causes a five-point
deduction; time is used only to rank equal scores.

Our acceptance gate is stricter than the point total alone. It requires full
official score, every required successful `grasp_end` event, zero collision
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
  -> stowed transport with official attachment semantics
  -> target-table release and multi-object slot assignment
  -> final state, collision and event verification
  -> official trajectory recorder and unmodified scorer
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
complete physical chain succeeds. The current final candidate uses one attempt
per object because all official-scene runs succeeded without retry; the state
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
reports a judged collision. Carrying motion uses a stowed arm posture and the
official attachment mechanism so the payload remains synchronized with the
base without writing final object poses directly.

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

### Official Agent entry validation

| Level | Score | Grasps | Collision frames | Maximum target distance | Wall time |
|---|---:|---:|---:|---:|---:|
| L1 | 10/10 | 1/1 | 0 | 0.162809 m | 70.906 s |
| L2 | 15/15 | 1/1 | 0 | 0.407609 m | 65.040 s* |
| L3 | 20/20 | 1/1 | 0 | 0.333074 m | 68.378 s* |
| L4 | 25/25 | 1/1 | 0 | 0.328942 m | 101.424 s* |
| L5 | 30/30 | 3/3 | 0 | 0.600000 m | 286.648 s* |

Total: `100/100`, `7/7` grasps, zero collision frames. Asterisks mark tasks
run concurrently; those wall times are not used for ranking claims.

Every manifest reports `runner.execution_mode=agent`, selected skill
`competition_task`, no planner output and every task object in `verified`
state. The protected `app.py`, `task_subprocess_runner.py` and
`knowledge/task_config.json` hashes match the official baseline.

### L2-L5 repeatability batch

| Level | Full-score runs | Wilson 95% interval | Grasps | Collision runs | Errors |
|---|---:|---:|---:|---:|---:|
| L2 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L3 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L4 | 20/20 | 83.8875%-100% | 20/20 | 0/20 | 0 |
| L5 | 20/20 | 83.8875%-100% | 60/60 | 0/20 | 0 |

The batch totals `1800/1800` official points and `120/120` required grasp
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

The final server validation used Ubuntu 24.04.3, Python 3.11.15, MuJoCo 3.9.0
and robosuite 1.5.2. The five validation JSON files in the separate prediction
ZIP are the exact trajectories from the clean official Agent entry validation.
The SOP generator has a separate locked environment because its model stack is
not part of the simulator. Reproduction commands, 25 image records and output
checksums are included under `sop_generated/`.

## 14. Limitations and next experiments

- VLM image descriptions are advisory and may be empty or imprecise; task facts
  remain grounded in the Prompt, published Erratum, official configuration and
  semantic map.
- The 80-run batch tests process-level repeatability, not perturbations of
  object pose, base localization or dynamics.
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
- `THIRD_PARTY_NOTICES.md`
- `config/upstream-lock.json`
