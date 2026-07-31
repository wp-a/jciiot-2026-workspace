# Video-Aligned Robust Full-Score Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a rules-compliant five-level JCIIOT solution that follows the official video, obtains each official maximum score with zero collision, and demonstrates repeatable robustness rather than a single lucky trajectory.

**Architecture:** Keep the official app, scorer, core, environments, and task configuration immutable. Use a deterministic workflow for semantic routing, collision-aware navigation, physical-grasp gating, official attachment-based long-distance transport, constrained lowering, release, and verification; insert a learned policy only behind the same pickup interface after the deterministic level-1 gate passes. Every change is driven by an official manifest and accepted only against nominal and explicit perturbation gates.

**Tech Stack:** Python 3.11/3.13, NumPy, MuJoCo, robosuite Tiago, official Streamlit app/scorer, robomimic BC-RNN, PyTorch Diffusion Policy, HDF5, pytest/unittest, Git, remote NVIDIA L40S server.

---

## Fixed Success Criteria

Level 1 is complete only when all of the following are true:

- the unmodified official app reports `10/10` and zero collision;
- five consecutive nominal runs report `10/10` and zero collision;
- at least 18 of 20 deterministic perturbation runs report `10/10`;
- all 20 perturbation runs report zero collision;
- every accepted trajectory records a successful physical grasp event before
  transport attachment capture and a stable released pose at the destination;
- protected hashes and the scored-path hard-violation count remain clean.

Levels 2-4 use the same gate with official maximum scores `15`, `20`, and `25`.
Level 5 uses maximum score `30`, requires three verified object transfers in one
run, and applies the same repeated-run and zero-collision requirements.

## Phase A: Make The Measurement Correct

### Task 1: Propagate Perturbations Through The Batch Runner

**Files:**
- Modify: `scripts/run_official_batch.py`
- Modify: `tests/test_official_batch.py`

**Step 1: Write the failing command-propagation test**

Add a test that builds a small-tier `BatchJob`, calls `_experiment_command`, and
asserts the command contains:

```python
self.assertIn("--perturbation-tier", command)
self.assertEqual(command[command.index("--perturbation-tier") + 1], "small")
```

Also assert that non-nominal labels include their tier so nominal and perturbed
manifests cannot overwrite each other.

**Step 2: Run the focused test and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_official_batch.py
```

Expected: failure because `BatchJob` and `_experiment_command` do not carry a
perturbation tier.

**Step 3: Implement the minimal propagation**

Add `perturbation_tier` to `BatchJob` and `build_jobs`. Preserve the existing
nominal label; append `-small`, `-medium`, or `-stress` for non-nominal jobs.
Append this exact pair in `_experiment_command`:

```python
"--perturbation-tier",
job.perturbation_tier,
```

Add the parser choice `nominal,small,medium,stress`, pass it through `run_batch`,
and record it in `batch-summary.json`.

**Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3.13 -m pytest -q tests/test_official_batch.py tests/test_perturbation_protocol.py tests/test_official_experiment.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/run_official_batch.py tests/test_official_batch.py
git commit -m "fix: propagate perturbations through official batches"
```

### Task 2: Add A Machine-Checkable Level-1 Robustness Gate

**Files:**
- Create: `scripts/evaluate_l1_full_score_gate.py`
- Create: `tests/test_l1_full_score_gate.py`

**Step 1: Write failing gate tests**

Create synthetic manifests and test these cases independently:

```python
report = evaluate_l1_gate(nominal_manifests, perturbation_manifests)
assert report["nominal_full_score_runs"] == 5
assert report["perturbation_full_score_runs"] == 18
assert report["collision_runs"] == 0
assert report["gate_passed"] is True
```

Also test rejection for 4 nominal runs, 17 perturbed full scores, one collision,
an invalid perturbation audit, a missing grasp event, and a final target distance
of `0.8` m or greater.

**Step 2: Verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_l1_full_score_gate.py
```

Expected: import failure because the evaluator does not exist.

**Step 3: Implement the pure evaluator and CLI**

Reuse `acceptance_met` from `scripts/run_official_experiment.py`. Require at
least five nominal and exactly the selected 20 perturbation manifests. Require
`perturbation_application.valid == true` for every non-nominal manifest. Emit a
JSON report with counts, score distribution, collision count, failure-stage
histogram, selected manifest paths, and `gate_passed`.

The CLI exits `0` only when the complete gate passes and `1` otherwise.

**Step 4: Verify GREEN**

Run:

```bash
python3.13 -m pytest -q tests/test_l1_full_score_gate.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/evaluate_l1_full_score_gate.py tests/test_l1_full_score_gate.py
git commit -m "feat: enforce the robust level-one score gate"
```

### Task 3: Rebuild And Audit A Clean Candidate

**Files:**
- Read: `config/upstream-lock.json`
- Read: `submission/JCIIOT/src/robot_agent/skills/`
- Read: `submission/JCIIOT/src/robot_agent/workflows/`
- Output: ignored temporary candidate directory
- Output: `artifacts/video-aligned-l1-local/scored-path-audit.json`

**Step 1: Run the complete owned test suite**

Run:

```bash
python3.13 -m pytest -q tests
```

Expected: all owned tests pass. Vendor collection is not part of this local
command because MuJoCo is available only in the server environment.

**Step 2: Run the submission audit**

Run:

```bash
python3.13 scripts/audit_scored_path.py --root submission --output artifacts/video-aligned-l1-local/scored-path-audit.json
```

Expected: exit `0` and `hard_violation_count` equals `0`.

**Step 3: Materialize from the pinned official checkout**

Run `scripts/materialize_submission.sh` with:

- official root: `vendor/JCIIOT2026`
- overlay: `submission`
- output: a new child of a `mktemp -d` directory

Expected: the script reports official commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`.

**Step 4: Verify protected hashes**

Check these materialized files against `config/upstream-lock.json`:

- `JCIIOT/app.py`
- `JCIIOT/knowledge/task_config.json`
- `JCIIOT/src/robot_agent/environments/robosuite_backend.py`

Expected: all three SHA-256 values match the lock file.

**Step 5: Record the candidate identity**

Write the overlay commit, official commit, candidate tree hash, audit hash, and
test command/result to `artifacts/video-aligned-l1-local/manifest.json`. Do not
commit the materialized official repository or large artifacts.

## Phase B: Obtain And Diagnose The First Official Task-1 Result

### Task 4: Restore The Official Server Evaluation Path

**Files:**
- No repository changes
- Output: remote `results/video-aligned-l1-<overlay-commit>/environment.json`

**Step 1: Verify network reachability**

Run:

```bash
nc -vz -w 5 211.87.224.136 28897
```

Expected: TCP connection succeeds. If it times out, reconnect to the known
campus network or authorized VPN before continuing; code changes cannot repair
this boundary.

**Step 2: Verify the server environment without changing it**

Connect as the registered competition user and record:

```bash
nvidia-smi
docker ps
df -h /home/user
git -C /home/user/jciiot-2026/official rev-parse HEAD
```

Expected: an L40S device is available, disk capacity is sufficient, and the
official checkout is the pinned commit. Never update the locked checkout in
place.

**Step 3: Upload and materialize into a new candidate directory**

Upload only the current overlay and experiment scripts. Run the remote
materializer into:

`/home/user/jciiot-2026/candidates/video-aligned-l1-<overlay-commit>`

Expected: remote overlay hashes equal the local overlay hashes and protected
hashes remain unchanged.

**Step 4: Preserve environment metadata**

Record Python, MuJoCo, robosuite, Torch, CUDA, GPU, official commit, overlay
commit, model hash, and candidate path in `environment.json` before executing a
trajectory.

### Task 5: Run One Nominal Official Diagnostic

**Files:**
- Output: remote `results/video-aligned-l1-<overlay-commit>/diagnostic/manifest.json`
- Output: remote `results/video-aligned-l1-<overlay-commit>/diagnostic/trajectory.json`
- Download: `artifacts/video-aligned-l1-<overlay-commit>/diagnostic/`

**Step 1: Run the flow entry point once**

Run `scripts/run_official_experiment.py` with task index `0`, perturbation tier
`nominal`, one attempt, the pinned official commit, and the current overlay
commit. Use the clean candidate root created in Task 4.

Expected: a complete manifest and trajectory exist even if the task fails.

**Step 2: Run the official app once**

Start the unmodified `app.py`, select task 1, click `Execute`, and save the
visible score output. This is a separate verification of the user-facing 8502
path; the automated runner is not a substitute for it.

**Step 3: Classify exactly one terminal failure stage**

Use this priority order:

1. `move_source`
2. `pregrasp` or `approach`
3. `close` or bilateral contact
4. `lift` or `hold`
5. `transport_attachment`
6. loaded `navigation` or collision
7. `official_place`
8. final `verify`

Record observed base/object pose, contacts, lift delta, hold duration,
attachment event, collision frames, source departure, destination distance,
score, and elapsed time. Do not tune a stage that the evidence shows passed.

**Step 4: Download all evidence before changing code**

Download the manifest, trajectory, stdout, environment manifest, and any GIF.
Verify SHA-256 after transfer. Zero-score GIFs remain evidence but are not
presented as successful demos.

## Phase C: Close Task-1 Failures One At A Time

### Task 6: Run A Metric-Driven Single-Stage Optimization Loop

**Files:**
- Modify only as indicated by the measured stage:
  - `submission/JCIIOT/src/robot_agent/skills/competition_navigation.py`
  - `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`
  - `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`
  - `submission/JCIIOT/knowledge/robot_params.json` when a public parameter is sufficient
- Test only the matching modules first:
  - `tests/test_competition_navigation.py`
  - `tests/test_competition_grasp.py`
  - `tests/test_competition_flow.py`
- Create per experiment: `autoresearch/classic-<date>-l1-<failure-stage>/`

**Step 1: Pre-register one hypothesis**

Write the observed failure, one causal hypothesis, one controlled change, the
primary metric, safety invariants, and keep/discard threshold. Examples:

- navigation: minimum clearance or path feasibility;
- grasp: bilateral contact and lift-hold success;
- transport: official attachment active plus zero collision;
- place: released target distance and stable support.

**Step 2: Write one failing regression test**

The test must encode the observed boundary, not merely assert a constant. For a
grasp-gate bug, construct a result with the exact missing contact/lift/hold
condition. For a route bug, test the measured obstacle geometry and clearance.
For a place bug, test active attachment, official release invocation, and final
verification separately.

**Step 3: Verify RED**

Run the single new test by node id and confirm it fails for the intended reason.

**Step 4: Implement the smallest allowed correction**

Do not modify `core`, `environments`, `app.py`, or `task_config.json`. Do not
write object qpos or call `sync_transport_attachment` from submission-owned
code. Keep attachment capture behind `verified_transport_grasp`.

**Step 5: Verify GREEN and local regression**

Run the focused module, then:

```bash
python3.13 -m pytest -q tests
python3.13 scripts/audit_scored_path.py --root submission --output artifacts/scored-path-audit-candidate.json
git diff --check
```

Expected: all tests pass and hard violations remain zero.

**Step 6: Run one official diagnostic**

Materialize a new clean remote candidate and rerun the same task, seed, and
perturbation. Never patch the previous candidate in place.

**Step 7: Keep or discard**

Keep only when the targeted physical metric improves without a collision,
protected-file change, attachment-before-grasp event, or regression in an
already-passed stage. Record both kept and discarded trials in `results.tsv`.

**Step 8: Commit a kept change**

Stage only the relevant allowed source, its regression test, and the experiment
record. Preserve unrelated dirty files.

Repeat Task 6 until one nominal official run reports task 1 `10/10` and zero
collision. This first success is a milestone, not the robustness gate.

### Task 7: Pass Five Consecutive Nominal Runs

**Files:**
- Output: remote `results/video-aligned-l1-<commit>/nominal-five/`
- Download: `artifacts/video-aligned-l1-<commit>/nominal-five/`

**Step 1: Freeze the candidate**

Record the candidate and dependency hashes. No code or parameter change is
allowed during the five runs.

**Step 2: Run five nominal seeds serially**

Use task index `0`, tier `nominal`, seeds `2026073101` through `2026073105`, one
worker, and one attempt. Serial execution avoids shared MuJoCo or recorder state.

**Step 3: Evaluate the nominal half of the gate**

Require all five manifests to have official score `10`, zero collision frames,
one verified grasp event, successful execution, and target distance `< 0.8` m.

**Step 4: Diagnose any failure without filtering it**

If one run fails, the series fails. Return to Task 6 with the failed seed and
stage. Do not replace or rerun a failed seed to improve the reported rate.

## Phase D: Prove Task-1 Robustness

### Task 8: Run The 20-Case Perturbation Matrix

**Files:**
- Create: `experiments/l1-video-aligned-robustness/protocol.md`
- Output: remote `results/video-aligned-l1-<commit>/perturbation-20/`
- Create after run: `experiments/l1-video-aligned-robustness/analysis.md`
- Download: `artifacts/video-aligned-l1-<commit>/perturbation-20/`

**Step 1: Pre-register the immutable matrix**

Use these selected cases:

- seeds `2026073111`-`2026073118`, tier `small`;
- seeds `2026073121`-`2026073128`, tier `medium`;
- seeds `2026073131`-`2026073134`, tier `stress`.

The existing protocol jointly samples robot XY/yaw, object XY/yaw, and, for
medium/stress, mass and friction. Record all generated values before running.

**Step 2: Run the matrix serially**

Use the corrected batch runner with task index `0`, one attempt, one worker,
and the frozen candidate. Run each tier into a distinct output directory.

**Step 3: Validate the perturbations**

Require `perturbation_application.valid == true` and measured nonzero changes
for every non-nominal run. An invalid injection invalidates the benchmark rather
than counting as a policy failure.

**Step 4: Evaluate the full gate**

Run `scripts/evaluate_l1_full_score_gate.py` across the five nominal and 20
perturbed manifests.

Expected:

- nominal full-score runs: `5/5`;
- perturbation full-score runs: at least `18/20`;
- collision runs: `0/25`;
- gate passed: `true`.

**Step 5: Iterate honestly if the gate fails**

Group failures by stage and perturbation magnitude. Return to Task 6 with the
most frequent safety-preserving failure stage. Any candidate change invalidates
the earlier five/20 gate and requires the whole gate to be rerun.

**Step 6: Produce accepted visual evidence**

Generate robot first-person, overhead, and third-person GIFs only for an
accepted official full-score trajectory. Confirm the frames show physical
pickup before transport and stable released placement.

## Phase E: Add Learning Only Where It Helps

### Task 9: Lock The Task-Native Tiago Dataset Contract

**Files:**
- Create: `config/tiago-pickup-dataset-schema.json`
- Create: `scripts/validate_tiago_pickup_dataset.py`
- Create: `tests/test_validate_tiago_pickup_dataset.py`

**Step 1: Write failing HDF5 contract tests**

Accept only task-native Tiago demonstrations with the observed official
observation keys, 20-dimensional actions, finite values, matching time lengths,
physical grasp/lift success metadata, and disjoint train/validation/test seeds.
Reject the provided dishwasher Fetch dataset explicitly.

**Step 2: Verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_validate_tiago_pickup_dataset.py
```

Expected: import failure.

**Step 3: Implement the validator**

Emit environment, robot, observation shapes, action width, demonstration count,
object-family counts, perturbation-tier counts, success labels, split hashes,
and validation errors. Exit nonzero on any mismatch.

**Step 4: Verify GREEN and the negative control**

Run the tests, then run the validator against
`table_setup_from_dishwasher_sample.hdf5`.

Expected: tests pass; the Fetch file is rejected for environment, robot,
observation, and action-schema mismatch.

**Step 5: Commit**

```bash
git add config/tiago-pickup-dataset-schema.json scripts/validate_tiago_pickup_dataset.py tests/test_validate_tiago_pickup_dataset.py
git commit -m "feat: validate native Tiago pickup data"
```

### Task 10: Collect Verified Pickup Demonstrations

**Files:**
- Create: `scripts/export_verified_pickup_hdf5.py`
- Create: `tests/test_export_verified_pickup_hdf5.py`
- Output: remote private dataset directory under `/home/user/jciiot-2026/data/`
- Create: `experiments/task-native-pickup-data/manifest.json`

**Step 1: Test segment extraction**

Given a trajectory with `grasp_start`, issued actions, observations,
`grasp_end`, and attachment capture, extract only the segment from grasp start
through the verified hold. Reject missing actions, failed gates, collision
frames, and attachment capture before grasp end.

**Step 2: Implement exact-action collection**

Record observations and issued 20-dimensional Tiago actions at control time;
do not train from visualization-only interpolated frames. Store scene, object,
robot/object initial pose, seed, contacts, lift, hold, and collision metadata.

**Step 3: Pass the one-demonstration replay gate**

Export one successful task-1 pickup, validate it, overfit a tiny BC-RNN, and
require one physical replay before collecting at scale. If replay fails, debug
observation/action alignment instead of adding data or epochs.

**Step 4: Collect the first balanced dataset**

Collect at least 50 verified task-1 pickup demonstrations covering nominal,
small, and medium perturbations. Keep failed attempts in a separate analysis
set, not as positive demonstrations.

**Step 5: Freeze split and hashes**

Split by initial-condition seed, not by time step. Save HDF5 SHA-256, split
manifest, environment versions, collector commit, and per-family counts.

### Task 11: Compare BC-RNN And Diffusion Policy Fairly

**Files:**
- Create: `config/training/bc-rnn-pickup.json`
- Create: `config/training/diffusion-pickup.json`
- Create: `scripts/evaluate_pickup_policy.py`
- Create: `tests/test_evaluate_pickup_policy.py`
- Create: `experiments/pickup-policy-comparison/protocol.md`

**Step 1: Test the promotion gate**

The evaluator must require physical bilateral contact, lift, stable hold, zero
collision, bounded inference latency, correct dataset/model hashes, and no
train/test seed overlap. Loss alone cannot promote a model.

**Step 2: Train BC-RNN first**

Use the official robomimic-compatible observation/action adapter. Train three
seeds with early stopping on held-out physical pickup success. Preserve config,
logs, checkpoints, and exact environment metadata.

**Step 3: Train Diffusion Policy on the identical split**

Use the same observations, action normalization, training examples, evaluation
seeds, and aggregate GPU-hour budget. Tune action horizon only on validation,
never on the official test cases.

**Step 4: Run the physical comparison**

Evaluate deterministic pickup, BC-RNN, and Diffusion Policy on the same held-out
matrix. Report success, bilateral contact, lift-hold, collision, mean attempts,
and inference latency with confidence intervals.

**Step 5: Promote only a real improvement**

Replace the deterministic pickup on a bounded object family only when a learned
policy improves held-out full pickup success and does not increase collision.
Otherwise keep the deterministic controller and retain the learned result as an
honest negative result.

**Step 6: Re-run the complete level-1 gate**

Any promoted policy must pass Tasks 7 and 8 from scratch through the same
workflow and official scorer.

## Phase F: Extend Sequentially To All Five Levels

### Task 12: Promote Levels 2 Through 4 One At A Time

**Files:**
- Modify only allowed skill/workflow/parameter files when evidence requires it
- Create per level: `experiments/l<level>-full-score/`

For each level:

1. Run one immutable nominal diagnostic.
2. Classify the terminal failure stage.
3. Use Task 6's one-change loop.
4. Obtain one official maximum-score, zero-collision run.
5. Pass five consecutive nominal runs.
6. Pass the 20-case perturbation gate.
7. Commit the accepted level before proceeding.

Do not tune a later level by regressing an earlier gate. After each promotion,
rerun one nominal regression for every previous level.

### Task 13: Complete The Three-Object Level 5 Workflow

**Files:**
- Modify when required: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`
- Modify when required: `submission/JCIIOT/src/robot_agent/skills/competition_navigation.py`
- Test: `tests/test_competition_flow.py`
- Test: `tests/test_competition_navigation.py`

**Step 1: Write state-isolation tests**

Assert that attachment, held-object metadata, selected grasp pose, and retry
state are cleared after each released and verified object. Assert that a failed
object stops the sequence and cannot mark later objects complete.

**Step 2: Verify RED and implement the minimal reset**

Only add reset behavior shown missing by the test. Keep official release and
physical gate invariants unchanged.

**Step 3: Optimize object order and delivery slots**

Use semantic geometry and collision-aware path cost. Maintain distinct scored
positions for all three boxes and verify each release before selecting the next.

**Step 4: Pass the full level-5 gate**

Require official `30/30`, three verified grasp events, all three final distances
`< 0.8` m, and zero collision, followed by five nominal and 20 perturbed runs.

## Phase G: Final Compliance And Submission

### Task 14: Produce The Reproducible Final Evidence Bundle

**Files:**
- Modify: `README.md`
- Modify: `docs/09-current-route-and-optimization-plan.md`
- Modify: `STATUS.md`
- Modify: `experiments/experiment-log.csv`
- Create: `submission-evidence/final-score-matrix.json`
- Create: `submission-evidence/model-manifest.json`
- Create: `submission-evidence/reproduction.md`

**Step 1: Run the completion audit**

Verify all five official maximum scores, zero collision, physical-grasp events,
stable release, repeated robustness gates, protected hashes, overlay scope, and
model/data hashes from authoritative manifests.

**Step 2: Run the final local checks**

```bash
git diff --check
python3.13 -m pytest -q tests
bash scripts/check_workspace.sh --require-private-remote
python3.13 scripts/audit_scored_path.py --root submission --output submission-evidence/final-scored-path-audit.json
```

Expected: all commands pass and the audit has zero hard violations.

**Step 3: Reproduce once from a clean materialization**

Materialize from the pinned official commit, install from documented dependency
locks, run all five official tasks through the unmodified app/scorer, and verify
the generated hashes match the evidence manifest.

**Step 4: Generate final media**

For accepted full-score runs, produce first-person, overhead, third-person, and
combined GIFs. Record source trajectory and GIF SHA-256 values.

**Step 5: Build the required ZIP**

Include only allowed submission code/configuration, reproducibility instructions,
technical report, novelty statement, model or public retrieval instructions,
score evidence, and optional videos. Verify the ZIP extracts cleanly and does
not contain secrets, private datasets, caches, or forbidden modified files.

**Step 6: Final claim rule**

Do not describe the solution as full score until every item in the completion
audit is proved by the unmodified official environment. A local test, one GIF,
or one successful seed is not sufficient evidence.
