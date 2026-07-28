# Honest Perturbation And Policy Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auditable perturbation benchmark and compliance gate, measure the current candidate honestly, and prepare the exact Tiago data interface required before policy training.

**Architecture:** Keep the locked official checkout and scorer unchanged. A research-only runner applies a deterministic, recorded perturbation after environment reset and before trajectory recording, then delegates to the same flow or Agent entry and unmodified scorer. Pure functions define perturbation sampling and compliance scanning so they can be test-driven locally; server-only MuJoCo integration is verified with nominal and perturbed smoke runs before any large batch or training.

**Tech Stack:** Python 3.11+, NumPy, MuJoCo 3.9, robosuite 1.5.2, bundled robomimic 0.5.0, HDF5, unittest/pytest, Git, four NVIDIA L40S GPUs.

---

### Task 1: Initialize The Autoresearch Ledger

**Files:**
- Create: `research-state.yaml`
- Create: `research-log.md`
- Create: `findings.md`
- Create: `literature/survey.md`
- Create: `literature/robomimic-v05.md`
- Create: `literature/mimicgen.md`
- Create: `literature/diffusion-policy.md`
- Create: `literature/act.md`

**Step 1: Record the fixed objective and baseline**

Set `research-state.yaml` to contain the official commit, current fixed-scene
baseline, score/safety/robustness metric order, hypotheses H1-H4, and the next
experiment. H1 predicts the apparent multi-seed stability will drop under
explicit pose perturbation. H2 predicts object-relative geometric teaching
will outperform fixed waypoint replay. H3 predicts BC-Transformer will exceed
BC-RNN only after the data and action interfaces are correct. H4 predicts
Diffusion Policy or ACT is useful only if simple BC failures remain multimodal.

**Step 2: Preserve the existing facts**

Write `findings.md` with the verified fixed-public-scene evidence, the failed
official checkpoint replay, the incompatible Fetch sample HDF5, and the known
private-state compliance risks. Do not convert any local score into an official
claim.

**Step 3: Save primary-source summaries**

Summarize the official robomimic v0.5 release, MimicGen, Diffusion Policy, and
ACT. Each file must include title, authors/project owner, year or release,
mechanism, evidence, limits for JCIIOT, URL, and access date.

**Step 4: Verify and commit**

Run: `git diff --check && test -s research-state.yaml && test -s findings.md && test -s literature/survey.md`

Expected: exit 0.

Commit: `research(init): robust JCIIOT policy research`

### Task 2: Define Deterministic Perturbation Specifications

**Files:**
- Create: `scripts/perturbation_protocol.py`
- Create: `tests/test_perturbation_protocol.py`

**Step 1: Write failing sampling tests**

Test that `tier_limits("small")` returns the locked small limits; the same
`seed`, `task_index`, and object name produce identical samples; different
seeds change at least one sampled field; nominal is exactly zero; and all
sampled values remain inside their tier bounds.

The desired public API is:

```python
sample = sample_perturbation(
    tier="small",
    seed=20260728,
    task_index=0,
    object_name="line_5_container_h01_near",
)
sample.as_dict()
```

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_perturbation_protocol.py`

Expected: FAIL because `scripts.perturbation_protocol` does not exist.

**Step 3: Implement the minimal pure model**

Create frozen dataclasses `TierLimits` and `PerturbationSample`. Define nominal,
small, medium, and stress constants from the approved design. Seed a local
`random.Random` with a stable SHA-256 digest of the input tuple; never use the
process-global random state. Sample uniform signed offsets for object XY/yaw,
base XY/yaw, mass scale, and friction scale.

**Step 4: Verify GREEN**

Run: `python3.13 -m pytest -q tests/test_perturbation_protocol.py`

Expected: all tests pass.

**Step 5: Commit**

Commit: `feat: define deterministic perturbation protocol`

### Task 3: Apply And Audit Perturbations In The Research Runner

**Files:**
- Modify: `scripts/run_official_experiment.py`
- Modify: `tests/test_official_experiment.py`

**Step 1: Write failing integration-unit tests**

Add tests for these desired functions:

```python
resolve_scored_object(task, requested_name=None) -> str
apply_perturbation(backend, task, sample) -> dict
```

Use a small fake MuJoCo model/data object. Assert object free-joint translation
and quaternion yaw are updated, joint velocity is zeroed, base perturbation is
delegated through an injected setter, mass/friction scales are applied only to
the scored object's geoms/body, `sim.forward()` is called, and the returned
audit records before/after values. Test nominal as a no-op.

Add parser flags `--perturbation-tier` and `--perturbation-object`. Assert the
manifest contains both the requested sample and measured application audit.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_official_experiment.py`

Expected: FAIL on missing perturbation API or parser flags.

**Step 3: Implement the research-only injector**

Use the official object's registered free joint. Compose yaw with the existing
quaternion, preserve roll/pitch and Z, zero its six-dimensional joint velocity,
scale only matching body mass and collision-geom friction, and call
`sim.forward()`. Apply base XY/yaw after object placement through a dedicated
research helper copied from the official coordinate mapping, not through the
submission overlay. Measure the resulting world pose and reject if it differs
from the request by more than 1 mm or 0.1 degrees.

Call the injector after `_load_scene` and before `start_recording`. Save the
sample and measured audit in every complete or error manifest.

**Step 4: Verify GREEN and regression**

Run: `python3.13 -m pytest -q tests/test_official_experiment.py tests/test_official_batch.py`

Expected: all tests pass.

**Step 5: Commit**

Commit: `feat: add audited perturbation runner`

### Task 4: Add A Scored-Path Compliance Scanner

**Files:**
- Create: `scripts/audit_scored_path.py`
- Create: `tests/test_scored_path_audit.py`

**Step 1: Write failing scanner tests**

Use temporary Python fixtures and assert that AST inspection reports assignments
to `.sim.data.qpos[...]`, assignments below a `relative_xy` subscript, imports or
calls of `sync_transport_attachment`, and private backend calls. Assert normal
`backend.step(action)` and read-only observations are accepted.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_scored_path_audit.py`

Expected: FAIL because the scanner does not exist.

**Step 3: Implement minimal AST scanning**

Scan only `submission/JCIIOT/src/robot_agent/skills` and `workflows`. Emit JSON
with file, line, rule, and source excerpt. The CLI exits 1 when violations exist
and 0 otherwise. Keep rules explicit; do not reject all private names because
trajectory recording currently uses documented private hooks that need a
separate organizer decision.

**Step 4: Verify GREEN and measure the incumbent**

Run: `python3.13 -m pytest -q tests/test_scored_path_audit.py`

Expected: all tests pass.

Run: `python3.13 scripts/audit_scored_path.py --root submission --output artifacts/scored-path-audit-current.json`

Expected: exit 1 and a complete, machine-readable list of the incumbent's known
violations. This nonzero result is a measurement, not a test-suite failure.

**Step 5: Commit**

Commit: `feat: audit scored-path compliance`

### Task 5: Lock The First Perturbation Experiment Protocol

**Files:**
- Create: `experiments/h1-explicit-pose-perturbation/protocol.md`
- Modify: `research-state.yaml`
- Modify: `research-log.md`

**Step 1: Pre-register H1**

Run L1 and L5 nominal smoke once, then L1 small-pose perturbation for five
preselected seeds. Primary metric is full-score rate. Secondary metrics are
verified grasp/lift, collision frames, final distance, failure stage, and
elapsed time. No candidate change is allowed during this experiment.

**Step 2: Define sanity gates**

The nominal smoke must reproduce the fixed-scene baseline. Every perturbed run
must record a nonzero measured pose change inside tolerance. If either condition
fails, label the experiment invalid and fix the runner before interpreting task
performance.

**Step 3: Verify and protocol commit**

Run: `git diff --check`

Expected: exit 0.

Commit before any server run: `research(protocol): test explicit pose robustness`

### Task 6: Run Server Smoke And H1 Batch

**Files:**
- Create after results: `experiments/h1-explicit-pose-perturbation/analysis.md`
- Create after results: `experiments/h1-explicit-pose-perturbation/results/summary.json`
- Modify after results: `findings.md`
- Modify after results: `research-state.yaml`
- Modify after results: `research-log.md`

**Step 1: Verify the server and materialize the candidate**

Check the remote official commit, GPU inventory, free disk, and absence of an
active stale experiment. Materialize the overlay onto a fresh copy of the
locked official checkout. Never modify the locked checkout in place.

**Step 2: Run nominal L1 and L5 smoke**

Use the existing pinned Python environment and unmodified scorer. Preserve
stdout, manifest, trajectory, package versions, candidate commit, and SHA-256.

Expected: nominal L1 and L5 meet their fixed-scene gates. If they do not, stop
H1 interpretation and diagnose environment or materialization drift.

**Step 3: Run five L1 small-pose seeds**

Use separate processes and retain every result. Do not retry or filter failures.

**Step 4: Independently aggregate**

Validate each manifest and trajectory, calculate full-score and collision rates
with Wilson intervals, and group failures by stage. Confirm the measured
perturbations differ across seeds.

**Step 5: Record results and commit separately from the protocol**

Commit: `research(results): measure explicit pose robustness`

### Task 7: Define And Validate The Tiago Dataset Contract

**Files:**
- Create: `scripts/validate_tiago_dataset.py`
- Create: `tests/test_validate_tiago_dataset.py`
- Create: `config/tiago-dataset-schema.json`

**Step 1: Write failing synthetic-HDF5 tests**

Generate tiny temporary HDF5 fixtures. Accept exact 20-dimensional actions and
required 128 x 128 robot-view plus two-arm proprioception keys. Reject Fetch
environment metadata, action width 10, missing observations, non-finite actions,
empty demonstrations, and train/validation seed overlap.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_validate_tiago_dataset.py`

Expected: FAIL because the validator does not exist.

**Step 3: Implement minimal validation**

Emit JSON containing demo/sample counts, action min/max, observation shapes,
environment, robot, perturbation-tier counts, split seed hashes, and all errors.
Exit nonzero on any contract violation.

**Step 4: Verify GREEN and reject the official sample**

Run: `python3.13 -m pytest -q tests/test_validate_tiago_dataset.py`

Expected: all tests pass.

Run the validator on the downloaded official Fetch sample.

Expected: nonzero exit with explicit robot, action-width, resolution, and
observation-schema failures.

**Step 5: Commit**

Commit: `feat: validate Tiago training datasets`

### Task 8: Gate One-Trajectory Overfit Before Scaling

**Files:**
- Create: `experiments/h2-one-demo-overfit/protocol.md`
- Create: `config/training/bc-rnn-lowdim.json`
- Create: `config/training/bc-transformer-image.json`
- Create: `scripts/run_policy_gate.py`
- Create: `tests/test_policy_gate.py`

**Step 1: Test the gate**

Require valid dataset schema, finite decreasing training loss, checkpoint
metadata matching the dataset hash and action schema, and at least one
successful physical replay through the exact evaluation adapter. A checkpoint
cannot pass from loss alone.

**Step 2: Collect one verified L1 teacher demonstration**

Record exact observations and actions from controller steps, not interpolated
trajectory frames that lack issued actions. Validate the HDF5 before training.

**Step 3: Train low-dimensional BC-RNN first**

Use one GPU and a bounded epoch/time budget. Overfitting the one demonstration
is expected; the real gate is physical replay.

**Step 4: Train image BC-Transformer only after BC-RNN interface proof**

Use context 10 and the exact evaluation observation preprocessing. Preserve
config, logs, checkpoint, dataset hash, versions, and three replay attempts.

**Step 5: Decide whether to scale**

Scale to 50 verified demonstrations per object family only if one of the two
policies physically replays the training grasp. Otherwise debug data/action
alignment; do not increase epoch count or data volume.

### Task 9: Final Regression And Documentation Gate

**Files:**
- Modify: `STATUS.md`
- Modify: `docs/09-current-route-and-optimization-plan.md`
- Modify: `experiments/experiment-log.csv`
- Modify: `research/source-ledger.csv`
- Modify: `CHANGELOG.md`

**Step 1: Run local verification**

Run:

```bash
git diff --check
python3.13 -m pytest -q tests
bash scripts/check_workspace.sh
```

Expected: all pass.

**Step 2: State only measured results**

Record nominal fixed-scene score, explicit perturbation success, collision rate,
and compliance findings separately. Do not create a final submission ZIP until
the scored-path scanner is clean and the official Agent clean reproduction
passes.

**Step 3: Commit the checkpoint**

Commit only the verified research checkpoint. Do not push or publish without
explicit user approval.
