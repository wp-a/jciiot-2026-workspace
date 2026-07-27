# L1 Scored Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a compliant deterministic L1 workflow that physically grasps, transports, and places the target object, then produces a trajectory scoring 10/10 with the unmodified official scorer.

**Architecture:** Keep official commit `0dcdddf` immutable and ship an overlay containing only allowed `skills/`, `workflows/`, and `knowledge/robot_params.json` changes. Navigation remains geometric; grasp first tries a two-arm scripted OSC controller using official object grasp sites and falls back to a public robomimic checkpoint when one is available. A remote experiment runner materializes the overlay on a clean official checkout, records every run, and invokes the unmodified official scorer.

**Tech Stack:** Python 3.11, NumPy, MuJoCo 3.9.0, official robosuite 1.5.2, `unittest`, Bash, JSON/TSV experiment artifacts.

---

### Task 1: Create the compliant submission overlay

**Files:**
- Create: `submission/README.md`
- Create: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`
- Create: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`
- Create: `scripts/materialize_submission.sh`
- Test: `tests/test_submission_boundaries.py`

**Step 1: Write the failing boundary tests**

Test that the overlay contains files only below these official paths:

```python
ALLOWED = (
    "JCIIOT/src/robot_agent/skills/",
    "JCIIOT/src/robot_agent/workflows/",
    "JCIIOT/knowledge/robot_params.json",
)
```

Also test that materialization rejects an official checkout whose Git HEAD differs from `config/upstream-lock.json`.

**Step 2: Verify RED**

Run: `python3 -m unittest tests/test_submission_boundaries.py -v`

Expected: FAIL because the overlay and materializer do not exist.

**Step 3: Implement the minimal overlay and materializer**

The materializer takes `--official-root`, `--overlay`, and optional `--output`. It verifies the official commit, copies the clean checkout to the output when requested, then copies only allowlisted overlay files. It never modifies `app.py`, `core/`, `environments/`, or `task_config.json`.

**Step 4: Verify GREEN**

Run: `python3 -m unittest tests/test_submission_boundaries.py -v`

Expected: all boundary tests pass.

**Step 5: Commit**

```bash
git add submission scripts/materialize_submission.sh tests/test_submission_boundaries.py
git commit -m "feat: add compliant submission overlay"
```

### Task 2: Implement the scripted physical grasp controller

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`
- Create: `tests/test_competition_grasp.py`

**Step 1: Write failing pure-function tests**

Cover:

- world delta normalization and clipping;
- stage completion only after all arm distances are within tolerance;
- timeout and failure reason preservation;
- success requires both physical gripper contacts and verified lift;
- `grasp_end success=true` is never emitted for contact-only or lift-only outcomes.

**Step 2: Verify RED**

Run: `python3 -m unittest tests/test_competition_grasp.py -v`

Expected: FAIL because controller functions are missing.

**Step 3: Implement the minimum physical controller**

Use lazy imports from the official factory-sorting helpers. The controller runs:

```text
open grippers
  -> move both gripper centers to pre-grasp sites
  -> descend/approach both grasp sites
  -> close and hold
  -> verify both grippers physically grasp the object
  -> lift 0.15 m and verify the object remains grasped
  -> attach for transport and set backend held-object state
```

Actions are generated through the robot's existing OSC and gripper controllers. Never write object qpos directly during grasp. Mark `grasp_start` before motion and `grasp_end` only with the actual combined contact-and-lift result.

**Step 4: Verify GREEN**

Run: `python3 -m unittest tests/test_competition_grasp.py -v`

Expected: all unit tests pass.

**Step 5: Commit**

```bash
git add submission/JCIIOT/src/robot_agent/skills/competition_grasp.py tests/test_competition_grasp.py
git commit -m "feat: add verified scripted grasp"
```

### Task 3: Add deterministic per-object execution

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`
- Create: `tests/test_competition_flow.py`

**Step 1: Write failing state-machine tests**

Test the state sequence:

```text
pending -> approached -> grasped -> lifted -> transported -> placed -> verified
```

Verify bounded retries, no transport after failed grasp, persistence of completed objects, and L5 continuation from the failed object's nearest recoverable state.

**Step 2: Verify RED**

Run: `python3 -m unittest tests/test_competition_flow.py -v`

Expected: FAIL because workflow behavior is missing.

**Step 3: Implement the deterministic workflow**

For L1, execute source move, scripted grasp, target move, physical place, and final object-distance verification. Keep the data model L5-capable but do not add unrelated planner features.

**Step 4: Verify GREEN**

Run: `python3 -m unittest tests/test_competition_flow.py -v`

Expected: all workflow tests pass.

**Step 5: Commit**

```bash
git add submission/JCIIOT/src/robot_agent/workflows/competition_flow.py tests/test_competition_flow.py
git commit -m "feat: add verified transport workflow"
```

### Task 4: Add the remote experiment and official scoring entry point

**Files:**
- Create: `scripts/run_official_experiment.py`
- Create: `tests/test_official_experiment.py`
- Modify: `artifacts/README.md`

**Step 1: Write failing experiment-contract tests**

Test that a run manifest records official/workspace commit, seed, scene, trajectory, official score, collision count, successful grasp events, elapsed time, status, and full exception details.

**Step 2: Verify RED**

Run: `python3 -m unittest tests/test_official_experiment.py -v`

Expected: FAIL because the runner does not exist.

**Step 3: Implement the runner**

The runner constructs the official backend, loads the current semantic map, executes `CompetitionFlow`, saves the official trajectory, and calls the scorer from the unmodified `app.py`. It writes JSON atomically and exits nonzero unless the requested gate is met.

**Step 4: Verify GREEN**

Run: `python3 -m unittest tests/test_official_experiment.py -v`

Expected: all unit tests pass without MuJoCo by using fake backends and scorer callbacks.

**Step 5: Commit**

```bash
git add scripts/run_official_experiment.py tests/test_official_experiment.py artifacts/README.md
git commit -m "feat: add scored experiment runner"
```

### Task 5: Run and tune the L1 physical loop on the GPU server

**Files:**
- Modify: `submission/JCIIOT/knowledge/robot_params.json`
- Modify: `experiments/experiment-log.csv`
- Create: `experiments/2026-07-27-l1-scripted-grasp.md`
- Create ignored artifacts below: `artifacts/remote-l1-*/`

**Step 1: Materialize a clean candidate**

Verify the upstream commit, apply only the overlay, and record hashes. Do not store the endpoint or credentials.

**Step 2: Run one L1 grasp-only attempt**

Run with fixed seed `20260727`. Preserve trajectory, event log, contact status, lift height, minimum clearance, and exception details.

**Step 3: Iterate one variable at a time**

Tune in this order: base approach pose, pre-grasp height, site offset, OSC gain/action cap, close steps, lift tolerance. Keep a candidate only when the physical grasp/lift metric improves without collision.

**Step 4: Run the full L1 workflow and official scorer**

Required result:

```text
score = 10/10
successful matching grasp_end = true
object leaves source by > 1 m in x or y
final target XY distance < 0.8 m
collision frames = 0
```

**Step 5: Repeat the accepted candidate**

Run five fixed seeds. The development gate is 5/5 score 10 with zero collisions. If the simulator ignores seed variation, record that fact and repeat five clean resets.

**Step 6: Record and commit the experiment**

```bash
git add submission/JCIIOT/knowledge/robot_params.json experiments
git commit -m "exp: establish L1 scored baseline"
```

### Task 6: Promote the L1 result into the full competition roadmap

**Files:**
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/08-module-roadmap.md`
- Create: `docs/plans/2026-07-28-l2-l5-competition-pipeline.md`

**Step 1: Run complete local verification**

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash tests/test_reference_scripts.sh
bash tests/test_workspace_check.sh
bash scripts/check_workspace.sh --require-private-remote
git diff --check
```

**Step 2: Verify submission boundaries and remote evidence**

Confirm no forbidden official file, secret, model, HDF5, generated trajectory, or server identifier is tracked.

**Step 3: Update status from evidence only**

Mark L1 complete only if the official score and repeat gate passed. Otherwise record the exact failed stage and retain L1 as active.

**Step 4: Write the next implementation plan**

Extend the proven workflow to L2-L4, then implement L5 three-object ordering and recovery. Do not start SOP/VLM innovation until at least one physical closed loop is proven.

