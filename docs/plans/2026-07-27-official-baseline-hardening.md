# Official Baseline Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Synchronize the workspace with official commit `0dcdddf`, make official task drift automatically detectable, and prove all five scenes can reset and step on the Linux GPU server without BC weights.

**Architecture:** Keep the official checkout immutable and ignored by the top-level repository. A standard-library Python validator compares tracked task facts with the locked official task config and semantic maps; a separate smoke runner imports the official robosuite checkout and records scene construction results as JSON. Remote setup uses an isolated Micromamba Python 3.11 environment and does not store credentials.

**Tech Stack:** Bash, Python 3.11 standard library, `unittest`, Git, Micromamba, MuJoCo, official robosuite.

---

### Task 1: Add a failing official-task consistency test

**Files:**
- Create: `tests/test_official_task_sync.py`
- Create: `scripts/check_official_tasks.py`

**Step 1: Write the failing tests**

Create `tests/test_official_task_sync.py` with temporary workspace fixtures for five fields: commit, environment name, source, target, object list, and max score. Include semantic-map fixtures where source and target stations expose a `center` array.

Test these cases:

```python
def test_matching_workspace_passes(): ...
def test_stale_source_fails_with_field_name(): ...
def test_stale_target_fails_with_field_name(): ...
def test_stale_coordinates_fail_with_field_name(): ...
def test_workspace_and_lock_commit_must_match(): ...
```

The test imports `scripts.check_official_tasks.validate_workspace` and asserts that mismatches raise `ValidationError` containing the task level and field.

**Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests/test_official_task_sync.py -v`

Expected: FAIL because `scripts/check_official_tasks.py` does not exist.

**Step 3: Implement the minimal validator**

Implement:

```python
class ValidationError(RuntimeError):
    pass

def validate_workspace(workspace: Path) -> list[str]:
    """Return validated level names or raise ValidationError."""
```

Load:

- `config/tasks.json`
- `config/upstream-lock.json`
- `<vendor>/JCIIOT/knowledge/task_config.json`
- each scene's generated semantic map

Compare `official_commit`, `scene/env_name`, `source`, `target`, `objects/object`, `max_score`, `source_center_xy`, and `target_center_xy`. Resolve maps using the official `scene_prefix`. Require exactly one workspace entry for every official level and reject extra levels.

Provide CLI:

```text
python3 scripts/check_official_tasks.py --workspace PATH
```

Print one concise success line; print validation errors to stderr and exit 1.

**Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests/test_official_task_sync.py -v`

Expected: 5 tests pass.

**Step 5: Commit**

```bash
git add scripts/check_official_tasks.py tests/test_official_task_sync.py
git commit -m "test: validate official task synchronization"
```

### Task 2: Synchronize the locked official baseline and tracked facts

**Files:**
- Modify: `config/tasks.json`
- Modify: `config/upstream-lock.json`
- Modify: `docs/00-competition-brief.md`
- Modify: `docs/01-official-baseline-audit.md`
- Modify: `README.md`
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`

**Step 1: Demonstrate the stale configuration failure**

Run the new validator against the current workspace before changing tracked facts.

Expected: FAIL for the L3 source and/or locked commit.

**Step 2: Fast-forward the ignored vendor checkout**

Run:

```bash
git -C vendor/JCIIOT2026 merge --ff-only origin/master
```

Verify HEAD is `0dcdddf18a9e694569aa1433cdfc04eb097fed78` and the vendor worktree is clean.

**Step 3: Update task facts and hashes**

Set:

- L3 source to `aux_input_1`, source center to `[0.144, 8.473]`;
- L5 target to `aux_output_1`, target center to `[0.144, 8.473]`;
- all snapshot and lock fields to the new commit and date.

Recompute SHA-256 values for every file listed in `config/upstream-lock.json`. Add the two changed semantic maps and `robosuite_backend.py` to the tracked file list because they define the new auxiliary-station behavior.

**Step 4: Update documentation conservatively**

Record the 2026-07-27 auxiliary-station change. Replace “完整可运行代码检出” with “完整源码检出”; state that local and remote smoke results are pending until Task 5 completes.

**Step 5: Run the validator**

Run: `python3 scripts/check_official_tasks.py --workspace .`

Expected: all 5 official tasks match the locked checkout and semantic maps.

**Step 6: Commit**

```bash
git add config README.md STATUS.md CHANGELOG.md docs/00-competition-brief.md docs/01-official-baseline-audit.md
git commit -m "docs: sync auxiliary task stations"
```

### Task 3: Integrate task consistency into workspace checks

**Files:**
- Modify: `scripts/check_workspace.sh`
- Modify: `tests/test_workspace_check.sh`

**Step 1: Write a failing integration fixture**

Extend `tests/test_workspace_check.sh` so a copied fixture with L3 source changed back to `input_6` must fail and mention `L3 source`.

**Step 2: Run the integration test and observe failure**

Run: `bash tests/test_workspace_check.sh`

Expected: FAIL because `check_workspace.sh` does not call the new validator.

**Step 3: Add the validator call**

Call:

```bash
python3 "${workspace_dir}/scripts/check_official_tasks.py" --workspace "${workspace_dir}"
```

after the vendor commit/hash checks and before reference checkout checks.

**Step 4: Run all local workspace tests**

Run:

```bash
bash tests/test_reference_scripts.sh
python3 -m unittest tests/test_official_task_sync.py -v
bash tests/test_workspace_check.sh
bash scripts/check_workspace.sh
```

Expected: all pass.

**Step 5: Commit**

```bash
git add scripts/check_workspace.sh tests/test_workspace_check.sh
git commit -m "test: reject stale official task facts"
```

### Task 4: Add a testable five-scene smoke runner

**Files:**
- Create: `scripts/smoke_official_scenes.py`
- Create: `tests/test_smoke_official_scenes.py`
- Modify: `artifacts/README.md`

**Step 1: Write failing unit tests**

Use fake environment classes and `unittest.mock` to test:

```python
def test_success_record_contains_dimensions_and_timing(): ...
def test_failure_record_preserves_stage_and_exception(): ...
def test_any_scene_failure_returns_nonzero_summary(): ...
```

No MuJoCo dependency is allowed in these unit tests.

**Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests/test_smoke_official_scenes.py -v`

Expected: FAIL because the runner does not exist.

**Step 3: Implement the runner**

CLI:

```text
python scripts/smoke_official_scenes.py \
  --official-root /path/to/JCIIOT2026 \
  --steps 1 --seed 20260727 --output result.json
```

After adding `<official-root>/JCIIOT/robosuite` to `sys.path`, lazily import all five official environment classes. For each class: construct with `robots="Tiago"`, renderer and offscreen renderer disabled, reset, compute a zero action from `action_spec`, step the requested number of times, capture model dimensions, and close in `finally`.

The top-level JSON records official Git commit, Python version, platform, seed, steps, UTC timestamps, and one result per scene. Exit 1 if any scene fails.

**Step 4: Run unit tests**

Run: `python3 -m unittest tests/test_smoke_official_scenes.py -v`

Expected: all pass without MuJoCo installed locally.

**Step 5: Document artifact handling and commit**

```bash
git add scripts/smoke_official_scenes.py tests/test_smoke_official_scenes.py artifacts/README.md
git commit -m "feat: add official scene smoke runner"
```

### Task 5: Build the isolated remote environment and run five scenes

**Files:**
- Remote create: `/home/user/jciiot-2026/source/JCIIOT2026`
- Remote create: `/home/user/jciiot-2026/tools/smoke_official_scenes.py`
- Remote create: `/home/user/jciiot-2026/results/scene-smoke-0dcdddf.json`
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`

**Step 1: Verify remote capacity and record versions**

Record Ubuntu, GPU model/driver, free disk, Micromamba version and available memory. Do not record host, username or credentials in Git.

**Step 2: Create a clean Python 3.11 environment**

Use Micromamba environment `jciiot-2026`. Install only the dependencies required by the official robosuite checkout and the smoke runner; do not install or fetch BC checkpoints.

**Step 3: Deploy fixed source and runner**

Clone the public official repository without LFS smudge and checkout exact commit `0dcdddf`. Transfer the smoke runner from the worktree. Confirm source HEAD and file hashes match the lock.

**Step 4: Run an import probe**

Verify Python, NumPy, MuJoCo and official robosuite imports before creating a scene.

Expected: all imports succeed from the isolated environment.

**Step 5: Run L1 only**

Run one reset and one zero-action step for L1. If it fails, preserve the full traceback and diagnose the exact dependency/render/resource boundary before changing anything.

**Step 6: Run L1-L5**

Run all scenes with seed `20260727`, one zero-action step, and JSON output under the remote results directory.

Expected: five successes, process exit 0.

**Step 7: Record verified results**

Update `STATUS.md` and `CHANGELOG.md` with the exact result. Clearly separate scene-loading success from BC grasp-policy readiness.

**Step 8: Commit**

```bash
git add STATUS.md CHANGELOG.md
git commit -m "docs: record remote scene smoke results"
```

### Task 6: Final verification and integration

**Files:**
- Modify if needed: `docs/plans/2026-07-27-official-baseline-hardening.md`

**Step 1: Run the complete local verification suite**

```bash
bash tests/test_reference_scripts.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash tests/test_workspace_check.sh
bash scripts/check_workspace.sh --require-private-remote
git diff --check
```

Expected: all pass.

**Step 2: Verify repository boundaries**

Confirm no credentials, server endpoint, model, HDF5, LFS object, generated result, vendor source or reference checkout is tracked.

**Step 3: Review all commits and merge**

Review `main..baseline-hardening`, fast-forward `main` only after tests pass, rerun the suite on merged `main`, then push the private remote.

**Step 4: Clean up**

Remove only the two verified worktree symlinks, remove the worktree, and delete the merged feature branch. Keep the main vendor checkout and downloaded references intact.
