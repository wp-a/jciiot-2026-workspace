# Official Execute Entry-Point Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the unmodified official Execute path run the validated deterministic competition workflow and reproduce maximum scores on all five scenes.

**Architecture:** Add one allowed `CompetitionTaskSkill` adapter and register it first from the allowed skill factory. Disable only the official LLM planner through its existing feature gate so `RobotAgent.run()` selects the adapter deterministically, while leaving the official recorder, environment, task configuration, and scorer unchanged.

**Tech Stack:** Python 3.11/3.13, `unittest`/pytest, MuJoCo 3.9.0, robosuite 1.5.2, Bash materializer, official JCIIOT scorer.

---

### Task 1: Add the competition task adapter

**Files:**
- Create: `submission/JCIIOT/src/robot_agent/skills/competition_task.py`
- Create: `tests/test_competition_task.py`

**Step 1: Write failing task-resolution tests**

Create tests that load the overlay module with lightweight `robot_agent` type
stubs and assert:

```python
def test_resolve_task_index_accepts_top_level_and_scene_metadata():
    assert module.task_index_from_metadata({"task_index": 2}) == 2
    assert module.task_index_from_metadata({"scene": {"task_index": "4"}}) == 4

def test_resolve_task_index_rejects_missing_or_invalid_values():
    with self.assertRaises(ValueError):
        module.task_index_from_metadata({})
    with self.assertRaises(ValueError):
        module.task_index_from_metadata({"task_index": "L1"})

def test_load_official_task_rejects_out_of_range_index():
    with self.assertRaises(IndexError):
        module.load_official_task(5, config_path)
```

**Step 2: Run tests and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_competition_task.py
```

Expected: collection fails because `competition_task.py` does not exist.

**Step 3: Implement minimal task-resolution helpers**

Implement:

```python
def task_index_from_metadata(metadata: dict) -> int:
    value = metadata.get("task_index")
    if value is None and isinstance(metadata.get("scene"), dict):
        value = metadata["scene"].get("task_index")
    if isinstance(value, bool):
        raise ValueError("task_index must be an integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("task_index is missing or invalid") from exc
    return index

def load_official_task(task_index: int, config_path: Path | None = None) -> dict:
    path = config_path or Path(__file__).resolve().parents[3] / "knowledge" / "task_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task_config.tasks must be a list")
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task_index out of range: {task_index}")
    task = tasks[task_index]
    required = {"level", "source", "target", "object", "max_score"}
    if not isinstance(task, dict) or not required.issubset(task):
        raise ValueError(f"malformed task entry at index {task_index}")
    return dict(task)
```

**Step 4: Run tests and verify GREEN**

Run the Task 1 command. Expected: resolution tests pass.

**Step 5: Write failing adapter execution tests**

Add tests proving:

- every non-empty prompt is handled;
- the adapter calls `run_official_task()` with its backend, scene context, grid,
  selected task, and `max_attempts=1`;
- workflow `success=false` is propagated as a failed `SkillResult`;
- exceptions become failed results with stable `error_code` and `error_type`;
- no invalid index silently falls back to another level.

**Step 6: Run tests and verify RED**

Expected: `CompetitionTaskSkill` is missing.

**Step 7: Implement the minimal adapter**

Implement a `BaseSkill` subclass named `competition_task`. Preserve the complete
workflow result in `payload["workflow"]`, plus `task_index` and `level`. Return
`success=False` for any lookup or execution error and include a stable
`error_code`; do not mutate object poses or call any fallback skill.

**Step 8: Run tests and verify GREEN**

Run the Task 1 command. Expected: all adapter tests pass.

**Step 9: Commit**

```bash
git add submission/JCIIOT/src/robot_agent/skills/competition_task.py tests/test_competition_task.py
git commit -m "feat: add deterministic competition entry skill"
```

### Task 2: Register the adapter in the official skill factory

**Files:**
- Create: `submission/JCIIOT/src/robot_agent/skills/library.py`
- Create: `tests/test_competition_entrypoint.py`

**Step 1: Write failing library integration tests**

Load the overlay `library.py` under stub official modules and assert:

```python
def test_library_disables_planner_for_scored_execution():
    assert os.environ["GATE_PLANNER"] == "false"

def test_competition_skill_is_registered_first():
    skills = module.wired_skills(fake_backend, fake_scene, fake_grid)
    assert skills[0].name == "competition_task"
```

Also assert the official `move`, `pick_up`, `place_down`, record, knowledge,
document, and optional memory skills remain available after the first adapter.

**Step 2: Run tests and verify RED**

Run:

```bash
python3.13 -m pytest -q tests/test_competition_entrypoint.py
```

Expected: overlay `library.py` is missing.

**Step 3: Copy the official factory into the overlay and add one integration**

Start from official commit `0dcdddf` `skills/library.py`. Add:

```python
os.environ["GATE_PLANNER"] = "false"
from robot_agent.skills.competition_task import CompetitionTaskSkill
```

Instantiate `CompetitionTaskSkill` as the first item returned by
`wired_skills()`. Do not change the construction or ordering of the remaining
official skills.

**Step 4: Run focused and complete tests**

Run:

```bash
python3.13 -m pytest -q tests/test_competition_entrypoint.py tests/test_competition_task.py
python3.13 -m pytest -q tests
```

Expected: focused tests pass; complete workspace suite passes.

**Step 5: Verify the submission boundary**

Run:

```bash
bash scripts/check_workspace.sh
git diff --check
```

Expected: only allowed overlay paths are present; all workspace checks pass.

**Step 6: Commit**

```bash
git add submission/JCIIOT/src/robot_agent/skills/library.py tests/test_competition_entrypoint.py
git commit -m "feat: route official agent through competition flow"
```

### Task 3: Add an official-agent experiment mode

**Files:**
- Modify: `scripts/run_official_experiment.py`
- Modify: `tests/test_official_experiment.py`

**Step 1: Write failing execution-mode tests**

Add tests proving the parser accepts `--execution-mode flow|agent`, defaults to
`flow` for historical evidence, and agent mode:

- creates `RobotAgent` with the existing backend, scene context, grid, and task
  metadata;
- calls `agent.run()` once with a non-empty deterministic prompt;
- serializes `TaskOutput.as_dict()` into `execution_result`;
- still records and scores through the unchanged official functions.

**Step 2: Run tests and verify RED**

```bash
python3.13 -m pytest -q tests/test_official_experiment.py
```

Expected: parser rejects `--execution-mode` or the agent branch is absent.

**Step 3: Implement the minimal mode switch**

Factor execution into a small helper. Preserve the current direct workflow as
`flow`; add `agent` using `robot_agent.core.agent.RobotAgent`. Construct scene
metadata with `task_index`, environment, map prefix, and the official task
object mapping. Do not duplicate controller code.

**Step 4: Run focused and complete tests**

```bash
python3.13 -m pytest -q tests/test_official_experiment.py
python3.13 -m pytest -q tests
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/run_official_experiment.py tests/test_official_experiment.py
git commit -m "test: exercise official RobotAgent entry path"
```

### Task 4: Materialize a clean candidate

**Files:**
- No source changes expected
- Output: server candidate under `/home/user/jciiot-2026/candidates/`

**Step 1: Clean generated overlay caches**

Delete only ignored `__pycache__` and `.pyc` files under `submission/`,
`scripts/`, and `tests/`. Do not delete source, manifests, or user data.

**Step 2: Run all local gates**

```bash
python3.13 -m pytest -q tests
bash scripts/check_workspace.sh
git diff --check
git status --short --branch
```

Expected: all tests and checks pass; tracked tree is clean.

**Step 3: Copy the committed workspace files to the server**

Transfer only the overlay, materializer, runner, and locked commit metadata.
Verify SHA-256 on both hosts. Do not push the private Git repository.

**Step 4: Materialize from a fresh official source checkout**

Run `scripts/materialize_submission.sh` against official commit `0dcdddf` into a
new, previously nonexistent candidate directory. Expected: materializer prints
the locked commit and succeeds.

**Step 5: Verify boundary hashes**

Compare every file outside allowed roots with the pristine official checkout.
Expected: no differences outside `skills/` and `workflows/`; scorer and
`task_config.json` hashes match pristine official files.

### Task 5: Run all five scenes through RobotAgent

**Files:**
- Output: remote `results/entrypoint-<commit>-20260728/`
- Download: `artifacts/remote-entrypoint-20260728/`

**Step 1: Run L1 smoke in agent mode**

Use the clean candidate, seed `20260727`, `--execution-mode agent`, and required
score 10. Expected: 10/10, 1/1 grasp, zero collision frames, distance below
0.8 m, and `execution_result.success=true`.

**Step 2: Inspect the manifest before expanding**

Confirm the reported skill is `competition_task`, the planner did not contact an
external model, the workflow history ends in `verified`, and official scorer
provenance is unchanged.

**Step 3: Run L2-L5 in agent mode**

Use the same fixed seed and required maxima 15, 20, 25, and 30. L5 must contain
three successful grasp events and three verified objects.

**Step 4: Run exact official subprocess smoke**

On L1, invoke the unmodified `task_subprocess_runner.py` with `DISPLAY=:0` and
the clean candidate. Score its saved trajectory with unmodified `app.py`.
Expected: successful subprocess manifest and official 10/10. If the server's
visible display is unavailable, record that environmental limitation and retain
the full RobotAgent path as the headless acceptance evidence.

**Step 5: Download and hash evidence**

Archive manifests, trajectories, logs, environment versions, candidate hashes,
and scorer hashes. Verify the local archive SHA-256 and extracted file count.

### Task 6: Close the multi-seed baseline

**Files:**
- Modify: `autoresearch/classic-260728-0434/results.tsv`
- Create: `experiments/2026-07-28-multiseed-stability.md`
- Modify: `experiments/experiment-log.csv`
- Modify: `experiments/README.md`
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/09-current-route-and-optimization-plan.md`

**Step 1: Verify all 80 terminal runs**

Require 80/80 manifests, no missing trajectories, no runner errors, and the
expected score/grasp/collision/distance gate for every run. Do not discard or
replace failures.

**Step 2: Populate the autoresearch ledger mechanically**

Generate one TSV row per manifest in deterministic level/seed order. Validate
that the header plus 80 rows are present and that paths resolve inside the
downloaded evidence archive.

**Step 3: Write the stability report**

Report per-level run count, full-score rate, Wilson 95% interval, collision
rate, grasp verification rate, target distance, and elapsed time. State that the
parallel wall-clock values are not official tie-break timing and that unchanged
scene geometry makes these seeds repeatability tests, not geometry-perturbation
tests.

**Step 4: Update status and experiment indexes**

Replace the obsolete claim that L2-L5 lack 20-run evidence. Keep initial-pose
and geometry perturbation as an explicit remaining experiment.

**Step 5: Commit**

```bash
git add autoresearch/classic-260728-0434/results.tsv experiments STATUS.md CHANGELOG.md docs/09-current-route-and-optimization-plan.md
git commit -m "docs: record multiseed stability evidence"
```

### Task 7: Produce final submission artifacts

**Files:**
- Modify: `submission/README.md`
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md` if dependency evidence changed
- Output: `/Users/wangpeng/jciiot-2026-deliverables/*.zip`

**Step 1: Document exact reproduction commands**

Describe official commit, supported Python/runtime versions, materialization,
official app Execute, headless verification, scoring, expected files, and known
limitations. Explicitly use `python3.13 -m pytest -q tests` for local tests so
pytest does not collect downloaded reference repositories.

**Step 2: Build separate ZIP artifacts**

- Validation predictions ZIP: exactly five scene JSON files at archive root.
- Code/report ZIP: materialized allowed source changes, reproduction README,
  technical report, notices, and checksums; exclude credentials, caches, Git
  metadata, reference repositories, and development trajectories unless the
  submission portal explicitly requests them.

**Step 3: Independently validate archives**

Run `unzip -t`, list every member, parse all JSON, verify expected scene names,
scan for secrets and forbidden paths, and record SHA-256. Extract to a temporary
directory and run the documented checks from that extraction.

**Step 4: Final completion audit**

Map every competition deliverable and each item in this plan to authoritative
evidence. Do not claim completion while official entry-point acceptance,
five-scene trajectories, reproducibility instructions, or archive checks are
missing.

**Step 5: Commit final tracked documentation**

```bash
git add README.md submission/README.md THIRD_PARTY_NOTICES.md
git commit -m "docs: finalize reproducible competition submission"
```
