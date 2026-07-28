# Three-Layer Physical Manipulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the real dataset and five-scene grasp baselines, produce one verified L1 cradle-transfer trajectory, and make an evidence-based BC-RNN versus Diffusion Policy decision.

**Architecture:** Research-only scripts inspect data and execute isolated official-checkpoint rollouts without changing protected competition files. Submission changes remain inside the permitted skill and workflow overlay, where a geometric teacher performs physical cradle transfer and a contact-aware controller later owns supported transport. Every promotion decision is derived from immutable JSON/TSV evidence rather than a UI score or training loss.

**Tech Stack:** Python 3.11, h5py, NumPy, PyTorch, robomimic, robosuite/MuJoCo, unittest/pytest, Git LFS, JSON/TSV experiment ledgers.

---

### Task 1: Materialized HDF5 inspector

**Files:**
- Create: `scripts/inspect_robomimic_hdf5.py`
- Create: `tests/test_inspect_robomimic_hdf5.py`

**Step 1: Write the failing tests**

Create a temporary HDF5 fixture containing `data/demo_1`, `data/demo_2`,
`actions`, `states`, `obs/robot0_robotview_image`, and low-dimensional
observations. Assert that `inspect_hdf5(path)` returns:

```python
{
    "materialized": True,
    "env_args": {"env_name": "FactorySorting1_TEST"},
    "demo_count": 2,
    "total_samples": 7,
    "action_dim": 20,
    "observation_keys": [
        "robot0_left_eef_pos",
        "robot0_robotview_image",
    ],
}
```

Add tests that a 134-byte Git LFS pointer is rejected as
`materialized=False`, inconsistent action dimensions raise `ValueError`, and
JSON output is deterministic.

**Step 2: Run the tests to verify RED**

Run:

```bash
python -m unittest tests.test_inspect_robomimic_hdf5 -v
```

Expected: FAIL because `scripts.inspect_robomimic_hdf5` does not exist.

**Step 3: Implement the minimal inspector**

Implement these public functions:

```python
def is_git_lfs_pointer(path: Path) -> bool: ...
def inspect_hdf5(path: Path) -> dict[str, Any]: ...
def classify_compatibility(summary: dict[str, Any]) -> dict[str, Any]: ...
def write_json_atomic(path: Path, value: dict[str, Any]) -> None: ...
```

Walk only metadata and shapes; do not load image arrays into memory. Decode
`env_args` and `env_info` as JSON when possible. Record each demo's sample
count, action/state shapes, observation keys/shapes/dtypes, masks, and root
attributes. Classify as `task-compatible` only when the environment is a
FactorySorting scene, action dimension is 20, and the required Tiago plus
`robot0_robotview_image` keys exist. Otherwise classify as `partially-reusable`
or `format-only` with explicit reasons.

**Step 4: Run the tests to verify GREEN**

Run the Task 1 command again.

Expected: all Task 1 tests PASS.

**Step 5: Commit**

```bash
git add scripts/inspect_robomimic_hdf5.py tests/test_inspect_robomimic_hdf5.py
git commit -m "feat: inspect robomimic dataset compatibility"
```

### Task 2: Audit the real official HDF5

**Files:**
- Create: `autoresearch/classic-260728-hdf5/config.md`
- Create: `autoresearch/classic-260728-hdf5/dataset-summary.json`
- Create: `autoresearch/classic-260728-hdf5/conclusion.md`

**Step 1: Materialize or locate the real asset**

Verify the file size is approximately 591,069,600 bytes. Do not analyze the
134-byte pointer. Prefer the already downloaded server copy; otherwise fetch
the single LFS object into a temporary research location without changing the
vendor checkout.

**Step 2: Run the inspector**

Run:

```bash
python scripts/inspect_robomimic_hdf5.py REAL_DATASET \
  --output autoresearch/classic-260728-hdf5/dataset-summary.json
```

Expected: exit 0 and a summary containing `env_args`, observation keys, action
dimension, demo count, and compatibility classification.

**Step 3: Write the conclusion**

Record the exact path, SHA-256, byte size, classification, reusable fields,
incompatible fields, and whether the data may enter a JCIIOT training split.
Do not infer unreported contents.

**Step 4: Commit the audit record**

```bash
git add autoresearch/classic-260728-hdf5
git commit -m "docs: audit official HDF5 metadata"
```

### Task 3: Isolated official-checkpoint grasp runner

**Files:**
- Create: `scripts/run_official_grasp_baseline.py`
- Create: `tests/test_official_grasp_baseline.py`

**Step 1: Write the failing tests**

Test pure helpers before loading MuJoCo:

```python
def test_build_jobs_expands_every_scored_object_and_seed(): ...
def test_physical_success_requires_bilateral_grasp_and_150mm_lift(): ...
def test_summary_keeps_failed_attempts_in_denominator(): ...
def test_cli_rejects_non_materialized_checkpoint(): ...
```

Expected job count for all tasks and one seed is 11: two L1 objects, two L2,
two L3, two L4, and three L5 objects.

**Step 2: Run the tests to verify RED**

```bash
python -m unittest tests.test_official_grasp_baseline -v
```

Expected: FAIL because the module does not exist.

**Step 3: Implement the runner**

The runner must:

- pin the official repository commit and checkpoint SHA-256;
- load `knowledge/task_config.json` without modifying it;
- create the official wrapped evaluation environment for one scene/object;
- derive the base XY from the semantic source approach and yaw toward the
  scored object's initial center;
- call the official checkpoint policy and official lift helper directly in
  the wrapped environment;
- avoid backend object synchronization and transport attachment;
- measure bilateral `_check_grasp`, object Z lift, fingerpad contacts,
  `has_judge_collision`, elapsed time, and failure stage;
- close the environment in `finally` and atomically write one JSON manifest.

Expose pure APIs:

```python
@dataclass(frozen=True)
class GraspJob: ...

def build_jobs(tasks, seeds) -> list[GraspJob]: ...
def physical_grasp_success(record, *, required_lift_m=0.15,
                           tolerance_m=0.02) -> bool: ...
def summarize(records, *, planned_runs: int) -> dict[str, Any]: ...
```

**Step 4: Run the tests to verify GREEN**

Run the Task 3 command again.

Expected: all Task 3 tests PASS.

**Step 5: Commit**

```bash
git add scripts/run_official_grasp_baseline.py tests/test_official_grasp_baseline.py
git commit -m "feat: measure official physical grasp baseline"
```

### Task 4: Five-level baseline execution

**Files:**
- Create: `autoresearch/classic-260728-grasp-baseline/config.md`
- Create: `autoresearch/classic-260728-grasp-baseline/results.tsv`
- Create: `autoresearch/classic-260728-grasp-baseline/summary.json`
- Create: `autoresearch/classic-260728-grasp-baseline/conclusion.md`

**Step 1: Dry-run one L1 attempt**

Run one L1 object and seed in EGL/headless mode. Verify the checkpoint loads,
action dimension matches, and the manifest contains measured physical fields.
The command is accepted even if the grasp fails; infrastructure errors are not.

**Step 2: Run the exploratory matrix**

Run at least 10 seeds for every scored object with one worker per safe EGL
device. Append every result, including crashes and failures, to `results.tsv`.
Do not auto-retry a failed physical attempt under the same seed.

**Step 3: Summarize and classify**

For each object family report attempt count, bilateral grasp rate, verified
150 mm lift rate, collision rate, mean time, and Wilson 95% interval. Freeze
families at or above 90% exploratory success; mark others for fine-tuning.

**Step 4: Commit only complete evidence**

```bash
git add autoresearch/classic-260728-grasp-baseline
git commit -m "docs: record five-level official grasp baseline"
```

### Task 5: Cradle-transfer contact model

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`
- Modify: `tests/test_competition_transport.py`

**Step 1: Write failing unit tests**

Add tests for:

```python
def test_cradle_support_requires_real_wrist_or_forearm_contact(): ...
def test_cradle_stability_resets_on_contact_or_height_loss(): ...
def test_cradle_delta_is_bounded_and_symmetric(): ...
def test_cradle_transfer_stops_on_judge_collision(): ...
def test_cradle_transfer_never_calls_attachment_or_object_pose_helpers(): ...
```

Use a fake driver with separate gripper contacts, support-link contacts, object
height, object-to-wrist drift, collision, and action history.

**Step 2: Run the focused tests to verify RED**

```bash
python -m unittest tests.test_competition_transport -v
```

Expected: FAIL because cradle APIs are missing.

**Step 3: Implement minimal pure logic**

Add:

```python
@dataclass(frozen=True)
class CradleObservation: ...

def is_cradle_supported(observation: CradleObservation) -> bool: ...
def next_cradle_stability(observation, stable_steps: int) -> int: ...
def bounded_symmetric_cradle_deltas(...) -> dict[str, np.ndarray]: ...
def run_physical_cradle_transfer(..., driver) -> dict[str, Any]: ...
```

Allow only `arm_4`, `arm_5`, `arm_6`, wrist, palm, or gripper-link collision
geometries as support contacts. Judge obstacle proxy contact, object drop,
single-arm escape, excessive relative drift, or step exhaustion terminates the
attempt. No fallback may mutate object state.

**Step 4: Run focused and full tests**

```bash
python -m unittest tests.test_competition_transport -v
python -m pytest -q
```

Expected: focused tests and the existing suite PASS.

**Step 5: Commit**

```bash
git add submission/JCIIOT/src/robot_agent/skills/competition_transport.py tests/test_competition_transport.py
git commit -m "feat: add physical cradle transfer controller"
```

### Task 6: L1 cradle gate and first successful trajectory

**Files:**
- Create: `scripts/run_l1_cradle_gate.py`
- Create: `tests/test_l1_cradle_gate.py`
- Create: `autoresearch/classic-260728-cradle/config.md`
- Create: `autoresearch/classic-260728-cradle/results.tsv`
- Create after success: `autoresearch/classic-260728-cradle/conclusion.md`

**Step 1: Write and fail the gate tests**

Test that acceptance requires all of:

```python
record["physical_grasp"] is True
record["lift_m"] >= 0.13
record["support_contact_steps"] >= 20
record["base_translation_m"] >= 0.5
record["attachment_calls"] == 0
record["object_pose_writes"] == 0
record["collision_frames"] == 0
record["dropped"] is False
```

Run the focused test and observe RED before implementation.

**Step 2: Implement the research runner**

Materialize the submission overlay onto the locked official tree, execute the
existing physical L1 grasp, then call `run_physical_cradle_transfer`. Record
all contact pairs, relative positions, action norms, events, and the original
trajectory. Do not invoke the official scorer for a partial gate.

**Step 3: Run bounded autoresearch iterations**

Use Classic mode with:

```text
Metric: accepted cradle gate, then maximize support_contact_steps and base_translation_m
Verify: unit suite + boundary audit + original trajectory gate
Iterations: 15
```

Change one bounded cradle parameter or target construction per iteration. Keep
only measured improvements; record every failure. Stop immediately on the
first fully accepted trajectory and repeat it from a clean process.

**Step 4: Generate physical evidence**

Render birdview and robotview GIFs from the accepted original trajectory. The
replay must visibly show the box resting on real robot links rather than moving
with a fixed relative transform.

**Step 5: Run the boundary audit and commit**

```bash
python scripts/audit_scored_path.py --submission-root submission/JCIIOT
python -m pytest -q
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py \
  autoresearch/classic-260728-cradle submission/JCIIOT/src/robot_agent/skills/competition_transport.py
git commit -m "feat: verify first physical cradle transfer"
```

Expected: zero hard boundary violations, full test suite PASS, and two clean
gate successes before proceeding.

### Task 7: Model decision report

**Files:**
- Create: `scripts/segment_manipulation_trajectory.py`
- Create: `tests/test_segment_manipulation_trajectory.py`
- Create: `experiments/three-layer-model-decision.md`

**Step 1: Write and fail segmentation tests**

Given trajectory events and contact transitions, assert deterministic segments
for `side_grasp`, `cradle_transfer`, `supported_transport`, and `placement`.
Assert frames are neither dropped nor shared across train and validation
episode groups.

**Step 2: Implement and verify the segmenter**

Export robomimic-compatible episodes using the exact 20-dimensional action and
available observation schema. Preserve source trajectory, seed, object,
segment, and physical-success metadata.

**Step 3: Run one-trajectory overfit probes**

Use identical data for BC-RNN and bundled robomimic Diffusion Policy. Verify
both through the evaluation action interface; do not compare training loss
alone. Record physical replay success, maximum slip, collision, inference
latency, and model size.

**Step 4: Decide**

Select BC-RNN unless Diffusion Policy improves physical rollout success or
slip by a predeclared margin without violating latency or reproducibility. If
neither replays the teacher, fix the data/action pipeline before collecting
more demonstrations.

**Step 5: Commit**

```bash
git add scripts/segment_manipulation_trajectory.py \
  tests/test_segment_manipulation_trajectory.py experiments/three-layer-model-decision.md
git commit -m "docs: decide manipulation policy from physical rollouts"
```

### Task 8: Final verification before 8502 promotion

**Files:**
- Update: `autoresearch/classic-260728-cradle/conclusion.md`
- Update only if behavior changed formally: `README.md`

**Step 1: Run independent verification**

```bash
python -m pytest -q
python scripts/audit_scored_path.py --submission-root submission/JCIIOT
python scripts/check_workspace.sh
```

Expected: all tests PASS, zero hard violations, workspace checks PASS.

**Step 2: Re-run L1 from a clean process**

Run the locked official entrypoint twice. Verify real grasp, cradle support,
at least 0.5 m physical transport, no collision, and no attachment/object pose
write. This gate does not claim full score until physical placement also
passes.

**Step 3: Promotion decision**

Do not switch the 8502 target unless the full L1 official workflow, including
physical placement and unmodified scoring, passes twice. Otherwise leave the
current service untouched and report the exact failed gate.
