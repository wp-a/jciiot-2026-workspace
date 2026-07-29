# Wrist-Seed Continuation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the invalid long straight-line joint seed path with an opt-in sequence of locally solved closure-axis IK nodes while preserving every position and collision gate.

**Architecture:** Add pure directed-axis interpolation, use it to generate simultaneous two-arm target nodes, initialize and regularize every bounded solve from the preceding node, then replay adjacent node segments through the existing waypoint gate. Keep one node behavior-compatible with the current endpoint experiment and record every node solver outcome.

**Tech Stack:** Python 3.11, NumPy, SciPy `least_squares`, unittest/pytest, MuJoCo 3.9.0, robosuite 1.5.2.

---

### Task 1: Directed-axis interpolation

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing tests**

Import `interpolate_directed_axis()` and require exact normalized endpoints at
fractions 0 and 1, unit-length intermediate values, monotonically decreasing
angular distance to the target, and rejection of non-finite/out-of-range
fractions or antipodal directed axes.

**Step 2: Verify RED**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k directed_axis_interpolation
```

Expected: import failure because the helper is absent.

**Step 3: Implement and verify GREEN**

Normalize both axes, require a non-negative dot product, validate a fraction in
`[0, 1]`, normalize `(1-f) * source + f * target`, and rerun the focused tests.

**Step 4: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: interpolate directed closure axes"
```

### Task 2: Continuation solver nodes and segmented path

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing allocation tests**

Add a pure `allocate_segment_steps(total_steps, segment_count)` helper. Require
positive integer inputs, exactly `segment_count` positive counts, a sum equal
to `total_steps`, and a maximum difference of one step.

**Step 2: Verify RED, implement, and verify GREEN**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k allocate_segment_steps
```

Expected: failure before implementation and pass after the minimal helper.

**Step 3: Refactor only the proposal solve**

Add `orientation_joint_seed_continuation_nodes` to the probe. Build fractions
`1/n ... 1`, interpolate each arm's directed target axis, and solve each node
from the preceding node. Use the preceding node as the residual regularization
origin. After each solve, set the node proposal temporarily and record solver,
node-axis errors, position errors, bound margin, and official collision pairs.

Reject and restore on solver failure, node error above 5 degrees, position error
above 15 mm, bounds violation, collision, or exception. Preserve one-node
behavior as the current direct solve.

**Step 4: Replay segmented path**

Allocate the existing 240 total steps across accepted nodes. Apply
`joint_interpolation_path()` to each adjacent pair and feed every global
waypoint to the unchanged `next_joint_seed_path_state()`. Record node and local
step ids. The final endpoint and top-level gate remain unchanged.

**Step 5: Verify locally and commit**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py
python -m py_compile scripts/run_l1_cradle_gate.py
git diff --check
```

Expected: all pass.

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: trace wrist seeds through local IK nodes"
```

### Task 3: CLI, full regression, and one official hypothesis test

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`
- Modify: `autoresearch/classic-260729-joint-wrist-refine/results.tsv`
- Create after experiments: `autoresearch/classic-260729-joint-wrist-refine/conclusion.md`

**Step 1: Extend the existing parser test**

Require `--orientation-joint-seed-continuation-nodes` to default to 1. Add the
flag and pass it through `run_probe()`.

**Step 2: Run the complete local gate**

```bash
python -m pytest -q tests
python -m py_compile scripts/run_l1_cradle_gate.py
python scripts/audit_scored_path.py --root submission/JCIIOT --output /tmp/jciiot-continuation-audit.json
bash scripts/check_workspace.sh
git diff --check
```

Expected: all tests pass, zero audit violations, and workspace check passes.

**Step 3: Commit and sync the research runner**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: expose wrist seed continuation"
```

Sync only the runner and verify SHA-256. Do not change candidate files or 8502.

**Step 4: Run one single-variable official experiment**

Use the prior 0.0185 m position scale and add only 24 continuation nodes. Keep
all other solver, OSC, path, and safety values fixed. Retain JSON and trajectory.

**Step 5: Decide from evidence**

If continuation passes joint seed, inspect OSC and physical regrasp. If it
stops at a local node, record the exact node and do not stack parameter changes.
Write the bounded-loop conclusion and commit compact evidence.
