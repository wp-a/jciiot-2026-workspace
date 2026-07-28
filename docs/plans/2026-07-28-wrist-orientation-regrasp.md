# Wrist-Orientation Regrasp Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align both Tiago Robotiq closure axes with the L1 container long-wall normals in a collision-free high-clearance pose, then test a real 0.13 m center regrasp lift.

**Architecture:** Add pure NumPy orientation geometry and hard-gate functions to the isolated L1 research runner, then extend its existing staged OSC loop with a closure-axis alignment stage. Promote only simulator records that prove bounded angular error, bounded position drift, zero collision/shortcuts, and physical lift; do not modify the current 8502 target or protected official code.

**Tech Stack:** Python 3.11, NumPy, unittest/pytest, MuJoCo 3.9.0, robosuite 1.5.2 OSC_POSE, existing JCIIOT research runner and audit tools.

---

### Task 1: Pure closure-axis rotation geometry

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:138-196`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write the failing tests**

Add tests that require:

```python
rotation = minimum_undirected_axis_rotation(
    source_axis=np.array([0.0, 0.6, 0.8]),
    target_axis=np.array([0.0, 1.0, 0.0]),
)
np.testing.assert_allclose(rotation @ normalized(source), target, atol=1e-8)
np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
```

Also test that target-axis sign is treated as equivalent, invalid vectors are
rejected, and `closure_axis_error_degrees()` returns zero for parallel or
antiparallel closure axes.

**Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k 'axis_rotation or closure_axis_error'
```

Expected: collection error or assertion failure because the helpers do not
exist.

**Step 3: Implement the minimum pure geometry**

Add:

```python
def minimum_undirected_axis_rotation(source_axis, target_axis):
    source = normalized_axis(source_axis)
    target = normalized_axis(target_axis)
    if float(np.dot(source, target)) < 0.0:
        target = -target
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        return np.eye(3)
    axis = cross / sine
    skew = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)
```

Implement finite-value and shape validation plus the error-angle helper.

**Step 4: Run focused and related tests**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: compute minimum gripper-axis alignment"
```

### Task 2: Bounded OSC orientation command and alignment gate

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing command-scaling tests**

Require a helper that converts a world-frame rotation delta to the controller
base frame and normalizes it against the OSC orientation range:

```python
command = normalized_osc_orientation_command(
    world_rotation_delta=rotation_z_90,
    controller_origin_rotation=rotation_z_180,
    output_min=np.array([-0.05] * 3 + [-0.5] * 3),
    output_max=np.array([0.05] * 3 + [0.5] * 3),
    max_action=0.30,
)
self.assertLessEqual(np.max(np.abs(command)), 0.30)
```

Test finite-value validation, non-positive scale rejection, zero rotation, and
the sign/direction of a known 90-degree rotation.

**Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k 'osc_orientation'
```

Expected: failure because the command helper is absent.

**Step 3: Implement minimal matrix-to-axis-angle and scaling helpers**

The undirected alignment angle is at most 90 degrees, so implement the stable
non-pi matrix conversion, transform it with
`origin.T @ world_delta @ origin`, divide by the controller orientation output
scale, and clip to the requested maximum action.

**Step 4: Write failing hard-gate tests**

Add `orientation_alignment_failures(record)` tests for:

- both arm errors at most 5 degrees;
- at least five stable alignment steps;
- maximum position drift at most 0.03 m;
- zero collision frames and no infrastructure error;
- rejection of missing, non-numeric, or non-finite evidence.

**Step 5: Implement the gate and run tests**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: gate bounded OSC wrist alignment"
```

### Task 3: Integrate the high-clearance orientation stage

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:790-1232`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing state-machine tests**

Extract and test pure decisions for stable-step accumulation and stop reasons:

```python
state = update_orientation_alignment_state(
    previous_stable_steps=4,
    right_error_deg=4.0,
    left_error_deg=3.5,
    position_drift_m=0.01,
    collision=False,
)
self.assertEqual(state.stable_steps, 5)
self.assertTrue(state.aligned)
```

Test reset on angular regression, termination on collision or excessive drift,
and no success from a single transient aligned frame.

**Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k 'orientation_alignment_state'
```

Expected: failure because the state helper is absent.

**Step 3: Add runtime closure-axis observations**

Read each `robot.eef_site_id[arm]` rotation. Record the full site matrix, local
X closure axis, selected target axis, angular error, end-effector position, and
drift on every stage step.

**Step 4: Extend the staged OSC action builder**

Allow `execute_stage()` to accept a target closure axis. Retain the existing
position action, fill the final three OSC action entries from the bounded
orientation command, and stop only after five aligned frames. Add CLI flags:

```text
--align-closure-axes
--orientation-max-action 0.30
--orientation-tolerance-deg 5.0
--orientation-stable-steps 5
--orientation-max-steps 160
```

Insert `align_closure_axes` after `raise_open_clearance` and before wall
clearance/center translation.

**Step 5: Run local verification**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py tests/test_competition_transport.py
python -m py_compile scripts/run_l1_cradle_gate.py
```

Expected: all tests pass and compilation exits zero.

**Step 6: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: align L1 gripper closure axes"
```

### Task 4: Bounded server experiments and traceable conclusion

**Files:**
- Create: `autoresearch/classic-260728-wrist-orientation/config.md`
- Create: `autoresearch/classic-260728-wrist-orientation/results.tsv`
- Create: `autoresearch/classic-260728-wrist-orientation/conclusion.md`
- Modify: `experiments/experiment-log.csv`
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`

**Step 1: Create the classic-loop configuration**

Record the official commit, candidate and runtime paths, seed 0, current 8502
non-promotion rule, and a maximum of four server variants.

Ordered keep metric:

1. no infrastructure error and zero official collisions;
2. both closure-axis errors at most 5 degrees for five steps;
3. position drift at most 0.03 m;
4. bilateral physical wall contact;
5. measured object lift at least 0.13 m and 20-step contact hold.

**Step 2: Materialize the research runner only**

Sync `scripts/run_l1_cradle_gate.py` to the existing isolated server tool path.
Do not restart or replace the current 8502 service.

**Step 3: Run the seed-0 orientation experiment**

Use the pinned venv, official candidate, `MUJOCO_GL=egl`, and logical GPU 2.
Save an atomic record JSON plus the original trajectory under the remote
results root.

Expected first decision: keep only if safety is preserved and closure-axis
error improves; a failed lift is still a valid diagnostic failure.

**Step 4: Run at most three evidence-driven variants**

Change only one of orientation maximum action, step budget, or target approach
orientation per iteration. Stop early on a full physical lift pass or after
the fourth valid variant. Never stack changes after an infrastructure error.

**Step 5: Pull record JSON only and write the conclusion**

Keep remote trajectory paths and SHA-256 references; do not commit multi-MB
trajectory duplicates. State pass/fail without converting an alignment-only
result into a score claim.

**Step 6: Run final verification**

Run:

```bash
python -m pytest -q tests
python scripts/audit_scored_path.py --root submission/JCIIOT --output /tmp/jciiot-wrist-audit.json
bash scripts/check_workspace.sh
git diff --check
```

Expected: complete test pass, zero scored-path violations, workspace check
success, and no diff whitespace errors.

**Step 7: Commit evidence**

```bash
git add autoresearch/classic-260728-wrist-orientation \
  experiments/experiment-log.csv STATUS.md CHANGELOG.md
git commit -m "docs: record L1 wrist-orientation gate outcome"
```
