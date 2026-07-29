# Joint-Space Wrist Seed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and experimentally validate a collision-checked 12-joint IK seed that moves the L1 high-clearance gripper pose away from Tiago joint limits before the existing OSC alignment and real center-regrasp route.

**Architecture:** Keep optimizer-independent validation and residual construction as pure NumPy helpers in the isolated L1 research runner. At runtime, solve both arms simultaneously, restore the simulator before judging the proposal, then apply it through an official-collision-checked interpolation with rollback; successful seeding only unlocks the existing OSC, contact, lift, and hold gates.

**Tech Stack:** Python 3.11, NumPy, SciPy `least_squares`, unittest/pytest, MuJoCo 3.9.0, robosuite 1.5.2, existing JCIIOT research runner and audit tools.

---

### Task 1: Pure joint-bound and objective helpers

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:22-405`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write the failing interior-bound tests**

Import and test a wished-for `interior_joint_bounds()` helper:

```python
lower, upper = interior_joint_bounds(
    [-1.0, -2.0],
    [1.0, 3.0],
    margin_rad=0.05,
)
np.testing.assert_allclose(lower, [-0.95, -1.95])
np.testing.assert_allclose(upper, [0.95, 2.95])
```

Reject mismatched shapes, non-finite values, negative margins, and a margin
that leaves any interval empty.

**Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k interior_joint_bounds
```

Expected: collection fails because `interior_joint_bounds` is not defined.

**Step 3: Implement the minimum bound helper**

Convert inputs to finite one-dimensional float arrays, add/subtract the scalar
margin, reject every non-positive interior interval, and return copies.

**Step 4: Run the focused test and verify GREEN**

Run the same command. Expected: all selected tests pass.

**Step 5: Write failing residual tests**

Import and test `joint_seed_objective_residual()` with two arms. A zero pose and
axis error at the start must return an all-zero 24-vector. A 10 mm right-arm
position error with a 10 mm position scale must contribute exactly one in the
corresponding residual entry. Test fixed directed axis targets and normalized
joint regularization independently. Reject missing arms, malformed vectors,
non-finite data, non-positive scales, and zero joint ranges.

The desired helper boundary is:

```python
residual = joint_seed_objective_residual(
    current_positions={"right": right_xyz, "left": left_xyz},
    target_positions={"right": right_hold, "left": left_hold},
    current_axes={"right": right_axis, "left": left_axis},
    target_axes={"right": right_target, "left": left_target},
    joints=current_q,
    start_joints=start_q,
    joint_ranges=upper - lower,
    position_scale_m=0.01,
    axis_scale=0.08715574274765817,
    regularization=0.02,
)
```

**Step 6: Verify RED, implement minimally, then verify GREEN**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k joint_seed_objective
```

Expected before implementation: import/definition failure. Expected after the
minimal pure helper is added: all selected tests pass.

**Step 7: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: define bounded wrist seed objective"
```

### Task 2: Joint-seed evidence gate

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:294-405`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing hard-gate tests**

Add `joint_seed_failures(record)` tests around this valid record:

```python
{
    "joint_seed_success": True,
    "joint_seed_right_error_deg": 9.9,
    "joint_seed_left_error_deg": 9.8,
    "joint_seed_max_endpoint_position_error_m": 0.015,
    "joint_seed_max_path_position_drift_m": 0.03,
    "joint_seed_min_bound_margin_rad": 0.0,
    "joint_seed_collision_frames": 0,
    "joint_seed_rolled_back": False,
    "infrastructure_error": None,
}
```

Reject each missing field, non-numeric or non-finite numeric value, endpoint
axis error above 10 degrees, endpoint position error above 0.015 m, path drift
above 0.03 m, negative bound margin, any collision, rollback, false success,
or infrastructure error.

**Step 2: Run the focused test and verify RED**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k joint_seed_gate
```

Expected: failure because the gate does not exist.

**Step 3: Implement the strict pure gate**

Add named constants for the 10-degree endpoint, 0.015 m endpoint position, and
0.03 m path drift thresholds. Treat booleans as invalid numeric evidence and
deduplicate returned failure names.

**Step 4: Run focused and full module tests**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: gate collision-checked wrist seeds"
```

### Task 3: Runtime simultaneous IK and atomic rollback

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:1067-1715`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing path-decision tests**

Add a pure `next_joint_seed_path_state()` helper and tests proving that it:

- accumulates maximum drift across both arms and all waypoints;
- accepts finite collision-free waypoints at or below 0.03 m;
- terminates on the first official collision or excessive drift;
- retains collision pairs and the failing waypoint index in returned evidence;
- rejects malformed/non-finite measurements.

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k joint_seed_path_state
```

Expected: failure because the helper is absent.

**Step 2: Implement the pure path state helper and verify GREEN**

Implement only the state transition and rerun the focused command. Expected:
all selected tests pass.

**Step 3: Add the runtime solver behind an explicit flag**

Extend `_center_regrasp_probe()` with:

```text
orientation_joint_seed: bool
orientation_joint_seed_margin_rad: float
orientation_joint_seed_max_nfev: int
orientation_joint_seed_steps: int
orientation_joint_seed_position_scale_m: float
orientation_joint_seed_axis_scale: float
orientation_joint_seed_regularization: float
orientation_joint_seed_max_error_deg: float
orientation_joint_seed_max_endpoint_position_error_m: float
```

Inside the probe, add `execute_joint_orientation_seed(target_axis)` that:

1. resolves the 12 official arm joint ids, qpos addresses, and model ranges;
2. captures both grip-site poses, start joints, and fixed target-axis signs;
3. constructs inward bounds with `interior_joint_bounds()`;
4. solves both arms with SciPy `least_squares` and the pure residual helper;
5. restores the complete start vector in `finally` before judging the result;
6. predicts and validates the endpoint, then restores again;
7. interpolates both arms simultaneously for the configured number of steps;
8. applies `_navigation_collisions` and the pure path state at every waypoint;
9. rolls all robot joints back and synchronizes controller goals on failure;
10. synchronizes controller goals at the new pose on success.

Only robot arm qpos may be assigned. Do not assign object qpos, step physics
during the solve, or call any attachment API.

**Step 4: Preserve complete evidence on every return**

Record joint names, start and proposal vectors, effective bounds, solver
success/status/message/nfev/cost, residual components, endpoint errors, bound
margin, path drift, waypoint count, collision pairs, exception, rollback, and
controller synchronization. Add the seed result to the probe's top-level JSON.

Insert the seed immediately before the existing OSC closure-axis alignment.
Stop the regrasp route if `joint_seed_failures()` is non-empty. Seed success
must not set physical-grasp or accepted status.

**Step 5: Run local verification**

```bash
python -m pytest -q tests/test_l1_cradle_gate.py
python -m py_compile scripts/run_l1_cradle_gate.py
```

Expected: tests pass and compilation exits zero.

**Step 6: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: seed wrist alignment in joint space"
```

### Task 4: CLI, summaries, and regression checks

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py:1716-1920`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write a failing parser-default test**

Extract `build_argument_parser()` if the parser is currently embedded in
`main()`. Assert the seed is opt-in and defaults are: 0.03 rad limit margin,
800 evaluations, 240 interpolation steps, 0.01 m position scale,
`sin(5 degrees)` axis scale, 0.02 regularization, 10-degree endpoint error, and
0.015 m endpoint position error.

**Step 2: Verify RED, implement CLI wiring, and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_l1_cradle_gate.py -k joint_seed_parser
```

Add the flags, pass them to `_center_regrasp_probe()`, and expose compact seed
fields plus `joint_seed_gate_failures` in the final atomic record. Rerun the
focused test and expect PASS.

**Step 3: Run the complete local gate**

```bash
python -m pytest -q tests
python -m py_compile scripts/run_l1_cradle_gate.py
python scripts/audit_scored_path.py --root submission/JCIIOT --output /tmp/jciiot-joint-seed-audit.json
bash scripts/check_workspace.sh
git diff --check
```

Expected: all tests pass, zero scored-path violations, workspace check passes,
and no whitespace errors. The two pre-existing untracked autoresearch folders
remain untouched.

**Step 4: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: expose joint-space wrist seed experiment"
```

### Task 5: Bounded official-runtime experiments

**Files:**
- Create: `autoresearch/classic-260729-joint-wrist-seed/config.md`
- Create: `autoresearch/classic-260729-joint-wrist-seed/results.tsv`
- Create: `autoresearch/classic-260729-joint-wrist-seed/conclusion.md`
- Modify only if a real milestone changes: `experiments/experiment-log.csv`
- Modify only if a real milestone changes: `STATUS.md`
- Modify only if a real milestone changes: `CHANGELOG.md`

**Step 1: Create the experiment ledger**

Pin official commit `0dcdddf18a9e694569aa1433cdfc04eb097fed78`, candidate
`/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`, scene L1,
seed 0, the pinned evaluation Python, EGL GPU 2, and a maximum of four valid
single-worker variants. Record that current port 8502 is not changed.

The keep order is:

1. no infrastructure error, object writes, attachments, or official collision;
2. valid non-rollback joint seed within the 10-degree/15-mm/30-mm gates;
3. OSC alignment within 5 degrees for five consecutive steps;
4. bilateral physical contact and at least 0.13 m lift;
5. 20-step bilateral hold and real transport evidence.

**Step 2: Sync only the research runner**

Copy the committed runner to
`/home/user/jciiot-2026/tools/competition-pipeline/run_l1_cradle_gate.py`.
Verify local and remote SHA-256 match. Do not restart Streamlit or replace its
candidate.

**Step 3: Run the default seed-0 experiment**

Use `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`,
`MUJOCO_GL=egl`, and `CUDA_VISIBLE_DEVICES=2`. Write the atomic JSON and retain
the full trajectory under a new remote results root.

**Step 4: Make at most three evidence-driven single-variable changes**

If the solver remains safe but misses the endpoint, vary only regularization,
position scale, or interior margin. If the endpoint passes but the interpolated
path collides, discard the straight-line route and stop rather than hiding the
collision with fewer samples. Do not change OSC parameters in the same
iteration as a joint-seed parameter.

**Step 5: Require a clean-process repeat after a complete pass**

A second run must repeat joint seed, OSC alignment, contact, lift, and hold with
zero collisions and shortcut calls. An alignment-only result is diagnostic and
cannot be reported as a competition score.

**Step 6: Pull compact evidence and write the conclusion**

Pull result JSON and SHA-256 metadata, not multi-megabyte trajectories. Append
one row per valid experiment to `results.tsv`, mark every variant keep/discard,
and state the exact blocker if no route passes.

**Step 7: Commit the traceable evidence**

```bash
git add autoresearch/classic-260729-joint-wrist-seed \
  experiments/experiment-log.csv STATUS.md CHANGELOG.md
git commit -m "docs: record joint-space wrist seed experiments"
```

Omit unchanged optional files from `git add`.

### Task 6: Promotion only after two complete physical passes

**Files:**
- Modify only after promotion gate: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`
- Modify only if orchestration is needed: `submission/JCIIOT/src/robot_agent/workflows/competition_task.py`
- Modify only for tuned legal parameters: `submission/JCIIOT/knowledge/robot_params.json`
- Test: `tests/test_competition_grasp.py`
- Test: `tests/test_scored_path_audit.py`

**Step 1: Write failing submission-path tests**

Require the proven robot-only joint seed, identical rollback behavior, official
collision checks at every waypoint, and no object-state or attachment calls.

**Step 2: Verify RED, port minimally, and verify GREEN**

Port only the exact verified variant into the allowed skill. Do not modify
`core/`, `environments/`, `app.py`, or `knowledge/task_config.json`.

**Step 3: Run full local and official five-scene verification**

Run the complete local gate from Task 4, then use `app.py` / Execute in the
official service for all five scenes. Retain score JSON, trajectories, and
multiple-view recordings. Report only scores produced by the official scorer.

**Step 4: Keep the existing 8502 candidate unless every promotion gate passes**

Switch the service candidate only after the full route has two clean L1 passes,
all five official scenes complete, the scored-path audit is clean, and the
official score artifacts are retained.
