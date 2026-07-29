# Sequential Under-Support Transition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the first collision-free, load-bearing hand or distal-arm contact under the L1 object while the opposite arm preserves the verified center grasp.

**Architecture:** Add two small allowed-skill helpers: one constructs a robot action with independent gripper commands, and one computes a single-arm descend-and-inset target from measured geometry. The opt-in research runner uses those helpers after the verified center hold, stops after the first-arm transition, and records contact, height, collision, pose-write, and attachment evidence before any second-arm or transport attempt.

**Tech Stack:** Python 3, NumPy, unittest/pytest, robosuite/MuJoCo, the existing center-regrasp runner, and the pinned official environment.

---

### Task 1: Add independent gripper action construction

**Files:**
- Modify: `tests/test_competition_grasp.py`
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`

**Step 1: Write the failing test**

Add a fake robot with two one-DOF grippers and a recording
`create_action_vector`. Test a wished-for helper that places `+1` in the
stationary arm's gripper action and `-1` in the moving arm's gripper action,
while preserving both arm commands.

```python
action = module.build_independent_gripper_action(
    robot,
    arm_actions={"right": np.ones(6), "left": np.zeros(6)},
    gripper_values={"right": 1.0, "left": -1.0},
    hold_targets={},
)
self.assertEqual(robot.action_dict["right_gripper"].tolist(), [1.0])
self.assertEqual(robot.action_dict["left_gripper"].tolist(), [-1.0])
```

**Step 2: Run the test and verify RED**

Run:
`pytest -q tests/test_competition_grasp.py::CompetitionGraspTests::test_independent_gripper_action_keeps_stationary_arm_closed`

Expected: fail because `build_independent_gripper_action` does not exist.

**Step 3: Implement the minimal helper**

Construct the same action dictionary as the official `build_action`, but accept
an explicit value for each arm. Reuse official camera/base hold helpers and
`robot.create_action_vector`; do not modify the official environment module.
Validate that both arms are present and each command is finite.

**Step 4: Run focused and module tests**

Run:
`pytest -q tests/test_competition_grasp.py`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add tests/test_competition_grasp.py submission/JCIIOT/src/robot_agent/skills/competition_grasp.py
git commit -m "feat: control competition grippers independently"
```

### Task 2: Add measured sequential support targets

**Files:**
- Modify: `tests/test_competition_transport.py`
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`

**Step 1: Write the failing tests**

Test `single_arm_under_support_targets` with live-style positions. The arm with
the larger separation-axis projection must move down and toward the midpoint;
the opposite arm must remain unchanged. Add mirrored-arm, zero-axis,
non-finite, negative-distance, and unknown-arm cases.

```python
targets = module.single_arm_under_support_targets(
    {"right": np.array([7.25, 4.82, 1.36]),
     "left": np.array([7.25, 4.42, 1.36])},
    moving_arm="right",
    separation_axis=np.array([0.0, 1.0, 0.0]),
    descent_m=0.12,
    inset_m=0.06,
)
np.testing.assert_allclose(targets["right"], [7.25, 4.76, 1.24])
np.testing.assert_allclose(targets["left"], [7.25, 4.42, 1.36])
```

**Step 2: Run the tests and verify RED**

Run the new test nodes and expect missing-function failures.

**Step 3: Implement the minimal pure helper**

Normalize the axis, compute the two arm projections, derive the moving arm's
inward sign from its projection relative to their midpoint, and return copied
targets with only the requested arm changed.

**Step 4: Run focused tests**

Run:
`pytest -q tests/test_competition_transport.py`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add tests/test_competition_transport.py submission/JCIIOT/src/robot_agent/skills/competition_transport.py
git commit -m "feat: plan sequential under-support targets"
```

### Task 3: Add an opt-in first-arm transition probe

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`
- Modify: `scripts/run_l1_cradle_gate.py`

**Step 1: Write failing parser and sequence tests**

Add defaults and overrides for:

- `--center-support-moving-arm` (`none`, `right`, or `left`)
- `--center-support-clearance-lift-m`
- `--center-support-descent-m`
- `--center-support-inset-m`

Add source-order assertions showing the transition occurs after
`hold_center_grasp` and before every transport branch.

**Step 2: Run the focused tests and verify RED**

Run the new test nodes and expect missing-argument or missing-stage failures.

**Step 3: Add the minimal runner stage**

When a moving arm is selected:

1. raise both closed arms by the configured clearance increment;
2. call `single_arm_under_support_targets` for that arm;
3. execute the target with the stationary gripper at `+1` and moving gripper at
   `-1` using `build_independent_gripper_action`;
4. record the stationary grasp, moving-arm allowed support contacts, object
   height, collision flag, and final geometry;
5. stop the probe after this stage regardless of outcome.

Extend `execute_stage` only enough to accept either a scalar gripper command or
a two-arm mapping. Keep all existing scalar call sites behavior-identical.

**Step 4: Run focused and full tests**

Run:

```bash
pytest -q tests/test_l1_cradle_gate.py
pytest -q
python -m py_compile scripts/run_l1_cradle_gate.py \
  submission/JCIIOT/src/robot_agent/skills/competition_grasp.py \
  submission/JCIIOT/src/robot_agent/skills/competition_transport.py
```

Expected: all tests pass and compilation succeeds.

**Step 5: Audit and commit**

Run the repository's scored-path audit, `git diff --check`, and verify the only
submission changes are under the allowed `skills/` path. Commit the runner and
tests with the allowed skill changes.

### Task 4: Run the first-arm physical experiment

**Files:**
- Create: `autoresearch/classic-260729-under-support-transition/config.md`
- Create: `autoresearch/classic-260729-under-support-transition/results.tsv`
- Create: `autoresearch/classic-260729-under-support-transition/conclusion.md`
- Create external artifact under: `artifacts/l1-under-support-transition-20260729/`

**Step 1: Establish infrastructure invariants**

Record the pinned official commit, candidate hashes, and current 8502 PID. Copy
only the allowed skill files plus external research runner. Do not restart the
official page.

**Step 2: Run a right-arm transition**

Start with measured bounded values: clearance lift `0.08 m`, descent `0.12 m`,
inset `0.04 m`. The run must stop after the right-arm stage.

**Step 3: Decide from measured gates**

Keep only if collision and forbidden-call counts are zero, object height remains
safe, the left arm preserves grasp, and the right side obtains an allowed hand,
wrist, or distal-arm contact. Otherwise record the exact failure boundary.

**Step 4: Iterate one parameter at a time**

If no allowed support contact is observed, vary in this order: descent, inset,
then moving-arm order. If object height or stationary grasp fails, vary
clearance lift first. Never stack parameter changes in one iteration.

### Task 5: Gate the complete transition and resume transport

Only after one arm passes, add the mirrored second-arm stage by repeating Tasks
1-4 with tests first. Require 20 consecutive bilateral allowed-support frames,
then run a `0.20 m` `-y` corridor carry. Progress to `1.05 m`, route completion,
official Task 1 score, and finally the other four scenes only after each prior
gate passes without collision or forbidden manipulation.
