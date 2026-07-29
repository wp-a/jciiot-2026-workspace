# High Precenter Regrasp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorder the existing L1 opposed-wall motions so open grippers align with the wall planes at high clearance before descending and closing.

**Architecture:** Reuse the existing target helpers and `execute_stage`; move the complete `squeeze_center_walls` block before `approach_center_walls`, without changing target magnitudes, controls, or physical gates. This is an isolated research-runner change.

**Tech Stack:** Python 3, pytest, robosuite/MuJoCo, official JCIIOT collision and grasp helpers.

---

### Task 1: Lock stage order with a red test

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`

Add a test asserting that `squeeze_center_walls` occurs before
`approach_center_walls` and both occur before `close_center_grasp` in
`_center_regrasp_probe`. Run the focused test and confirm the current code
fails because approach precedes squeeze.

### Task 2: Reorder only the existing stages

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`

Move the squeeze target calculation and complete squeeze call ahead of the
approach target calculation and call. Preserve open-gripper commands, step
limits, failure names, collision checks, close schedule, and downstream lift
and hold behavior. Run focused and module tests, compilation, and diff checks;
commit source and test together.

### Task 3: Verify and execute iteration 4

Run the full test suite, scored-path audit, workspace checks, and diff checks.
Sync only the research runner, verify its hash plus the immutable official
commit and unchanged 8502 PID, then run seed 0 with the retained 0.10 m base
advance, zero center shift, and 0.040 m precenter motion. Record high precenter
contacts, descent contact transitions, official grasp frames, lift, hold,
collision count, artifact hash, and verdict.
