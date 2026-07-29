# Pregrasp Base Advance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a collision-checked 0.10 m unloaded base advance before the center regrasp so both arms can reach the tabled object.

**Architecture:** Test a pure bounded world-velocity helper, add an opt-in stage using the existing official physical driver, and stop on collision or any premature object contact. Preserve the accepted joint seed and real gripper-close sequence.

**Tech Stack:** Python, NumPy, unittest/pytest, MuJoCo, robosuite direct-base driver.

---

### Task 1: Bounded advance velocity

**Files:** `scripts/run_l1_cradle_gate.py`, `tests/test_l1_cradle_gate.py`

1. Write failing tests for `bounded_base_advance_world_velocity()` covering
   direction to object, max speed, final-step clipping, zero remaining distance,
   and invalid inputs.
2. Run the focused test and verify RED.
3. Implement the minimum pure helper and verify GREEN.
4. Commit.

### Task 2: Official runtime base-advance stage

**Files:** `scripts/run_l1_cradle_gate.py`, `tests/test_l1_cradle_gate.py`

1. Add an opt-in `regrasp_base_advance_m` probe argument.
2. Implement `execute_base_advance()` using `OfficialPhysicalCarryDriver.step`,
   `world_velocity_to_base_frame`, zero arm deltas, open grippers, current hold
   targets, and 0.05 s control intervals.
3. Record base/EEF/object/contact/collision evidence every step. Abort on any
   official collision or object contact; require measured translation.
4. Insert after orientation and before retreat. Return the stage failure without
   attempting wall contact.
5. Run module tests and syntax compilation; commit.

### Task 3: CLI, full verification, and official experiment

**Files:** `scripts/run_l1_cradle_gate.py`, `tests/test_l1_cradle_gate.py`,
`autoresearch/classic-260729-real-center-grasp/results.tsv`

1. Extend the parser test to require `regrasp_base_advance_m == 0.0`, verify RED,
   add `--regrasp-base-advance-m`, and wire it through.
2. Run all tests, scored-path audit, workspace check, syntax, and diff check.
3. Sync only the committed runner and verify SHA-256, official commit, and 8502.
4. Repeat the retained arm-only, 24-node, 10-degree-entry route with only
   `--regrasp-base-advance-m 0.10` added.
5. Retain JSON/trajectory evidence and stop on the next measured blocker.
