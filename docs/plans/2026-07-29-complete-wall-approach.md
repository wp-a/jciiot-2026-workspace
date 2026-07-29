# Complete Wall Approach Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the open grippers complete the existing bounded descent around the L1 long walls before squeezing and closing.

**Architecture:** Remove only the first-contact completion condition from `approach_center_walls`. Preserve the target, 220-step cap, OSC controller, trajectory recording, official collision checks, and all downstream physical gates.

**Tech Stack:** Python 3, pytest, robosuite/MuJoCo, official JCIIOT collision and grasp helpers.

---

### Task 1: Add the failing regression

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`

Add `test_complete_wall_approach_does_not_stop_on_first_contact`, extracting
the `approach_center_walls` call and asserting that it does not contain
`stop_bilateral_contact_steps`. Run the focused test and confirm it fails on
the current parameter.

### Task 2: Make the minimum behavior change

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`

Delete only `stop_bilateral_contact_steps=1` from the approach call. Run the
focused test, module tests, syntax compilation, and diff check; then commit the
source and test together.

### Task 3: Verify and run the isolated experiment

Run all tests, the scored-path audit, workspace checks, and diff checks. Sync
only the research runner and verify the local/remote hash, immutable official
commit, and unchanged 8502 PID. Re-run seed 0 with the retained 0.10 m base
advance, zero center shift, and 0.040 m squeeze. Record approach/squeeze heights,
contact geoms, grasp frames, lift, hold, collision, artifact hash, and verdict.
