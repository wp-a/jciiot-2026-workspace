# Contact-Constrained Close Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Permit an L1 physical close attempt when two distinct object walls are explicitly bracketed by the official fingerpad pairs despite a contact-limited pose target.

**Architecture:** Add one pure bracket-evidence function and two read-only MuJoCo geometry readers to the research runner. Use the bracket only as an alternative completion mode for the approach stage; preserve official close, lift, hold, collision, write, and attachment gates.

**Tech Stack:** Python 3, NumPy, pytest, robosuite/MuJoCo, official gripper `important_geoms` and grasp helpers.

---

### Task 1: Test the pure bracket contract

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`

Write failing tests for: two distinct walls correctly bracketed; a wall outside
one finger pair rejected; both arms assigned to the same wall rejected; invalid
axis or non-finite inputs rejected. Run the focused tests and confirm failure
because the desired helper does not exist.

### Task 2: Implement pure evidence and read-only readers

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`

Implement `fingerpad_bracket_evidence`, `fingerpad_world_positions`, and
`opposed_object_wall_centers`. Use only model/data reads in the runtime readers.
Run focused tests, module tests, compilation, and diff checks.

### Task 3: Integrate the alternative approach completion mode

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Modify: `tests/test_l1_cradle_gate.py`

Add a red source-sequence regression proving the bracket fallback occurs only
after a failed approach and only when the stage did not report an official
collision. Record `pose_target_reached`, `contact_constrained_ready`,
`completion_mode`, fingerpad positions, wall centers, and projections in the
stage result. Do not update any grasp or score field. Run focused and module
tests, then commit.

### Task 4: Verify and execute the first isolated run

Run all tests, scored-path audit, workspace checks, and diff checks. Sync only
the runner; verify SHA-256, official commit, and unchanged 8502 PID. Run the
retained seed-0 high-precenter command with 0.10 m base advance, zero center
shift, and 0.040 m lateral motion. Record readiness evidence, official grasp
frames, lift, hold, collision, artifact hash, and verdict.
