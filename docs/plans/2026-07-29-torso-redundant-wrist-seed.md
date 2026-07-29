# Torso-Redundant Wrist Seed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in Tiago torso degree of freedom to the proven continuation IK seed and preserve the resulting torso target through OSC.

**Architecture:** Keep a pure helper responsible for the ordered solver joint list, apply separate arm and torso interior margins, and reuse the existing residual, continuation, segmented path, collision, rollback, and evidence code for a 13-element vector. Refresh only the torso hold target after a successful seed.

**Tech Stack:** Python 3.11, NumPy, SciPy, unittest/pytest, MuJoCo, robosuite.

---

### Task 1: Ordered solver joint list and CLI

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1:** Write failing tests for `joint_seed_joint_names(False)` returning the
12 right/left arm names and `joint_seed_joint_names(True)` appending exactly
`robot0_torso_lift_joint`.

**Step 2:** Run `python -m pytest -q tests/test_l1_cradle_gate.py -k joint_seed_joint_names`
and verify RED, implement the pure helper, then verify GREEN.

**Step 3:** Extend the parser test before implementation to require
`orientation_joint_seed_include_torso == False` and
`orientation_joint_seed_torso_margin_m == 0.005`; verify RED, add the flags,
and pass them into `_center_regrasp_probe()`.

**Step 4:** Commit the pure/CLI change.

### Task 2: Thirteen-variable runtime and torso hold target

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1:** Resolve the ordered names through the tested helper. Apply the arm
margin to the first 12 bounds and the torso margin to the final bound when
enabled. Record both margins and whether torso is included.

**Step 2:** Reuse the existing objective and continuation logic unchanged over
the resulting vector length. Verify every restore and path assignment uses the
same complete qpos address list.

**Step 3:** On seed success, read the current torso qpos and replace
`hold_targets["torso"]` with a one-element array before OSC starts. Record the
refreshed value in joint-seed evidence. Do not update it on rollback.

**Step 4:** Run module tests, syntax compilation, and diff check; commit.

### Task 3: Full verification and bounded official experiment

**Files:**
- Create: `autoresearch/classic-260729-torso-wrist-seed/config.md`
- Create: `autoresearch/classic-260729-torso-wrist-seed/results.tsv`
- Create after run: `autoresearch/classic-260729-torso-wrist-seed/conclusion.md`

**Step 1:** Run all tests, scored-path audit, workspace check, and diff check.

**Step 2:** Sync only the committed research runner and verify SHA-256 plus the
unchanged official candidate and 8502 process.

**Step 3:** Run seed 0 with the retained 24 nodes, 0.0185 m position scale,
0.01 rad arm margin, and torso inclusion with a 0.005 m margin. Change no other
parameter.

**Step 4:** Retain compact JSON and hashes. Report joint seed, OSC, contact,
lift, and collision gates separately; do not promote or switch 8502 without two
complete physical passes.
