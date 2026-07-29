# Complete Wall Squeeze Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the L1 near-wall regrasp complete its bounded open-gripper squeeze before attempting physical closure.

**Architecture:** Keep the existing near-wall geometry, 25 mm opposed target, controller, recorder, and official collision checks. Remove only the one-frame object-contact completion condition from `squeeze_center_walls`, then run the same server command against the immutable official candidate.

**Tech Stack:** Python 3, pytest, robosuite/MuJoCo, official JCIIOT collision and grasp helpers.

---

### Task 1: Lock the failed behavior with a regression test

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`

**Step 1:** Add a test that extracts the `squeeze_center_walls` call from
`_center_regrasp_probe` and asserts that the call does not contain
`stop_bilateral_contact_steps`.

**Step 2:** Run:
`python -m pytest -q tests/test_l1_cradle_gate.py -k complete_wall_squeeze`

Expected: FAIL because the current call contains
`stop_bilateral_contact_steps=1`.

### Task 2: Implement the minimal squeeze correction

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`

**Step 1:** Remove only `stop_bilateral_contact_steps=1` from the
`squeeze_center_walls` call. Preserve open grippers, target calculation,
maximum steps, collision checks, and downstream grasp gate.

**Step 2:** Run the focused test and module tests; both must pass.

**Step 3:** Run `python -m py_compile scripts/run_l1_cradle_gate.py` and
`git diff --check`.

**Step 4:** Commit the source and regression test together.

### Task 3: Verify the complete local safety envelope

**Files:**
- Read: `submission/JCIIOT/`

**Step 1:** Run `python -m pytest -q tests`.

**Step 2:** Run
`python scripts/audit_scored_path.py --root submission/JCIIOT --output /tmp/jciiot-complete-squeeze-audit.json`.

Expected: zero hard violations, warnings, and total violations.

**Step 3:** Run `bash scripts/check_workspace.sh` and `git diff --check`.

### Task 4: Run the isolated server experiment

**Files:**
- Create: `autoresearch/classic-260729-complete-wall-squeeze/config.md`
- Create: `autoresearch/classic-260729-complete-wall-squeeze/results.tsv`
- Create: `autoresearch/classic-260729-complete-wall-squeeze/conclusion.md`

**Step 1:** Synchronize only `scripts/run_l1_cradle_gate.py`; verify its SHA-256,
official candidate commit `0dcdddf18a9e694569aa1433cdfc04eb097fed78`, and
the unchanged 8502 PID.

**Step 2:** Run seed 0 with the retained 24-node wrist seed, 10-degree runtime
entry, 0.10 m base advance, zero center shift, and unchanged 25 mm squeeze.

**Step 3:** Record stage counts, contact geom names, consecutive official grasp
frames, lift, hold, collision frames, artifact paths, and hashes.

**Step 4:** Keep the change only if it improves the ordered physical metric
without weakening a higher gate. If it fails, form one new evidence-based
hypothesis and change one variable in the next bounded loop.
