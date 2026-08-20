# Strict Physical Carry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove floor-push and attachment shortcuts from the official competition entrypoint, then establish a measurable data and controller path for real continuous physical carrying.

**Architecture:** Keep the existing SOP parser, object-family grasp profiles, navigation, and physical state machine. Change the entrypoint to use only the physical-carry transport contract; a failed physical hold must fail closed instead of falling back to floor push or transport attachment. Improve the physical controller only after the strict routing guard is verified, using one contact-topology hypothesis per experiment and accepting only zero-collision, zero-shortcut trajectories.

**Tech Stack:** Python 3.11, MuJoCo/robosuite, NumPy, unittest/pytest, HDF5 via h5py, existing JCIIOT skills/workflows.

---

### Task 1: Lock the strict transport contract

**Files:**
- Modify: `tests/test_competition_flow.py`
- Modify: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py:211-227,926-981`

**Step 1: Write failing tests**

Add tests asserting that every official level selects `physical_carry`, that `l1_floor_push` is rejected by the driver constructor, and that attachment activation is not reachable from the strict entrypoint.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_competition_flow.py -k 'strict_physical or transport_mode'`

Expected: FAIL because the current entrypoint selects `l1_floor_push` for L1-L4 and permits only `attachment`/`l1_floor_push` modes.

**Step 3: Implement the minimal guard**

Permit only `physical_carry` in the official driver and route every level through it. Remove the floor-push and attachment branches from the official carrying path; preserve historical helper functions for audit/replay but make them unreachable from `run_official_task`.

**Step 4: Run tests to verify they pass**

Run the focused test command, then the full competition-flow test module.

**Step 5: Commit**

`git add tests/test_competition_flow.py submission/JCIIOT/src/robot_agent/workflows/competition_flow.py && git commit -m "fix: fail closed on nonphysical transport shortcuts"`

### Task 2: Add physical-carry data and integrity audits

**Files:**
- Create: `scripts/audit_physical_carry_hdf5.py`
- Create: `tests/test_audit_physical_carry_hdf5.py`
- Modify: `docs/12-data-and-algorithm-register.md`

**Step 1:** Add a failing audit test for HDF5 schema, 20-dimensional actions, finite values, complete seed-level splits, and zero attachment/object-pose writes.

**Step 2:** Run the focused audit tests and confirm failure on invalid fixtures.

**Step 3:** Implement the smallest schema/integrity auditor and a manifest writer. Keep this workspace-only tool outside the competition submission tree because it is not part of the runtime Agent.

**Step 4:** Run the audit tests and inspect the generated manifest.

**Step 5:** Record the verified current datasets, their limitations, and the accepted teacher-data gate.

### Task 3: Improve only the physical grasp controller

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`
- Modify: corresponding grasp/transport tests

**Step 1:** Add tests for contact-topology profiles and fail-closed minimum lift/continuous bilateral contact.

**Step 2:** Run them red against the current side-friction profile.

**Step 3:** Implement one bounded topology change at a time: direction-aligned opposed contact, guarded vertical support, and contact-aware reseat. Do not alter core, environment, app.py, or task_config.json.

**Step 4:** Run unit tests, then one server experiment per registered gate: static lift, 0.10 m withdrawal, 0.50 m carry.

**Step 5:** Keep only trajectories passing zero collision, bilateral contact, minimum lift, zero attachment, zero object-pose writes, and real object displacement gates.

### Task 4: Define the data expansion and learning gate

**Files:**
- Modify: `docs/12-data-and-algorithm-register.md`
- Modify: `experiments/experiment-log.csv`

Collect only task-native TIAGo trajectories after a complete physical teacher exists. Split by seed, not frames; cover object pose, yaw, friction, mass, approach drift, and asymmetric contact recovery. Retrain the same Diffusion configuration as the controlled comparison. Promote a learned residual only at `>=8/10` unseen closed-loop physical successes, zero collisions, and zero shortcut events.

### Task 5: Verification and handoff

Run focused tests, full allowed-submission tests, static shortcut scans, and one server smoke run. Report actual results and explicitly distinguish fixed-scene public scores from strict physical-carry evidence.
