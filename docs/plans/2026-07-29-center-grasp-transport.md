# Center-Grasp Physical Transport Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the repeatable L1 center grasp into a collision-checked physical carry with measured object translation.

**Architecture:** Add an opt-in carry distance to the research runner, derive one straight waypoint from live base/object geometry, and delegate motion to the existing `run_physical_transport`. Add a route-specific diagnostic gate that keeps gripper grasp stability separate from cradle-link support.

**Tech Stack:** Python 3, NumPy, pytest, robosuite/MuJoCo, existing `PhysicalCarryConfig`, `OfficialPhysicalCarryDriver`, and `run_physical_transport`.

---

### Task 1: Define the carry target and gate by TDD

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`
- Modify: `scripts/run_l1_cradle_gate.py`

Write red tests for `forward_carry_target`: target lies the requested distance
from base toward object; zero distance is allowed; negative/non-finite distance
and coincident base/object with positive distance are rejected. Implement the
minimal pure helper.

Write red tests for `center_grasp_transport_failures`: a complete record passes;
each missing physical grasp, lift, hold, transport success, >1 m object motion,
collision/write/attachment/drop/infrastructure condition fails. Implement the
gate without reusing cradle support fields.

### Task 2: Add the opt-in physical transport stage

**Files:**
- Modify: `tests/test_l1_cradle_gate.py`
- Modify: `scripts/run_l1_cradle_gate.py`

Add a red sequence test proving `run_physical_transport` occurs after
`hold_center_grasp`. Add `--center-carry-distance-m` with default zero and pass
it through `run_probe` to `_center_regrasp_probe`.

After the hold, construct a waypoint with `forward_carry_target`, configure the
existing helper conservatively, run it, append structured stage evidence, and
fail the probe if transport fails. Record hold grasp steps, transport success,
requested distance, and measured object planar translation. Select the new
gate only when carry distance is positive.

Run focused tests, module tests, compilation, and diff checks; commit.

### Task 3: Complete local verification

Run all tests, scored-path audit, workspace checks, and diff checks. Confirm no
submission file changed.

### Task 4: Execute the bounded carry series

**Files:**
- Create: `autoresearch/classic-260729-center-grasp-transport/config.md`
- Create: `autoresearch/classic-260729-center-grasp-transport/results.tsv`
- Create: `autoresearch/classic-260729-center-grasp-transport/conclusion.md`

Sync only the runner and verify its hash, official commit, and unchanged 8502
PID. Run 0.20 m first. If all higher gates remain true, run 0.50 m, then 1.05 m.
Record requested/base/object translation, contact-preserving transport steps,
minimum object height, collision, failure stage, artifact hash, and decision.
If a distance fails, keep it fixed and change only one controller parameter per
subsequent experiment.
