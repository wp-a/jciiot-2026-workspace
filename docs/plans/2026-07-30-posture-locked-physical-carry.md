# Posture-Locked Physical Carry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove or reject a 0.10 m attachment-free physical carry after the candidate's verified bilateral L1 grasp.

**Architecture:** Extend only the external L1 research runner with a pure evidence gate, a scoped transport-attachment audit, and an opt-in posture-locked carry probe. The probe reuses the candidate's physical grasp and official posture-locked base navigation, but rejects any run that activates attachment, writes object qpos through the attachment helper, collides, drops, slips, or loses bilateral terminal contact.

**Tech Stack:** Python 3.13, NumPy, unittest/pytest, robosuite/MuJoCo, official pinned server environment, JSON trajectory evidence.

---

### Task 1: Define the short-carry evidence gate

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write the failing tests**

Add a valid record fixture and tests that require:

```python
{
    "posture_carry_success": True,
    "projected_object_progress_m": 0.08,
    "lateral_object_drift_m": 0.03,
    "object_gripper_drift_m": 0.03,
    "final_object_lift_m": 0.10,
    "terminal_bilateral_contact": True,
    "collision_frames": 0,
    "attachment_activations": 0,
    "object_pose_writes": 0,
    "infrastructure_error": None,
}
```

Test each boundary independently and reject non-finite numeric values.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_l1_cradle_gate.py -k posture_carry_gate -q
```

Expected: FAIL because `posture_carry_failures` is undefined.

**Step 3: Implement the minimal pure gate**

Add `POSTURE_CARRY_THRESHOLDS`, `POSTURE_CARRY_REQUIRED_FIELDS`,
`posture_carry_failures(record)`, and `posture_carry_accepted(record)`. Use
inclusive pass thresholds and return stable field names as failures.

**Step 4: Run tests and verify GREEN**

Run the focused test, then:

```bash
python -m pytest tests -q
```

Expected: focused tests pass and the project suite remains green.

**Step 5: Commit**

```bash
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "test: gate posture-locked physical carry"
```

### Task 2: Add a scoped transport-attachment audit

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write the failing tests**

Test a small context manager against a fake transport module. It must count
calls to `capture_transport_attachment` and `set_object_qpos`, report an active
attachment before or after the scope, and restore the original module
functions even when the body raises.

**Step 2: Run the focused test and verify RED**

```bash
python -m pytest tests/test_l1_cradle_gate.py -k transport_attachment_audit -q
```

Expected: FAIL because the audit helper is undefined.

**Step 3: Implement the minimal audit helper**

Add a context manager that temporarily wraps only the transport module's
`capture_transport_attachment` and `set_object_qpos`. Record:

```python
{
    "attachment_activations": 0,
    "object_pose_writes": 0,
    "active_before": False,
    "active_after": False,
}
```

The helper must never suppress the wrapped call or an exception.

**Step 4: Run tests and commit**

```bash
python -m pytest tests -q
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "test: audit transport attachment use"
```

### Task 3: Implement the 0.10 m posture-locked carry probe

**Files:**
- Modify: `scripts/run_l1_cradle_gate.py`
- Test: `tests/test_l1_cradle_gate.py`

**Step 1: Write failing pure-geometry tests**

Add tests for a helper that decomposes object displacement into projected
progress and absolute lateral drift for a finite normalized world direction.
Reject zero or non-finite direction vectors.

**Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_l1_cradle_gate.py -k directed_planar_progress -q
```

**Step 3: Implement the geometry helper**

Normalize the requested direction and return:

```python
projection = float(np.dot(displacement, direction))
lateral = float(abs(np.cross(direction, displacement)))
```

**Step 4: Add the runtime probe**

Implement `_posture_locked_carry_probe` with this sequence:

1. reject an active transport attachment before movement;
2. capture base, object, both EEF positions, and object-relative EEF offsets;
3. call `backend.follow_path` for exactly one bounded waypoint;
4. run inside the scoped attachment audit;
5. read terminal object pose, EEF poses, bilateral contacts, collision state,
   and object-to-gripper transform drift;
6. return compact evidence without altering protected code or object qpos.

Do not add retries, long paths, placement, or policy inference.

**Step 5: Add CLI integration**

Add:

```text
--posture-locked-carry-distance-m
--posture-locked-carry-world-direction-x
--posture-locked-carry-world-direction-y
```

When distance is positive, run the probe immediately after a successful
candidate physical grasp. Select `posture_locked_physical_carry` as the record
mode and use `posture_carry_failures` for final acceptance.

**Step 6: Run local verification and commit**

```bash
python -m pytest tests -q
python -m py_compile scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git diff --check
git add scripts/run_l1_cradle_gate.py tests/test_l1_cradle_gate.py
git commit -m "feat: probe posture-locked physical carry"
```

Expected: all project tests pass.

### Task 4: Run the official pinned 0.10 m experiment

**Files:**
- Modify after result: `autoresearch/classic-260730-posture-carry/results.tsv`
- Create after result: `autoresearch/classic-260730-posture-carry/config.md`
- Create after result: `autoresearch/classic-260730-posture-carry/conclusion.md`

**Step 1: Verify local and remote runner hashes**

Copy only the external research runner to the server tools directory and
require matching SHA-256 hashes. Do not change the candidate or the 8502
process.

**Step 2: Run one 0.10 m trial**

Use the immutable candidate, official commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`, seed 0, and a world direction away
from the base through the grasped object.

**Step 3: Inspect the hard evidence**

Report and archive:

- base displacement;
- projected object progress and lateral drift;
- final object lift and minimum recorded object height;
- terminal bilateral contacts and object-to-gripper drift;
- collision frames and pairs;
- attachment activations and object-pose writes;
- infrastructure errors and elapsed time.

**Step 4: Keep or reject**

If every gate passes, repeat at seed 0 for determinism and then 0.25 m. If any
gate fails, change exactly one controller or direction parameter and preserve
the failed record.

**Step 5: Commit the evidence index**

```bash
git add autoresearch/classic-260730-posture-carry
git commit -m "research: record posture-locked carry trial"
```

### Task 5: Decide the L1 scale-up route

**Files:**
- Modify: `docs/plans/2026-07-30-posture-locked-physical-carry-design.md`
- Create only after short gate passes: `docs/plans/2026-07-30-l1-physical-carry-route.md`

**Step 1: If the short gate passes**

Build a route plan from the official semantic map and collision proxy AABBs,
with 0.25 m checkpoints and an immediate stop on slip, drop, or collision.

**Step 2: If the short gate fails**

Document the measured failure mechanism. Do not integrate the carry mode into
the candidate and do not claim an official score.
