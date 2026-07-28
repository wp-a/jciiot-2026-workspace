# Fully Physical Carry And Placement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace attachment-driven object transport with controller-driven physical carrying and placement, then verify all five public scenes through the unmodified 8502 UI.

**Architecture:** Preserve the verified two-arm OSC grasp and lift, but end grasp with a physical hold instead of an attachment. A new allowed skill will follow A* paths using Tiago's mobile-base action controller while keeping both OSC arms stationary relative to the base and both grippers closed; it will fail on contact loss, object drop, or collision. The same skill will physically descend, release, settle, and measure final distance at the destination.

**Tech Stack:** Python 3.11+, NumPy, MuJoCo, robosuite Tiago BASIC composite controller, unittest/pytest, Streamlit `app.py`, Git, remote NVIDIA L40S server.

---

### Task 1: Define Physical Carry Geometry And Guards

**Files:**
- Create: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`
- Create: `tests/test_competition_transport.py`

**Step 1: Write failing pure-function tests**

Test the desired API:

```python
world_velocity_to_base_frame(world_xy, yaw)
slew_limited_command(previous, requested, max_delta)
transport_base_goal(object_target_xy, base_xy, base_yaw, object_xy)
next_contact_stability(contacts, stable_steps)
```

Assert correct frame rotation, bounded acceleration, preservation of the held
object's base-relative offset, and immediate reset of bilateral-contact
stability when either side is false.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py`

Expected: FAIL because `competition_transport.py` does not exist.

**Step 3: Implement minimal pure helpers and configuration**

Create `PhysicalCarryConfig` with explicit path tolerance, step budget, linear
and angular limits, acceleration limit, object-drop tolerance, descent limit,
support-stability steps, release steps, and settle steps. Implement only the
four tested helpers with NumPy and standard-library math.

**Step 4: Verify GREEN**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py`

Expected: all pure-function tests pass.

**Step 5: Commit**

Commit: `feat: define physical carry control geometry`

### Task 2: Implement Contact-Closed-Loop Path Following

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`
- Modify: `tests/test_competition_transport.py`

**Step 1: Write failing driver-level tests**

Define a deterministic fake driver and test:

```python
run_physical_transport(
    backend,
    path=path,
    object_name=name,
    hold_yaw=yaw,
    minimum_object_z=z,
    config=config,
    driver=fake,
)
```

Verify that successful transport reaches all waypoints through step actions;
the action contains base, both arms, both grippers, torso, and head; a lost
left or right grasp fails with `failure_stage="contact"`; an object drop fails
with `failure_stage="object_drop"`; collision fails with
`failure_stage="collision"`; and max-step exhaustion fails without fallback.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py`

Expected: FAIL on the missing runner or driver.

**Step 3: Implement the official adapter**

`OfficialPhysicalCarryDriver` will use the robot's normal composite action
vector and official read-only grasp/object-position helpers. Each action will
contain zero-delta arm OSC commands, closed grippers, absolute torso/head hold
targets, and a bounded base velocity. Each iteration calls `env.step(action)`
exactly once and records through the existing backend recorder. Do not call
`backend.follow_path`, `_restore_upper_body_posture`, attachment helpers, or
any object-joint setter.

**Step 4: Verify GREEN and regression**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py tests/test_competition_grasp.py`

Expected: all tests pass.

**Step 5: Commit**

Commit: `feat: carry objects through physical base actions`

### Task 3: Remove Post-Grasp Kinematic Attachment

**Files:**
- Modify: `tests/test_competition_grasp.py`
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_grasp.py`

**Step 1: Write the failing grasp contract test**

Change the fake grasp driver so no attachment method exists. Assert a verified
physical grasp succeeds immediately after lift and returns measured hold
metadata needed by transport. Add a source assertion that this skill neither
imports `transport_attachment` nor calls an attachment function.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_competition_grasp.py`

Expected: FAIL because `run_scripted_grasp` still calls
`attach_for_transport` and the source still imports the attachment helper.

**Step 3: Remove attachment and stow behavior**

Delete the attachment import, adapter method, object-relative attachment
profile helper, and success-stage attachment call. Return physical hold yaw,
object Z, and bilateral-contact evidence from the completed grasp. Preserve
the existing physical approach and lift behavior.

**Step 4: Verify GREEN**

Run: `python3.13 -m pytest -q tests/test_competition_grasp.py tests/test_competition_transport.py`

Expected: all tests pass.

**Step 5: Commit**

Commit: `fix: keep grasp physical after lift`

### Task 4: Route Workflow Through Physical Transport

**Files:**
- Modify: `tests/test_competition_flow.py`
- Modify: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`

**Step 1: Write failing workflow tests**

Assert that carrying movement:

- computes a destination base goal from the current physically held object
  offset and the deterministic target slot;
- plans an A* route to that base goal;
- calls `run_physical_transport`, never `MoveSkill.run` or
  `backend.follow_path`;
- propagates contact, drop, collision, and timeout failures;
- never reads, changes, clears, or synchronizes attachment state.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_competition_flow.py`

Expected: FAIL because carrying currently delegates to direct navigation and
final placement uses attachment synchronization.

**Step 3: Implement minimal carrying integration**

Store the physical hold metadata returned by grasp. For carrying, compute the
slot target, derive the matching base goal, plan with the same official map
planner used by empty navigation, and invoke the physical transport skill.
Remove `_extend_held_object_toward_target` and every transport-attachment
branch. Keep empty-handed navigation unchanged.

**Step 4: Verify GREEN and workflow regression**

Run: `python3.13 -m pytest -q tests/test_competition_flow.py tests/test_competition_task.py tests/test_competition_entrypoint.py`

Expected: all tests pass.

**Step 5: Commit**

Commit: `feat: route carrying through physical controller`

### Task 5: Implement Physical Descent And Release

**Files:**
- Modify: `submission/JCIIOT/src/robot_agent/skills/competition_transport.py`
- Modify: `submission/JCIIOT/src/robot_agent/workflows/competition_flow.py`
- Modify: `tests/test_competition_transport.py`
- Modify: `tests/test_competition_flow.py`

**Step 1: Write failing placement tests**

Test `run_physical_place` with a fake driver. Require bilateral contact during
descent, a measured downward object displacement or stable support plateau
before release, open-gripper actions only after that condition, bounded settle
steps, collision rejection, and final planar distance below 0.8 metres.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py tests/test_competition_flow.py`

Expected: FAIL because physical placement is not implemented.

**Step 3: Implement descent, release, and settle**

Command equal bounded world-Z deltas for both OSC arms with the base stopped
and grippers closed. Detect support from a stable object-height plateau under
continued descent command. Open both grippers only after support evidence,
then step physics through the release and settle windows. Record place events
and return measured distance and failure stage.

**Step 4: Verify GREEN and full local suite**

Run: `python3.13 -m pytest -q tests/test_competition_transport.py tests/test_competition_flow.py tests/test_competition_grasp.py`

Run: `python3.13 -m pytest -q`

Expected: all tests pass.

**Step 5: Commit**

Commit: `feat: place held objects through physical release`

### Task 6: Add Hard Submission Compliance Gates

**Files:**
- Modify: `tests/test_submission_boundaries.py`
- Modify: `scripts/audit_scored_path.py`
- Modify: `tests/test_scored_path_audit.py`

**Step 1: Write failing boundary tests**

Scan submitted `skills/` and `workflows/` source. Reject
`transport_attachment`, `sync_transport_attachment`,
`capture_transport_attachment`, attachment-relative state writes, and writes
to task-object free-joint qpos. Keep read-only object pose inspection legal.

**Step 2: Verify RED**

Run: `python3.13 -m pytest -q tests/test_submission_boundaries.py tests/test_scored_path_audit.py`

Expected: FAIL until all old attachment and object-pose write paths are gone.

**Step 3: Complete the scanner rules and remove remaining violations**

Make rule names explicit and report file and line. Do not broaden the scanner
to unrelated test or research code. Remove only scored-path violations exposed
by the new hard rules.

**Step 4: Verify GREEN and audit**

Run: `python3.13 -m pytest -q tests/test_submission_boundaries.py tests/test_scored_path_audit.py`

Run: `python3.13 scripts/audit_scored_path.py --root submission --output artifacts/scored-path-audit-physical.json`

Expected: no attachment/object-pose violations and exit 0 for the hard
competition rules. Any remaining private instrumentation warning must be
listed separately and must not mutate object state.

**Step 5: Commit**

Commit: `test: enforce physical scored-path boundaries`

### Task 7: Run The L1 Physical Autoresearch Gate

**Files:**
- Create: `autoresearch/classic-260728-<time>/config.md`
- Create: `autoresearch/classic-260728-<time>/results.tsv`
- Create: `artifacts/remote-physical-l1-<commit>/`

**Step 1: Materialize a clean remote candidate**

Materialize the locked official commit plus only the submission overlay into a
new remote candidate directory. Restore the official checkpoint by verified
SHA-256 without modifying protected sources. Record protected-file hashes.

**Step 2: Run one L1 diagnostic attempt**

Use the same public L1 scene, seed, official scorer, and remote pinned
environment. Capture score, collision frames, grasp events, transport contact
checks, minimum object height, final distance, and elapsed time.

**Step 3: Inspect the first-person replay**

Render birdview and `robot0_robotview` from the unchanged scored trajectory.
Reject the candidate if the box exhibits unsupported motion, contacts disappear
during carry, or the scorer result cannot be reproduced.

**Step 4: Iterate one hypothesis at a time**

Tune only one of speed, acceleration, waypoint tolerance, descent rate, or
support stability per iteration. Log every attempt and keep only changes that
improve the ordered metric: physical validity, collision-free completion,
score, then elapsed time.

**Step 5: Run L1 through 8502**

Restart only the candidate Streamlit service if needed, reload the existing
page, click the L1 `Execute` button, and preserve the UI result. Do not proceed
unless the unmodified page reproduces the physical L1 success.

### Task 8: Expand To Five Public Scenes And Package Evidence

**Files:**
- Create: `artifacts/remote-ui-physical-five-level-<commit>/`
- Modify: `research-log.md`
- Modify: `research-state.yaml`

**Step 1: Run L2-L4 sequentially**

For each level, require the same physical, collision, score, and replay gates.
Fix a level before moving to the next; never infer success from L1.

**Step 2: Run L5 sequential multi-object transfer**

Verify all three totes are independently grasped, physically transported,
released, and remain within the official target radius without disturbing
previous placements.

**Step 3: Execute all five through 8502**

Click each `Execute` button in one UI session. Record displayed scores and
times, original trajectories, logs, result JSON, and source hashes.

**Step 4: Render and inspect multiple views**

Generate birdview and `robot0_robotview` GIFs from unchanged trajectories.
Record frame counts and checksums. Do not canonicalize or alter the scored
trajectory for physical-validity inspection.

**Step 5: Final verification**

Run: `git diff --check && python3.13 -m pytest -q`

Expected: clean diff and all tests pass. Report exact public scores and any
remaining limitations; do not call unmeasured or hidden performance full
score.

