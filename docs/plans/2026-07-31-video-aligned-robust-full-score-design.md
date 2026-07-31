# Video-Aligned Robust Full-Score Design

## Status

Approved on 2026-07-31. The selected success criterion is not a single lucky
official run: level 1 must achieve the official full score with zero collision
and pass a repeated perturbation suite before work moves to level 2.

## Evidence Base

This design is grounded in the official technical-sharing video, its slides,
the pinned official source at commit
`0dcdddf18a9e694569aa1433cdfc04eb097fed78`, and the local source audit in
`docs/10-official-technical-sharing-analysis.md`.

The official task sequence is:

`move -> pick_up -> move -> place_down`

The video requires visible physical contact and lift during pickup. The
official implementation also explains that navigation edits the Tiago base
qpos directly, so a physically held free object does not naturally follow over
long distances. The official baseline therefore captures a transport
attachment after pickup, synchronizes it during navigation, then lowers and
releases the object under gravity at the output table.

## Considered Approaches

### Deterministic-only

Use geometric navigation, a fixed bilateral grasp, the official transport
attachment, and a fixed place routine. This is the shortest route to a level-1
score but is sensitive to object pose and scene variation and offers limited
innovation.

### Learned end-to-end

Train BC-RNN or Diffusion Policy to generate the continuous manipulation
actions. This offers eventual generalization but requires a large amount of
task-native Tiago data, makes collision behavior harder to bound, and delays
the first trustworthy official score.

### Layered hybrid (selected)

Use deterministic workflow, route planning, safety checks, attachment gating,
placement, and verification. Put learning only at the manipulation boundary
where the official video identifies the baseline weakness: approach and
pickup. This provides an auditable level-1 path now and a controlled route to
cross-scene generalization later.

## Architecture

The scored path consists of nine explicit stages:

1. Parse the SOP and resolve source, object, and destination semantics.
2. Plan a collision-aware path to a pre-grasp base pose.
3. Generate bounded bilateral grasp candidates from observed object geometry.
4. Execute either the deterministic pickup controller or a trained pickup
   policy through the same skill interface.
5. Gate transport on bilateral contact, measured lift, finite state, a stable
   hold interval, and no collision.
6. Capture the official transport attachment only after the gate passes.
7. Execute collision-aware loaded navigation while checking object clearance.
8. Align with the destination, lower under the official constraint, clear the
   attachment, open the grippers, and allow gravity to settle the object.
9. Verify the official success predicates and record evidence.

The workflow owns state transitions and failure recovery. Continuous control
is owned by the deterministic controller or learned pickup policy. An LLM may
parse a new SOP or select among bounded skills, but it is not permitted to emit
joint actions or bypass a physical gate. The five published tasks do not need
an online API key on their primary scored path.

## Data Flow

Each run starts from an immutable experiment manifest containing the official
commit, overlay commit, configuration hash, model hash, scene, object, initial
pose perturbation, and seed. Runtime observations flow into navigation and
pickup. Stage results flow into a structured trace containing contacts, lift,
hold stability, collisions, attachment capture and release, object displacement,
destination error, elapsed time, and official score.

Only successful task-native Tiago pickup segments may enter the training
dataset. Training, validation, and held-out pose splits are fixed by manifest;
model selection uses physical success and safety metrics rather than training
loss alone.

## Failure Handling

- A navigation failure stops before pickup.
- A failed pickup never enables the transport attachment.
- A non-finite observation, one-sided contact, insufficient lift, unstable
  hold, or collision fails the pickup gate.
- Recovery is bounded to one retreat and a small, configured candidate set.
- A loaded-navigation or placement failure clears the attachment only through
  the official release path and marks the run failed.
- No recovery may directly write the object freejoint or modify a forbidden
  competition file.

Failures are retained as experiment evidence and as candidates for offline
analysis. They are not silently counted as successful demonstrations.

## Validation Gates

### Level 1 full-score gate

- The unmodified official app and scorer report `10/10` with zero collisions.
- Five consecutive nominal runs report `10/10` with zero collisions.
- A 20-case suite varies robot XY, robot yaw, object XY, and control seed.
- At least 18 of 20 perturbation cases report `10/10`.
- All 20 perturbation cases report zero collisions.
- Successful videos visibly show physical pickup before attachment capture and
  stable gravity-supported placement after attachment release.
- Protected-file hashes match the pinned official baseline and the scored-path
  audit reports zero hard violations.

Level 2 work cannot begin until this gate passes.

### Cross-scene gate

For each subsequent level, first obtain one official full-score zero-collision
run, then repeat the same nominal and perturbation protocol. Level 5 additionally
requires all three objects to be delivered in one run without state leakage
between transfers.

### Learned-policy gate

BC-RNN is the first learned baseline. Diffusion Policy is trained only against
the same data split and compute budget. A learned policy is promoted only when
its held-out pickup success rate exceeds the deterministic controller without
increasing collision rate or violating runtime constraints.

## Scope and Compliance

Competition changes remain inside:

- `src/robot_agent/skills/`
- `src/robot_agent/workflows/`
- `knowledge/robot_params.json`

The implementation must not modify:

- `src/robot_agent/core/`
- `src/robot_agent/environments/`
- `app.py`
- `knowledge/task_config.json`

The official attachment is used as published, only after verified physical
pickup. Submission-owned code must not call a direct object-pose synchronization
helper or write an object freejoint.

## Delivery Order

1. Restore official server access and validate the current gated task-1 path.
2. Close task-1 pickup, transport, placement, or collision failures one measured
   failure stage at a time.
3. Pass the complete level-1 full-score gate.
4. Build the task-native pickup dataset and compare BC-RNN with Diffusion Policy.
5. Extend and revalidate levels 2 through 5 sequentially.
6. Package the reproducible code, model, manifests, report, trajectories, score
   evidence, and multi-view GIFs.
