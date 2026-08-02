# Quality-Gated Physical Transport Data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and apply a fail-closed audit that admits only attachment-free physical transport evidence with sufficient measured object displacement.

**Architecture:** A standalone pure-Python auditor reads compact result JSON, derives object translation from recorded start and final object positions, and emits a classification plus explicit failures. A batch command writes an immutable JSON/TSV ledger for existing server results; no scored submission file or live service is modified.

**Tech Stack:** Python 3.11 standard library, `unittest`, JSON, TSV, existing JCIIOT result records.

---

### Task 1: Single-Record Physical Transport Audit

**Files:**
- Create: `tests/test_audit_physical_transport_dataset.py`
- Create: `scripts/audit_physical_transport_dataset.py`

**Step 1: Write the failing tests**

Cover one valid 0.50 m physical transport and rejection cases for missing audit
fields, attachment calls, object-pose writes, collision frames, insufficient
object translation, insufficient lift, loss of bilateral contact, excess drift,
drop, and infrastructure error. Assert that displacement is derived from the
transport probe's start and final object positions rather than base motion or a
pre-grasp position.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py`

Expected: collection failure because the audit module does not exist.

**Step 3: Implement the minimal auditor**

Implement:

```python
def audit_record(record, *, minimum_object_translation_m=0.50,
                 minimum_object_lift_m=0.13,
                 maximum_object_gripper_drift_m=0.05) -> dict: ...
```

The output contains `classification`, `eligible`, `failures`, derived metrics,
and the integrity counters used by the decision. Required fields fail closed.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/audit_physical_transport_dataset.py \
  tests/test_audit_physical_transport_dataset.py
git commit -m "research(data): gate physical transport evidence"
```

### Task 2: Batch Ledger and Duplicate Protection

**Files:**
- Modify: `tests/test_audit_physical_transport_dataset.py`
- Modify: `scripts/audit_physical_transport_dataset.py`

**Step 1: Write the failing tests**

Create temporary result files and assert that batch auditing sorts records,
stores SHA-256 provenance, separates `transport_success`, `recovery`, and
`rejected`, and flags duplicate file content. Verify that the CLI writes JSON
and TSV atomically.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py`

Expected: failures for missing batch and CLI behavior.

**Step 3: Implement the minimal batch command**

Support repeatable input paths and directory globs, `--json-output`, and
`--tsv-output`. Do not copy or mutate source records.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/audit_physical_transport_dataset.py \
  tests/test_audit_physical_transport_dataset.py
git commit -m "research(data): add transport evidence ledger"
```

### Task 3: Audit Existing Server Evidence

**Files:**
- Create: `autoresearch/classic-260802-physical-data-gate/config.md`
- Create: `autoresearch/classic-260802-physical-data-gate/results.tsv`
- Create: `autoresearch/classic-260802-physical-data-gate/ledger.json`
- Create: `autoresearch/classic-260802-physical-data-gate/conclusion.md`
- Modify: `research-log.md`
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`

**Step 1: Verify local regression tests**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py tests/test_l1_cradle_gate.py`

Expected: all tests pass.

**Step 2: Synchronize only the auditor to an isolated server tool directory**

Record local and remote SHA-256 values and require equality. Do not synchronize
the submission overlay or restart ports 8502/8503.

**Step 3: Audit the existing full-physical L1 compact result files**

Run the auditor against `/home/user/jciiot-2026/results/full-physical-l1-20260802/`
and write the JSON/TSV ledger under an isolated server result directory.

**Step 4: Download and verify the ledger**

Copy the ledger into `autoresearch/classic-260802-physical-data-gate/`, verify
hash equality, and summarize the eligible count, failure-mode counts, best true
object translation, and integrity violations.

**Step 5: Update the append-only records**

Document H6's verified 36 demonstrations and 9,866 samples, its offline-only
status, and the new transport-data eligibility result. Do not claim a score or
closed-loop success that was not measured.

**Step 6: Commit**

```bash
git add autoresearch/classic-260802-physical-data-gate \
  research-log.md STATUS.md CHANGELOG.md
git commit -m "docs(research): audit physical transport data eligibility"
```

### Task 4: Select One Next Physical Experiment

**Files:**
- Create: `autoresearch/classic-260802-physical-carry-next/config.md`
- Create after execution: `autoresearch/classic-260802-physical-carry-next/results.tsv`
- Create after execution: `autoresearch/classic-260802-physical-carry-next/conclusion.md`

**Step 1: Select the closest eligible-adjacent record**

Choose by maximum derived object translation subject to zero collision,
attachment, object-pose write, drop, and infrastructure error. Inspect the first
failing boundary and freeze every unrelated controller parameter.

**Step 2: State one falsifiable hypothesis**

Document one controller change and its expected effect before editing or
running. Do not relax the final 0.50 m object-translation gate.

**Step 3: Execute one seed-0 diagnostic**

Run in a new isolated server tool and result directory. Preserve compact JSON
and full trajectory. Do not change the live app service.

**Step 4: Keep or discard against the fixed metric**

Keep only if true object translation improves without any integrity or safety
regression. Otherwise restore the incumbent and document the failed hypothesis.

**Step 5: Commit the evidence**

```bash
git add autoresearch/classic-260802-physical-carry-next
git commit -m "research(control): test next physical carry hypothesis"
```

### Task 5: Full Verification

**Files:**
- Modify if needed: `experiments/README.md`

**Step 1: Run focused tests**

Run: `python -m pytest -q tests/test_audit_physical_transport_dataset.py tests/test_collect_native_grasp_demo.py tests/test_merge_native_grasp_demos.py tests/test_l1_cradle_gate.py`

Expected: all tests pass.

**Step 2: Run workspace checks**

Run: `bash scripts/check_workspace.sh`

Expected: exit 0.

**Step 3: Verify repository scope**

Run: `git status --short` and `git diff --check main...HEAD`.

Expected: clean worktree and no whitespace errors after commits.
