# L1 Reverse-Egress Physical Carry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce and independently verify the first collision-free, attachment-free L1 physical transport with at least `0.50 m` true object translation by reversing away from the source production line.

**Architecture:** Reuse the immutable server runner and interface-matched candidate. Invoke the runner's existing explicit-route mode so path direction is the only experimental variable, then classify the compact result with the fail-closed physical-data auditor. Preserve raw evidence and hashes; do not modify the live app or submission overlay.

**Tech Stack:** Python 3.11, MuJoCo 3.9.0, robosuite 1.5.2, JSON trajectory evidence, SHA-256, existing physical-data auditor.

---

### Task 1: Pre-register the Reverse-Egress Experiment

**Files:**
- Create: `autoresearch/classic-260802-l1-reverse-egress/config.md`

**Step 1: Record immutable inputs**

Record:

- official commit `0dcdddf18a9e694569aa1433cdfc04eb097fed78`;
- candidate `/home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3`;
- candidate grasp SHA-256 `7128fcb14ebeccdfeb7bbe283cb99d823ac9fa9ac0710667884814964cfc5fd2`;
- candidate transport SHA-256 `2e71651e5e7fdae64c0c3d5557574909eee5ce8f01de20e80c89025f6c2080a2`;
- runner `/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py`;
- runner SHA-256 `50ade2e8609a6c7ea1dac7fbb59ae6a2e6b99b3de240e10d19d8cdce6e405732`;
- auditor SHA-256 `944bcc7a040ee2bf198b29a7a637018844bdc75fe9d37c3112df7b68f047b79e`.

**Step 2: Freeze the command**

Use explicit route waypoint `(8.500015, 4.599998)`, seed 0, actuator-only
transport, `0.04 m/s` maximum linear speed, default `0.03 m` internal drift
guard, zero arm/inward feedforward, and disabled recovery.

**Step 3: State keep/discard rules**

Keep only if the auditor reports `transport_success`; otherwise preserve the
failure boundary and do not change more than one variable in the next trial.

**Step 4: Commit**

```bash
git add autoresearch/classic-260802-l1-reverse-egress/config.md
git commit -m "research(control): preregister L1 reverse egress"
```

### Task 2: Verify Server Inputs and Execute One Diagnostic

**Files:**
- Remote create: `/home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50.json`
- Remote create: `/home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50-trajectory.json`

**Step 1: Verify source hashes**

Run `sha256sum` on the runner, candidate grasp/transport modules, and auditor.
Expected: exact equality with Task 1.

**Step 2: Verify live services remain untouched**

Record existing 8502/8503 process IDs. Do not restart or overwrite either
candidate.

**Step 3: Execute the isolated route**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/user/jciiot-2026/envs/official-pinned-eval/bin/python \
/home/user/jciiot-2026/tools/full-physical-l1-c8a6bf2/run_l1_cradle_gate.py \
  --candidate-root /home/user/jciiot-2026/candidates/l4-target-margin-cc1b5b3 \
  --expected-official-commit 0dcdddf18a9e694569aa1433cdfc04eb097fed78 \
  --output /home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50.json \
  --trajectory /home/user/jciiot-2026/results/l1-reverse-egress-20260802/seed0-xplus-0p50-trajectory.json \
  --seed 0 \
  --full-physical-stage route \
  --full-physical-waypoint 8.500015 4.599998 \
  --full-physical-actuator-only \
  --posture-locked-carry-max-linear-m-s 0.04
```

Expected: the runner writes both files. Its process exit code may be nonzero
when its legacy route gate disagrees; the independent auditor is authoritative.

**Step 4: Preserve compact metrics and hashes**

Record grasp, start/final base and object positions, contact continuity, lift,
drift, collision, integrity counters, failure stage, elapsed time, result hash,
and trajectory hash.

### Task 3: Independently Audit and Decide

**Files:**
- Create: `autoresearch/classic-260802-l1-reverse-egress/ledger.json`
- Create: `autoresearch/classic-260802-l1-reverse-egress/results.tsv`
- Create: `autoresearch/classic-260802-l1-reverse-egress/conclusion.md`

**Step 1: Run the remote auditor on only the compact result**

Expected classification: `transport_success`. Any collision, missing integrity
field, attachment/state write, short motion, lost contact, drop, insufficient
lift, or excess drift must prevent success.

**Step 2: Download compact evidence**

Copy the result, ledger, and TSV into the experiment directory. Keep the full
trajectory remote and record its path and SHA-256 instead of committing it.

**Step 3: Verify local/remote hashes**

Expected: compact result, ledger, and TSV hashes match their remote sources.

**Step 4: Decide**

- If `transport_success`, retain reverse egress as the incumbent and proceed to
  Task 4.
- If not, document the first failure boundary and pre-register exactly one
  causal change. Do not scale data or train a model.

**Step 5: Commit**

```bash
git add autoresearch/classic-260802-l1-reverse-egress
git commit -m "research(control): test L1 reverse egress"
```

### Task 4: Escalate a Passing Egress to the Official Departure Threshold

**Files:**
- Create after execution: `autoresearch/classic-260802-l1-source-exit/config.md`
- Create after execution: `autoresearch/classic-260802-l1-source-exit/results.tsv`
- Create after execution: `autoresearch/classic-260802-l1-source-exit/conclusion.md`

**Step 1: Pre-register a same-direction extension**

If Task 3 passed, change only the explicit `+x` waypoint distance so measured
object displacement can exceed the official strict `1.0 m` departure rule.
Retain every controller and safety setting.

**Step 2: Execute seed 0 and audit**

Require the same integrity gates plus measured `x` or `y` object displacement
strictly greater than `1.0 m` from the task's pre-grasp object position.

**Step 3: Keep or discard**

Keep only a collision-free result. A passing departure is evidence for the
first 50% of L1 objective score, not a full-score claim.

**Step 4: Commit evidence**

Commit compact evidence and append the result to `research-log.md` and
`experiments/experiment-log.csv`.

### Task 5: Regression and Next-Route Handoff

**Files:**
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`
- Modify: `research-log.md`
- Modify: `experiments/experiment-log.csv`

**Step 1: Run focused tests**

Run:

```bash
python -m pytest -q \
  tests/test_audit_physical_transport_dataset.py \
  tests/test_l1_cradle_gate.py
```

Expected: all tests pass.

**Step 2: Run workspace checks**

Run `bash scripts/check_workspace.sh` from the main workspace after merge.
Expected: exit 0.

**Step 3: Update the route state honestly**

Record whether `0.50 m` transport and `>1.0 m` source departure were proved.
Do not call L1 full score until physical arrival within `0.8 m` of `output_4`,
zero collision, and official scoring are also verified.

**Step 4: Prepare the next design**

If source egress passes, the next design uses collision-proxy and semantic-map
geometry to build an explicit corridor route toward `output_4`. If it fails,
the next design targets the measured physical failure boundary.

**Step 5: Commit**

```bash
git add STATUS.md CHANGELOG.md research-log.md experiments/experiment-log.csv
git commit -m "docs(research): record L1 reverse-egress outcome"
```
