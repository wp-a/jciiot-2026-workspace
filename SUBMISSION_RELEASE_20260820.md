# JCIIOT 2026 Submission Release

Release date: 2026-08-20 (Asia/Shanghai)

## What is being submitted

This Git revision contains the official JCIIOT overlay and the reproducibility
material needed to inspect it. The runtime modification surface is limited to:

- `submission/JCIIOT/src/robot_agent/skills/`
- `submission/JCIIOT/src/robot_agent/workflows/`
- `submission/JCIIOT/knowledge/generated_sop_l1.md` through `generated_sop_l5.md`

The protected official files are not modified by the overlay:
`app.py`, `src/robot_agent/core/`, `src/robot_agent/environments/` and
`knowledge/task_config.json`.

## Runtime route

The submission entrypoint uses one transport contract for L1-L5:

```text
SOP/task metadata
  -> deterministic object state machine
  -> collision-aware base approach
  -> two-arm OSC grasp and measured lift
  -> physical_carry (continuous hold required)
  -> physical_place (measured descent/support)
  -> official trajectory and score audit
```

`OfficialCompetitionDriver` rejects `attachment` and `l1_floor_push` at
construction time. `run_official_task` passes only `physical_carry`. The object
pose is never written by the overlay. If bilateral contact, lift, transport or
place gates fail, the route returns failure instead of changing the object state
through an attachment or floor-push fallback.

## Reproduction

1. Check out the official baseline at the exact commit in
   `config/upstream-lock.json` (`0dcdddf18a9e694569aa1433cdfc04eb097fed78`).
2. Materialize the overlay with `scripts/materialize_submission.sh`.
3. Install the official Linux dependencies, including Python 3.11, MuJoCo 3.9.0
   and robosuite 1.5.2.
4. Run `PYTHONDONTWRITEBYTECODE=1 python3.11 -m pytest -q tests`.
5. Run each task through `scripts/run_official_experiment.py` as shown in
   `submission/README.md`; use the current Git revision for `--workspace-commit`.

The unmodified official scorer is the only score authority. A trajectory is
accepted locally only if it has the required score, required successful grasp
events, zero collision frames, target distance below `0.8 m`, and a successful
workflow result.

## Data and models

No private data, checkpoint or API key is included. The official sample HDF5 is
not a JCIIOT Tiago training set: the historical audit found five Fetch/iGibson
demos with ten-dimensional actions. The local copy is currently a Git LFS
pointer, so this release does not pretend that its arrays were re-read locally.
Task-native H2/H5b/H6 data and Diffusion/BC-RNN experiments are research
artifacts, not hidden dependencies of the scored route. Their evidence level and
limitations are recorded in `docs/12-data-and-algorithm-register.md`.

## Verification performed for this release

- Strict physical-carry entrypoint tests pass, including rejection of shortcut
  transport modes.
- HDF5 physical-data audit tests pass for dimensions, finite values, event order,
  seed-level splits and zero shortcut counters.
- Submission boundary tests confirm the overlay contains only allowed files.
- `git diff --check` and Python compilation checks pass.

These are code and data-integrity checks. They are not a claim that five-level
strict-physical transport has already succeeded on the 8502 server or in the
organizer's hidden evaluation. Historical fixed-scene attachment and L1
floor-contact scores remain explicitly labelled as comparisons in the research
archive and are not used as evidence for this strict release.

## Third-party and licensing

The scored path uses the public official JCIIOT baseline, its bundled MuJoCo /
robosuite interfaces, Python and NumPy. Offline SOP image evidence uses the
public Apache-2.0 Qwen3-VL-2B-Instruct model; its weights are not redistributed
and it is not loaded during scoring. See `THIRD_PARTY_NOTICES.md` for the source
and license ledger.

## Contact for evaluation

The repository root is the reproducibility workspace. Reviewers should begin at
`submission/README.md`, then read this release note, `TECHNICAL_REPORT.md`, and
`docs/06-submission-compliance.md`. The commit and official-baseline hashes in
the generated manifests are the authoritative provenance fields.
