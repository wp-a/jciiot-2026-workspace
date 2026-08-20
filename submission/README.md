# JCIIOT 2026 Competition Submission Overlay

This directory is the exact code overlay for the official JCIIOT 2026
baseline. Every file is inside a modification root allowed by the contestant
manual. The release policy is strict physical carry: the official entrypoint
rejects `attachment` and `l1_floor_push`, and a failed physical hold stops the
workflow instead of silently falling back to a shortcut.

This repository contains reproducible code and evidence, not an organizer
score claim. Historical fixed-scene attachment and floor-contact results remain
in the research archive as labelled comparisons; they are not the
strict-physical release result.

## Locked baseline

- Repository: `https://github.com/JCIIOT2026/JCIIOT2026.git`
- Commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Runtime: Linux, Python 3.11, MuJoCo 3.9.0, robosuite 1.5.2
- Scored-path checkpoints or API calls: none
- Offline SOP image model: public, hash-locked, not bundled and never loaded by the scored path

The overlay does not modify `app.py`, `task_subprocess_runner.py`,
`src/robot_agent/core/`, `src/robot_agent/environments/` or
`knowledge/task_config.json`.

## Overlay contents

```text
JCIIOT/src/robot_agent/skills/competition_grasp.py
JCIIOT/src/robot_agent/skills/competition_navigation.py
JCIIOT/src/robot_agent/skills/competition_task.py
JCIIOT/src/robot_agent/skills/competition_transport.py
JCIIOT/src/robot_agent/skills/sop_generator.py
JCIIOT/src/robot_agent/skills/library.py
JCIIOT/src/robot_agent/workflows/competition_flow.py
JCIIOT/knowledge/generated_sop_l1.md ... generated_sop_l5.md
```

The runtime chain is:

```text
app.py -> task_subprocess_runner.py -> RobotAgent.run()
       -> CompetitionTaskSkill -> CompetitionFlow
       -> bilateral physical grasp/lift -> physical_carry -> physical_place
```

`CompetitionTaskSkill` is registered first in the official skill library. The
scored path does not depend on a live LLM response.

## Materialize the candidate

```bash
git clone https://github.com/JCIIOT2026/JCIIOT2026.git official
git -C official checkout 0dcdddf18a9e694569aa1433cdfc04eb097fed78

bash scripts/materialize_submission.sh \
  --official-root "$PWD/official" \
  --output "$PWD/candidate"
```

The materializer rejects a different upstream commit and every file outside the
allowed overlay boundary. Install dependencies from the official repository;
do not float MuJoCo above the locked `3.9.0` version.

## Run and score one task

Task indices `0..4` correspond to L1..L5. Replace `RELEASE_COMMIT` with the
Git revision of this repository (for example, `git rev-parse HEAD`).

```bash
python scripts/run_official_experiment.py \
  --candidate-root "$PWD/candidate" \
  --expected-official-commit 0dcdddf18a9e694569aa1433cdfc04eb097fed78 \
  --workspace-commit RELEASE_COMMIT \
  --task-index 0 \
  --seed 20260820 \
  --execution-mode agent \
  --trajectory "$PWD/results/trajectory-l1.json" \
  --output "$PWD/results/manifest-l1.json" \
  --required-score 10
```

Repeat with required scores `15`, `20`, `25` and `30` for task indices `1..4`.
The command uses the unmodified official scorer. A zero exit code requires the
requested score, all required physical `grasp_end` events, zero collision
frames, a target distance below `0.8 m`, and a successful workflow result.

## Tests and audits

```bash
PYTHONDONTWRITEBYTECODE=1 python3.11 -m pytest -q tests
bash scripts/check_workspace.sh --require-private-remote
git diff --check
```

The physical-carry HDF5 gate is run separately after a task-native dataset is
materialized:

```bash
python scripts/audit_physical_carry_hdf5.py DATASET.hdf5 \
  --output artifacts/data-audit/MANIFEST.json
```

It rejects wrong action/state widths, non-finite values, split leakage,
missing event order, collisions, attachment calls and object-pose writes.
Low-lift or low-translation trajectories are retained only as recovery data.

## SOP knowledge

The five `generated_sop_l*.md` files were generated from the official DOCX
inputs. The generator does not read the official hand-written SOP Markdown. It
cross-checks Prompt text, Erratum, immutable task configuration and semantic
maps. Image evidence from the public Qwen3-VL-2B-Instruct model is advisory and
cannot generate robot actions. Do not place model weights or API keys in this
repository.

## Evidence boundary

Local unit tests and static boundary audits are reproducible in this workspace.
At the time of this release, no new 8502 server run is presented as proof of
five-level strict-physical success, and no BienData or organizer score is
claimed. The corresponding limitations and historical baselines are recorded
in `SUBMISSION_RELEASE_20260820.md` and `TECHNICAL_REPORT.md`.
