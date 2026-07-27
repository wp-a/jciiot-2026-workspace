# JCIIOT 2026 Competition Submission Overlay

This directory is the minimal code overlay for the official JCIIOT 2026
baseline. It contains only files under the modification roots allowed by the
official contestant manual.

## Locked baseline

- Repository: `https://github.com/JCIIOT2026/JCIIOT2026.git`
- Commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Recommended runtime: Linux, Python 3.11, MuJoCo 3.9.0
- Scored-path external checkpoints: none
- External inference services during scored execution: none
- Offline SOP image model: public, hash-locked, not bundled and not scored

The scored path uses the official environment, trajectory recorder and scorer.
It does not modify `app.py`, `task_subprocess_runner.py`, `core/`,
`environments/` or `knowledge/task_config.json`.

## Overlay contents

```text
JCIIOT/src/robot_agent/skills/competition_grasp.py
JCIIOT/src/robot_agent/skills/competition_navigation.py
JCIIOT/src/robot_agent/skills/competition_task.py
JCIIOT/src/robot_agent/skills/sop_generator.py
JCIIOT/src/robot_agent/skills/library.py
JCIIOT/src/robot_agent/workflows/competition_flow.py
JCIIOT/knowledge/generated_sop_l1.md
JCIIOT/knowledge/generated_sop_l2.md
JCIIOT/knowledge/generated_sop_l3.md
JCIIOT/knowledge/generated_sop_l4.md
JCIIOT/knowledge/generated_sop_l5.md
```

`CompetitionTaskSkill` is registered first in the official skill library. The
official `GATE_PLANNER=false` feature gate is used for scored execution, so the
selected task runs through a deterministic, physically verified workflow
instead of depending on a live LLM response.

## Materialize a candidate

From the code-and-report package root:

```bash
git clone https://github.com/JCIIOT2026/JCIIOT2026.git official
git -C official checkout 0dcdddf18a9e694569aa1433cdfc04eb097fed78

bash scripts/materialize_submission.sh \
  --official-root "$PWD/official" \
  --output "$PWD/candidate"
```

The materializer rejects a different official commit and rejects files outside
the allowed overlay boundary. Install the official repository exactly as
described in its README and root `requirements.txt`; do not allow `mujoco` to
float above the official `3.9.0` pin.

## Run one headless scored task

The following command exercises the official `RobotAgent.run()` entry path,
records the trajectory and invokes the unmodified official scorer. Task indices
`0..4` correspond to L1..L5.

```bash
python scripts/run_official_experiment.py \
  --candidate-root "$PWD/candidate" \
  --expected-official-commit 0dcdddf18a9e694569aa1433cdfc04eb097fed78 \
  --workspace-commit 260839a7915c8327fcd2a2611b16053c582d5dc4 \
  --task-index 0 \
  --seed 20260727 \
  --execution-mode agent \
  --trajectory "$PWD/results/trajectory-l1.json" \
  --output "$PWD/results/manifest-l1.json" \
  --required-score 10
```

Repeat with required scores `15`, `20`, `25` and `30` for task indices
`1`, `2`, `3` and `4`. A zero exit code requires full score, all required
successful grasp events, zero collision frames, target distance below 0.8 m
and a successful workflow result.

On a graphical competition workstation, starting the official application and
pressing **Execute** follows:

```text
app.py -> task_subprocess_runner.py -> RobotAgent.run()
       -> CompetitionTaskSkill -> CompetitionFlow
```

The submitted validation archive is separate from this overlay. Its root must
contain exactly the five official scene-named JSON trajectory files.

## Regenerate SOP knowledge offline

The five submitted `generated_sop_l*.md` files were generated from the original
official DOCX files. The generator never reads the official hand-written
`knowledge/sop*.md` files. It deterministically extracts DOCX text, parses the
task-specific Prompt, checks the official Erratum, task configuration and
semantic map, and uses a public VLM only for hash-addressed image evidence.

Prepare the public model version in `config/sop-vlm-lock.json`, then run:

```bash
python candidate/JCIIOT/src/robot_agent/skills/sop_generator.py \
  --app-root candidate/JCIIOT \
  --output-dir results/sop_generated \
  --use-vision \
  --require-vision \
  --local-vlm-model /path/to/Qwen3-VL-2B-Instruct \
  --vision-model-id 'Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0' \
  --local-vlm-device cuda:0
```

An OpenAI-compatible or Ollama vision endpoint can be used instead through
`--vlm-base-url`, `--vlm-model` and `--vlm-api-type`. Do not put API keys in the
repository; the key is read from the environment variable named by
`--vlm-api-key-env`.

The VLM evidence is advisory and cannot generate robot actions. The scored
workflow does not load the VLM or its weights. Full generated Markdown,
provenance, checksums and the fixed model runtime are documented in
`sop_generated/`, `config/sop-vlm-lock.json` and the SOP experiment report.

## Tests

The Python unit and boundary tests are self-contained and do not require the
simulator or the 1.7 GB official checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.13 -m pytest -q tests
```

The complete workspace audit additionally requires a Git checkout and the
locked official repository at `vendor/JCIIOT2026`; reference checkouts may be
absent:

```bash
bash scripts/check_workspace.sh
git diff --check
```

The final Linux validation used Python 3.11.15, MuJoCo 3.9.0 and robosuite
1.5.2. See `TECHNICAL_REPORT.md` and the experiment reports included in the
code-and-report package for results, evidence boundaries and limitations.
