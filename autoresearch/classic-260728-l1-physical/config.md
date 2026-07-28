# L1 Physical Support and Regrasp Diagnostics

- Mode: classic, sequential evidence-driven experiments
- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `/home/user/jciiot-2026/candidates/robust-l1-cradle-20260728a`
- Runtime: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`
- Scene: public fixed L1 / task index 0
- Seed: `0`
- GPU isolation: `CUDA_VISIBLE_DEVICES=2`
- Research entrypoint: `scripts/run_l1_cradle_gate.py`
- Remote record root: `/home/user/jciiot-2026/results/l1-cradle-20260728a/`
- Local evidence: record JSON only; full trajectories remain at the remote root
- Current 8502 service: unchanged because no full gate passed twice

## Submission Boundary

The experiments use only simulator actions and physical contacts. The records
require zero task-object pose writes and zero transport attachment calls. No
file under official `core/`, `environments/`, `app.py`, or
`knowledge/task_config.json` was modified.

## Hard Gates

Cradle transfer:

- bilateral physical grasp;
- at least 0.13 m measured lift;
- at least 20 consecutive non-finger bilateral support-contact steps;
- at least 0.50 m base translation;
- zero attachment calls, object-pose writes, and collision frames;
- no drop or infrastructure error.

Physical push:

- at least 20 consecutive object-contact steps;
- at least 0.50 m object translation;
- at least 0.30 m base translation;
- zero attachment calls, object-pose writes, and collision frames;
- no infrastructure error.

The push base threshold is 0.30 m because a direct 0.3315 m advance produced
an official collision between the Tiago torso proxy and the production-line
proxy. The object threshold remains 0.50 m.

## Record Validity

`center-pinch-sync-z.json` and `physical-push-500.json` contain infrastructure
errors and are retained only as invalid-debug records. All other JSON files are
valid diagnostic outcomes, not successful gate evidence.
