# 2026-07-27 L1 Scripted OSC Baseline

## Objective

Produce a submission-generated L1 trajectory with a physical grasp, zero
collisions, and 10/10 from the unmodified official scorer. This experiment
tests whether public geometric grasp sites and the official Tiago OSC
controllers can remove the missing-BC-checkpoint blocker.

## Candidate

- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Workspace implementation commit: `2e5cce9d3a3bcfebe30bcb3997ca16eacd5796f9`
- Environment: Ubuntu 24.04.3, Python 3.11.15, MuJoCo 3.9.0,
  robosuite 1.5.2, four NVIDIA L40S GPUs
- Grasp method: two-arm geometric OSC controller using official object grasp
  sites; both contacts and the official lift verifier are required
- Workflow: deterministic per-object state machine
- Scoring: unmodified `app.py::_score_steps(0)`
- BC checkpoint: none

Only allowed overlay paths were added under `skills/` and `workflows/`.
`app.py`, `core/`, `environments/`, and `task_config.json` remained unchanged.

## Results

| Seed | Official score | Grasp events | Collision frames | Final distance (m) | Elapsed (s) |
|---:|---:|---:|---:|---:|---:|
| 20260727 | 10/10 | 1 | 0 | 0.164965 | 76.064765 |
| 20260728 | 10/10 | 1 | 0 | 0.164965 | 64.959380 |
| 20260729 | 10/10 | 1 | 0 | 0.164965 | 64.003480 |
| 20260730 | 10/10 | 1 | 0 | 0.164965 | 64.374842 |
| 20260731 | 10/10 | 1 | 0 | 0.164965 | 65.185490 |

Summary: 5/5 full-score runs, 0 collision frames across 7,275 recorded
frames, and 5/5 physically verified grasp-and-lift events. The post-warmup
mean elapsed time was 64.63 seconds.

## Physical Evidence

Every run reported both right and left gripper grasp status as true. The
first run raised the object from z=1.249906 m to z=1.376765 m, an observed
lift of 0.126859 m within the configured 0.02 m tolerance of the 0.15 m
target. Transport attachment was enabled only after this check.

## Artifacts

The ignored local directory `artifacts/remote-l1-scripted-20260727/` contains
all five official trajectories, five manifests, and raw logs. Representative
SHA-256 values:

- `manifest-iter1.json`: `903bba81f978a4cb231fd0a9dda06553df4ec16be9a6c0baf7d0c43938943593`
- `trajectory-iter1.json`: `793471abc70c37b9dfd75e38d49303daf7471aad7b70dd65d42c6624ce466c2b`
- `trajectory-20260731.json`: `5e573aa1a6bed0770b4161b1c9cd4b78a9981eb81528ac15c8d0c7e693f4a87d`

The five manifests were produced before the runner's `started_at` assignment
was corrected, so that field reflects manifest assembly time. Per-run
`elapsed_s`, trajectory frame timestamps, and `finished_at` remain valid. The
runner regression is fixed for subsequent experiments.

## Conclusion and Boundary

L1 meets the development stability gate and no longer depends on the missing
official BC checkpoint. This does not yet prove L2-L5 performance: their
objects, approach poses, reachability, and L5 multi-object interactions still
require separate physical experiments. BC remains an optional fallback if a
scene's grasp sites cannot be reached reliably by the geometric controller.
