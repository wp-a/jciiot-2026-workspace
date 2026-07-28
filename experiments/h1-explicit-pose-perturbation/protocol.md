# H1 Protocol: Explicit Pose Perturbation

Date locked: 2026-07-28 (Asia/Shanghai)

Type: confirmatory baseline measurement.

## Hypothesis

Explicit object and base pose perturbations will reveal failures that were not
visible in the previous deterministic multi-seed batch. The earlier L2-L5
80-run result repeated identical trajectory lengths and final geometry, so it
cannot be used as the H1 baseline.

## Frozen code and environment

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Research branch protocol commit parent: `349036c`.
- Submission overlay: unchanged during this experiment.
- Execution entry: official `RobotAgent`, `--execution-mode agent`.
- Attempts: one; no retry, filtering, or candidate modification.
- Environment: existing pinned Python 3.11 / MuJoCo 3.9 / robosuite 1.5.2
  server environment.
- Scorer: unmodified official `app.py::_score_steps`.

## Runs

Sanity runs:

| Run | Level | Tier | Seed | Object | Required local public score |
|---|---|---|---:|---|---:|
| sanity-l1 | L1 | nominal | 20260727 | `line_5_container_h01_near` | 10/10 |
| sanity-l5 | L5 | nominal | 20260727 | `white_tote_b01_left_center` | 30/30 |

Confirmatory H1 runs:

| Run | Level | Tier | Seed | Object |
|---|---|---|---:|---|
| h1-01 | L1 | small | 20260728 | `line_5_container_h01_near` |
| h1-02 | L1 | small | 20260729 | `line_5_container_h01_near` |
| h1-03 | L1 | small | 20260730 | `line_5_container_h01_near` |
| h1-04 | L1 | small | 20260731 | `line_5_container_h01_near` |
| h1-05 | L1 | small | 20260732 | `line_5_container_h01_near` |

The locked small tier is object XY +/- 2 cm, object yaw +/- 5 degrees,
base XY +/- 1 cm, base yaw +/- 2 degrees, and nominal dynamics.

## Metrics

Primary metric: L1 full-score rate across the five H1 runs.

Secondary metrics:

- successful bilateral grasp and physical lift event rate;
- collision-run and collision-frame rates;
- maximum final target distance;
- first workflow failure stage;
- elapsed time;
- Wilson 95% interval for full-score and collision-free rates.

Every manifest and trajectory is retained. A failed process counts as a failed
run. Results are not repeated to replace failures.

## Sanity and validity gates

1. Both nominal runs must reproduce the corresponding fixed-public-scene local
   score and zero-collision requirement. If not, the experiment is invalid due
   to environment, materialization, or entrypoint drift.
2. Each H1 manifest must contain the requested sample and a
   `perturbation_application.valid=true` audit.
3. Each H1 run must have at least one nonzero measured object or base pose
   change, and the five generator digests must be distinct.
4. Measured XY error must be at most 1 mm and measured yaw error at most
   0.1 degrees. The runner aborts before task execution if this gate fails.
5. The materialized candidate and locked official files are hashed before the
   first run and after the last run.

## Interpretation rule

- A 5/5 result is preliminary small-perturbation evidence only; its Wilson
  interval remains wide and it does not prove medium/stress robustness.
- Any failure supports H1 and determines the next engineering target from its
  first failed stage.
- An invalid sanity or perturbation audit produces no performance conclusion.
- Scores remain local public-scorer results, not BienData or organizer scores.
