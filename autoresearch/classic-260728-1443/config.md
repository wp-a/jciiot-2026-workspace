# L1 Fully Physical Carry Gate

- Mode: classic
- Candidate commit: `98e7e6df2b7352f96d9f16c54e95c564d3b69e8b`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Scene: public fixed L1 / task index 0
- Seed: `20260728`
- Entrypoint: `scripts/run_official_experiment.py`, followed by unmodified
  `app.py` on port 8502 only after the diagnostic gate passes
- Iteration limit: 10
- Remote GPU: a free L40S device selected at run time
- Hard constraints: no transport attachment, no task-object qpos write, no
  protected-source modification, bilateral grasp during descent and carry

## Ordered Metric

1. Physical validity: bilateral grasp and lift, controller-stepped carrying,
   measured support before release, and no unsupported object motion.
2. Safety: zero official judge-collision frames.
3. Official public-scene score, maximum 10 for L1.
4. Elapsed time.

An iteration is kept only if it improves this ordering without regressing a
higher-priority metric. No attachment fallback is permitted.

## Verification

- Candidate source audit: hard violation count must be zero.
- Protected source hashes must match before and after the run.
- Original scored trajectory must contain physical transport and placement
  events and must pass contact/object-height checks.
- Birdview and `robot0_robotview` replays must show continuous physical
  support; no transformed trajectory may be used for this validity decision.
- The unmodified scorer result must be reproducible from the saved trajectory.

## Final Status

The ten-iteration budget was exhausted without passing L1. See `results.tsv`
and `conclusion.md`. No candidate from this loop may replace the current 8502
service or be reported as a valid score.
