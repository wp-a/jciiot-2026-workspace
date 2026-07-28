# Research Log

Chronological, append-only record of research decisions and evidence.

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-07-28 | bootstrap | Locked official commit `0dcdddf18a9e694569aa1433cdfc04eb097fed78`. Preserved the five-level local fixed-public-scene result as a baseline, not an official score. The L2-L5 80-run batch repeated identical geometry, so explicit perturbation is the first active hypothesis. |
| 2 | 2026-07-28 | bootstrap | Reviewed primary sources for robomimic v0.5, MimicGen, Diffusion Policy, and ACT. Selected a model ladder of BC-RNN, BC-Transformer, then conditional escalation to Diffusion Policy or ACT. General VLA training is outside the first research cycle. |
| 3 | 2026-07-28 | bootstrap | Defined the compliance boundary: no robot/object qpos assignment or transport-attachment relative-state mutation in the final scored path. Current geometry code remains a teacher until the boundary is clean. |
