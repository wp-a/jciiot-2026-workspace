# Autoresearch Configuration

- Mode: classic
- Objective: quantify L2-L5 stability of the fixed geometric candidate before final submission
- Candidate code commit: `05a7b0dc0eefafd7f12f14feed0ffbe3975c6332`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Tasks: L2, L3, L4, L5
- Seeds: `20260727..20260746` inclusive (20 per task, 80 runs total)
- Primary metric: per-level official full-score rate
- Secondary metrics: collision-run rate, verified-grasp rate, target distance, elapsed time, runner-error rate
- Verify: unmodified official `app.py::_score_steps` plus trajectory event/frame audit
- Acceptance per run: official level maximum, required grasp events, zero collision frames, maximum target distance below 0.8 m, successful workflow result
- Stability report: retain all terminal manifests and trajectories; report Wilson 95% intervals without discarding failures
- Iteration policy: do not change the candidate during this baseline batch; classify failures before starting a new candidate batch

