# Autoresearch Configuration

- Mode: classic
- Objective: produce a compliant L1 trajectory with the deterministic hybrid pipeline
- Primary metric: official L1 score
- Secondary metrics: successful physical grasp, verified lift, collision frames, final target distance, elapsed time
- Verify: official unmodified `app.py::_score_steps(0)` plus trajectory event/frame audit
- Acceptance: 10/10, matching `grasp_end success=true`, zero collisions, five repeated successes
- Iteration limit: 25
- Keep rule: retain only a candidate that improves the lexicographic tuple `(score, grasp_and_lift, -collisions, -target_distance, -elapsed_time)`

