# Competition Submission Overlay

This directory contains only files that the official JCIIOT rules permit a
team to modify. Apply it to the official commit locked in
`config/upstream-lock.json` with `scripts/materialize_submission.sh`.

The overlay deliberately excludes `app.py`, `task_config.json`, `core/`, and
`environments/`. Generated trajectories, datasets, checkpoints, and secrets
must remain outside Git.

