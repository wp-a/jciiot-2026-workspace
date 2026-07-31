# Official attachment gate iteration

- Date: 2026-07-31
- Mode: classic
- Objective: task 1 official score 10/10 with zero collision
- Candidate route: physical bilateral grasp and lift, gated official transport
  attachment, official constrained lowering, gravity release
- Pinned official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Local verification:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=submission/JCIIOT/src /opt/anaconda3/bin/python -m pytest -q tests`
- Compliance verification:
  `PYTHONPATH=submission/JCIIOT/src /opt/anaconda3/bin/python scripts/audit_scored_path.py --root submission --output artifacts/scored-path-audit-official-attachment.json`
- Runtime metric: unmodified official `app.py` task 1 score and collision flag

The local gate can promote a candidate to server evaluation, but cannot prove
the competition score.
