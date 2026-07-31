# Conclusion

The strict physical carry controller plateaued at approximately 0.158 m because
direct base-qpos navigation does not preserve a free object's gripper contact.
The official technical sharing and official source establish that the provided
baseline addresses this simulator boundary with `transport_attachment` after a
successful physical grasp and lift.

The promoted candidate therefore enforces this order:

1. scripted physical approach and bilateral close;
2. measured lift and finite hold-state gate;
3. official attachment capture only after the gate;
4. official collision-aware navigation;
5. official constrained lowering, attachment clear, and gravity release.

Local evidence:

- workflow RED: 7 expected failures before implementation;
- workflow GREEN: 17/17 tests passed;
- owned regression: 357/357 tests passed;
- scored-path audit: 0 hard violations, 28 pre-existing warnings;
- materialization from the pinned official commit succeeded;
- overlay files are byte-identical in the materialized candidate;
- `app.py`, `knowledge/task_config.json`, and the official backend retain their
  locked SHA-256 hashes.

The official task 1 score remains pending because the current Mac network does
not route to `211.87.224.136:28897`. No score is claimed from local tests.
