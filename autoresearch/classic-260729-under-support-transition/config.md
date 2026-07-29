# L1 Sequential Under-Support Transition

## Objective

Establish measured non-finger support on one side of the L1 container while the
opposite gripper preserves the verified center grasp. This is an intermediate
physical gate, not an official score claim.

## Metric

Primary: `support_transition.success`.

Required evidence:

- verified center grasp and 20-step hold before transition;
- clearance lift, lower, and inset stages complete;
- moving arm has hand, wrist, or distal-arm object contact;
- stationary arm retains official grasp contact;
- object remains at least 0.10 m above its table reference;
- judge collision frames, object-pose writes, and attachment calls remain zero.

## Environment Invariants

- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
- Pinned interpreter: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`.
- Experimental candidate: `/home/user/jciiot-2026/candidates/robust-l1-inchworm-eb48310`.
- External runner: `/home/user/jciiot-2026/tools/under-support-3260782/run_l1_cradle_gate.py`.
- Result root: `/home/user/jciiot-2026/results/l1-under-support-transition-20260729`.
- The 8502 service PID was `1769287` before synchronization and runs from
  `/home/user/jciiot-2026/candidates/robust-h1-758e8b1/JCIIOT`; it must not be
  restarted or modified by this loop.

Only `src/robot_agent/skills/competition_grasp.py` and
`src/robot_agent/skills/competition_transport.py` are synchronized into the
experimental candidate. The runner is external instrumentation and is not part
of the competition submission.

## Initial Probe

- moving arm: right;
- clearance lift: 0.08 m;
- descent: 0.12 m;
- inward inset: 0.04 m;
- seed: 0;
- stop after the first-arm transition.

If the run fails, change one parameter per iteration in this order: descent,
inset, moving-arm order, then clearance lift. Retain every compact result and do
not promote a transition that only shows finger contact.

## Revision After Iteration 1

The first valid run showed that opening the moving gripper immediately removes
the bilateral load path: the stationary grasp failed after four lowering steps.
Iteration 2 changes only the moving gripper command from open to closed during
descent and inset. All geometry and safety parameters remain fixed.
