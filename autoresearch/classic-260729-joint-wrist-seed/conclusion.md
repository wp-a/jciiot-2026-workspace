# Conclusion

The simultaneous 12-joint seed is viable as an endpoint optimizer, but none of
the four bounded variants passed every joint-seed gate. All four runs were
valid official-runtime experiments with no infrastructure error, no trajectory
collision, no object-pose write, and no attachment call. Every failure rolled
all robot arm joints back and synchronized the controllers.

## Findings

- The default 0.03 rad interior margin converged in 24 evaluations to 7.52 and
  11.10 degrees with only 4.27 mm endpoint position error. The left arm missed
  the 10-degree seed gate.
- Reducing only the interior margin to 0.01 rad improved the errors to 7.20 and
  10.83 degrees. The proposal again reached an interior bound, so margin alone
  was not the dominant tradeoff.
- Increasing only the position residual scale to 0.015 m used 9.13 mm of the
  15 mm endpoint budget and improved the errors to 6.90 and 10.35 degrees.
- A 0.020 m position scale produced the first endpoint with both angular errors
  inside the seed gate: 6.51 and 9.74 degrees. It was correctly rejected because
  the endpoint position error was 15.265 mm, 0.265 mm above the hard gate.

No variant reached joint interpolation, OSC correction, center regrasp, or
physical lift. These results are not a competition score.

## Decision

Discard all four variants as complete routes. Retain the implementation and
the monotonic position/orientation tradeoff as evidence. The next bounded loop
will refine only the position residual scale between 0.015 and 0.020 m, starting
at 0.0185 m. The 10-degree angular and 15 mm endpoint gates remain unchanged.

## Artifacts

- Remote root: `/home/user/jciiot-2026/results/l1-joint-wrist-seed-20260729`
- Local compact records: `/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-joint-wrist-seed-20260729`
- `iter01-default.json`: `063a7e467f5297bc14a214b33d1274440f524b47b2f63f44b9c78c81eaa0a610`
- `iter02-margin001.json`: `20d5bedaddee7d966d1d00e1d91b28ac3826b17fdc7ff51e8bb87dca4cf5f91c`
- `iter03-pos015.json`: `e2dff7c4f793acec9a131181b769a7941a175c4c9167bbc8b8c7af3332efa401`
- `iter04-pos020.json`: `9593ee49833072aea2904b161db3cf6c3459e917c6ff5fd0b365da5a0e7a2720`

The original remote trajectories are retained with their SHA-256 values in the
remote result root. They are not duplicated in Git.
