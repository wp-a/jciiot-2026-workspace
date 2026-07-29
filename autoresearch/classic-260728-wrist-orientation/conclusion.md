# Conclusion

No variant passed the high-clearance wrist-orientation gate, so no result was
promoted to the submission or the current 8502 service. All four runs used the
official pinned MuJoCo environment, public L1 scene, seed 0, unchanged 5-degree
and 0.03 m hard thresholds, and produced zero judge-collision frames.

## Findings

- The initial 0.20 action reduced both angular errors monotonically, but the
  position gate stopped the run after 9 steps at 0.030440 m drift.
- The first position measurement mixed the fingerpad midpoint with the OSC
  grip-site orientation. Official source inspection showed that OSC controls
  `grip_site`, so commit `e0a785e` made the position and orientation boundary
  consistent. A same-parameter repeat still reached 0.030291 m drift after 10
  steps, disproving that mismatch as the primary cause.
- The official Robotiq 85 XML confirms that grip-site x is the closure axis:
  the fingers are separated along gripper-base y and the fixed eef transform
  maps that direction to grip-site x.
- Reducing the orientation limit from 0.20 to 0.08 improved rotation per unit
  of drift, reaching 37.89 and 41.58 degrees before the 0.03 m gate stopped it.
- At 0.02, maximum position drift stayed at 0.006204 m and collisions stayed
  at zero for 1000 steps. Errors reached 29.66 and 33.34 degrees. The final 30
  steps still improved by about 0.02 degree per step, so this is a slow
  convergence timeout, not evidence of a hard reachability limit.

## Decision

Discard all four variants. There is still no physical center regrasp, lift, or
valid score claim. Do not change the official submission or the live 8502
candidate based on these runs.

The next bounded experiment should retain `orientation_max_action=0.02`, all
hard gates, and the grip-site boundary, while extending only the orientation
timeout to roughly 2600 steps. If it reaches five stable steps safely, evaluate
the existing retreat, center approach, bilateral contact, and physical lift.
If it does not, stop per-step OSC residual control and investigate a staged
joint-space wrist seed within the allowed skill/workflow boundary.

## Evidence

- Remote: `/home/user/jciiot-2026/results/l1-wrist-orientation-20260728`
- Local: `/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-wrist-orientation-20260728`
- Research script commit: `e0a785e`
- Result table: `results.tsv`

Result JSON SHA-256 values:

- `iter01-a020.json`: `7a9c9e4a3ceed18a1b17e2faec52c662333f88b390ba5a25664f2425b68e0fd8`
- `iter02-gripsite-a020.json`: `11d13b4ce5387b0537a30ec1d5417faffafc3c56e6da20aa6bcae20750703e0a`
- `iter03-gripsite-a008.json`: `5c348fbca9ff669793b2c89322e2669562f2d1fbc49114a2a1a58b774f2f8a7b`
- `iter04-gripsite-a002.json`: `0a093cc11bb61d6abe20978dcf9bdcfd0176773ebc39e26c4fe0b7a9063eb17b`
