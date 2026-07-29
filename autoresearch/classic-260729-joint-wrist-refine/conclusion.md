# Conclusion

The 24-node continuation implementation solved the invalid straight-line path
and produced the first accepted robot-only joint seed. It is retained as the
new research incumbent. It did not pass OSC alignment, physical regrasp, or the
competition score gate.

## Evidence

- The direct 0.0185 m endpoint met the angular and 15 mm endpoint gates but its
  straight joint interpolation exceeded 30 mm grip-site drift at waypoint 7.
- The first 24-node run followed the local IK branch through node 22 with zero
  collision and 8.00 mm position error. It stopped only because the initial
  implementation incorrectly applied the 5-degree OSC gate to every seed node.
- Applying the formal 10-degree seed gate allowed all 24 nodes and all 240
  recorded waypoints to pass. Endpoint and maximum path drift were both 13.324
  mm, angular errors were 6.63 and 9.93 degrees, and collision count was zero.
- The subsequent 2600-step OSC stage remained safe with 6.73 mm additional
  drift and zero collision, but timed out at 5.98 and 9.20 degrees. It did not
  reach center contact or lift.

The accepted joint seed changed only robot joints, never wrote object qpos,
and never used an attachment. This is a valid intermediate physical-control
result, not a score claim.

## Next decision

The final seed and OSC trajectory place the torso at approximately 0.349744 m
against its official 0.35 m upper limit while multiple arm joints also approach
their limits. The next bounded hypothesis is to include the official
`robot0_torso_lift_joint` as a thirteenth IK variable so the solver can trade
torso height against arm posture while holding both grip sites fixed. The same
node, endpoint, path-drift, collision, rollback, and OSC gates remain.

## Artifacts

- Remote root: `/home/user/jciiot-2026/results/l1-joint-wrist-refine-20260729`
- Local compact root: `/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-joint-wrist-refine-20260729`
- `iter01-pos0185.json`: `9ca149494b26fd4f0c82d4eac4427ed574f13228c850346fa4ecb5d1a4fb6f15`
- `iter02-cont24.json`: `c7ae59e164b025544c904be3870b87a25a300ce4ec00d03b36e3303be875857c`
- `iter03-cont24-node10.json`: `fdc3920ddb9eaf969db23a6c1d1c2164b91c3f0c3c03f21cd433068811c89c31`

Original trajectories remain on the server and are not duplicated in Git.
