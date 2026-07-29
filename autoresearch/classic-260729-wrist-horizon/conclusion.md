# Conclusion

No extended-horizon OSC variant passed the 5-degree wrist-alignment gate. All
three runs had zero judge collisions and stayed below the 0.03 m position-drift
limit. None reached center regrasp or physical lift, so all were discarded.

## Evidence

- Extending the original component-clipped controller to 2600 steps reached
  11.71 and 18.07 degrees with 9.75 mm maximum drift.
- L2-norm action limiting preserved the commanded rotation axis and improved
  the same 2600-step result to 9.68 and 15.22 degrees with 6.87 mm drift.
- A per-arm 0.005 fine limit below 15 degrees did not converge. After 5000
  steps it ended at 9.76 and 15.51 degrees with 7.10 mm drift.
- The final trajectory placed both `arm_*_1_joint` values at approximately
  1.5708 rad. Official Tiago XML defines their upper limit as
  1.57079632679 rad. The right `arm_*_6_joint` also reached its lower limit.

The current high-clearance pose is therefore joint-limit constrained. Further
OSC gain and timeout sweeps are not supported by the evidence. The next route
must change the collision-free robot joint seed before resuming OSC alignment.
No object pose writes or transport attachments were used.

## Artifacts

- Remote: `/home/user/jciiot-2026/results/l1-wrist-horizon-20260729`
- Local: `/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-wrist-horizon-20260729`
- `iter01-a002-s2600.json`: `ae2aa39a8723eed3bc5591e39cea8f47cc58449fddf67428ba8d41658f56cbcb`
- `iter02-norm-a002-s2600.json`: `89e0e06f0d1b80a9c6a46c611120d430b8827b9a5e3ff69e9634b257bf7641cc`
- `iter03-scheduled-s5000.json`: `e6745f3c28cc1a5ee7a2371291a064b102bd0504865943c053f6ac4b9d6f6e92`
