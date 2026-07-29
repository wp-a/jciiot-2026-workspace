# Conclusion

The torso-redundant seed was safe and functionally correct but did not improve
the ordered task metric enough to retain.

The solver moved the torso from 0.349748 m to 0.300942 m, refreshed the
subsequent torso hold target to the same value, completed all 24 IK nodes and
240 path waypoints, and recorded zero collision. Seed errors were 6.61 and 9.93
degrees. After 2600 OSC steps they were 5.91 and 9.20 degrees. This differs only
marginally from the arm-only 5.98 and 9.20 degree result, while maximum seed
path drift increased from 13.32 to 17.05 mm.

The torso branch is therefore discarded as the current incumbent. The code
remains opt-in and the 8502 candidate remains unchanged. No physical center
contact, lift, or official score was produced.

The next experiment returns to the accepted arm-only continuation seed and
tests whether its sub-10-degree closure-axis pose is already sufficient for a
real center regrasp. This changes only the research-stage orientation tolerance
from 5 to 10 degrees; collision, contact, lift, hold, shortcut, and score gates
remain unchanged.

## Artifact

- Remote: `/home/user/jciiot-2026/results/l1-torso-wrist-seed-20260729/iter01-torso005.json`
- Local: `/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-torso-wrist-seed-20260729/iter01-torso005.json`
- SHA-256: `649e2b89c9a51ec10ac1a8b08834caf20248f492848e774c1892717c460071e4`
