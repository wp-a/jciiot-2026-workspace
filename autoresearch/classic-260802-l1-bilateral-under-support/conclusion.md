# L1 Bilateral Under-Bottom Support Conclusion

Date: 2026-08-02 (Asia/Shanghai)

## Decision

Discard the simultaneous closed-gripper descent-and-inset transition. Do not
merge its experimental controller into the stable competition candidate and do
not use its trajectory as successful transport training data.

## Valid Physical Result

The first two invocations were setup diagnostics: the later center-regrasp
runner failed `close_center_grasp` with no object contact, so neither reached
the registered support change. The eligible invocation instead reused the
current candidate's repeatedly verified raw scripted grasp.

That invocation established:

- real raw grasp and bilateral contact: true;
- raw grasp lift: approximately `0.2045 m`;
- 20-step physical hold: inherited from the official container grasp profile;
- additional closed-gripper clearance raise: successful;
- object rise during that clearance stage: `0.121405 m`;
- collision frames: `0`;
- attachment calls: `0`;
- object-pose writes: `0`;
- infrastructure error: none.

The simultaneous transition then failed after 240 control steps. Both end
effectors nearly reached the requested target: final target errors were
`0.011457 m` on the right and `0.013259 m` on the left. Nevertheless, the tote
moved downward by `0.219540 m` during the transition, while the end effectors
moved downward by about `0.163 m`. The final contacts remained fingerpad and
fingertip contacts on both arms; no palm, wrist, hand, or distal-arm support was
observed. The final lift over the original table reference was only
`0.106774 m`, so the proof raise was correctly not attempted.

## Root Cause

This is not an OSC reachability failure. With both grippers closed, the
vertical side-wall contacts drag the tote down with the fingers. The hand
collision centers remain roughly `0.28 m` above the tote bottom, so a `0.16 m`
vertical end-effector descent plus lateral midpoint inset cannot transform the
same-side pinch into bottom support. More steps or a looser Cartesian target
tolerance would only complete the wrong geometry.

The official tote model also has solid collision walls and no usable handle
hole. A valid next transition must change the contact topology: preserve the
load with one arm while the other releases, moves down and forward beneath the
bottom, and proves contact specifically against the tote bottom geom before
raising. It must use the successful raw grasp, not the regressed center-regrasp
path, and must stop immediately on stationary-contact or height loss.

## Evidence

- valid result SHA-256:
  `1d278ac9b22e33f66b157511027504982da623c66325dd3ac9eed6a3d322b139`;
- valid trajectory SHA-256:
  `e73a3b53d13ab0d93bf18715721bc151a3495872144430afa862baeee774b75a`;
- independent audit ledger SHA-256:
  `7cd44f2c44881442357b7df68b5ada5bd39bf9c187bd847116d096e5c325ebfd`;
- independent audit TSV SHA-256:
  `bcfa46a37a1a7c840d9f702420847c7f1890c59a94588a00e58b79cc0aa2da1a`;
- result classification: `rejected`;
- full trajectory frames: `1042`;
- trajectory events: physical `grasp_start` and successful `grasp_end` only.

All local artifacts are stored in `artifacts/` beside this report. The live
8502/8503 services and the frozen candidate were not modified.
