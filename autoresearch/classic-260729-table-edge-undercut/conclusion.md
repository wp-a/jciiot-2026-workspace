# L1 Table-Edge Undercut Conclusion

## Status

Experiment in progress. No open-gripper support or official score is claimed.

Iteration 0 stopped safely in the first clearance stage. The right end effector
moved from `z=1.161` to `z=1.324` without object contact or judge collision but
could not reach the conservative `z=1.400` target. The object did not move, and
the run had zero pose writes and attachment calls.

The next single-variable trial lowers only the above-container clearance from
`0.15 m` to `0.07 m`, producing a target near the measured reachable height.
The table edge, outside clearance, undercut depth, lower target, and raise target
remain frozen.

Iteration 1 passed the lowered clearance stage and reached the requested outside
`y` and `z` without contact. Its world `x` plateaued at `7.483` for a target of
`7.259`, which is also outside the container's maximum `x` support footprint of
approximately `7.359`. The next trial adds only the previously validated
`0.10 m` physical base advance before running the same open-gripper targets.

The first execution with base advance was invalid because the clearance target
had been captured before the base moved and pulled the arm back to its old world
position. A regression test now requires target capture after the advance; the
invalid JSON is archived but is not counted as a physical parameter result.

Corrected iteration 2 advanced the base by `0.100000 m`, passed clearance, and
improved the outside-stage `x` plateau from `7.483` to `7.372`. It still stopped
`0.113 m` short of the requested `x=7.259`, with zero contact or collision. The
response was approximately one-for-one with base advance, so the next trial
changes only base advance from `0.10 m` to `0.20 m`.

Iteration 3 was stopped at `0.150 m` of the requested `0.20 m` advance when the
open left fingers first touched the container. There was no judge collision or
meaningful object movement. Rather than advancing into the object, the next
trial restores the safe `0.10 m` base advance and changes only the target point
from object-local `x=+0.20 m` to `x=+0.31 m`. The hand collision width still
overlaps the bottom footprint while making the target reachable.

Iteration 4 passed base advance, clearance, and the reachable outside target
with zero contact or collision. The descent then saturated at `z=1.173`, about
`0.173 m` above the container bottom. The recorded torso joint remained fixed at
its `0.35 m` upper limit. The next trial changes only the torso hold target after
the outside stage to `0.15 m`; all world-space hand targets remain unchanged.

Iteration 5 reached torso `0.150001 m` and right EEF `z=0.972`, already about
`0.028 m` below the collision bottom, without contact or collision. It missed
the unnecessarily deep `z=0.950` target by `0.022 m`. The next trial changes
only the below-bottom clearance from `0.05 m` to `0.03 m`, matching the measured
safe reach while preserving real geometric clearance.
