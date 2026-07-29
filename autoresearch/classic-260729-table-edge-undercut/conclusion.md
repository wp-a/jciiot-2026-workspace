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
