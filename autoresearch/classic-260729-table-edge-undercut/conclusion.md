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
