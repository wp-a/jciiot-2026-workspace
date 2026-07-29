# L1 Sequential Under-Support Conclusion

## Status

The post-grasp sequential transition did not produce under-support contact in
four valid pinned-simulator trials. No under-support transition or official
score is claimed.

Keeping the moving gripper closed preserved contact longer than opening it, but
the closed fingertips dragged the box downward with the end effector. Combining
the 0.12 m descent and 0.04 m inset into one Cartesian target extended the
stationary-contact window from 38 to 42 steps; it still ended with right-side
finger contact only and no palm, wrist, or distal-arm support. The run had zero
judge collisions, object-pose writes, and attachment calls.

The important geometric observation is that the L1 container already overhangs
the `input_5` table edge by approximately 0.132 m along world `+y`. The next
route therefore starts with the object resting on the table, keeps the gripper
open, descends outside the table edge, moves under the exposed bottom strip, and
raises into measured bottom contact. This directly tests non-prehensile lifting
without first creating an unstable pinch-to-support transition.
