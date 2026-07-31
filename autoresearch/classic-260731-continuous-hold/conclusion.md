# Conclusion

The visible tote separation was caused by two official transport mechanisms
being active at once. The gated relative attachment kept the tote at the
measured gripper offset, while the legacy backend held-crate callback rewrote
the tote to the mobile-base center before trajectory recording.

The accepted candidate keeps the legacy handle disabled throughout navigation,
enables it only at the constrained-lowering boundary, and chooses an output
base goal that preserves the measured post-grasp tote offset.

Official task-1 evidence for commit `94db515`:

- score: `10/10`;
- collision frames: `0`;
- verified bilateral physical grasp and lift: `1`;
- final target distance: `0.114646 m`;
- elapsed time: `65.804 s`;
- held trajectory frames: `709`;
- minimum tote-to-base distance while held: `0.950637 m`;
- held frames below `0.80 m`: `0`;
- base-relative tote XY range: `0.000602 m, 0.019632 m`;
- first-person replay: all sampled pickup, travel, turn, and placement frames
  show the tote between both grippers.

This result must not be described as a purely friction-driven 15 m carry. The
initial bilateral grasp and lift are physical, while long-distance transport
uses the official post-grasp attachment helper required by the baseline's
direct-base navigation model. The correction removes the unintended legacy
teleport and maintains continuous gripper-relative presentation; it does not
replace the official attachment with a learned or purely dynamic carry.
