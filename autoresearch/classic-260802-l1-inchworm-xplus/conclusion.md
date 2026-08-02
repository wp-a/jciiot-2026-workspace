# Conclusion

Decision: discard the current four-step center-directed reseat configuration.

The clean replacement run used the frozen candidate and raw physical grasp
through the attachment-free guard. It recorded zero collision frames, zero
attachment activations, both attachment-active flags false, and zero object
pose writes. The initial grasp and lift were physical and bilateral.

The controller completed one full stroke-reset-reseat cycle. In cycle 2 it
completed a second arm stroke and all 30 compensated base-reset steps with
small planar gripper drift (`0.004668 m` maximum). The first following reseat
step then caused both contacts to be lost. Effective object progress was only
`0.128528 m`; after failure the object settled to `z=1.178617 m`. This is not a
task success, not a scored result, and not eligible training data.

The failure is more specific than the previous longitudinal-slip diagnosis:
`bilateral_planar_reseat_deltas` points each gripper toward the object center in
the full XY plane. After a +X stroke/reset, that diagonal correction can add a
longitudinal component to a side-wall grasp. The next falsifiable test disables
only the four reseat steps while keeping grasp, stroke, reset, compensation,
route, seed, and integrity guards fixed.

The earlier interface-incompatible launch is retained only as negative audit
evidence. It had attachment active before and after transport and is excluded
from every physical comparison.
