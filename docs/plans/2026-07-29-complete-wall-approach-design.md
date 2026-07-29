# Complete Wall Approach Design

## Evidence

With the corrected squeeze controller and a 0.040 m squeeze target, the two end
effector centers reached the long-wall planes within about 1--2 mm. Both still
contacted only `left_inner_knuckle` geoms and closing lost all contact.

The preceding `approach_center_walls` stage requested an end-effector height of
`table_object_z + 0.115`, but passed `stop_bilateral_contact_steps=1`. It stopped
on the first bilateral knuckle contact at approximately 1.276--1.282 m. The
container walls top out near 1.250 m, so the grippers remained 26--32 mm above
the rim and never inserted their fingerpads around the wall.

## Selected change

Retain the 0.040 m wall-centered squeeze and remove only the one-frame contact
completion condition from `approach_center_walls`. The open grippers continue
toward the existing bounded height target, with the existing 220-step cap and
per-step official judge-collision check. Object contact remains diagnostic and
does not bypass collision handling.

The alternative of lowering the target further is deferred because the current
target has not yet been executed. Closing while descending is also deferred
because it changes two control dimensions at once.

## Acceptance

The approach must either reach its existing target tolerance or report an
official collision failure; first contact alone is not success. Downstream
requirements remain unchanged: bilateral official grasp for three consecutive
frames, at least 0.13 m measured lift, at least 20 closed-gripper hold steps,
zero official collision, zero object writes, and zero attachments.
