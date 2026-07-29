# High Precenter Regrasp Design

## Architectural finding

Three experiments have now falsified the low-first opposed-wall sequence:

1. completing the 25 mm squeeze left both grippers outside the wall planes;
2. a 40 mm squeeze centered both end effectors on the wall planes but only
   after their inner knuckles were already resting on the top rim;
3. completing the preceding descent for 220 steps left the right gripper near
   the target height and the left gripper about 17 mm high, with the same inner
   knuckle contacts and no fingerpad grasp.

The controller is trying to descend outside the walls and then move laterally
through the rim. More steps or a lower target would only press the same rigid
geometries together.

## Options considered

1. **High precenter, then descend** (selected): while the open grippers are at
   high clearance, move them laterally to the known wall planes; then descend
   with each wall entering the open jaw, and close only after the descent.
2. Extend the low descent horizon. The right arm already reached the height
   tolerance and both arms retained inner-knuckle contact, so this does not
   address the geometry.
3. Add a spiral or compliant search. This could help later, but it changes more
   control dimensions and is unnecessary before testing the deterministic
   ordering implied by the measured geometry.

## Sequence

After the existing high-clearance wrist alignment, base advance, wall retreat,
and zero center shift:

1. compute the existing 40 mm opposed inward targets and execute
   `squeeze_center_walls` at high clearance with open grippers;
2. compute the existing vertical targets from the new centered poses and
   execute `approach_center_walls` with open grippers;
3. hold the resulting poses while applying the existing gradual close schedule;
4. require three consecutive bilateral official grasp frames before lift;
5. retain the existing lift, hold, collision, object-write, and attachment
   gates unchanged.

Every motion remains controller-stepped and recorded. Any official collision
immediately fails the stage. The current 8502 service and official candidate
remain unchanged during this diagnostic experiment.

## Verification

A regression test must fail on the current low-first order and pass only when
the `squeeze_center_walls` call precedes `approach_center_walls`. Existing tests
continue to prove that close precedes lift, the gripper stays closed during
lift/hold, and no first-contact shortcut remains in either motion stage.
