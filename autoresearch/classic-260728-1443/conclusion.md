# L1 Fully Physical Carry Gate Conclusion

## Status

The ten-iteration L1 gate did not pass. No candidate from this loop is valid
for the unmodified 8502 UI or for a score claim. Every official diagnostic run
scored 0/10, although the initial two-gripper grasp and lift were repeatedly
physical and collision-free.

## Established Evidence

- The locked Tiago wheel-action controller moved only about 5.6 mm in 100
  full-scale actions. Cross-scene action-only navigation is not viable.
- Bounded official direct-base increments move the robot, while the free box
  moves only through MuJoCo gripper contact. This removed the former attachment
  and object-qpos shortcut but exposed the true grasp limitation.
- The L1 box is grasped by two grippers on the same near side wall. Replay
  reconstruction confirmed both fingerpads per arm at transport start.
- During lateral transport the box slips both horizontally and vertically.
  At failure, one left fingerpad loses contact while the grippers have moved up
  by roughly 0.12 m relative to the box.
- Smaller base increments, physics settle steps, arm-led cycles, 10 mm and
  20 mm inward grasp insertion, continuous height feedback, periodic physical
  relift, and pre-relift gripper recentering did not produce a valid transfer.
- A simultaneous underbody transition carried the box back to the source
  table instead of placing the grippers below it.
- A sequential underbody probe showed that one gripper cannot independently
  support the 0.45 kg box long enough for the other arm to reach the bottom.
  This remained true after an additional physical 0.13 m lift.

## Competition Boundary Result

The submitted overlay still has zero hard scored-path violations: no transport
attachment call and no task-object qpos write. The current research branch is
not a submission candidate because it does not pass L1 performance.

## Recommended Next Route

The next evidence-based route is a fully physical supported-transport strategy,
not another side-grasp parameter sweep:

1. Lower and release the box onto a real support surface.
2. Use closed-loop non-prehensile pushing or dragging while the table or floor
   carries the weight.
3. Plan the robot and object jointly through collision-free corridors.
4. Preserve the same hard audit: no object pose writes, no attachment, and
   score only the original trajectory.

This changes the manipulation mode from long-distance carrying to physical
pushing. It should be implemented as a separately approved route because its
visual behavior and innovation claim differ from the selected carry design.
