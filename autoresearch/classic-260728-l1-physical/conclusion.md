# L1 Physical Support and Regrasp Conclusion

## Status

No experiment in this directory passed either hard gate. No result here is a
valid score claim, and none may replace the current 8502 service. All five final
opposed-wall experiments had zero official collision frames, zero object-pose
writes, and zero attachment calls, but none lifted the box from the table.

## Established Physical Evidence

- The official L1 scripted grasp is a real two-gripper contact and lift. It
  lifts the near container by about 0.204 m in the diagnostic record.
- The grasp sites are both on the same local `x=+0.29 m` end of the 60 x 40 x
  25 cm container. The grasp is therefore cantilevered rather than centered.
- Holding, moving 40/80/140 mm inward, and recentering preserve finger contact
  only. They never establish bilateral wrist, palm, hand, or forearm support.
- Table-assisted release is physical and repeatable: the box settles at about
  `z=1.125478 m` without a pose write.
- Apparent open-cradle support was top-rim pressure while the table carried the
  box. When the arms rose, contact disappeared and the object stayed on the
  table; it was not load-bearing support.
- The first center-regrasp implementation placed end-effectors about 0.12 m on
  either side of the object center while the long walls are at about 0.193 m.
  It descended inside the open container and could not grasp it.
- A 0.10 m high-clearance outward move corrected that geometry. The arms then
  reached the opposed walls with one real fingertip contact per arm and zero
  collisions.
- Closing the Robotiq grippers at that pose removed the contacts. A 25 mm arm
  squeeze produced only transient bilateral contact. Starting a vertical lift
  at the first bilateral contact moved the arms for 180 steps but changed box
  height by approximately zero.
- The Robotiq closure axes in the resulting pose have vertical components of
  about 0.86. Fixed wrist orientation is suitable for top-rim grasping, not
  normal-force grasping of the vertical long walls.
- Direct base pushing reached 0.3315 m but moved the object 0 m and collided
  with the production-line proxy. Controller-only base motion moved 0.0021 m
  in 600 steps. A staged high arm shift physically moved the box about 0.175 m,
  but the available lowering stage did not return it to table support.

## Decision

Stop position-only and fixed-wrist parameter sweeps. The next grasp experiment
must explicitly rotate both wrists so the Robotiq closure axes align with the
opposed wall normals while preserving a collision-free approach. The alternate
route is a separately gated table-supported push/drag controller with staged
robot repositioning.

Do not start BC-RNN or Diffusion Policy comparison training yet. The official
HDF5 is Fetch/iGibson with 10-dimensional actions, and this loop has produced
no valid Tiago transport demonstrations. Training becomes justified only after
the orientation-aware controller or a teleoperation collector yields a small
set of real successful trajectories and passes a single-trajectory overfit
check.

## Traceability

The final opposed-wall test is indexed as
`L1-OPPOSED-WALL-LIFT-ON-CONTACT` in `experiments/experiment-log.csv`. The
official five-level checkpoint baseline is indexed as
`OFFICIAL-GRASP-5LEVEL-SEED0`. Per-run diagnostic values are in `results.tsv`;
full simulator trajectories remain under the remote root recorded in
`config.md`.
