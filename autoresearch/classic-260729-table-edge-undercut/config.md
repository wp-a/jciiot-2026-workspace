# L1 Table-Edge Undercut Autoresearch

## Objective

Obtain measured open-gripper support under the L1 container with zero judge
collisions, object-pose writes, and attachment calls.

## Frozen Geometry

- Table edge world y: `4.688 m`.
- Container half-depth: `0.200 m`.
- Container half-height: `0.125 m`.
- Outside clearance: `0.080 m`.
- Undercut clearance beyond the table edge: `0.060 m`.
- Raise above the measured bottom: `0.120 m`.
- Moving arm: right.
- Gripper: open throughout.

Change one parameter per iteration and preserve every compact result JSON.

## Later Evidence

- Remote root: `/home/user/jciiot-2026/results/l1-table-edge-undercut-20260729/`
- Static support geom: `line_5_container_h01_near_support`
- Static support half-size: `0.340 x 0.240 x 0.010 m`
- Movable bottom half-size: approximately `0.300 x 0.200 m`

Trajectory-state replay showed the proposed fork deeply penetrating the static
support before any attempted lift. The route is therefore rejected.
