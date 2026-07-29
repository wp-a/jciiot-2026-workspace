# JCIIOT L1 wrist-seed continuation design

Date: 2026-07-29 (Asia/Shanghai)

Status: scheme-1 implementation amendment based on official-runtime path
evidence. Competition gates and promotion rules are unchanged.

## Root cause

The 0.0185 m position-scale experiment found a valid joint endpoint with 6.63
and 9.93 degree closure-axis errors, 13.32 mm endpoint position error, and zero
collision. A straight 240-waypoint joint interpolation did not preserve that
task-space constraint: right grip-site drift grew from 4.89 mm at waypoint 1
to 33.70 mm at waypoint 7, where the unchanged 30 mm gate rejected the path.

The endpoint is far from the high-clearance start in several joints, including
about 1.79 rad on joint 2 and about 1.59 rad on joint 3. Linear interpolation
between two inverse-kinematic solutions does not stay on the grip-site pose
manifold. Increasing waypoint count only samples the same invalid curve more
densely and cannot solve the problem.

## Considered paths

### Relax the path drift gate

The first 3% of the path already exceeds 30 mm and the drift is monotonic over
the observed segment. Relaxing the gate would hide an unconstrained sweep and
does not establish safety. Rejected.

### Joint-space RRT with IK projection

A constrained RRT could explore multiple IK branches and project samples back
to the task manifold. It is a valid fallback but introduces stochastic search,
many collision queries, and substantially more code before local reachability
has been tested. Deferred.

### Closure-axis continuation IK

Interpolate each arm's fixed directed closure-axis target from its measured
start direction to the final wall normal. For each fraction, solve a bounded
two-arm IK initialized and regularized at the preceding node while keeping both
original grip-site positions fixed. Validate each node, then interpolate only
between adjacent node solutions with the existing per-waypoint collision and
drift gates.

This is selected as the smallest test of whether the endpoint belongs to a
continuous, position-preserving IK branch reachable from the high-clearance
start.

## Geometry and solver behavior

A pure helper performs normalized interpolation between directed unit axes.
The selected target sign guarantees a non-negative dot product, so normalized
linear interpolation cannot cross a zero vector and yields deterministic start
and end axes.

The runtime option is opt-in and specifies a positive node count no greater
than the total interpolation step count. One node preserves the existing
direct-endpoint experiment. For multiple nodes:

1. build one target-axis pair per increasing fraction;
2. initialize each bounded least-squares solve from the preceding node;
3. regularize joint displacement relative to that preceding node;
4. require finite node state, position error at most 15 mm, closure-axis error
   to the node target at most 5 degrees, and zero official collision;
5. restore the original robot joints after all unrecorded solver probes;
6. replay adjacent node segments with the existing 30 mm path drift and
   official collision checks at every recorded waypoint.

Any failed node or segment rolls all 12 robot joints back, synchronizes the
controllers, and records the exact node, solver, pose, and collision evidence.

## Verification

Pure tests cover directed-axis interpolation endpoints, unit length,
monotonicity, and invalid fractions/vectors. Parser tests keep continuation
opt-in with a deterministic default node count. Existing 211 tests, syntax,
submission audit, and workspace checks must remain green.

The first official-runtime experiment changes only continuation node count
from one to 24 while retaining the proven 0.0185 m endpoint scale and every
hard gate. If local continuation cannot reach the final target, the result is
evidence that the current local IK branch is blocked; the next decision is a
constrained planner or controlled Cartesian excursion, not more gain tuning.
