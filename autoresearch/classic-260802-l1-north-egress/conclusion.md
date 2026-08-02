# Conclusion

Decision: discard the `+y` north-egress route for the current side-wall grasp.

The physical grasp and lift succeeded. The actuator-only carry stopped after 19
control steps when planar object-to-gripper drift reached `0.031484 m`; the base
had moved only `0.031000 m`, while independently recomputed true object motion
was `0.003423 m`. The tote remained lifted (`0.195191 m` minimum lift), did not
drop, retained bilateral contact samples, and recorded zero collision,
attachment, teleport, object-pose-write, or robot-state-write evidence.

The canonical physical-data auditor classified the record as recovery solely
because object translation was below `0.50 m`. This is strong negative evidence
for route planning: motion along the current grasp separation axis shears the
grippers around an almost stationary load. It is markedly worse than the clean
`-x` incumbent (`0.265401 m`) and cannot reach the production line's north end.

Stop route-direction experiments with this side-wall grasp. The next physical
gate is a simultaneous bilateral transition from the already lifted pinch to
measured under-bottom support. The official collision model has solid side
walls and no handle holes, so visual handle insertion is not a valid MuJoCo
contact strategy.
