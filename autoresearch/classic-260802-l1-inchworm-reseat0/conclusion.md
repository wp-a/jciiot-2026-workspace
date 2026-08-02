# Conclusion

Decision: discard `reseat_steps=0`.

The clean run reproduced the physical bilateral grasp and lift with zero
collision frames, zero attachment activation or active flags, and zero object
pose writes. It completed two arm strokes and two 0.06 m compensated base
resets, then lost both contacts on the first step of cycle 3. Final effective
progress was `0.128768 m`, below the pre-registered `0.149215 m` keep gate and
far below the `0.50 m` structural success threshold.

Removing reseat changed neither the progress nor the underlying failure. The
trajectory boundary audit shows why: during reset 2, the stored object joint
height fell from `1.320868 m` to `1.179251 m` and pitch changed from `+11.565°`
to `-10.012°`. The broad bilateral contact flags remained true through that
reset, but they represented marginal edge contact after the tote had already
rolled and dropped. The first cycle-3 action exposed the failure; it did not
cause the preceding loss of support.

The next intervention must address load pose stability during base reset, not
reseat or the later contact check. At minimum, reset needs a fail-closed object
height/attitude gate. A successful controller also needs structural vertical
support (for example an undercut/handle or cradle contact) or an actuator-only
attitude stabilization action; further changes to the coarse contact boolean
alone cannot improve the physics.
