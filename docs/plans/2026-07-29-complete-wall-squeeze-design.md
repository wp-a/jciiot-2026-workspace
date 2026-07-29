# Complete Wall Squeeze Design

## Status and scope

This is a scheme-1 research-runner change. It modifies only
`scripts/run_l1_cradle_gate.py` and its tests. It does not modify the official
candidate or any competition submission file. The current 8502 service remains
unchanged until a later candidate passes the complete physical and scoring
gates twice.

## Evidence and root cause

The retained run `iter03-nearwall-close.json` used a 0.10 m collision-checked
base advance and zero center shift. It reached both container walls with zero
official collision frames and avoided the previous forearm contacts. The first
contacts were the left inner knuckle of each open gripper.

`squeeze_center_walls` requested a 25 mm inward motion, but it also passed
`stop_bilateral_contact_steps=1`. Consequently the stage returned success on
its first contact frame after one simulator step. The gripper centers had not
completed the requested inward motion. Closing from that pose lost all contact
and never satisfied the unchanged three-frame bilateral official grasp gate.

## Options considered

1. Complete the existing bounded squeeze before closing. This isolates the
   observed premature termination and retains per-step official collision
   checks. This is the selected first experiment.
2. Sweep the near-wall center shift. That changes geometry without first
   testing whether the existing 25 mm target was ever executed, so it is not
   yet justified.
3. Close while continuously servoing inward. This may be more compliant, but
   changes both arm and gripper control at once and is deferred unless option 1
   falsifies the simpler hypothesis.

## Behavior

The squeeze stage keeps the grippers open and executes its existing 25 mm
opposed inward targets until both end effectors are within the standard 12 mm
target tolerance. Object contact is observed but no longer treated as stage
completion. Any official judge collision still terminates the stage as a
failure. Only after successful completion may `close_center_grasp` run.

The downstream gates remain unchanged: three consecutive bilateral official
grasp frames, at least 0.13 m measured lift, at least 20 closed-gripper hold
steps, zero object-pose writes, zero attachments, and zero official collision
frames.

## Verification

A regression test extracts the `squeeze_center_walls` call and proves it no
longer uses the one-frame bilateral-contact stop. The focused test must first
fail on the current code, then pass after the one-line behavior change. The full
test suite, scored-path audit, workspace checks, and diff checks must pass
before the runner is synchronized to the server.

The first server experiment changes only this source behavior relative to the
retained near-wall run. Its structured stage record must show more than one
squeeze step or an explicit target completion/failure; physical success is
decided only by the unchanged grasp, lift, hold, and collision gates.
