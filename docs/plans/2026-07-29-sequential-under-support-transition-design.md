# Sequential Under-Support Transition Design

## Goal

Convert the verified L1 center pinch into real bilateral bottom support before
long-distance base motion. The transition must preserve simulated contact
physics, stop on judge collisions or height loss, and never use attachments or
object-pose writes.

## Evidence and Decision

The center pinch already passes a 20-step bilateral hold and a real lift. It can
move the object 0.226 m along the grasp axis and 0.0796 m along the outer
`-y` corridor. Repeated base resets eventually lose one side contact; increasing
pinch duration or repeating shorter resets therefore treats the symptom rather
than the load-path failure.

Three alternatives were considered:

1. Continue tuning side-pinch force and reset distance. This is the smallest
   change, but four independent extraction or lateral trials already lost one
   contact.
2. Use table-supported pushing or dragging. The public displacement score may
   permit it, but the known approach corridor has produced torso collisions and
   it is weaker evidence for SOP-compliant manipulation.
3. Briefly pinch to create clearance, then transfer the load to both hands or
   distal arm links under the object. This changes the load path and is the
   selected design.

## Transition Sequence

1. Start only after the existing center grasp has passed real bilateral contact,
   lift, and 20 stable hold steps.
2. Raise both end effectors by a bounded clearance increment while retaining the
   closed grasp. Stop if object height, bilateral grasp, or collision gates fail.
3. Keep one arm fixed and closed. Open the other gripper, descend it by a bounded
   amount, then move it inward toward the object midpoint.
4. Require measured contact between the object and an allowed hand, wrist, or
   distal-arm collision geometry before treating that side as load-bearing.
5. Repeat the open-descend-inset sequence for the other arm while preserving the
   first support contact.
6. Hold for at least 20 consecutive bilateral support observations before any
   base motion. Finger-only or table-only contacts do not count.

The first simulator experiment stops after the first arm transition. It does not
attempt the second arm or transport until one-sided support is physically
observed. This isolates the new variable and reduces the risk of losing an
otherwise valid grasp.

## Controller Interface

The existing research-stage controller gains per-arm gripper commands so the
stationary arm can remain closed while the moving arm opens. A pure target
builder computes the moving-arm descent and inward displacement from the current
end-effector positions and the measured separation axis. It determines the
inward sign from each arm's actual projection, not from arm names.

The eventual submission implementation remains under
`src/robot_agent/skills/` or `src/robot_agent/workflows/`. The external runner
under `scripts/` is diagnostic instrumentation and is not part of the submitted
competition code.

## Safety and Acceptance Gates

- Judge collision frames: `0`.
- Attachment calls and object-pose writes: `0`.
- Object height must remain above the configured table-clearance threshold.
- The stationary arm must retain grasp contact during the first-arm transition.
- First-arm support must include a hand, wrist, or distal-arm collision geometry;
  finger contacts alone are insufficient.
- The full transition requires 20 consecutive bilateral support observations.
- Every failed simulator run is retained in the experiment ledger; no failed run
  may be represented as an official score.

## Verification Strategy

Pure target and per-arm command construction are test-driven. The full local
suite and allowed-file audit run before a candidate is copied to the pinned
official environment. Simulator iterations change one physical parameter at a
time: clearance lift, descent, inward inset, then arm order. Only measured
improvements are retained.

## Experiment 1 Revision

The first valid pinned-simulator run falsified the assumption that the
stationary arm can carry the object while the moving gripper opens. The
clearance lift succeeded and added 0.094 m of object height, but opening the
right gripper caused the left official grasp to fail after four lowering steps.
The run had zero collision, pose-write, and attachment events and stopped above
the height gate.

The next single-variable experiment therefore keeps the moving gripper closed
during both descent and inset. This lets the existing bilateral pinch retain
the load while the moving hand changes position. The stage is accepted only if
the final contact set includes an allowed hand, wrist, or distal-arm geometry;
finger-only contact still does not count. Opening is deferred until measured
support exists.

The closed-gripper run retained object contact on both arms for 33 lowering
steps and moved the right end effector down 0.069 m. The object remained 0.173 m
above its table reference with zero collision, but the stationary arm's
official complete-grasp boolean became false when one fingertip contact was
lost. Four other stationary-arm contacts remained. The transition safety gate
is therefore revised to require measured stationary-arm object contact and
minimum height, rather than the stricter complete-grasp heuristic. The initial
center grasp and hold still require the official bilateral grasp check.

With the physical-contact gate, the right arm reached 0.080 m of descent before
the stationary-side contacts disappeared. The object followed downward by
0.064 m and the right side still had finger contacts only, so completing pure
descent before starting inset is not a viable path. The next experiment combines
the same 0.12 m descent and 0.04 m inset in one Cartesian target. This changes
only path timing and tests whether the hand can enter the bottom region during
the measured stationary-contact window.

The combined path lasted 42 steps, compared with 38 for pure descent, but it
did not change the load path. The right end effector moved about 0.096 m down
and 0.026 m inward while the container moved about 0.087 m down and 0.044 m in
the same lateral direction. The final contact set contained right fingertips
only; left contact was lost and no allowed support geometry touched the object.
This falsifies the post-grasp diagonal-inset hypothesis at the frozen geometry.

The container geometry and station bounds expose a better non-prehensile
entry. The `input_5` edge is at world `y=4.688`, while the container reaches
approximately `y=4.820`, leaving about 0.132 m of bottom unsupported by the
table. Further transition work should begin from the table-supported object:
move an open hand outside that edge, descend below the bottom, inset only within
the overhang, and raise until measured bottom support appears. This is a new
table-edge undercut route, not a parameter continuation of the failed
pinch-to-support sequence.
