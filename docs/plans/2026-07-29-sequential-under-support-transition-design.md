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
