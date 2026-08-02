# Physical Transport Data Gate Conclusion

## Verdict

No existing compact L1 result qualifies as a `0.50 m` attachment-free physical
transport demonstration. The 41 unique records contain 0 transport successes,
13 recovery candidates, and 28 rejected records.

## Closest Valid Attempt

`s0-actuator-g0p35-vertical-0p50-seed0.json` is the closest clean record:

- true planar object translation: `0.265401 m`;
- minimum object lift: `0.197830 m`;
- maximum object-to-gripper drift: `0.031537 m`;
- continuous bilateral contact: true;
- drop: false;
- collisions, attachment calls/activations, object-pose writes, and robot-state
  writes: all zero.

The controller stopped when its internal drift guard crossed approximately
`0.03 m`; relaxing the dataset gate is prohibited. This record is recovery data,
not a successful transport demonstration.

The old `0.25 m` accepted candidate moved the object only `0.199200 m`, despite
`0.249891 m` base translation. This proves that base motion and the runner's old
acceptance flag are not valid dataset-success labels.

## Dataset and Training Consequence

H5b produced 24 formal recovery demonstrations with 6,521 samples. H6 merged
those with 12 eligible H2 grasp demonstrations for 36 demonstrations and 9,866
samples (`24/6/6` train/validation/held-out). Its epoch-140 Diffusion checkpoint
reached median held-out MSE `0.029003` versus `0.125100` for the constant
baseline, but clipped 477, 599, and 530 of 1,631 held-out steps over the three
sampling seeds. H6 has no closed-loop success result.

Therefore no larger training run is authorized from the same data. The next
experiment must first improve true physical object translation while preserving
the strict integrity, contact, lift, drift, and collision gates.
