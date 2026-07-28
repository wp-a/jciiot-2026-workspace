# Official checkpoint baseline conclusion

## Result

The materialized official `model_epoch_150.pth` did not physically grasp or lift any of the 11 configured object entries at seed 0.

| Level | Runs | Physical success | Collision | Unsupported by official grasp interface |
|---|---:|---:|---:|---:|
| L1 | 2 | 0 | 0 | 1 |
| L2 | 2 | 0 | 0 | 0 |
| L3 | 2 | 0 | 2 | 0 |
| L4 | 2 | 0 | 0 | 0 |
| L5 | 3 | 0 | 0 | 0 |
| Total | 11 | 0 | 2 | 1 |

Every measured object Z displacement was effectively zero (small negative values of about 5-6 micrometres are simulator settling). No record passed bilateral `_check_grasp`; the L3 auxiliary-table attempts also set the official collision flag. There were no infrastructure errors in the final records.

`line_5_container_h01_far` is a structural exception: the official L1 scene source only creates bilateral grasp sites for the `near` replacement. The baseline therefore records this object as unsupported by the official checkpoint interface, not as a crash and not as a physical attempt.

## Interpretation

This is a one-seed coverage diagnostic, not a statistically converged success-rate estimate. It is nevertheless sufficient to reject the assumption that the supplied checkpoint is a working five-scene grasp solution: all ten objects that expose the required policy sites failed before lift, and the only remaining object cannot be addressed by the official policy interface.

The checkpoint remains useful as an architecture and observation-interface reference. It should not be the primary competition manipulation controller, and it should not be described as a trained high-score solution.

## Training decision

Do not start BC-RNN versus Diffusion Policy training yet:

1. The supplied HDF5 is an incompatible iGibson Fetch dataset with only five demonstrations and 10-dimensional actions.
2. The official checkpoint provides no successful five-scene demonstrations to distill.
3. A learned policy comparison would currently measure data mismatch, not algorithm quality.

The next gate is a repeatable L1 scripted physical cradle/grasp-and-lift trajectory with no qpos writes, no attachment, bilateral contact evidence, at least 0.13 m measured lift, and no collision. Only after successful competition-native rollouts exist should they be converted into robomimic HDF5 and split into held-out seeds for BC-RNN and Diffusion Policy comparison.
