# Official HDF5 compatibility conclusion

## Verdict

`table_setup_from_dishwasher_sample.hdf5` is **format-only** for the JCIIOT FactorySorting competition. It is a valid robomimic-style demonstration file, but it is not a JCIIOT Tiago FactorySorting training set and must not be used to fine-tune the official 20-dimensional grasp policy as if it were task data.

## Verified evidence

- Environment: `SemanticOrganizeAndFetch`, not `FactorySorting`.
- Simulator/task: iGibson kitchen `Pomaria_1_int`; move a YCB bowl from a dishwasher to a table.
- Robot: `FetchGripper`, not the competition's dual-arm Tiago configuration.
- Demonstrations: `5`.
- Total samples: `1916`.
- Action dimension: `10`; the official JCIIOT checkpoint interface expects `20`.
- Observation keys: `depth`, `depth_wrist`, `gt_nav`, `object`, `proprio`, `proprio_nav`, `rgb`, `rgb_wrist`, `scan`.
- Missing all required JCIIOT policy observations: left/right EEF position and quaternion, left/right gripper qpos, and `robot0_robotview_image`.

The complete machine-readable extraction is in `dataset-summary.json`. The file hash and inspection environment are recorded in `config.md`.

## Consequence for the technical route

1. Do not train BC-RNN or Diffusion Policy on this file for the five competition scenes.
2. Use it only to verify robomimic HDF5 conventions and data tooling.
3. First measure the official checkpoint's physical grasp baseline on all scored objects.
4. Collect competition-native Tiago demonstrations only after the deterministic L1 physical controller produces repeatable successful rollouts.
5. Decide BC-RNN versus Diffusion Policy using those native rollouts and a held-out seed set, not this sample file.
