# L1 real center-grasp conclusion

No run in this three-iteration loop passed the physical cradle gate. The
current 8502 candidate remains unchanged, and none of these results is an
official score claim.

The retained arm-only 24-node wrist seed and 10-degree runtime entry completed
with zero official collision frames in all three valid runs. Adding 0.10 m of
collision-checked base advance resolved the previous reach failure: both arms
completed `translate_to_center` and reached the object walls.

The default 0.24 m center shift placed the end effectors above the container
interior. Both forearms contacted the long walls before the fingerpads could
grasp. Setting the center shift to zero removed those forearm contacts. The
first opposed contacts then came from both grippers' inner knuckles, but the
current squeeze stage stopped after that first contact frame. Closing from this
pose lost contact and never produced three consecutive bilateral official
grasp frames.

The next experiment retains the zero-center-shift geometry and changes the
squeeze controller, not the collision or grasp gates: it must complete the
bounded inward target while checking every simulator step for official judge
collision, then require the same bilateral fingerpad grasp, lift, and hold
evidence.

## Artifacts

- Iteration 1 JSON SHA-256:
  `097f948071493f95c7b23acf2a18944a6364ec0224424a98c0f2e55046238166`
- Iteration 2 JSON SHA-256:
  `63855b9ddd36db0cf166cbf29e186a3e26e87379978bae03dbb9fbc6d35da714`
- Iteration 3 JSON SHA-256:
  `85e11630b1911beb41bc0fa0c01b1d1508f0232fee52dca8fddd247c96309718`

The compact JSON files are stored locally under
`/Users/wangpeng/jciiot-2026-workspace/artifacts/l1-real-center-grasp-20260729/`;
full trajectories remain in the remote result root recorded in `config.md`.
