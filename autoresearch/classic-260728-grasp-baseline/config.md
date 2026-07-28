# Official checkpoint physical grasp baseline

- Experiment date: 2026-07-28
- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Source checkout: `/home/user/jciiot-2026/candidates/competition-entry-260839a/JCIIOT`
- Checkpoint: `/home/user/jciiot-2026/assets/official-lfs-0dcdddf/model_epoch_150.pth`
- Checkpoint size: `139543773` bytes
- Checkpoint SHA-256: `ef5910f6a9f6309b5ced617762dffeb1169a8b0cfcea892d158e6b483252169f`
- Python: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`
- GPU: one NVIDIA L40S, selected through `CUDA_VISIBLE_DEVICES`
- MuJoCo rendering: headless EGL
- Runner: `scripts/run_official_grasp_baseline.py`
- Seed: `0`

The runner verifies the source commit and checkpoint hash before importing the model. It uses the unmodified official wrapped policy rollout and `lift_grasped_object` helper directly. It does not call backend object synchronization, write object qpos, or create a transport attachment.

Official parameters read from `knowledge/robot_params.json`:

- policy rollout: 360 steps, 5 initial-view steps, 5 post-hold steps;
- physical validation: 0.15 m lift, 0.02 m tolerance, 300 maximum lift steps, 20 hold steps.

Representative command:

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /home/user/jciiot-2026/envs/official-pinned-eval/bin/python \
  /home/user/jciiot-2026/tools/run_official_grasp_baseline.py \
  --app-dir /home/user/jciiot-2026/candidates/competition-entry-260839a \
  --checkpoint /home/user/jciiot-2026/assets/official-lfs-0dcdddf/model_epoch_150.pth \
  --output /home/user/jciiot-2026/results/official-grasp-baseline-20260728/all-levels-seed0.json \
  --seed 0 --device cuda:0
```

The first foreground matrix was intentionally interrupted after all L1-L3 records had been atomically written because L3 collision diagnostics emitted thousands of repeated lines. L4-L5 were then run separately with stdout redirected. `raw-l1-l3-seed0.json` and `raw-l4-l5-seed0.json` are the original manifests; `summary.json` and `results.tsv` are deterministic derivatives of their 11 records.
