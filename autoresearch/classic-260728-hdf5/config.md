# Official HDF5 audit configuration

- Audit date: 2026-07-28
- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Source URL: `https://github.com/JCIIOT2026/JCIIOT2026/raw/refs/heads/master/JCIIOT/robosuite/dataset/table_setup_from_dishwasher_sample.hdf5`
- Server asset: `/home/user/jciiot-2026/assets/official-lfs-0dcdddf/table_setup_from_dishwasher_sample.hdf5`
- File size: `591069600` bytes
- SHA-256: `e7f8fd98aa70ba5cea4cb5fec963d3534083b6d2fa9be7128fa33e9146f79eea`
- Inspector: `scripts/inspect_robomimic_hdf5.py`
- Python: `/home/user/jciiot-2026/envs/official-pinned-eval/bin/python`
- h5py: `3.16.0`

Reproduction command on the evaluation server:

```bash
/home/user/jciiot-2026/envs/official-pinned-eval/bin/python \
  /home/user/jciiot-2026/tools/inspect_robomimic_hdf5.py \
  /home/user/jciiot-2026/assets/official-lfs-0dcdddf/table_setup_from_dishwasher_sample.hdf5 \
  --output /home/user/jciiot-2026/assets/official-lfs-0dcdddf/dataset-summary.json
```

The inspector reads HDF5 metadata and dataset shapes only. It does not load image arrays into memory.
