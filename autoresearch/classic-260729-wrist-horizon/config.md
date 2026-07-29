# L1 extended wrist-alignment horizon

- Mode: classic
- Workspace branch: `robust-hybrid-20260728`
- Workspace commit: `5673e43`
- Research script commit: `e0a785e`
- Official commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Candidate: `robust-l1-cradle-20260728a`
- Scene: public L1 `FactorySorting1_3FO3ERFHISEM`
- Seed: `0`
- Scheduling: sequential, one worker, EGL GPU 2
- Protected source changes: none
- Remote result root: `/home/user/jciiot-2026/results/l1-wrist-horizon-20260729`

## Single-variable experiment

Retain the verified safe `orientation_max_action=0.02` controller and extend
only `orientation_max_steps` from 1000 to 2600. All hard gates remain:

- both closure-axis errors at or below 5 degrees for five consecutive steps;
- maximum OSC grip-site position drift at or below 0.03 m;
- zero official judge-collision frames;
- no object pose writes or transport attachments;
- physical bilateral contact, lift, hold, and transport remain required for
  promotion beyond this research gate.

The previous 1000-step trace ended at 29.66 and 33.34 degrees with 0.006204 m
drift and was still improving. This experiment tests whether the same safe
controller reaches the existing angular gate with a sufficient horizon.
