# H2 Results: Competition-Native Tiago Grasp Dataset

Date completed: 2026-08-01 (Asia/Shanghai)

## Outcome

All 14 pre-registered L1 runs passed the full workflow gate. Every run scored
10/10 with the unmodified public scorer, had zero collision frames, emitted one
verified grasp-and-lift event, and produced an aligned 20-dimensional action
demonstration. The dataset contains 4,065 grasp-window samples in total.

This confirms the competition-native data interface and teacher coverage over
the registered nominal, small, and medium pose perturbations. It does not
establish that a learned policy can reproduce the teacher in closed loop.

## Aggregate audit

- Official source commit: `0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- Recorder workspace commit: `976186b034578898799313f3b96277039f478065`
- Candidate: `l4-target-margin-cc1b5b3`
- Passed runs: 14/14
- Tiers: 3 nominal, 9 small, 2 medium
- Samples: 4,065 total, 265 to 360 per demonstration
- Action shape: `[T, 20]`
- State shape: `[T, 87]`
- Images: `[T, 128, 128, 3]`, `uint8`, non-constant
- Inspector classification: `task-compatible`
- Server artifact root:
  `/data01/user/jciiot-2026/model-research/h2-native-tiago-grasp-dataset/`

## Run ledger

| Seed | Tier | Samples | Score | Collisions | Target distance (m) | HDF5 SHA-256 |
|---:|---|---:|---:|---:|---:|---|
| 20260840 | nominal | 360 | 10/10 | 0 | 0.114646 | `29d6c25eff0bd4d111dfd2ec562e6e6c0381e1b2cf07021239bde88b9413aa4e` |
| 20260841 | small | 276 | 10/10 | 0 | 0.150382 | `bdd7cb055da1b4433ddf129be9075233b9b3253d7b074df1458d77da93c8f383` |
| 20260850 | nominal | 360 | 10/10 | 0 | 0.114646 | `b38224148691b338eeb3f34fa107907c4338b97aeececadd9d1c59ee97d3f797` |
| 20260851 | nominal | 360 | 10/10 | 0 | 0.114646 | `adf4571ed7a5c805b509120f1c644733a4b0657282f4b244e7fc1bdeaea147c2` |
| 20260852 | small | 267 | 10/10 | 0 | 0.151721 | `a13d1671af11b4c4061774e7e26ec9e11901e4b139f765e7949af1482eb480aa` |
| 20260853 | small | 271 | 10/10 | 0 | 0.129117 | `5b6134a10c18e05d511fea59a522e2b67bb37823bd9eada8addef31767846f62` |
| 20260854 | small | 274 | 10/10 | 0 | 0.157106 | `9b7c573bfdd32a26836faec0f4ffa8451d894d63b856f6b2461a90e131692d78` |
| 20260855 | small | 276 | 10/10 | 0 | 0.129135 | `b92f9a18a4871b131cf49295ecb44f7263d4bc15cf214711a78e9859db4182b6` |
| 20260856 | small | 274 | 10/10 | 0 | 0.154902 | `f607049364810f6387fa217b9fccc6bfee6888378dec18aecb4c9c2e929beed0` |
| 20260857 | small | 272 | 10/10 | 0 | 0.131074 | `e6c3ee11b7acfe70a47df785a2f4c6abf1cdb06b923b9b1020f23c7986c8b6aa` |
| 20260858 | small | 272 | 10/10 | 0 | 0.164764 | `909512b0a3c31bda6cc04e96a2938533c5b6e7ab21edb0cba1940a4dbe6643f3` |
| 20260859 | small | 271 | 10/10 | 0 | 0.158646 | `eaf3f1f8b927308d10bf4976fe208c75912337675a02fbdb1e3639450d738c36` |
| 20260860 | medium | 265 | 10/10 | 0 | 0.129080 | `9275f27fb72b285645bf86666661f7354e6303fe24d816296f5d2a398960bae4` |
| 20260861 | medium | 267 | 10/10 | 0 | 0.147497 | `0ae5dd3b8cc96affcab6f6d004b5fd9487a82d3faca18949f99e08e6b0e1831d` |

## Interpretation

The nominal teacher is deterministic, so the three nominal runs have identical
action sequences despite different run seeds. H3 uses only seed 20260840 and
excludes nominal seeds 20260850 and 20260851 from model fitting and evaluation
to prevent artificial cross-split leakage.
