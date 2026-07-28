# Official assets and score-reality audit

Date: 2026-07-28 (Asia/Shanghai)

## Bottom line

The existing `100/100` evidence is a valid **local fixed-public-scene baseline**:
the trajectories were generated in the five published MuJoCo scenes and scored
with the unmodified public scorer. It is not a BienData result, not an
organizer-reproduced result, and not evidence of robustness to hidden pose or
dynamics changes. The current validation ZIP must therefore remain a baseline
artifact until the compliance and perturbation gates below pass.

The newly accessible official checkpoint is also not a shortcut to a credible
score. Its metadata says it is an L1 Tiago BC-Transformer trained for 500
epochs, but its saved best rollout success rate is `0.0`. Direct execution with
the official evaluator failed all three repeated L1 resets.

## Sources and trust level

1. Official repository: `JCIIOT2026/JCIIOT2026`, locked at
   `0dcdddf18a9e694569aa1433cdfc04eb097fed78`.
2. Local training deck: `直播分享..pptx`, SHA-256
   `9a7d9d631d75c117395845f8d2292912080c80087d45ba874fbdc03739b704f9`.
   The deck is useful guidance but is not independently authenticated as an
   organizer-issued rule document. Its metadata names Kimi/Moonshot, contains
   12 slides, has no notes or citations, and appears to be a partial 22-slide
   deck.
3. The competition rules supplied by the team remain the binding local rule
   snapshot unless the organizer publishes a newer clarification.

The deck describes the standard execution sequence as
`move -> pick_up -> move -> place_down`, with planning followed by Agent skill
execution. It recommends changing `skills/`, `workflows/`, knowledge files and
the robomimic checkpoint, while keeping core, environment, app and task-config
files unchanged. It also says skills should call `EnvBackend` rather than touch
MuJoCo directly. These are treated as strong submission-review guidance, not
as a proven formal disqualification rule.

## Asset integrity

All 11 user-listed Git LFS objects were downloaded from the official repository
and match both the stated byte count and the Git LFS object SHA-256.

Local root:
`/Users/wangpeng/jciiot-2026-assets/official-lfs-0dcdddf/`

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `model_epoch_150.pth` | 139543773 | `ef5910f6a9f6309b5ced617762dffeb1169a8b0cfcea892d158e6b483252169f` |
| `table_setup_from_dishwasher_sample.hdf5` | 591069600 | `e7f8fd98aa70ba5cea4cb5fec963d3534083b6d2fa9be7128fa33e9146f79eea` |
| `3FO3ERFHISEM.zip` | 71105956 | `f7b8694291720eaa287d6fd4450ad853d9e24f1862b83c188b0b5f0a4f0bec95` |
| `3FO3ERRPH7X9.zip` | 69722599 | `55909af1c33975bed6e7b413904770c051f3491da5977ddaf31769f911c3ff9e` |
| `3FO3ERTPXEUT.zip` | 70210383 | `84df2417c250067aab647877a6f7dfc1251e47fe3a17cf38b58222448f80c376` |
| `3FO3ERFKY9RN.zip` | 70315486 | `84271a7a76ca5a23d015977c69882053c65b8c15a521110751a44bb2a94650e7` |
| `3FO3ERT2C5FP.zip` | 72266391 | `39654b030e01461159e3d7932263c7273e166f52499c4b2e2f2376e29726db2c` |
| `lowered_table_meshes_70cm_v2.zip` | 448906 | `b80479ee21b77c1d05c9f10dedc9fd999aad870f6bd4aac5a85076e4e8f7f8f9` |
| `lowered_table_meshes_70cm_v3_fix_front_pillars.zip` | 448637 | `60feb57188bd342bf967e05dab7ac284a5a1dd2e38b2dd66809a605765b9bc1e` |
| `lowered_table_meshes_70cm_v4_shorten_front_legs.zip` | 667273 | `e9fff5ffbb75cd859cf67593374fd25ca556bf6e3e81b8ef9d902137986dcdf1` |
| `lowered_table_meshes_70cm_v5_piecewise_front_supports.zip` | 667263 | `18249745eac09d239e80349f8bd7cb9cd8a274706b30aee852b6807ab9a15c8c` |

## HDF5 finding: format example, not competition demonstrations

The 591 MB HDF5 file is for iGibson
`SemanticOrganizeAndFetch` in the `Pomaria_1_int_custom_kitchen_cleanup_full`
scene. It contains five long demonstrations (`demo_0`, `demo_1`, `demo_10`,
`demo_101`, `demo_102`) and reports 45,941 total samples. Its robot is Fetch,
the action width is 10, and its observations include 90 x 120 RGB/depth,
navigation scans and Fetch proprioception.

JCIIOT uses Tiago with a 20-dimensional action and 128 x 128
`robot0_robotview` RGB plus two-arm end-effector/gripper state. The sample HDF5
can be used to study robomimic layout, masks and tooling, but it must not be
mixed into JCIIOT training or cited as official task demonstrations.

## Checkpoint finding: relevant model, weak recorded result

The checkpoint was inspected with `torch.load(weights_only=True)` and explicit
safe globals. It contains:

- algorithm: behavior cloning with a two-layer, four-head BC-Transformer;
- input: six two-arm low-dimensional keys plus 128 x 128 robot-view RGB;
- action width: 20;
- frame stack and transformer context: 10;
- training data path: `robomimic/datasets/202606222228_low_dim.hdf5`;
- environment: `FactorySorting1_3FO3ERFHISEM`, Tiago, robosuite 1.5.2;
- configured and saved epoch: 500;
- saved `best_success_rate`: `0.0` and `best_return`: `0.0`.

Although the downloaded file is named `model_epoch_150.pth`, its archive root
is `model_epoch_500/` and `variable_state.epoch` is 500. This naming mismatch
must be documented rather than guessed away.

Server evaluation used the unmodified official
`load_factory_sorting_evalization.py`, the exact LFS checkpoint, CUDA, EGL,
the checkpoint's L1 scene/base pose and target object, 360 policy steps and no
viewer. The final replay used an isolated pinned inference environment with
Python 3.11, MuJoCo 3.9.0, robosuite 1.5.2, NumPy 1.26.4, PyTorch 2.7.0,
torchvision 0.22.0, h5py 3.16.0 and the official supporting pins. Three repeated
seed-1 resets all failed identically:

- right gripper-end distance to target: `1.1347917259445428 m`;
- left gripper-end distance to target: `1.1183357172947588 m`;
- bilateral finger-pad contact: false;
- final grasp success: false;
- result: `0/3` grasp successes.

Because the resets were identical, this is a deterministic reproduction, not
three independent random samples. It establishes that the public checkpoint
does not solve its own published L1 reset under the official direct evaluator
in a requirements-pinned environment.

Raw pinned log (git-ignored evidence):
`artifacts/remote-official-checkpoint-20260728/official-bc-l1-seed1-3resets-pinned.log`,
SHA-256
`b9f6f993be0ad37d06d5df3ab1abd6d35b939ff06b1f2e324949076de6ada3ff`.
An earlier combined-runtime replay is retained beside it for dependency
comparison and is not the primary result.

## USD and mesh finding

The five USD packages contain `world.usda` plus 578-610 referenced geometry,
material, texture and metadata files. They are the visual factory source
packages and do not provide Tiago action demonstrations or robomimic records.
Keep them for visual cross-checks, screenshots and future USD/MuJoCo geometry
comparison; do not add them to model training.

The mesh ZIPs are four historical table-height revisions. The official current
tree already contains all 44 files from
`lowered_table_meshes_70cm_v5_piecewise_front_supports.zip` byte-for-byte.
Older packages only partially match the current tree:

| Package | Files | Match current | Different | Missing |
|---|---:|---:|---:|---:|
| v2 | 32 | 25 | 7 | 0 |
| v3 | 32 | 13 | 19 | 0 |
| v4 | 44 | 33 | 11 | 0 |
| v5 | 44 | 44 | 0 | 0 |

Do not extract any of these archives over the locked source tree. The active
MuJoCo source already incorporates v5.

## Assessment of the current candidate

What is real and useful:

- It solves the five published fixed scenes with the unmodified public scorer.
- It has repeatable state-machine, collision, grasp, lift and final-position
  evidence on those exact scenes.
- Object-relative grasp sites and family-specific geometry are stronger than
  one fixed world-coordinate trajectory.
- The retained failure trajectories are valuable teacher data and diagnostics.

What prevents calling it final:

1. There is no BienData score or organizer reproduction.
2. The 80-run batch repeated identical geometry; it did not test distribution
   shift.
3. The current first-match `competition_task` skill disables the planner and
   captures every non-empty task. That may be technically reproducible, but it
   diverges from the training deck's standard atomic-skill execution story and
   weakens the innovation presentation.
4. The current L5 placement path mutates transport-attachment relative state
   and calls private synchronization helpers. It does not write object qpos
   directly, but it is a compliance and reviewer-perception risk because the
   object is moved kinematically without a corresponding arm/base action.
5. Several scored-path helpers read private simulator state or private backend
   helpers directly from allowed skill/workflow files. The deck explicitly
   recommends the `EnvBackend` boundary.

The current ZIP should therefore be labeled `fixed-public-scene-baseline` and
not submitted as a claimed final validation result until points 3-5 are
resolved or explicitly cleared by the organizer.

## Recommended technical route

Use a hybrid system, but change which component owns the physical grasp and
placement:

```text
validated task compiler / optional LLM proposal
  -> atomic move, pick_up, move, place_down skills
  -> risk-aware geometric navigation
  -> object-family robomimic grasp policy
  -> contact + lift verifier and bounded recovery
  -> physical place controller through EnvBackend
  -> public scorer + perturbation audit
```

The current geometric grasp controller becomes an offline teacher and safety
reference, not the final shortcut around the backend. Navigation and task
state remain deterministic. Learning is limited to the manipulation segment
where appearance, pose and contact make a policy useful.

### Phase 0: make the measurement honest

1. Keep the current trajectory ZIP unchanged as a baseline artifact.
2. Add explicit object-pose, base-pose, camera and dynamics perturbations to a
   separate evaluation harness. Do not modify the locked official scenes used
   for the public-score reproduction.
3. Report fixed-scene score, perturbation success, collision rate and timing as
   separate columns.
4. Ask the organizer one narrow question in the official group: whether a
   deterministic validated plan is acceptable when the LLM is unavailable,
   and whether skills may call private simulator/attachment helpers. Preserve
   the answer as a dated rule artifact.

### Phase 1: validate the data and trainer before scaling

1. Extend the official L1 collection pattern to all five scenes and four object
   families: container, green tote, blue tote and white tote.
2. Record the exact competition observation keys, 128 x 128 robot-view RGB and
   20-dimensional actions. Reject a dataset at load time if its environment,
   action width or observation schema differs.
3. First overfit one successful trajectory. The checkpoint must replay that
   grasp successfully before collecting a large dataset.
4. Then overfit 5-20 demonstrations and verify that image orientation,
   frame-stack order, action scaling and gripper signs match evaluation.

### Phase 2: collect task-specific data

Start with 50 successful demonstrations per object family. Sample explicit
tiers rather than relying on `seed`:

| Tier | Object XY | Object yaw | Base XY | Base yaw | Dynamics |
|---|---:|---:|---:|---:|---:|
| small | +/- 2 cm | +/- 5 deg | +/- 1 cm | +/- 2 deg | nominal |
| medium | +/- 4 cm | +/- 10 deg | +/- 3 cm | +/- 5 deg | mass/friction +/- 10% |
| stress | +/- 6 cm | +/- 15 deg | +/- 5 cm | +/- 8 deg | mass/friction +/- 20% |

Only keep physically successful teacher trajectories, but retain failed
attempt metadata for coverage analysis. Split by perturbation seed before
training so near-duplicate trajectories cannot leak into validation. Scale to
200-500 demonstrations per family only when the 50-demo learning curve shows
that more data, rather than a pipeline bug, is the bottleneck.

### Phase 3: train the smallest useful policies

1. Baseline A: robomimic BC-RNN.
2. Baseline B: the official-style BC-Transformer with context 10.
3. Train three seeds and select by held-out grasp success, not training loss or
   epoch count.
4. Route an object family to its own checkpoint through the allowed
   `pick_up`/workflow layer and public backend configuration method.
5. Use DAgger-style correction collection only after the first policy exposes
   its actual failure states.
6. Evaluate Diffusion Policy or ACT only if actions remain demonstrably
   multimodal after the simpler baselines. They are not the first step.

### Phase 4: remove gray-zone execution

1. Replace direct transport-attachment relative-position edits with physical
   base/arm actions and the official placement backend.
2. Keep atomic `move`, `pick_up`, `move`, `place_down` skills visible in the
   execution trace.
3. Let an LLM propose only symbolic steps; validate entities, order and safety
   deterministically. Use a deterministic plan as an explicit documented
   fallback if the organizer confirms it is allowed.
4. No final scored-path module should directly set MuJoCo qpos, attachment
   relative pose or protected backend state.

### Phase 5: acceptance gates

Do not label a package final unless all gates pass:

- locked public five-scene scorer: full score, zero collision, 20 clean runs
  per level;
- grasp-only held-out set: at least 95% small-tier and 90% medium-tier success
  for every object family;
- full-task perturbation: at least 20 runs per level per small/medium tier,
  with zero collision and reported Wilson intervals;
- three training seeds with configs, logs, checkpoints and hashes preserved;
- no forbidden/protected-file changes and no private-state kinematic placement;
- one clean-machine reproduction through the official Agent entry;
- technical report clearly distinguishes local public-score results from any
  later BienData or organizer result.

## Realistic schedule on the available four-L40S server

- Day 1: data schema guards, one-demo overfit and L1 collector reproduction.
- Day 2: four-family 50-demo pilot and BC-RNN/Transformer three-seed training.
- Day 3: held-out evaluation, DAgger correction collection and retraining.
- Day 4: physical placement replacement and full-task perturbation batch.
- Day 5: ablations, timing optimization, final clean reproduction and report.

This is an experiment plan, not a promise that five days guarantees a winning
score. A single training run may take hours rather than days on an L40S; the
time-consuming part is collecting diverse successful trajectories and running
enough held-out physical evaluations. The previous result was fast because it
did not train a model and exercised deterministic published geometry.
