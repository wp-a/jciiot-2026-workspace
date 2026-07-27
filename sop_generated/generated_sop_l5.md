# Generated SOP Knowledge - L5

## Provenance

| Field | Value |
|---|---|
| Generator | `competition-sop-generator/1.0` |
| Source DOCX | `JCIIOT 2026 case 9 SOP.docx` |
| Source SHA-256 | `de1a3779d119a17031a17d4ca7812366b5bc1f6c66982617272aa66006b7e5ba` |
| Prompt evidence | `paragraph:1` |
| Parse status | `ready` |

This file was generated from the original DOCX. The official hand-written
`knowledge/sop*.md` reference files were not generation inputs.

## Resolved Task Contract

| Field | Resolved value |
|---|---|
| Material | three white-rimmed storage bins |
| Quantity | 3 |
| Raw source label | Pick Station 6 |
| Effective source label | Pick Station 6 |
| Source resolution | prompt |
| Target label | Place Station 1 |
| Official source entity | `input_1` |
| Official target entity | `aux_output_1` |
| Official candidate objects | `white_tote_b01_left_center`, `white_tote_b01_left_front`, `white_tote_b01_left_back` |

## Verified Operating Procedure

1. Confirm the task identity, target material, quantity, source and target.
2. Navigate to the official source through a collision-free route and stop at a verified approach pose.
3. Identify a valid candidate object; confirm that the grasp path is clear.
4. Execute a smooth grasp, require bilateral contact, and verify physical lift before transport.
5. Stow the payload and navigate to the official target while monitoring collisions and load stability.
6. Confirm placement space, lower and release the object, then verify final stability and target distance.
7. Repeat the complete cycle until 3 objects are verified.
8. Record grasp events, trajectory, collision state, final coordinates and completion status.

## Safety And Recovery Contract

- Stop immediately on collision or uncontrolled contact; do not continue from an unverified state.
- If the load drops, stop and re-grasp only when a collision-free recovery is available.
- If a path is blocked, stop at clearance and re-plan; never force passage.
- If SOP evidence conflicts with the scene, preserve the conflict and require the published correction or task/map agreement.
- Mark an object complete only after grasp, lift, transport, release and final-position evidence all pass.

## Cross-Checks

| Check | Result |
|---|---|
| Prompt matches case specification | `true` |
| Official task configuration matches | `true` |
| Semantic source/target ports exist | `true` |

## Conflicts And Resolutions

No task-specific conflict was detected.

## Image Evidence

| Image | Input SHA-256 | VLM model | Status | Attempts | Response SHA-256 |
|---|---|---|---|---:|---|
| `image1.png` | `f0436132913aeb0cb38f7d604a92aa18df2c244b24668a58b4cf73331ae9eb4f` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `d14f1abc4ca22826d7c6573f51d2c80f0685645d74ae21fc4c21d8787f18558f` |
| `image2.png` | `82fdf0aa849b7c1f79661619f62106d1a1330dab3b685d707a57146a79e90ff1` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `fb6ba30d95896d130c60de1e0b8426fdc4bc6ea2e78a939af7217512f0514470` |
| `image3.png` | `3374db9ea4d64eb73e51fd4437aaa625ff1aa58afb3e9191242d4677e148437d` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 2 | `4d61749ee65172ef4ebe6e419af5e1625a776c26e01d849dc0cbe24cf7df794c` |
| `image4.png` | `21ce184a4a64bf73632854c5933e80ca62b61deccb9d6255c61a4925deb33b18` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `597c88064f1b64a3cc2f489a34d49366e7e9abe3d91bc05a6fc842f74702cecb` |
| `image5.png` | `30333720204eac53f3bc607ac43ef2cf0114219eb96161401eb7f17707353308` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `df06188b4a10fc76b89f05ab7f028c93a03dada6ac99e8ff50234bc0b811d067` |

### `image1.png`

{"material_observations": ["metal", "plastic", "wood", "rubber"], "route_or_arrow_observations": ["horizontal conveyor", "vertical path", "blue platform"], "safety_observations": ["yellow warning on cabinet", "color-coded indicator", "no visible emergency stop"], "uncertainties": ["no explicit station labels", "no clear route instructions", "no safety interlocks visible"], "visible_labels": ["machine", "platform", "cabinet", "control panel"]}

### `image2.png`

{"material_observations": ["plastic", "metal", "wood", "glass"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety signs", "no warning labels"], "uncertainties": ["no station labels", "no route instructions"], "visible_labels": ["boxes", "table", "shelf", "drawer"]}

### `image3.png`

{"material_observations": [], "route_or_arrow_observations": [], "safety_observations": [], "uncertainties": [], "visible_labels": []}

### `image4.png`

{"material_observations": ["teal tabletop", "gray legs", "light blue floor", "light gray wall"], "route_or_arrow_observations": ["diagonal line from top to bottom", "green line from top to bottom"], "safety_observations": ["no visible hazards", "no warning signs"], "uncertainties": ["no station labels", "no explicit route instructions"], "visible_labels": ["table", "legs", "floor", "wall"]}

### `image5.png`

{"material_observations": ["smooth matte finish", "uniform color", "no visible markings"], "route_or_arrow_observations": ["green line indicating path", "no visible arrows"], "safety_observations": ["no visible safety signs", "no warning labels"], "uncertainties": ["no station labels", "no explicit instructions"], "visible_labels": ["three gray containers", "green platform", "white poles", "light blue background"]}
