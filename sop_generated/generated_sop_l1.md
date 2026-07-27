# Generated SOP Knowledge - L1

## Provenance

| Field | Value |
|---|---|
| Generator | `competition-sop-generator/1.0` |
| Source DOCX | `JCIIOT 2026 case 1 SOP.docx` |
| Source SHA-256 | `32a446a8395b03b46c9581d3e4978bd84bd90f31096e675da77366fd1cdc9c1c` |
| Prompt evidence | `paragraph:1` |
| Parse status | `ready` |

This file was generated from the original DOCX. The official hand-written
`knowledge/sop*.md` reference files were not generation inputs.

## Resolved Task Contract

| Field | Resolved value |
|---|---|
| Material | blue, hollow plastic box |
| Quantity | 1 |
| Raw source label | Pick Station 2 |
| Effective source label | Pick Station 2 |
| Source resolution | prompt |
| Target label | Place Station 3 |
| Official source entity | `input_5` |
| Official target entity | `output_4` |
| Official candidate objects | `line_5_container_h01_near`, `line_5_container_h01_far` |

## Verified Operating Procedure

1. Confirm the task identity, target material, quantity, source and target.
2. Navigate to the official source through a collision-free route and stop at a verified approach pose.
3. Identify a valid candidate object; confirm that the grasp path is clear.
4. Execute a smooth grasp, require bilateral contact, and verify physical lift before transport.
5. Stow the payload and navigate to the official target while monitoring collisions and load stability.
6. Confirm placement space, lower and release the object, then verify final stability and target distance.
7. Repeat the complete cycle until 1 object is verified.
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
| `image1.png` | `a7e1a0093261dc5747cda82b635f2a36603a8417a331190f6755874ef5941108` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `41f66202eb256e633543b9134abeb139d31db9b18be9a15b51d4cbfba236ce35` |
| `image2.png` | `390d4e596cb91455ef2a1895c43a3ef13b612521cddc0b35ea0312d562638b22` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `af0433790cb4c46969c31fedd791b8dfe2f2195b0a9071f1be39c24b7dd5882b` |
| `image3.png` | `73f7aee6a6cb7b47d4d517da7909eca4dc766b4056e83d8d71a53dbf34454231` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 2 | `4d61749ee65172ef4ebe6e419af5e1625a776c26e01d849dc0cbe24cf7df794c` |
| `image4.png` | `4d8a640cc5fde0ba8066e82e482d39e212d06e35c27c77024d9bfa3435a2c0ea` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `2dfd2d8a4c182b9e88552f3cb2b56b49b45279492fb00c8c552338603582de83` |
| `image5.png` | `6d43722df3d1e7bea7586ad039b6a153735fd8c394d6e607e8bb116b6fa64884` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `5af8100e3104874274b44ec119c165fc03e9ba9750a8e7a1491eb2e23af09812` |

### `image1.png`

{"material_observations": ["plastic", "metal", "wood", "glass"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety signs", "no warning labels", "no protective equipment"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows", "no safety equipment"], "visible_labels": ["workbench", "storage bins", "computer monitor", "pallet"]}

### `image2.png`

{"material_observations": ["plastic", "metal", "wood", "polypropylene"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety signage", "no warning labels", "no protective equipment"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows", "no safety instructions"], "visible_labels": ["workbench", "storage bins", "workstation", "computer monitor"]}

### `image3.png`

{"material_observations": [], "route_or_arrow_observations": [], "safety_observations": [], "uncertainties": [], "visible_labels": []}

### `image4.png`

{"material_observations": ["metal", "plastic", "wood", "teal surface"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety guards", "no warning signs", "no emergency stop", "no clear signage"], "uncertainties": ["no station labels", "no route arrows", "no safety equipment", "no material handling instructions"], "visible_labels": ["workbench", "power outlet", "workstation", "machine"]}

### `image5.png`

{"material_observations": ["plastic", "metal", "wood", "concrete"], "route_or_arrow_observations": ["conveyor belt", "horizontal rail", "vertical rail"], "safety_observations": ["electrical outlet", "metal guard", "no visible warning signs"], "uncertainties": ["no station labels", "no route arrows", "no task materials"], "visible_labels": ["blue plastic crate", "white metal frame", "gray floor", "white table"]}
