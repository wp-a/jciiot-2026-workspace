# Generated SOP Knowledge - L2

## Provenance

| Field | Value |
|---|---|
| Generator | `competition-sop-generator/1.0` |
| Source DOCX | `JCIIOT 2026 case 3 SOP.docx` |
| Source SHA-256 | `8eb46479564acfefe6abc6929eb749ea9b96abd458b5066e968424aa88080e00` |
| Prompt evidence | `paragraphs:1-5` |
| Parse status | `ready_with_resolved_conflicts` |

This file was generated from the original DOCX. The official hand-written
`knowledge/sop*.md` reference files were not generation inputs.

## Resolved Task Contract

| Field | Resolved value |
|---|---|
| Material | Green-rimmed storage bin |
| Quantity | 1 |
| Raw source label | Pick Station 1 |
| Effective source label | Pick Station 1 |
| Source resolution | prompt + ERRATUM Case 2 |
| Target label | Place Station 3 |
| Official source entity | `input_6` |
| Official target entity | `output_4` |
| Official candidate objects | `green_tote_b01_upper`, `green_tote_b01_lower` |

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

| Field | Observed | Selected | Resolution |
|---|---|---|---|
| source_label | Pick Station 2 | Pick Station 1 | ERRATUM Case 2 and task-specific Prompt take precedence |
| material | blue hollow plastic bin | Green-rimmed storage bin | task-specific Prompt overrides the generic body template |

## Image Evidence

| Image | Input SHA-256 | VLM model | Status | Attempts | Response SHA-256 |
|---|---|---|---|---:|---|
| `image1.png` | `e999038469990d174941701d73bd70b2a02d7275e4cd56c002927d4b00bb073a` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `0679abff61f5b2cd266483c257e646c7dde6cf30fd5f9791e6b7b5c822108009` |
| `image2.png` | `8908d23c48a45d12dea283ff1f330f044c5d5c59a4cdbb83b52fa61f3ef04217` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `3a873da3582ce62fe6d1b710e53984965d234ac7997aebb7f1e2b482ef0f982f` |
| `image3.png` | `73f7aee6a6cb7b47d4d517da7909eca4dc766b4056e83d8d71a53dbf34454231` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 2 | `4d61749ee65172ef4ebe6e419af5e1625a776c26e01d849dc0cbe24cf7df794c` |
| `image4.png` | `71c0bf94b99dbef4602a753b23cc0aa0508a5c99115ed6498557b836486855c1` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `ca271d8a4f51df3412e34ae323592d6f4c2738fd7cec5843c3cf8f8affac6efd` |
| `image5.png` | `690c8afaa65574f0c5d4efdc886b12931693d8782b5615affd3966b422e06b98` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `ebb4416d837140970be104b4ed539caf1c95d309ac8e85d5082cf3209e39ab8d` |

### `image1.png`

{"material_observations": ["plastic", "metal", "wood", "glass"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["red emergency stop button", "no visible warning signs"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows"], "visible_labels": ["factory", "workbench", "storage bins", "workstation"]}

### `image2.png`

{"material_observations": ["plastic", "metal", "wood", "glass"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety signs", "no warning labels", "no protective equipment"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows", "no safety equipment"], "visible_labels": ["workbench", "storage bins", "workstation", "pallet"]}

### `image3.png`

{"material_observations": [], "route_or_arrow_observations": [], "safety_observations": [], "uncertainties": [], "visible_labels": []}

### `image4.png`

{"material_observations": ["metal", "plastic", "wood", "teal surface"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible warning signs", "no emergency stop", "no protective guards"], "uncertainties": ["no station labels", "no clear task instructions", "no safety protocols"], "visible_labels": ["workbench", "power outlets", "storage rack", "machine"]}

### `image5.png`

{"material_observations": ["plastic container", "metal frame", "white panels", "blue flooring"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["electrical outlets", "no visible warning signs", "no safety barriers"], "uncertainties": ["no station labels", "no process flow", "no operator instructions"], "visible_labels": ["green container", "white shelf", "blue floor", "white wall"]}
