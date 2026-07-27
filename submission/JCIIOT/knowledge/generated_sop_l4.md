# Generated SOP Knowledge - L4

## Provenance

| Field | Value |
|---|---|
| Generator | `competition-sop-generator/1.0` |
| Source DOCX | `JCIIOT 2026 case 7 SOP.docx` |
| Source SHA-256 | `df637fcb1e558cbcacfc895d38b435008aa9eb15e1f85744631cea12b137e568` |
| Prompt evidence | `paragraph:1` |
| Parse status | `ready` |

This file was generated from the original DOCX. The official hand-written
`knowledge/sop*.md` reference files were not generation inputs.

## Resolved Task Contract

| Field | Resolved value |
|---|---|
| Material | blue, hollow plastic box |
| Quantity | 1 |
| Raw source label | Pick Station 5 |
| Effective source label | Pick Station 5 |
| Source resolution | prompt |
| Target label | Place Station 2 |
| Official source entity | `input_2` |
| Official target entity | `output_5` |
| Official candidate objects | `blue_container_h01_back_upper`, `blue_container_h01_back_lower` |

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
| `image1.png` | `981b032136cddbdb439d69e2819d7672f11d64cff1ff5c6119efc000addd7d83` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `b94ed47cce9c8d69a6e9c0760d56d04c5c8fee35a2ebfef933d071875e629670` |
| `image2.png` | `ee57dd143186a4a73de327b1812d39b32a0f602e686bdd5aa88e54f9b6f527a9` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `f57304ad7ca617d910ee479e25b4a805eb149f91c1dcda20f0c7fcdb6e75c66a` |
| `image3.png` | `3374db9ea4d64eb73e51fd4437aaa625ff1aa58afb3e9191242d4677e148437d` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 2 | `4d61749ee65172ef4ebe6e419af5e1625a776c26e01d849dc0cbe24cf7df794c` |
| `image4.png` | `dafbe08d11e2bfefe0eba1178b3936e726797982c5f3c74e4e8042e89eb78ab6` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `db74ac4675949871516631c23b8bd38eb19a6ab3395470898293f3ca7dba7bcd` |
| `image5.png` | `e3d52eb2c898dd489a34f44e6f9207649035029f027b4ae1c9eeb89b86e30733` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `b8daeedfd10835ae3ec4cb34ab23cc15aed57f30569cf8927b0f105e1d3b21df` |

### `image1.png`

{"material_observations": ["metal", "plastic", "glass", "wood"], "route_or_arrow_observations": ["black conveyor belt", "teal platform", "white support structure"], "safety_observations": ["safety light", "green floor", "no visible hazards"], "uncertainties": ["no station labels", "no route arrows", "no safety instructions"], "visible_labels": ["machine", "workbench", "green boxes", "monitor"]}

### `image2.png`

{"material_observations": ["plastic crates", "metal support structure", "concrete floor", "white painted surfaces"], "route_or_arrow_observations": ["black conveyor belt", "white platform", "white support pillars"], "safety_observations": ["color-coded pole", "white platform", "white cylindrical tank"], "uncertainties": ["no explicit station labels", "no route arrows", "no safety signs"], "visible_labels": ["blue crates", "black conveyor belt", "white machinery", "green floor"]}

### `image3.png`

{"material_observations": [], "route_or_arrow_observations": [], "safety_observations": [], "uncertainties": [], "visible_labels": []}

### `image4.png`

{"material_observations": ["plastic", "metal", "wood", "composite"], "route_or_arrow_observations": ["horizontal conveyor", "vertical lift", "no visible path"], "safety_observations": ["red and green safety lights", "no visible emergency stop", "no visible warning signs"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows", "no material handling instructions"], "visible_labels": ["machine", "control panel", "safety light", "workbench"]}

### `image5.png`

{"material_observations": ["plastic", "blue", "lattice pattern"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["no visible safety signs", "no warning labels"], "uncertainties": ["no station labels", "no route arrows", "no safety equipment"], "visible_labels": ["blue plastic crate", "white table", "green surface", "gray cabinet"]}
