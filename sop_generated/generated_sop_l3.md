# Generated SOP Knowledge - L3

## Provenance

| Field | Value |
|---|---|
| Generator | `competition-sop-generator/1.0` |
| Source DOCX | `JCIIOT 2026 case 5 SOP.docx` |
| Source SHA-256 | `e193fc7d16436d07a8091ecabee712cd9461066603df4d1efa09e9d6eb363927` |
| Prompt evidence | `paragraph:1` |
| Parse status | `ready` |

This file was generated from the original DOCX. The official hand-written
`knowledge/sop*.md` reference files were not generation inputs.

## Resolved Task Contract

| Field | Resolved value |
|---|---|
| Material | blue material transfer bin |
| Quantity | 1 |
| Raw source label | Pick Station 1 |
| Effective source label | Placement Point 1 |
| Source resolution | ERRATUM Case 3 |
| Target label | Place Station 2 |
| Official source entity | `aux_input_1` |
| Official target entity | `output_5` |
| Official candidate objects | `blue_tote_b01_far_right`, `blue_tote_b01_near_right` |

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
| `image1.png` | `166d9d9f779d15e64735e3902e4b89e957dcbb73ffb55924a187b7c383d41414` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `bda79ed4f0e72672bb4c8cf793f09b2e5d00b1741c4d96a82236eb771715ef07` |
| `image2.png` | `eea412abcc929de68ebe0a8ff8b7f44a501556f7612700d048548d399ddfdeec` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `2dd6db31bee606f9ed90cad06946800b0b3553b8cb8a7256bdf2145f76a0573b` |
| `image3.png` | `3374db9ea4d64eb73e51fd4437aaa625ff1aa58afb3e9191242d4677e148437d` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 2 | `4d61749ee65172ef4ebe6e419af5e1625a776c26e01d849dc0cbe24cf7df794c` |
| `image4.png` | `d00eb526f6067d63dd9edb74324c2d5fc45505c1d4ce270a26b9993d9a68f71a` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `7add5f0898699abddd2605e961cfa2630035ff54de00552eff0b770434495e80` |
| `image5.png` | `6bbd7d16016670732b62c39bb499e1ea1493e533492ebfe2959c4979c3790388` | `Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` | `analyzed` | 1 | `9e6f8c5306e9f831458808a5b43933d04c2335f7584d3f97cc9c81ca4ad255ed` |

### `image1.png`

{"material_observations": ["teal top", "gray supports", "light background"], "route_or_arrow_observations": ["horizontal line", "vertical line"], "safety_observations": ["no visible hazards", "no warning signs"], "uncertainties": ["no text or labels", "no equipment details"], "visible_labels": ["table", "support pillars", "background"]}

### `image2.png`

{"material_observations": ["plastic", "smooth surface", "uniform color", "no visible wear"], "route_or_arrow_observations": ["no visible route", "no arrows", "no directional indicators"], "safety_observations": ["no visible safety signs", "no warning labels", "no protective equipment"], "uncertainties": ["no explicit SOP text", "no station labels", "no hazard indicators", "no equipment status"], "visible_labels": ["blue crates", "table", "white wall", "light gray floor"]}

### `image3.png`

{"material_observations": [], "route_or_arrow_observations": [], "safety_observations": [], "uncertainties": [], "visible_labels": []}

### `image4.png`

{"material_observations": ["plastic", "metal", "wood", "concrete"], "route_or_arrow_observations": ["no arrows", "no route indicators"], "safety_observations": ["red-green light", "black light", "no visible warning signs"], "uncertainties": ["no explicit SOP text", "no station labels", "no route arrows"], "visible_labels": ["blue box", "white platform", "red-green light", "black light"]}

### `image5.png`

{"material_observations": ["plastic", "metal", "composite", "wood"], "route_or_arrow_observations": ["horizontal conveyor", "vertical lift", "pathway"], "safety_observations": ["red and green lights", "safety barrier", "warning signs"], "uncertainties": ["no explicit instructions", "no operator presence", "no material flow details"], "visible_labels": ["machine", "control panel", "warning lights", "safety barrier"]}
