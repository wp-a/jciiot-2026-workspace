# Robot Policy Literature Survey For JCIIOT 2026

Accessed: 2026-07-28.

## Selection rule

This survey prioritizes primary papers, project pages, and official code. A
method is promoted only when it fits the official Tiago 20-dimensional action
interface, the small five-scene distribution, the collision penalty, and the
available four-L40S experiment budget.

## Decision table

| Method | Useful mechanism | JCIIOT role | Decision |
|---|---|---|---|
| robomimic BC-RNN | recurrent behavioral cloning | low-dimensional data/action pipeline proof | implement first |
| robomimic BC-Transformer | visual temporal context and optional future actions | primary learned grasp/local-place candidate | implement after one-demo proof |
| MimicGen | object-relative subtask transformation and physical replay | generate diverse teacher demonstrations | reimplement concept only |
| robomimic Diffusion Policy | multimodal action sequence generation | challenger after simple BC failure analysis | conditional |
| ACT | action chunks and temporal aggregation | challenger for measured compounding error | conditional |
| general VLA | language-vision-action generalization | unnecessary for five known tasks in the first cycle | defer |

## Main gap

The limiting evidence is not model availability. It is the absence of a valid,
competition-native Tiago dataset and an honest perturbation benchmark. The first
research cycle therefore measures the incumbent under explicit shifts, then
proves one-demonstration physical replay before scaling data or model capacity.

## Sources

- [robomimic releases](https://github.com/ARISE-Initiative/robomimic/releases)
- [MimicGen project](https://mimicgen.github.io/)
- [Diffusion Policy official repository](https://github.com/real-stanford/diffusion_policy)
- [ACT / Mobile ALOHA repository](https://github.com/MarkFzp/act-plus-plus)
