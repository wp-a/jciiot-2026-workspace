# robomimic v0.5

- Project owner: ARISE Initiative.
- Release: v0.5.0, official GitHub release available by 2026-07-28.
- Source: https://github.com/ARISE-Initiative/robomimic/releases
- Accessed: 2026-07-28.
- License: MIT in the official repository.

## Mechanism and evidence

The release adds a UNet Diffusion Policy implementation, action dictionaries,
action normalization, multi-dataset training, language conditioning, resumable
training, and future-action prediction for BC-Transformer. The maintainers
describe Diffusion Policy and BC-Transformer as strong robomimic baselines that
often outperform BC-RNN on their datasets.

## Relevance and limits

The official JCIIOT tree already embeds robomimic `0.5.0`, so the competition
can compare BC-RNN, BC-Transformer, and Diffusion Policy without adding a new
framework. Maintainer benchmark claims are not JCIIOT evidence. We must first
validate the Tiago action/observation schema and compare held-out physical
rollouts under the same perturbations.
