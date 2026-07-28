# Diffusion Policy

- Title: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.
- Authors: Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song.
- Venue/year: Robotics: Science and Systems, 2023.
- Code: https://github.com/real-stanford/diffusion_policy
- Paper: https://arxiv.org/abs/2303.04137
- Accessed: 2026-07-28.
- Code license: MIT.

## Mechanism and evidence

Diffusion Policy treats visuomotor control as conditional denoising over action
sequences. Receding-horizon sequence prediction can represent multimodal actions
and smooth high-dimensional control. The official repository provides multi-seed
training and evaluation tooling and publishes raw metrics and checkpoints for
its benchmark tasks.

## Relevance and limits

JCIIOT may benefit if contact corrections have multiple valid action modes that
unimodal BC averages poorly. The method costs more training and inference than
BC and cannot correct an invalid action scale, observation order, or gripper
sign. We will use the bundled robomimic v0.5 implementation only after BC-RNN
and BC-Transformer are evaluated on the same valid dataset.
