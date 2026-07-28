# MimicGen

- Title: MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations.
- Authors: Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox.
- Venue/year: CoRL 2023; data-generation code released July 2024.
- Project: https://mimicgen.github.io/
- Code: https://github.com/NVlabs/mimicgen
- Accessed: 2026-07-28.

## Mechanism and evidence

MimicGen segments source demonstrations into subtasks, selects source segments,
transforms object-relative waypoints to a new scene context, composes them, and
keeps successful physical rollouts. The project reports more than 50,000
generated demonstrations from fewer than 200 human demonstrations across 18
tasks, including mobile manipulation and broad reset distributions.

## Relevance and limits

The transferable idea is well matched to JCIIOT: segment the existing teacher
into approach, contact, lift, and release; transform object-relative waypoints;
then replay through the actual Tiago controllers. Direct code integration is
not planned because the repository license is restrictive for competition use
and its task wrappers do not directly support JCIIOT. We will independently
implement the narrow transformation and verification logic under our license.
