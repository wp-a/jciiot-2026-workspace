# Action Chunking With Transformers

- Title: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.
- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn.
- Year: 2023.
- Paper: https://arxiv.org/abs/2304.13705
- Code: https://github.com/MarkFzp/act-plus-plus
- Accessed: 2026-07-28.
- Code license: MIT.

## Mechanism and evidence

ACT predicts chunks of future bimanual actions with a conditional variational
Transformer and uses temporal aggregation at execution. The paper reports
80-90% success on six precise real-world tasks from about ten minutes of
demonstrations per task, addressing compounding error and non-stationary human
demonstrations.

## Relevance and limits

Action chunking may help Tiago execute smooth bilateral contact and lift
segments. ACT's ALOHA embodiment, action representation, cameras, and control
rate differ from JCIIOT, so copying reported hyperparameters would be
unjustified. It is a conditional challenger only if temporal compounding error
remains after the exact Tiago data and BC evaluation path is verified.
