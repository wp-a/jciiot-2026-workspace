# 架构决策记录

本目录使用简化 Architecture Decision Record（ADR）保存会影响技术路线、合规或复现方式的决定。

## 状态

- `accepted`：当前执行决定；
- `proposed`：待实验或团队确认；
- `superseded`：被新 ADR 替代，但原记录保留；
- `rejected`：已评估并明确不采用。

## 规则

1. 一个 ADR 只解决一个长期决策。
2. 记录背景、候选方案、决定、理由、后果、证据和复审条件。
3. 改变路线时新增 ADR，并在旧 ADR 标记替代关系；不要覆盖历史理由。
4. 实验性参数调整写入实验日志，不为每个参数创建 ADR。

## 索引

- [ADR-0001：采用可验证混合竞赛架构](0001-hybrid-competition-architecture.md)
- [ADR-0002：隔离第三方与其他参赛者代码](0002-reference-code-policy.md)
