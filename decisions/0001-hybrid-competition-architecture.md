# ADR-0001：采用可验证混合竞赛架构

- 状态：accepted
- 日期：2026-07-22
- 决策人：JCIIOT 2026 参赛团队

## 背景

比赛只有五个已知 MuJoCo 工厂场景，静态地图、工位和目标对象集合可以读取。Performance 关注任务完成、碰撞和时间，Innovation 关注方法新颖性。官方 baseline 已经把 VLM/LLM 语义层、A* 导航、robomimic 抓取和物理放置分层。

端到端 VLA 具备研究吸引力，但需要大量同构数据、观测/动作适配和 GPU 训练，其分布外稳定性和调试可解释性在提交周期内存在高风险。纯硬编码方案能够快速得分，但难以满足 SOP 原创解析、失败恢复和创新说明。

## 候选方案

1. 端到端 VLA：由视觉语言模型直接产生长程机器人动作。
2. 完全确定性脚本：每个场景使用固定路线、固定姿态和固定抓放序列。
3. 可验证混合系统：VLM 只做 SOP 语义抽取，确定性 workflow 编排，几何规划导航，局部模仿学习抓取，执行后显式验证和有限恢复。

## 决定

采用方案 3。系统边界为：

```text
DOCX/VLM schema + evidence
  -> deterministic task graph
  -> risk-aware grid navigation
  -> robomimic local grasp policy
  -> physical placement
  -> object/event/collision verification
  -> bounded recovery
```

LLM/VLM 不直接输出底盘或关节动作。任何学习策略只负责已有接口定义清楚、可独立评测的局部技能。

## 理由

- 能利用已知地图、对象和宽松目标容差，提高零碰撞和重复成功率。
- 每个模块都有明确前置条件、后置条件、指标和替换边界。
- L5 多对象状态、恢复行为和 SOP 证据链可以形成有说服力的集成创新。
- 与官方允许修改的 `skills/`、`workflows/` 和 `robot_params.json` 边界一致。
- 失败能够定位到语义、导航、抓取、运输、放置或验证阶段。

## 后果

- 第一阶段必须先完成固定计划 L1，而不是立即训练大模型。
- 需要维护对象状态账本、验证器和统一实验日志。
- 几何与规则约束会占据主要工程工作，但这些工作同时支撑性能和创新报告。
- VLA 保留为 P2 支线；如果尝试，必须与同数据的 robomimic 基线公平比较。

## 证据

- `docs/01-official-baseline-audit.md`
- `docs/02-technology-landscape.md`
- `docs/07-similar-projects.md`
- `research/notes/github-project-audit-2026-07-22.md`

## 复审条件

满足以下任一条件时复审：

- 官方宣布隐藏开放场景或大幅随机化，固定几何方法不再有效；
- L1-L5 混合基线稳定后，轻量 VLA 在相同数据和算力下显著提高重复成功率；
- 官方修改允许目录或评测接口，使现有分层架构无法提交。
