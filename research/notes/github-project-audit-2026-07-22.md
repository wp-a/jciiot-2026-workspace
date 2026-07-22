# GitHub 同类项目审计记录

审计日期：2026-07-22

## 方法

- 先按 JCIIOT 模块拆分检索：SOP/VLM、行为树、导航、抓取模仿学习、数据生成、移动操作、执行验证。
- 只把官方仓库、作者项目页和论文原文作为技术事实来源。
- 对 GitHub 仓库核对默认分支、HEAD commit、许可证、最近提交、README 和目录结构。
- 对同赛题 fork 使用 GitHub compare 检查相对上游的 ahead/behind 和修改文件。
- 许可证、依赖和官方允许修改边界优先于算法新颖度。

## 远程审计结论

### 最优先

| 来源 | 固定 commit | 远程证据状态 | 初步结论 |
|---|---|---|---|
| robomimic | `e10526b9...` | remote-verified | 与官方栈直接匹配，优先复用现有版本和训练接口 |
| PythonRobotics | `b38c510e...` | remote-verified | 适合提取 A*/Theta*/D* Lite 和样条参考实现 |
| py_trees | `48d0f7af...` | remote-verified | 适合 L5 状态共享、重试和恢复语义 |
| multimodal-bt-generation | `49969844...` | remote-verified | 有实际代码和数据，不是只有论文占位；模拟器不匹配 |
| MimicLabs | `0b81d7e8...` | remote-verified | robosuite 数据采集结构高度相关，依赖许可证需拆分审查 |

### 架构和创新参考

| 来源 | 固定 commit | 远程证据状态 | 初步结论 |
|---|---|---|---|
| KIOS | `e9f16f5b...` | remote-verified | 世界状态和计划修复有价值，完整依赖过重 |
| CP-Gen | `1e7c1fc4...` | remote-verified | 约束保持增强有创新价值，不直接接入依赖栈 |
| OK-Robot | `174c742b...` | remote-verified | 系统模块边界相似；AnyGrasp 明确不可用 |
| ACT++ | `26bab078...` | remote-verified | action chunking 可研究，动作空间不匹配 |
| MobileManiBench | `13546663...` | remote-verified | 2026 数据/评测设计参考，Isaac Sim 不接入 |
| RoboMonkey | `db9f8d31...` | remote-verified | verifier 思路可简化，完整 VLA 栈过重 |

### 同赛题公开 fork

官方仓库当时有 8 个公开 fork。只有 `QiShengZhao/JCIIOT2026` 和 `jiangzizi/JCIIOT2026` 相对上游存在实质提交。

`QiShengZhao/JCIIOT2026@95fa2bed...` 包含固定 L1 计划、BC/DAgger 数据脚本和运行诊断，但还修改了 `app.py`、environment、robomimic 和场景文件。其 commit 说明称脚本路径 10/10、DAgger 约 1/5；这些都没有在本地未修改 scorer 中复现，且评分实现发生变化，证据状态只能是 `remote-verified`。

## 许可证排除项

- OK-Robot 中 AnyGrasp 需要单独注册/授权，禁止使用。
- MimicGen 使用 NVIDIA 非商业许可证；MimicLabs/CP-Gen 的依赖链不能按顶层 MIT 许可证简单判断。
- 其他参赛者 fork 即使公开，也不构成可复制实现；比赛规则明确禁止抄袭。

## 本地审计状态

本笔记创建时，参考仓库尚未下载到 `references/repos/`。完成固定 commit 下载后，需要在下方追加：实际 HEAD、许可证文件路径、关键源码路径、磁盘大小和本地代码审计结论。

## 尚未完成

- 没有下载模型权重、数据集或 Git LFS 资产。
- 没有在 JCIIOT 环境运行任何外部策略。
- 没有把任何作者自报指标标为本赛题已复现。
