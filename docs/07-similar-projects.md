# 同类项目与可复用方案

核对日期：2026-07-22。仓库版本、许可证和采用状态以 [`research/source-ledger.csv`](../research/source-ledger.csv) 与 [`references/repositories.json`](../references/repositories.json) 为准。

## 结论

当前没有公开、可复现、覆盖 JCIIOT 五个场景且符合官方修改边界的完整方案。最可行的组合是：官方框架和评分器作为固定边界，robomimic 负责局部抓取，PythonRobotics/Nav2 概念增强导航，轻量行为树或状态机负责编排和恢复，MimicLabs/CP-Gen 只提供数据方法参考。

## 同一比赛公开实现

### QiShengZhao/JCIIOT2026

- 固定版本：`95fa2bedc3b4bc286cb86258169bde046090d065`。
- 有用内容：L1 固定四步计划、扰动示范采集、BC 配置、DAgger 状态分布纠正、HDF5 聚合和无头环境诊断。
- 作者记录：脚本抓取路径自报 L1 10/10；DAgger 重训闭环表现约 1/5 后被放弃。
- 关键问题：仓库同时修改 `app.py`、environment、robomimic 和场景文件，并改变评分口径；这些超出当前官方允许目录。
- 使用规则：只记录“如何诊断”的线索，不复制实现，不将自报分数写成已复现结果。

其他公开 fork 多数没有领先上游的实现提交。`jiangzizi/JCIIOT2026` 主要是无头运行修改和文档导出，未提供可核实的新客观成绩。

## 直接复用候选

### robomimic

- 匹配模块：抓取策略、数据格式、训练、rollout 和 checkpoint。
- 可用方案：先建立低维/图像 BC-RNN 基线，再比较 BC-Transformer；只有 BC 在多峰动作上明确失败且数据充足时才试 Diffusion Policy。
- 本赛题做法：继续使用官方已内嵌版本，先核对 RGB 翻转、相机键、动作维度、normalization、horizon 和 base approach pose。
- 不做事项：第一阶段不迁移 LeRobot，不额外接入原版 Diffusion Policy 仓库。

### PythonRobotics

- 匹配模块：静态栅格导航和路径后处理。
- 可用方案：A* 保留为基线；加入障碍距离代价、Theta* 视线捷径、D* Lite 备选、B-Spline/三次样条平滑。
- 本赛题做法：只移植必要函数到允许修改的 `skills/move.py`，每条平滑线段重新做连续碰撞检查。
- 风险：示例算法不是生产导航栈，机器人 footprint、全向底盘和载荷状态必须自行建模。

### py_trees

- 匹配模块：L5 多物体编排、重试、超时和状态共享。
- 可用方案：Sequence 表达正常流程，Selector 表达有限恢复，Blackboard 保存每个物体的 `pending/grasped/placed/verified`。
- 本赛题做法：先比较引入依赖和自研小状态机的复杂度；若只需要十余个节点，保持最小实现可能更稳。

## 设计和创新参考

### multimodal-bt-generation

该仓库包含多模态指令到行为树的训练数据、prompt、对象映射和 OmniGibson 执行集成。可复用的是类型化 BT schema、grounding、单任务重试和错误反馈；OmniGibson、BEHAVIOR-1K 和大模型微调管线与本赛题接口不匹配。

### KIOS

KIOS 把世界状态、前置条件、动作效果、LLM 规划和行为树执行联系起来。适合参考计划修复和失败解释，但完整项目包含数据库、LangChain、机器人服务等历史依赖。本赛题应采用其数据结构思想，而不是部署整套系统。

### OK-Robot

OK-Robot 是与“移动到目标、抓取、运输、放置”最相似的系统架构参考。其模块拆分、开放词汇地图和失败分类有价值，但 AnyGrasp 需要单独授权，仓库还包含相关编译二进制；比赛明确禁止需要授权的工具，因此绝不使用该抓取组件。

### ACT++

ACT++ 提供 action chunking、temporal ensembling、Diffusion Policy 和 HDF5 处理示例。它能缓解长动作序列抖动，但 Mobile ALOHA 双臂动作空间、相机和控制周期与 Tiago 不同，只在官方 BC-RNN 基线稳定后做离线对比。

### RoboMonkey

RoboMonkey 在测试时产生多个动作候选并用 verifier 选择。完整实现依赖 OpenVLA、SIMPLER 和多张 RTX 4090，不适合比赛主线。可转化的创新是确定性 verifier：候选轨迹必须同时满足 clearance、速度/加速度、夹爪状态、物体抬升和目标距离约束。

### MobileManiBench

该 2026 基准覆盖移动操作技能、对象、场景和多模态轨迹。Isaac Sim/IsaacLab 环境不能直接并入 robosuite，但其“技能级策略生成数据、统一记录语言/视觉/状态/动作、跨场景验证”结构适合作为实验和报告设计参考。

## 数据生成参考

### MimicLabs

MimicLabs 提供 robosuite/LIBERO/RoboCasa 任务描述、遥操作采集、回放、后处理和大规模数据扩充。最有价值的是三段式流水线：固定任务配置、采集少量成功源示范、自动生成并筛选更多轨迹。

### CP-Gen

CP-Gen 在改变对象或场景几何时保持任务约束，可用于设计 base pose、对象位置和 yaw 扰动。其实现依赖 MimicGen、cuRobo 和自定义 robosuite 分支，现阶段只提取“约束保持增强”的方法，不直接绑定代码。

### MimicGen 许可边界

MimicGen 代码采用 NVIDIA 非商业许可证。比赛包含奖金且获奖代码需要公开，使用范围存在需要进一步确认的解释空间。当前策略是不复制、不绑定、不派生其实现，只根据论文思想独立设计小规模数据转换；任何进一步使用必须先完成许可和组委会确认。

## 明确不作为首阶段依赖

- Nav2：成熟但需要 ROS 2；只采用 inflation、footprint 和 collision critic 概念。
- LeRobot：活跃且功能完整，但迁移官方 HDF5、观测和动作接口的成本过高。
- OpenVLA-OFT、SmolVLA、GR00T：保留为研究支线，不能替代 L1 物理闭环。
- PDDLStream：规划能力强但依赖和许可证会扩大提交风险，当前没有必要。
- AnyGrasp：需要授权，明确禁止。

## 对 JCIIOT 的组合建议

```text
DOCX/VLM schema + evidence
    -> deterministic task graph
    -> risk-aware A*/Theta* navigation
    -> robomimic BC-RNN grasp
    -> physical place and object-pose verification
    -> bounded retry / per-object state recovery
```

第一阶段只实现能够提高官方重复分数的部分。任何新模型或大框架必须通过相同数据、相同种子、相同 scorer 的消融实验后才能进入主线。
