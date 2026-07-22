# 模块路线图

本文件把外部项目经验映射到 JCIIOT 的实际模块、指标和实验顺序。状态以 [`STATUS.md`](../STATUS.md) 为准，实验结果写入 [`experiments/experiment-log.csv`](../experiments/experiment-log.csv)。

## 模块总览

| 模块 | 官方现状 | 主要风险 | 首选改进 | 参考项目 | 核心指标 |
|---|---|---|---|---|---|
| SOP 解析 | DOCX 段落 + 每图 VLM 描述 | 自由文本、不留证据、图文冲突 | 类型化 schema、页/图证据、地图交叉校验 | multimodal-bt-generation、GLM-5V | 字段准确率、证据覆盖率、五 SOP 一致性 |
| 任务编排 | LLM 生成固定四步计划 | L5 三对象、失败后全局重来 | 确定性任务图、对象状态账本、有限恢复 | py_trees、KIOS | 完整任务成功率、无效动作数、恢复成功率 |
| 导航 | 二值栅格 A* + 路径简化 | 贴障、平滑后穿障、终点姿态不稳 | clearance cost、Theta*、连续碰撞复核、approach pose | PythonRobotics、Nav2、OK-Robot | 碰撞率、最小间隙、路径长度、耗时 |
| 抓取 | robomimic BC + 抬升验证 | 观测不一致、base pose 偏移、跨场景泛化 | 先校准接口，再扰动示范、BC-RNN 和有限 DAgger | robomimic、MimicLabs、CP-Gen、ACT++ | 抓取/抬升成功率、重试数、闭环成功率 |
| 运输 | 导航复用 | 载荷 footprint、掉落、机械臂姿态 | payload-aware clearance、周期性持物验证 | OK-Robot、Nav2 | 掉落率、载荷碰撞率、运输成功率 |
| 放置 | 面向工位、下降、松爪 | 边缘滑落、释放后位移 | 中心安全区、稳定等待、最终物体校验 | MobileManiBench | 到达率、最终目标距离、稳定时间 |
| 恢复 | 基本依赖单技能返回值 | 无阶段化原因、重复错误动作 | 前后置条件、错误分类、有限重试和替代 approach | py_trees、KIOS、RoboMonkey | 恢复成功率、额外时间、错误升级率 |
| 数据 | 主要为官方 L1 数据 | 五场景不足、分布偏移 | 成功源示范、几何扰动、失败回放、严格筛选 | MimicLabs、CP-Gen、robomimic | 有效轨迹数、覆盖度、验证集/闭环成功率 |
| 评测 | 单轨迹官方评分 | 只看最好结果、配置不可追溯 | 固定种子、重复运行、失败全保留、置信区间 | MobileManiBench | 平均分、任务成功率、碰撞率、95% CI |

## P0：规则、环境和 L1 闭环

1. 补齐 Git LFS 资产并记录 SHA-256。
2. 在目标 Linux 环境运行未修改 baseline 和 scorer。
3. 用固定结构化计划隔离 LLM 不确定性。
4. 检查 RGB 上下方向、observation key、action shape、normalization、`sim.forward()` 和 base pose。
5. 至少运行 5 个固定种子，得到抓取、到达、碰撞和耗时基线。

退出标准：真实 `grasp_end success=true`、两种离开口径都通过、最终距离小于 0.8 m、零碰撞、官方 10/10，并有重复统计。

## P1：导航安全和五场景抓放

### 导航实验顺序

1. `NAV-B0`：官方 A* 参数扫描，不改算法。
2. `NAV-E1`：occupancy inflation 和距离衰减代价。
3. `NAV-E2`：路径简化后逐线段碰撞复核。
4. `NAV-E3`：工位 approach pose 与最终 yaw 分离优化。
5. `NAV-E4`：载荷状态使用更大 footprint。
6. `NAV-E5`：在零碰撞方案之间比较耗时。

### 抓取实验顺序

1. `GRASP-B0`：官方 checkpoint 和固定 base pose。
2. `GRASP-E1`：只修正观测/动作和渲染一致性。
3. `GRASP-E2`：base XY/yaw 小扰动数据采集。
4. `GRASP-E3`：BC-RNN 与官方 BC 的等数据比较。
5. `GRASP-E4`：失败状态专家标注/有限 DAgger。
6. `GRASP-E5`：数据充足后比较 BC-Transformer 或 Diffusion Policy。

每一步只改变一个主要变量。训练 loss 不能代替闭环抓取和整任务分数。

## P1：L5 多对象 workflow

每件对象维护独立状态：

```text
pending -> approached -> grasped -> lifted -> transported -> placed -> verified
```

失败只回退到最近可恢复状态，并设置每阶段最大重试。完成对象立即持久化，后续导航和放置必须把已放置对象视为不可碰撞区域。对象顺序先按路线成本和抓取可靠性排序，再用消融实验确认。

退出标准：三件对象都有匹配的成功抓取事件，全部到达目标，整段轨迹无碰撞，官方 30/30。

## P1：原创 SOP 流水线

建议中间表示：

```json
{
  "task_id": "L1",
  "object_count": 1,
  "pick_station_label": "Pick Station 2",
  "place_station_label": "Place Station 3",
  "ordered_steps": [],
  "safety_constraints": [],
  "evidence": [],
  "confidence": {}
}
```

文字、表格、图片和关系文件分别解析；VLM 只能填 schema，不能直接发控制动作。随后用 Erratum、semantic map 和允许的实体集合校验。生成物记录输入哈希、模型/API 版本、prompt 版本和逐字段证据。

## P2：创新增强

- 约束保持数据增强：扰动 base/object 后验证抓取几何和动作边界。
- 轻量候选 verifier：在多个 approach/path/action 候选间按安全和完成概率排序。
- 自适应安全裕量：空载、载荷、靠近工位使用不同 footprint 和速度。
- 失败驱动数据闭环：按失败阶段选择补采数据，而不是无差别扩充。

创新必须同时报告性能收益、额外计算成本、失败情况和消融结果。

## 暂缓事项

- 端到端 VLA 微调；
- GR00T/Isaac Sim 全栈迁移；
- ROS 2/Nav2 整体接入；
- 未经许可复核的 MimicGen/AnyGrasp；
- 修改 `app.py`、core、environment 或 `task_config.json` 来让结果通过。
