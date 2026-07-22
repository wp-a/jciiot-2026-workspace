# 2025-2026 技术地图

调研截止：2026-07-22。这里只收录与本赛题可落地性有关的技术；新论文的作者结果均视为待独立复现，不把“最新”等同于“最适合比赛”。

## 技术判断

本赛题的场景、工位、地图和对象集合都是已知的，客观评分只关心完成度、碰撞和最后位置。端到端 VLA 的训练成本、接口适配和随机失败，暂时大于它带来的泛化收益。主线应是可验证的混合架构：VLM 负责 SOP 多模态理解，结构化/行为树负责长程编排，几何规划负责移动，BC 或轻量策略负责局部抓取。

## 第一优先级：直接提升比赛表现

### 1. 有证据的 SOP Grounding

使用 GLM-5V-Turbo 或同级 VLM，把 DOCX 文本和图片解析为类型化 schema，而不是自由文本摘要。建议字段：

```text
task_id, object_count, object_visual_description,
pick_station_label, place_station_label,
ordered_steps, safety_constraints,
evidence(page, image, text_span), confidence
```

随后把人类工位名称映射到 semantic map 实体，并用任务图、Erratum 和地图坐标做一致性检查。GLM-5V-Turbo 官方文档支持图像/视频/文件、视觉定位、文档理解、函数调用、200K 上下文，且比赛提供额度；它适合做这层语义前端，但输出仍必须通过 schema 和地图验证。

### 2. 行为树/契约式任务编排

把 LLM 产出的意图编译成确定性 workflow 或行为树，每个节点带前置条件、后置条件、超时和有限重试：

```text
Locate object -> Navigate to approach -> Verify pose -> Grasp
-> Verify lift -> Navigate with payload -> Place -> Verify final state
```

L5 对三件物体循环执行，并在每件完成后持久化状态。2024-2026 的 LLM-to-Behavior-Tree、LLM-to-PDDL-to-BT 工作说明了这种分层路线，但比赛中无需引入完整 ROS/BT.CPP 依赖；关键是可检查的状态机语义。

### 3. 带安全余量的导航

从 Nav2 的成熟设计中借用概念，而不是整体移植 ROS2：

- 对 occupancy grid 做 inflation / distance transform；
- A* 代价加入离障碍物距离惩罚；
- 携带物体后使用更大的 footprint 和碰撞边界；
- 路径简化后的每条线段重新做连续碰撞检查；
- 工位接近点单独优化最终位置与 yaw；
- 先获得零碰撞稳定路线，再做速度和路长优化。

Hybrid-A*、MPPI 的 footprint/collision critic 是有用参考，但当前全向底盘和静态场景未必需要完整实现。

### 4. 抓取数据闭环

先修复 base pose、对象映射、观测方向和 lift verification，再扩展 L2-L5 演示采集。数据不足时优先采用：

- 场景参数扰动与示教轨迹增广；
- MimicGen 式基于子任务变换的自动数据生成；
- robomimic BC 的场景专用/对象专用 checkpoint；
- 失败样本回放与有限 DAgger，而不是不受控在线学习。

## 第二优先级：可作为创新支线

| 技术 | 2026 状态 | 与本赛题的关系 | 结论 |
|---|---|---|---|
| OpenVLA-OFT | 论文/官方代码可用；并行解码、action chunking、连续动作 | LIBERO 上作者报告 76.5% -> 97.1%、26x 吞吐；训练通常需要 4-8 张 A100/H100，推理约 16 GB | 有数据后做研究支线，不做首周主线 |
| SmolVLA | 450M、LeRobot、flow matching、异步推理 | 消费级硬件友好，适合验证轻量 VLA；Tiago 观测/动作仍需对齐和数据 | 最现实的轻量 VLA 备选 |
| NVIDIA GR00T N1.7 | 2026-04 早期访问，3B，支持自定义机器人后训练 | 官方最低推理建议 16 GB GPU，但生态更偏人形机器人，Tiago 接入成本高 | 监测，不先集成 |
| π0.5 | 2025 开放世界移动操作研究 | 分层高层子任务预测很有启发，但不是可直接替换的比赛组件 | 借鉴架构，不直接落地 |
| Xiaomi-Robotics-1 | 2026-07-16 新预印本；代码/权重尚称“将发布” | 宣称面向移动操作和高效适配，但距离可复现太近 | 仅技术观察 |
| cuRobo 2.0 | 2026-04 官方代码；GPU IK、碰撞检查、轨迹优化 | 可增强机械臂局部规划，但 CUDA 集成和允许修改边界需确认 | 仅在抓放已成为瓶颈时评估 |
| Diffusion Policy / BAKU | 成熟模仿学习参考 | 对多峰动作与 action chunking 有价值，训练数据仍是先决条件 | BC 不足且数据充足时尝试 |

## 研究证据给出的反向提醒

- OK-Robot 的经验支持“强基础模型 + 简单状态机 + 可靠技能”的系统路线，而非把所有控制交给单一大模型。
- Colosseum V2 等 2026 基准继续显示 VLA 在分布变化下会明显退化。五个固定工厂场景中，几何先验和显式状态不是负担，而是优势。
- 最新预印本的自报结果不能替代本赛题轨迹和官方评分器的重复实验。

## 推荐创新叙事

最适合 40% 专家评分、同时能转化为客观分的创新组合是：

1. **可追溯多模态 SOP 编译器**：逐字段证据、置信度、Erratum/地图交叉校验。
2. **契约式长程执行器**：行为树/状态机、三对象循环、闭环验证和有限恢复。
3. **载荷感知安全导航**：footprint 膨胀、clearance 成本、路径后验碰撞检查。
4. **场景高效抓取适配**：少量示教 + 自动增广 + 统一重复评测。

这比“换一个更大的 LLM”更容易形成清晰 Novelty Statement，也更容易用消融实验支撑。
