# 资料与证据索引

访问日期均为 2026-07-22。优先列官方页面、官方代码、项目主页和论文原文；社区 fork 单独标注，不能替代官方规则。

## 比赛一手资料

- [Biendata 比赛主页](https://www.biendata.net/competition/jciiot/)：报名入口、页面状态、队伍/参与人数与奖金；2026-07-22 补充的规则正文已保存为[本地规则快照](../research/rules-snapshot-2026-07-22.md)。
- [主办方赛事公告（智源社区转发）](https://hub.baai.ac.cn/view/56351)：报名/提交/评审日期、奖金、QQ 群、免费 GLM 额度和技术直播信息。
- [官方 GitHub 仓库](https://github.com/JCIIOT2026/JCIIOT2026)：场景、代码、SOP、评分器与更新记录。
- [固定版本选手手册](https://github.com/JCIIOT2026/JCIIOT2026/blob/f4ab8fd2158b919a41b2ce350432259cd1ee6a11/JCIIOT/README.md)：允许修改边界、SOP 原创性、地图、抓取训练和参数。
- [固定版本 Erratum](https://github.com/JCIIOT2026/JCIIOT2026/blob/f4ab8fd2158b919a41b2ce350432259cd1ee6a11/ERRATUM.md)：2026-07-22 的 Case 2/3 更正。
- [官方技术直播回放](https://www.bilibili.com/video/BV1uiK86QEWL/)：AI TIME 发布，时长 59:16；覆盖赛题、环境、baseline 演示和 Q&A。当前无公开字幕。
- [GLM-5V-Turbo 官方文档](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo)：模型输入、上下文、视觉定位、文档理解和函数调用能力。

## 官方仓库中应优先阅读的文件

本地路径都位于 `vendor/JCIIOT2026`：

- `JCIIOT/README.md`：完整选手手册。
- `ERRATUM.md`：最新 SOP 更正。
- `competition description/'RunningRobot'+Competition+Description.docx`：比赛描述。
- `competition description/'RunningRobot'+Competition+FAQ.docx`：FAQ。
- `competition description/sop+prompt/*.docx`：五份原始 SOP，自动解析的唯一正确输入。
- `JCIIOT/app.py`：实际评分逻辑。
- `JCIIOT/knowledge/task_config.json`：关卡、内部工位和对象映射，只读。
- `JCIIOT/knowledge/robot_params.json`：允许修改的运行参数。
- `JCIIOT/src/robot_agent/skills/`：允许修改的技能实现。
- `JCIIOT/src/robot_agent/workflows/`：允许修改的流程实现。
- `JCIIOT/robosuite/robosuite/environments/factory_sorting/maps/`：原始场景地图。
- `JCIIOT/robosuite/robosuite/environments/factory_sorting/generated_maps/`：语义地图与 occupancy grid。

## 可复现技术资料

### VLA 与模仿学习

- [OpenVLA-OFT 论文](https://arxiv.org/abs/2502.19645)、[官方代码](https://github.com/moojink/openvla-oft)、[项目页](https://openvla-oft.github.io/)：并行解码、action chunking、连续动作与 L1 回归。
- [SmolVLA 官方发布](https://huggingface.co/blog/smolvla)：450M、LeRobot、flow matching 和异步推理。
- [NVIDIA Isaac GR00T 官方仓库](https://github.com/NVIDIA/Isaac-GR00T)、[硬件建议](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)：N1.7 与自定义机器人后训练。
- [π0.5 项目说明](https://www.pi.website/blog/pi05)、[论文](https://www.pi.website/download/pi05.pdf)：开放世界移动操作与分层高层子任务预测。
- [Xiaomi-Robotics-1 论文](https://arxiv.org/abs/2607.15330)、[项目页](https://robotics.xiaomi.com/xiaomi-robotics-1.html)：2026-07 新工作；代码/权重尚未可用。
- [BAKU 论文](https://arxiv.org/abs/2406.07539)：多任务机器人 Transformer 与 action chunking。
- [Diffusion Policy 官方代码](https://github.com/real-stanford/diffusion_policy)：扩散式视觉运动策略。
- [MimicGen 项目页](https://mimicgen.github.io/)：从少量人工示教自动生成大规模操作数据。
- [OK-Robot 论文](https://arxiv.org/abs/2401.12202)：开放词汇移动操作的系统集成经验。
- [Colosseum V2 论文](https://arxiv.org/abs/2605.27759)：VLA 在系统性分布变化下的评测。

### 任务规划与恢复

- [LLM as BT-Planner](https://arxiv.org/abs/2409.10444)：LLM 生成和维护机器人行为树。
- [H-AIM](https://arxiv.org/abs/2601.11063)：LLM、PDDL 和行为树分层执行。
- [Multimodal Behavior Tree Generation](https://arxiv.org/abs/2603.06084)：从多模态任务描述生成 BehaviorTree.CPP 结构；2026 预印本。

### 导航与运动规划

- [Nav2 Smac Hybrid-A*](https://docs.nav2.org/configuration/packages/smac/configuring-smac-hybrid.html)：代价感知 Hybrid-A*、转弯半径和启发式缓存。
- [Nav2 Inflation Layer](https://docs.nav2.org/configuration/packages/costmap-plugins/inflation.html)：障碍物膨胀与距离衰减代价。
- [Nav2 MPPI Controller](https://docs.nav2.org/configuration/packages/configuring-mppic.html)：SE(2) footprint 碰撞与近碰惩罚。
- [Nav2 Concepts](https://docs.nav2.org/concepts/index.html)：动态 footprint 等架构概念。
- [cuRobo 官方仓库](https://github.com/NVlabs/curobo)、[cuRobo 2.0 论文](https://arxiv.org/abs/2603.05493)：GPU 碰撞检查、IK 和轨迹优化。

## 社区线索，非官方证据

- [QiShengZhao fork](https://github.com/QiShengZhao/JCIIOT2026)：L1 固定计划、BC/DAgger 和 headless 调试线索；修改过评分器，自报成绩不可直接比较。
- [jiangzizi fork](https://github.com/jiangzizi/JCIIOT2026)：headless 运行和文档导出线索。
- 官方仓库根目录 Leaderboard：`SOP-MapGuard 100/100` 只有未经核验的自报主观分，无代码和证据。

## 使用原则

- 比赛事实以固定版本官方代码和后续组委会明确答复为准。
- 论文中的作者自报数字只用于技术筛选，最终决策以本赛题重复实验为准。
- 新增来源时记录访问日期、版本/提交、它支持的具体判断以及是否已经复现。
- 网页动态正文无法由公开视图抓取时，保留带日期的人工摘录，并把“原文规则”和“工程解释”分开存放。
