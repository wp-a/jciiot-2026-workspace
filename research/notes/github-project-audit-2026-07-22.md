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

## 本地 checkout 验证

12 个选定仓库已下载到顶层工作空间的 `references/repos/`，总磁盘占用约 649 MB。所有 checkout 都是 detached、固定 commit、工作区干净；未初始化 submodule，未下载 Git LFS、模型和数据集。`scripts/check_references.sh` 已验证 12/12 的 origin 和 HEAD。

| 本地目录 | 大小 | 已检查的关键文件 | 本地代码审计结论 |
|---|---:|---|---|
| `robomimic` | 109 MB | `examples/train_bc_rnn.py`、`docs/tutorials/multi_dataset_training.md`、`robomimic/config/diffusion_policy_config.py` | `local-code-audited`；BC-RNN、多数据集加权和 rollout 直接相关，优先使用官方内嵌版本 |
| `pythonrobotics` | 1.5 MB | `PathPlanning/AStar/a_star.py`、`ThetaStar/theta_star.py`、`DStarLite/d_star_lite.py` | `local-code-audited`；代码是教学实现，含全局绘图和简单障碍遍历，只移植算法思想并针对 NumPy 栅格重写 |
| `py_trees` | 19 MB | `py_trees/composites.py`、`blackboard.py`、`decorators.py` | `local-code-audited`；带 memory 的 Sequence/Selector 和 Blackboard 适合长任务，但需与小状态机比较依赖成本 |
| `multimodal_bt` | 41 MB | `bt_checks/control_flow.py`、`bt_repair/llm_repair.py`、`vlm/object_mapping.py` | `local-code-audited`；已实现 XML parse、Retry/Timeout 参数检查、对象映射和修复回路，schema 思路可用 |
| `kios` | 174 MB | `data/examples/world_state.json`、`behavior_tree_skeleton.json`、`kios_plan/dynamic_planning.py` | `local-code-audited`；目标节点 + Selector + 前置条件 Sequence + Action 的结构非常适合显式恢复，完整服务栈不采用 |
| `mimiclabs` | 9.8 MB | `data_collection/sim/scripts/collect_data.py`、`mimicgen/scripts/prepare_src_dataset.py` | `local-code-audited`；任务配置、源示范采集、后处理、扩充三段流水线可参考，MimicGen 依赖仍受限 |
| `cpgen` | 176 MB | `demo_aug/configs/base_config.py`、`augmentor/augmentor.py`、`constraint_segmentation.py` | `local-code-audited`；支持 SE(3)、关节、外观和相机扰动及碰撞启发式，但有大量任务硬编码和未完成检查，不直接移植 |
| `ok_robot` | 96 MB | `ok-robot-navigation/a_star/path_planner.py`、`configs/path.yaml` | `local-code-audited`；终点选择同时考虑可达性、理想距离和避障很有价值；实现含硬编码距离和外部模型栈，不复制 |
| `act_plus_plus` | 2.6 MB | `policy.py`、`imitate_episodes.py`、`utils.py` | `local-code-audited`；ACT/Diffusion 和 temporal aggregation 清晰，但存在 Mobile ALOHA 假设及本机绝对路径残留 |
| `mobilemanibench` | 11 MB | `unimanip/rsl_ppo/train.py`、`record_parallel.sh` | `local-code-audited`；按机器人-对象-技能训练专用策略再批量记录的结构可借鉴，脚本依赖 Isaac Sim 和多 GPU |
| `robomonkey` | 3.3 MB | `monkey-verifier/src/infer_server.py`、`action_processing.py`、`scripts/env_verifier.sh` | `local-code-audited`；实际 verifier 依赖 LLaVA reward model、固定 token/action 编码和额外权重，不进入主线 |
| `jciiot_qisheng` | 5.2 MB | `run_l1_fixed_plan.py`、`collect_l1_dagger.py`、L1 设计文档 | `local-code-audited`；固定计划和失败分布诊断有参考价值，但实现与自评分超出官方允许边界，禁止复制 |

## 代码级收获

1. **robomimic 多数据集训练可以直接支持“专家示范 + 恢复示范”。** 每个 HDF5 可设置采样权重，并选择是否按数据集大小归一化。正式采用前必须保证所有数据的 observation/action space 和预处理一致。
2. **行为树应先做确定性校验，再考虑 LLM 修复。** multimodal-BT 对 XML、Retry 次数、Timeout 和 Parallel threshold 做静态检查；JCIIOT 可把同样原则应用到 JSON workflow，避免把格式错误留到仿真阶段。
3. **KIOS 的恢复模式比自由重规划更适合比赛。** Selector 的第一分支检查目标是否已满足，第二分支执行带前置条件的动作序列。这能自然避免 L5 已完成对象被重复搬运。
4. **OK-Robot 的终点不是直接使用语义目标坐标。** 它在可达点中平衡目标距离和障碍距离，再让机器人朝向真实目标。JCIIOT 的工位 approach pose 应采用同类分离设计。
5. **CP-Gen 的配置空间适合转化为我们自己的小型扰动矩阵。** 首阶段只需要 base XY/yaw、对象位置、EEF 小噪声和相机轻微扰动，不需要 NeRF、尺度/剪切变换或 cuRobo 全栈。
6. **RoboMonkey 完整 verifier 不具备比赛性价比。** 它把候选动作编码进视觉语言 reward model，并需要另行下载 7B 权重。我们应使用仿真可直接读取的碰撞、物体高度、夹爪和目标距离作为确定性 verifier。
7. **公开 fork 反而证明了先固定物理闭环的重要性。** 其 BC/DAgger 记录显示训练 loss 或示范拟合不等于闭环成功；任何抓取模型只以真实 rollout 和整任务官方分数验收。

## 尚未完成

- 没有下载模型权重、数据集或 Git LFS 资产。
- 没有在 JCIIOT 环境运行任何外部策略。
- 没有把任何作者自报指标标为本赛题已复现。
