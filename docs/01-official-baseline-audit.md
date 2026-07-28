# 官方基线审计

审计对象：官方仓库提交 `f4ab8fd2158b919a41b2ce350432259cd1ee6a11`（2026-07-22）；2026-07-26 增量再审计至 `f948609f2f281176272287fa991fce96d1f9ff98`；2026-07-27 同步至 `0dcdddf18a9e694569aa1433cdfc04eb097fed78`。增量见文末补遗。

## 架构

```text
SOP DOCX + 图片
    -> ReadDocumentSkill / VLM
    -> Markdown 知识库
    -> LLM Planner 生成结构化计划
    -> move / pick_up / place_down / record_trajectory
    -> MuJoCo + robosuite + robomimic
    -> trajectory JSON
    -> app.py 评分
```

这是“语言规划器 + 确定性技能 + BC 抓取”的混合系统，不是严格意义上的端到端 VLA。

## 各模块现状

### SOP 读取

`read_document.py` 用 `python-docx` 提取段落，对每张内嵌图片调用 VLM。现有图片提示主要要求描述工厂布局，没有强制 JSON schema、页面/图片证据、字段置信度或跨模态一致性校验。它可以演示能力，但不够适合评审和稳定执行。

### 任务规划

Planner 让 LLM 输出严格结构化 JSON，但示例主路径被压成固定四步：`move -> pick_up -> move -> place_down`。这对 L1-L4 合理，对 L5 的三物体任务不够：L5 至少需要显式的三次抓放循环、每件物体的状态追踪和局部失败恢复。

### 导航

`MoveSkill` 在静态 occupancy grid 上运行 A*，路径简化后交给后端跟踪。默认参数包括 `max_steps=3000`、`waypoint_tolerance=0.01`、`max_linear=0.70`、`max_angular=1.20`、全向底盘、`drive_mode=direct`、20 Hz。

主要缺口：

- 成本只表达可走/不可走，缺少离障碍物距离代价；
- 没有根据抓取后的载荷/机械臂外形扩大 footprint；
- 路径简化可能把安全余量吃掉；
- 接近工位的最终位姿和朝向没有作为独立约束优化；
- 默认追求几何短路，不等于碰撞风险最小。

### 抓取

`PickUpSkill` 调用 robomimic BC policy，再做 0.15 m 抬升验证。默认 checkpoint 为 `model_epoch_500.pth`，回退到 `model_epoch_150.pth`，执行 360 步。官方只提供 L1 演示采集脚本，其他四个场景需要自行扩展。

最大风险是训练/评估观测一致性、每个工位的准确 base pose、对象名称映射与模型分布外泛化。公开 fork 已发现无头运行和 `sim.forward()` 初始化问题，但这些修复位于官方禁止修改的 environment 目录，不能未经组委会许可直接作为提交改动。

### 放置

放置使用转向、下降、释放的物理过程。当前评分只看最终 XY 距离，不直接检查 Z 或是否稳定落桌。策略仍应把物体放在目标桌中心并等待稳定，因为异常穿透/滑落可能改变最后一帧或触发碰撞。

### 评分器

当前 `app.py` 的实际行为：

- L1-L4：必须先找到匹配源工位/物体且 success 为真的 `grasp_end`；相对源工位中心 X 或 Y 位移大于 1 m得一半；最终 XY 到目标中心小于 0.8 m 得一半。
- L5：三个白色 tote 分别计 5 分离开、5 分到达；需要各自成功抓取；全任务任意碰撞只扣一次 5 分。
- 碰撞判定来自轨迹帧 `has_collision`；总分下限为 0。
- 时间只用于同分排序。

代码注释与旧说明中仍残留过时口径，正式实验必须固定提交并用未修改官方评分函数复算。

## 高价值发现

1. **L5 是最大结构性缺口。** 默认四步计划无法自然完成三个对象，必须在允许修改的 workflow/skills 中显式循环，并记录每个对象的成功状态。
2. **零碰撞优先于最短时间。** 一次碰撞扣 5 分，足以抹掉 L1 的半项得分；时间只在同分时起作用。
3. **SOP 解析是 40% 创新分的好入口。** 官方要求自动生成知识库，且默认解析器缺少 schema、证据和校验，改进空间大而风险低。
4. **目标容差较宽。** 0.8 m 是 XY 阈值，优先稳定放到目标中心，不需要为毫米级终点牺牲整体可靠性。
5. **精确对象名是硬依赖。** 每关对象名和工位映射已写入 `task_config.json`；SOP 解析结果必须映射到这些内部实体。

## 公开 fork 审计

截至 2026-07-22，官方仓库有 8 个公开 fork。大多数只是同步上游；两个有实质改动：

- [`QiShengZhao/JCIIOT2026`](https://github.com/QiShengZhao/JCIIOT2026)，审计提交 `95fa2bed...`：增加 L1 固定计划、BC/DAgger 脚本和无头运行修复，自报 L1 10/10。它同时修改 `app.py`，移除了抓取成功门控并改为相对出生点计分，因此该自评分不能与当前官方评分器直接比较。可借鉴 `headless`、`sim.forward()`、图像上下翻转一致性等诊断线索，不能把分数当作已复现结果。
- [`jiangzizi/JCIIOT2026`](https://github.com/jiangzizi/JCIIOT2026)，审计提交 `d2d86270...`：主要增加无头运行与文档导出；没有可核实的新客观成绩。

官方根目录 Leaderboard 的 `SOP-MapGuard 100/100` 明确标为参赛者自报主观分，没有方法、代码、证据或官方复现，不应作为 SOTA。

## 环境注意事项

- 官方推荐 Python 3.11，并需安装项目自身、内嵌 robosuite 及 requirements。
- 必须按根 `requirements.txt` 固定 `mujoco==3.9.0`。仅安装 robosuite 的宽松依赖会得到 3.10.0，其 `mj_fullM` Python 绑定签名已经变化，官方控制器会在环境构造阶段报 `TypeError`。
- 官方 `.pth`、`.hdf5`、`.zip` 由 LFS 管理，轻量快照中的指针不能替代权重。
- macOS 的 MuJoCo 可视 viewer 与普通 Python 子进程可能冲突；批量实验宜使用正确配置的 offscreen 渲染或 Linux GPU 环境。
- 官方参考硬件是 Linux + RTX 4090 级别。正式复现应记录操作系统、CPU/GPU、驱动、MuJoCo 后端与随机种子。

## 2026-07-26 增量再审计（f4ab8fd → f948609）

改动范围：`app.py`（+73/-31）、`knowledge/task_config.json`（+28/-12）、`skills/pick_up.py`（+14/-2）、`task_subprocess_runner.py`（+14/-2）。`ERRATUM.md`、`README.md`、`robot_params.json` 无变化。

1. **候选物体机制**：每关 `object` 改为数组；`_object_name_matches` 对候选做双向子串匹配，空名或空列表视为通过。成功 `grasp_end` 事件记录的物体名优先作为计分对象；无事件时兜底“距目标最近”只在候选内选取。抓取门控本身没有放松，先前“两种离开口径都要满足”的策略保持不变。
2. **L3 目标物体更换**：`orange_tote_b01_upper` → `blue_tote_b01_far_right` / `blue_tote_b01_near_right`（源/目标工位不变）。旧物体名的固定计划、抓取位姿和 SOP 笔记必须迁移。
3. **技能侧取主物体**：`pick_up.py` 与 `task_subprocess_runner.py` 用 `_primary_object_name` 取数组第一项，意味着官方管线默认仍抓第一个候选；把候选选择做成决策点（按可达性/抓取可靠性挑选）是合规的差异化空间。
4. 注释乱码修复无行为影响。
5. **LFS 状态已恢复**：2026-07-28 重新访问时，用户列出的 11 个 LFS 对象均可完整下载，字节数和 SHA-256 与 Git LFS pointer 一致。旧的“配额仍超限”结论不再成立，但本地仍按提交 `0dcdddf` 和对象哈希缓存，避免上游再次不可用。
6. **公开 checkpoint 不是可靠现成策略**：安全解析显示 `model_epoch_150.pth` 实际保存 epoch 500 的 L1 Tiago 20 维 BC-Transformer，训练记录 `best_success_rate=0.0`。服务器用未修改官方 evaluator 在其 L1 固定 reset 重复三次，结果 0/3，夹爪无接触。
7. **示例 HDF5 不是赛题数据**：`table_setup_from_dishwasher_sample.hdf5` 只有 5 条 Fetch/iGibson 厨房整理演示，动作维度 10；JCIIOT Tiago 为 20 维动作。该文件只能参考 robomimic 格式，不可直接训练赛题策略。
8. **USD/mesh 的用途**：五个 USD 包是 `world.usda` 加视觉资产，不含动作演示；四个 mesh 包是桌面修订历史，当前源码中 44 个相关 OBJ 已与 v5 包逐字节一致，不应再覆盖解压。完整证据见 `research/notes/official-assets-and-score-reality-audit-2026-07-28.md`。

## 2026-07-27 增量再审计（f948609 -> 0dcdddf）

改动范围：`knowledge/task_config.json`、FactorySorting5/9 两份生成语义地图和 `robosuite_backend.py`，共 4 个文件。

1. **L3 改用辅助输入桌**：源工位由 `input_6` 改为 `aux_input_1`。新工位中心 `[0.144, 8.473]`，接近点 `[0.11, 7.55]`，包含两个 blue tote 候选。
2. **L5 改用辅助输出桌**：目标工位由 `output_6` 改为 `aux_output_1`，中心和接近点与 L3 辅助桌相同。L5 仍需搬运三个 white tote。
3. **后端初始化顺序调整**：启用物理策略时，`_ensure_physics_policy()` 从导航环境 reset 之后提前到导航环境构造之前。该文件属于禁止修改目录，只记录并使用官方版本。
4. 官方 `grasp_poses` 没有增加 `aux_input_1`；后端可以使用导航后传入或当前 base pose，但 L3 抓取策略是否适配新辅助桌必须通过实际闭环验证，不能仅凭任务配置判定可用。

## 2026-07-27 远程场景冒烟

在 Ubuntu 24.04.3、Python 3.11.15、MuJoCo 3.9.0 和官方 robosuite 1.5.2 环境中，固定 seed `20260727`，五个场景都成功完成构造、reset 和一个零动作 step，action shape 均为 20。该实验不加载 checkpoint，不启用 renderer，不执行任务 workflow，也不调用评分器；它只证明普通 Git 场景资产和基础动力学链可用。完整记录见 `experiments/2026-07-27-scene-smoke.md`。
