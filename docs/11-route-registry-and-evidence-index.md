# 路线注册表与证据索引

更新时间：2026-08-05（Asia/Shanghai）

这份文档是当前工作区关于“任务一是否满分、是否使用 attachment、结果能否复核”的唯一索引。其他实验目录保留历史过程，但不能覆盖这里的路线分类和证据状态。

## 结论先行

任务一确实存在一条**不使用 transport attachment 的官方 10/10 路线**：

- 路线 ID：`L1-PD-FLOOR-64797D3`
- 工作区/候选提交：`64797d3941f7899c7fee173941097d8f8e9ea593-compliant`（短名 `64797d3-compliant`）
- 官方上游锁定：`0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- 未修改官方 `app._score_steps(0)`：`10/10`
- 碰撞帧：`0`
- attachment 调用/激活：`0`
- 物体位姿写入：`0`
- 轨迹帧：`14,299`（完整 iter33/iter34 证据）
- 物体真实接触推运步数：`6,985`
- 离开源工位：`dx=7.37 m, dy=11.98 m`
- 到目标桌中心：`0.748201 m`（官方阈值 `<0.8 m`）

这里的“纯物理”必须按证据边界表述：箱体从抓取、放置到地面推运均由 MuJoCo 接触和重力产生，没有 attachment 或物体位姿捷径；底盘跨场景导航使用了 direct base-qpos 仿真导航抽象。因此它是“无 attachment 的物体真实物理 + 官方评分器满分”，不是完整移动底盘执行器动力学的证明。这个边界不影响它作为当前任务一的无 attachment 满分 incumbent，但提交报告不能把它夸大为全系统纯动力学。

## 路线分类

| 路线 ID | 机制 | 官方客观分 | 完整性证据 | 当前用途 |
|---|---|---:|---|---|
| `L1-PD-FLOOR-64797D3` | 双臂物理抓取/抬升，物理放到地面，底盘与箱体接触推运到目标 | **10/10** | 无 attachment、无物体位姿写入、零碰撞；完整事件账本 | **任务一正式无 attachment 基线** |
| `L1-ATTACH-94DB515` | 物理抓取后使用官方 transport attachment 同步运输 | 10/10（本地） | 物理抓取前置成立，但运输依赖 attachment | 旧提交/对照，不作为纯物理路线 |
| `L1-PHYS-AERIAL-V16` | 尝试全程悬空、侧壁夹持和真实底盘移动 | 0/10（最终目标未到） | attachment=0、位姿写入=0、零碰撞；只证明局部物理承重 | 研究支线，不是满分基线 |
| `L1-SCRIPTED-OSC-V1` | 确定性抓取与官方 attachment 运输 | 10/10（多次本地重复） | 结果稳定，但运输不是无 attachment | 历史 attachment 对照 |

## `L1-PD-FLOOR-64797D3` 证据链

1. 评分回放：`autoresearch/classic-260802-supported-transport-inventory/official-score-replay.json`。评分函数是官方 `app._score_steps(0)`，评分器文件未修改，两个独立轨迹均为 `10`。
2. 路线矩阵：`autoresearch/classic-260802-supported-transport-inventory/results.tsv`。其中 `floor-push-target` 和 `floor-push-target-repeat` 的 `attachment_calls=0`、`object_pose_writes=0`、`collision_frames=0`。
3. 完整资产清单与 SHA-256：`autoresearch/classic-260802-supported-transport-inventory/asset-manifest.md`。
4. 完整轨迹本地归档目录：`/Users/wangpeng/jciiot-2026-assets/physical-floor-push-20260730/`。
   - `trajectories/iter33-floor-base-route-safe-turn.jsonl`
   - `trajectories/iter34-floor-base-route-seed1.jsonl`
   - `iter33-task1-10of10-robot0_robotview.gif`
   - `iter33-task1-contact-sheet.jpg`
5. 任务一官方结果摘要：`/Users/wangpeng/Downloads/JCIIOT_任务一_10分_严格合规_官方结果.json`。该文件记录 `official_score=10`、`collision_frames=0`、`successful_grasp_events=2`、最终距离 `0.7481709 m`，并绑定 `64797d3-compliant`。

### 事件级物理检查

iter33 完整轨迹的事件账本包含：

- 两次 `grasp_start/grasp_end`，左右接触和 `lift_success` 均为真；
- 两段 `inchworm_transport`，随后 `physical_place`；
- `floor_corridor_push_start`；
- 三段 `floor_base_push_segment`，累计 `6,985` 个物理接触步；
- `floor_corridor_push_end`，最终目标距离 `0.7482008 m`。

对完整 JSON 的字符串审计结果为：不包含 `attachment`、`object_pose_write` 或 `object qpos`。这比只看最终 JSON 分数更可靠，因为分数本身不能证明运输机制。

### 提交包和可视化产物

- 五关预测包：`/Users/wangpeng/jciiot-2026-deliverables/JCIIOT2026_validation_predictions_20260801_L1-pure-dynamics.zip`
- 其中 L1 文件：`FactorySorting1_3FO3ERFHISEM.json`
- L1 可视化轨迹 SHA-256：`c86808e4b72db7ba5d25621b14cbc959c1dcd25ed5f838cddd97155c57f331dc`
- 第一视角 GIF：`/Users/wangpeng/Downloads/JCIIOT_任务一_10分_严格合规纯物理_第一视角.gif`

五关 ZIP 中 L2-L5 仍是当时的 attachment 基线，不能因为文件名含有 `pure-dynamics` 就把整个五关 ZIP 声称为五关无 attachment。当前只有 L1 已被这条证据链证明为无 attachment 满分。

## 工作区导航规则

- 查任务一“无 attachment 满分”：先看本文件、`official-score-replay.json`、`results.tsv` 和 `asset-manifest.md`。
- 查五关旧本地满分：看 `docs/09-current-route-and-optimization-plan.md`、`experiments/2026-07-28-five-level-performance-baseline.md`，并明确标注 attachment。
- 查全程悬空纯物理研究：看 `autoresearch/classic-260802-l1-physical-rim-carry/`；它没有替代 `L1-PD-FLOOR-64797D3`。
- 任何新路线进入正式提交前，必须同时满足：官方评分器结果、碰撞统计、attachment/物体位姿写入审计、完整轨迹 SHA-256，以及明确的导航状态边界。
- 不允许用 GIF 单帧、JSON 自报字段或最终坐标单独宣称“真实抓取”或“纯物理”。

## 当前决策

任务一的默认 incumbent 恢复为 `L1-PD-FLOOR-64797D3`。后续任务一优化只有两个方向：

1. 在不改变该证据链的前提下，把路线接入允许目录内的官方 Agent 入口；或
2. 以同样的硬门禁取得真正的悬空物理运输，并且官方评分仍为 10/10 后再替换 incumbent。

在这两个条件都没有发生前，不再用 attachment 路线覆盖任务一的无 attachment 满分记录，也不把 `L1-PHYS-AERIAL-V16` 的 0/10 研究结果当成回归。
