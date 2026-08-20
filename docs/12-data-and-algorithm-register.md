# 数据与算法登记册

更新时间：2026-08-15

## 1. 赛题真正考察什么

JCIIOT 不是单一模型准确率比赛，而是已知 MuJoCo 工厂环境中的移动操作系统集成：

1. 从 SOP、任务配置和场景状态识别源工位、目标工位、候选物体和顺序。
2. 生成无碰撞的底盘接近路径与双臂抓取姿态。
3. 用真实接触完成双侧抓取、抬升、运输和释放。
4. 对碰撞、滑移、掉落、目标距离和任务状态做可复核验证。
5. 提交可复现代码、轨迹、报告和创新证据。

官方公开评分只检查抓取事件、离开源工位、目标距离和碰撞，因此公开 `10/10` 不能单独证明全程真实抱持。当前项目的严格提交口径是：`attachment=0`、物体位姿写入=0、地面/桌面推运=0、碰撞=0，并且运输阶段持续双侧接触和抬升。

## 2. 当前数据

| 数据集 | 规模 | 形状 | 用途与限制 |
|---|---:|---|---|
| 官方 `table_setup_from_dishwasher_sample.hdf5` | 历史远端审计为 5 demos | Fetch/iGibson，动作 `[T,10]` | 只作格式参考；当前本地文件只是 134-byte Git LFS 指针，不能在本机重新审计数组 |
| H2 native Tiago grasp | 14 demos，4065 steps | 状态 `[T,87]`，动作 `[T,20]`，图像 `[T,128,128,3]` | 真实比赛接口；只覆盖接近、接触、抬升窗口 |
| H5b recovery | 24 demos，6521 steps | 同上 | 覆盖左右接近漂移和单臂接触恢复；没有完整长途成功 teacher |
| H6 merged | 36 demos，9866 steps | 状态 `[T,87]`，动作 `[T,20]` | 离线训练集；动作裁剪严重，尚未通过闭环晋级 |
| AIST-Bimanip | 50 demos，25000 steps | ALOHA，动作 `[T,14]` | 可作视觉/阶段预训练；不能直接拼接原始动作 |

H2/H5/H6 的 demo 必须按完整 seed 划分 train/validation/held-out，不能按帧随机切分。每条合格物理轨迹还必须有事件账本、碰撞帧数、接触侧、最小抬升、真实物体位移、attachment 调用数、物体位姿写入数和轨迹 SHA-256。

### 2.1 数据是否真正落地

- 官方样例的本机检查报告为 `artifacts/data-audit-20260815/official-sample-inspection.json`：`materialized=false`，LFS 声明大小 591,069,600 bytes。表中的 5 demos / 10维动作来自此前远端审计记录，不是本轮重新读取的结果。
- H2/H5b/H6 的规模、形状、划分和哈希由 `experiments/`、`research-log.md` 与远端路径记录交叉支持；当前工作区没有对应的原始 HDF5 payload。因此它们可以支撑历史实验结论，但还不能由本机新审计器重新签发数据准入清单。
- H3/H4 的本地报告和闭环 JSON 已落地；大模型 checkpoint 仍登记为远端文件。任何复训前必须先同步原始 HDF5 和选定 checkpoint，并逐文件核验 SHA-256。

## 3. HDF5 推荐格式

```text
data/
  demo_000/
    obs/                 # float32 state or image observations
      state              # [T, 87]
      image              # [T, 128, 128, 3], uint8, optional for lowdim policy
    actions              # [T, 20], float32 normalized Tiago action
    timestamps           # [T], float64
    events               # JSON string or structured event table
    seed                 # scalar metadata
    object_name          # scalar metadata
    task_level           # scalar metadata
    integrity/            # shortcut and collision audit fields
      collision_frames
      attachment_calls
      object_pose_writes
      min_lift_m
      true_object_translation_m
```

每个 demo 必须包含 finite 数值、`T > 0`、动作宽度20、状态宽度87、时间单调递增、seed唯一且事件顺序为 `grasp_start -> grasp_end -> transport_start -> transport_end -> place_end`。严格物理 teacher 还必须满足：双侧接触、最小抬升至少0.13m、运输中无接触丢失、零碰撞、零 attachment、零物体位姿写入。

工作区审计命令：

```bash
python scripts/audit_physical_carry_hdf5.py DATASET.hdf5 \
  --output artifacts/data-audit/MANIFEST.json
```

审计器 fail closed：格式、动作/状态维度、非有限值、seed重复、split 泄漏、事件缺失或 shortcut 计数任一异常都会拒绝；抬升/位移/连续双侧接触未达标的无 shortcut 轨迹只进入 recovery，不得标成完整 teacher。

## 4. 当前算法登记

### 确定性主线

SOP/配置解析 → 对象候选排序 → A* / 安全走廊导航 → 对象族几何抓取 profile → 约束 IK + OSC → 渐进闭合 → 双侧接触与抬升门禁 → 物理运输 → 物理放置 → 轨迹审计。

这是当前唯一可以作为工程 teacher 的路线。它不依赖 LLM API 直接输出动作。

### BC-RNN

两层 LSTM，隐藏维度400，20维动作输出。H2 数据上离线 MSE 相对常数基线改善92.76%，但未见扰动闭环 `0/5`。失败原因是成功样本没有包含 learner-reachable 的偏移和恢复状态。

### Diffusion Policy

条件 UNet，预测 horizon 16、action horizon 8、20维动作。相同12条数据上闭环 `2/5`，优于 BC-RNN 但仍不能部署；输出裁剪和接触恢复缺失是主要问题。H6 的36条合并数据只通过离线检查，不能称训练成功。

## 5. 数据扩充顺序

1. 先取得一条 L1 完整真实悬空运输 teacher；没有成功 teacher 时禁止继续堆 epoch。
2. 对 teacher 做对象 XY、yaw、夹爪初始偏差、摩擦、质量和观测噪声分层扰动。
3. 专门收集 learner-state correction：左/右接近漂移、左/右单臂接触、抬升期滑移、转向期姿态误差。
4. 训练集按 seed 划分；恢复数据不能把同一条轨迹的相邻帧拆到不同 split。
5. 保持 H4 Diffusion 架构不变，只改变数据覆盖，做受控对照。
6. 只有十个全新闭环 seed 达到至少 `8/10`、零碰撞、零 shortcut，才允许学习策略进入局部抓取残差；它不能直接接管长距离运输。

## 6. 为什么当前一直抓不住

当前 L1 箱体碰撞几何只有 bottom/front/back/left/right 五个 box geom，视觉 mesh 的碰撞通道关闭；因此“从视觉把手插入”没有可用物理通道。同壁侧夹持主要依赖摩擦，底盘启动后切向载荷使箱体相对夹爪滑移。现有 41 条纯物理记录没有一条完成0.5m稳定悬空运输，所以问题首先是接触拓扑和底盘控制接口不匹配，不是训练轮数不足。

## 7. 禁止回归

官方入口只允许 `transport_mode="physical_carry"`。历史 `l1_floor_push` 和 `transport_attachment` 代码保留在实验归档中用于复盘，但不可由 `run_official_task` 调用。任何回归结果必须同时报告公开评分、碰撞、接触、真实位移和 shortcut 审计。
