# ADR-0004：物理证据门槛优先，训练推迟到有效 Tiago 数据之后

- 状态：accepted
- 日期：2026-07-28
- 决策人：JCIIOT 2026 参赛团队
- 关系：细化 ADR-0003，不替代其对象族几何主策略

## 背景

公开固定场景基线可以获得未修改公开评分器 100/100，但部分运输仍依赖 attachment 相对状态，不能据此宣称真实抓取搬运。进一步实测发现：官方 checkpoint 五关 seed 0 为 0/11 物理成功；示例 HDF5 是 Fetch/iGibson 的 10 维动作，不兼容 Tiago 的 20 维动作；L1 端壁抓取虽能抬升，但运输滑落，固定腕姿的支撑、重抓和推移诊断均未通过 hard gate。

## 候选方案

1. 继续优化固定场景 attachment/qpos 路线，以公开评分器分数为唯一目标。
2. 立即用示例 HDF5 或失败轨迹训练 BC-RNN、BC-Transformer 或 Diffusion Policy。
3. 先通过真实物理硬门槛：腕姿对齐的几何控制器或独立桌面推拖路线产生成功 Tiago 轨迹，再进行单轨迹 overfit 和学习策略对照。

## 决定

采用方案 3。

- 任何候选必须在原始仿真轨迹中证明物理接触、抬升、运输和放置；禁止 task-object pose write 和 transport attachment。
- L1 首先在高位无接触区重定向双腕，使 Robotiq 闭合轴对齐箱体中段侧壁法向。
- 若腕姿路线不可达或不安全，转入单独的桌面分段推拖 gate，不与抓取成功混记。
- 单次 gate 通过只允许继续复验；连续通过两次之前不替换当前 8502 服务，也不形成成绩声明。
- BC-RNN 与 Diffusion Policy 对照推迟到自采 Tiago 成功数据通过 schema、回放和单轨迹 overfit 后。

## 理由

- 当前瓶颈是抓取几何和接触力学，不是缺少更大的模型。
- 不兼容 HDF5 训练只能制造无法复现的结果，不能解决 Tiago 控制接口问题。
- 物理 gate 能把真实改进与评分器/状态同步捷径分开，符合提交可复现和创新评审要求。
- 失败轨迹仍可用于安全约束和难例分析，但不能冒充成功示范。

## 后果

- 短期内保留公开固定场景 100/100 作为历史基线，但不称真实物理满分或最终候选。
- 训练启动时间后移，计算资源先用于姿态可达性、接触和碰撞验证。
- 需要新增腕姿控制、关节可达性和两次重复 gate；若均失败，必须明确切换到推拖路线，而不是继续位置参数扫描。

## 证据

- `autoresearch/classic-260728-grasp-baseline/conclusion.md`
- `autoresearch/classic-260728-hdf5/conclusion.md`
- `autoresearch/classic-260728-1443/conclusion.md`
- `autoresearch/classic-260728-l1-physical/conclusion.md`
- `experiments/experiment-log.csv`
- research implementation commit `694bd99`

## 复审条件

- 组委会明确要求或禁止某类物理推拖操作；
- 官方发布兼容 Tiago 的新成功示范或 checkpoint；
- 腕姿重定向通过 0.13 m 抬升和 0.50 m 接触保持运输门槛；
- 可靠遥操作能够提供足量、可回放的比赛专用成功数据。
