# 2026-07-28 官方 Agent 入口五关验收

> 口径更正（2026-07-28）：这里的 Agent 和 scorer 来自锁定公开源码；100/100 是本地固定公开场景结果，不是 BienData 成绩或组委会复现。当前聚合 skill/planner gate 与私有 attachment 使用仍需提交前合规处理。

## 目的

此前满分轨迹由工作区 runner 直接调用 `run_official_task()` 生成，尚不能证明评委运行官方 `app.py` 时会进入同一方案。本实验验证干净物化候选通过未修改的 `RobotAgent.run()` 和官方 skill registry 执行方案 B。

## 接入方式

官方 Execute 调用链固定为：

```text
app.py -> task_subprocess_runner.py -> RobotAgent.run()
       -> skills/library.py -> registered skill
```

候选只在允许目录增加 `CompetitionTaskSkill`，并在 `skills/library.py` 中：

1. 使用官方已有的 `GATE_PLANNER=false`，避免评委现场 LLM 波动；
2. 将 `competition_task` 注册为首个 skill；
3. 根据 `RobotAgent` 传入的 `task_index` 只读官方 `task_config.json`；
4. 调用已经验证的 `CompetitionFlow`，再由官方 backend 录轨和官方 scorer 评分。

未修改 `app.py`、`task_subprocess_runner.py`、`core/`、`environments/` 或 `task_config.json`。

## 干净物化证明

- 官方源提交：`0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- 入口候选提交：`260839a7915c8327fcd2a2611b16053c582d5dc4`
- 新物化目录：`competition-entry-260839a`
- Git 差异：1 个允许目录内修改文件、4 个允许目录内新增文件

受保护文件的官方/候选 SHA-256 完全相同：

| 文件 | SHA-256 |
|---|---|
| `JCIIOT/app.py` | `4834b2ac4f6d97fe2c39462c6dabef3c4f52b8a357121f0a40507baaef9782eb` |
| `JCIIOT/knowledge/task_config.json` | `4ab0b19ed3599421f1de91aebdcc777d0bded83ed56cebe12f937cfc95d62331` |
| `JCIIOT/src/robot_agent/task_subprocess_runner.py` | `18283fb9fd776a45351400f844eb99019fd5cf78e3920161cc11ab23cc332860` |

## 五关结果

| 关卡 | 官方得分 | 抓取事件 | 验证对象 | 碰撞帧 | 最终最大距离 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 10/10 | 1/1 | 1 | 0 | 0.162809 m | 70.906 s |
| L2 | 15/15 | 1/1 | 1 | 0 | 0.407609 m | 65.040 s* |
| L3 | 20/20 | 1/1 | 1 | 0 | 0.333074 m | 68.378 s* |
| L4 | 25/25 | 1/1 | 1 | 0 | 0.328942 m | 101.424 s* |
| L5 | 30/30 | 3/3 | 3 | 0 | 0.600000 m | 286.648 s* |

总计 `100/100`、`7/7` 抓取事件、0 碰撞。每份 manifest 均满足：

- `runner.execution_mode=agent`；
- `execution_result.skill_name=competition_task`；
- `planner_raw` 为空；
- 选择原因为 `Planner disabled via GATE_PLANNER`；
- workflow 中所有对象最终状态均为 `verified`。

带 `*` 的 L2-L5 与其他关卡并行运行，因此时间不用于速度排名。

## 图形入口边界

服务器没有可访问的 X 显示：`:0` 探测失败且未安装 Xvfb。官方 `task_subprocess_runner.py` 默认创建可见 MuJoCo viewer，因此未在该服务器伪造图形点击实验。验证使用同一干净候选和未修改的 `RobotAgent.run()`，但 backend 设为 headless；这覆盖了实际 skill 注册、Agent 选择、控制、录轨和评分链，未覆盖 Streamlit 按钮与可见窗口本身。

在有图形桌面的官方环境中，`task_subprocess_runner.py` 会构造相同的 `RobotAgent`，其受保护文件哈希已证明未修改。

## 证据

- 本地证据目录：`artifacts/remote-entrypoint-20260728/entrypoint-260839a-20260728/`
- 原始证据 ZIP：`artifacts/remote-entrypoint-20260728/entrypoint-260839a-20260728-evidence.zip`
- ZIP SHA-256：`f8250f24367b26eb24974e79032636953c20fae8ef76f64da43694a511f63e21`
- `entrypoint-summary.json`：五关机器可读审计与受保护文件哈希
- `checksums.sha256`：11 个内部文件，远端与本地均验证通过
