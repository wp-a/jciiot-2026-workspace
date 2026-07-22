# JCIIOT 2026 工业具身智能挑战赛工作空间

调研快照日期：2026-07-22（Asia/Shanghai）

这是后续参赛工作的单一入口。当前结论是：这不是一个适合优先押注端到端 VLA 的开放世界赛题，而是五个已知 MuJoCo 工厂场景中的 SOP 理解、移动操作与可靠执行问题。最有胜率的主线是“多模态 SOP 结构化 + 可验证任务编排 + 安全导航 + 稳定抓放”，再用轻量学习方法增强抓取泛化。

## 当前最紧急事项

- 报名截止：**2026-07-24 23:59**。
- 作品提交截止：**2026-08-16 23:59**。
- 比赛主页顶部显示 2026-09-01 关闭，但主办方赛程明确写明 8 月 16 日作品截止；执行时按更早日期规划。
- 报名后加入官方 QQ 群 `726582909`，获取通知及 GLM-5.2、GLM-5V-Turbo 免费额度。
- 所有队员都要完成系统注册；队伍最多 5 人、名称不超过 15 个字符，并在截止前锁定名单和队长。
- 方案只能依赖公开发布、符合赛事许可要求的代码/数据/工具；获奖队伍须公开实现系统的全部代码。

## 从这里开始

1. [当前状态](STATUS.md)：当前阶段、阻塞、下一里程碑和更新规则。
2. [比赛简报](docs/00-competition-brief.md)：规则、赛程、任务和交付物。
3. [官方基线审计](docs/01-official-baseline-audit.md)：代码结构、评分实现、限制与已发现风险。
4. [最新技术地图](docs/02-technology-landscape.md)：2025-2026 可用技术及与本赛题的适配度。
5. [竞赛技术路线](docs/03-winning-strategy.md)：分阶段方案、实验门槛与优先级。
6. [同类项目调研](docs/07-similar-projects.md)：公开仓库、可复用内容、成本和许可证风险。
7. [模块路线图](docs/08-module-roadmap.md)：SOP、编排、导航、抓放、恢复、数据与评测的优化顺序。
8. [资料索引](docs/04-source-index.md)和[来源台账](research/source-ledger.csv)：网页、代码、论文、固定版本和采用状态。
9. [架构决策](decisions/README.md)：路线选择和第三方代码隔离政策。
10. [提交合规清单](docs/06-submission-compliance.md)：组队、依赖、复现、报告和发布要求。

## 工作区结构

```text
config/       已核对的任务事实和评分约束
docs/         比赛与技术调研文档
research/     检索记录和后续论文笔记
decisions/    不覆盖历史的架构与合规决策记录
references/   外部项目 manifest；实际 checkout 不入顶层 Git
experiments/  实验协议与结果日志
src/          后续自研代码入口，当前不放占位实现
data/         数据说明；大文件不入库
artifacts/    轨迹、模型和评测产物说明；大文件不入库
vendor/       官方基线的只读快照
tests/        工作空间管理脚本的离线测试
```

## 常用命令

```bash
# 下载或更新固定版本参考仓库，不下载 Git LFS/模型/数据
bash scripts/fetch_references.sh

# 校验参考仓库的 origin、commit 和干净状态
bash scripts/check_references.sh

# 校验资料、来源、上游哈希、忽略规则和禁跟踪文件
bash scripts/check_workspace.sh

# 运行工作空间管理脚本测试
bash tests/test_reference_scripts.sh
bash tests/test_workspace_check.sh
```

## 上游状态

- 官方仓库：<https://github.com/JCIIOT2026/JCIIOT2026>
- 已审计提交：`f4ab8fd2158b919a41b2ce350432259cd1ee6a11`
- 上游提交时间：2026-07-22 17:17:47 +08:00
- 本地 `vendor/JCIIOT2026` 是轻量代码快照。由于本机未安装 Git LFS，`.pth`、`.hdf5`、`.zip` 大文件不会自动下载；运行前需要补齐 Git LFS 和官方权重。

## 下一阶段的最小成功标准

先完成官方环境的 L1 可复现闭环：固定计划执行、真实 `grasp_end success`、无碰撞、轨迹由未修改的官方 `app.py` 评分为 10/10，并连续多次运行统计成功率。达到这一门槛前，不启动大规模 VLA 微调。
