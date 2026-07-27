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
8. [当前技术路线与优化计划](docs/09-current-route-and-optimization-plan.md)：已实现架构、竞争优势、创新边界和下一阶段门槛。
9. [五关性能基线](experiments/2026-07-28-five-level-performance-baseline.md)：固定种子 100/100、零碰撞和 L5 收敛证据。
10. [资料索引](docs/04-source-index.md)和[来源台账](research/source-ledger.csv)：网页、代码、论文、固定版本和采用状态。
11. [架构决策](decisions/README.md)：路线选择和第三方代码隔离政策。
12. [提交合规清单](docs/06-submission-compliance.md)：组队、依赖、复现、报告和发布要求。

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
- 已审计提交：`0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- 上游提交时间：2026-07-27 17:56:06 +08:00
- 本地 `vendor/JCIIOT2026` 已扩展为完整源码检出（约 1.7 GB，含全部 robosuite 场景/网格/地图，均为普通 git 对象）。源码完整不等于运行环境已验证，五场景冒烟结果以 `STATUS.md` 为准。
- 本机已安装 Git LFS，但**上游仓库 LFS 配额已超限**（"exceeded its LFS budget"），`model_epoch_150.pth`、示例 `.hdf5`、USD `.zip` 目前从 GitHub 无法下载，仅保留 133/134 字节指针。默认权重 `model_epoch_500.pth` 从未入库。桌面网格 zip 只是建模源档案，解压后的 `.obj/.stl` 已在库内，不阻碍仿真运行。需通过官方 QQ 群获取权重或反馈配额问题。

## 下一阶段的最小成功标准

固定种子五关满分闭环已经完成。下一阶段最小成功标准是：L2-L5 各至少 20 种子，保留全部失败并统计满分率/碰撞率；在干净 Linux 环境一键复现；完成 SOP 证据链、模块消融和技术报告。达到这些门槛前，不启动大规模 VLA 微调。
