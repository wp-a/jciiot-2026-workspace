# 检索记录

检索日期：2026-07-22。

## 问题拆解

1. 比赛的正式赛程、交付物、评分、代码边界和硬件要求是什么？
2. 官方 baseline 的真实架构和评分实现是什么？
3. 是否存在可复现的公开参赛方案或成绩？
4. 2025-2026 的 VLA、模仿学习、行为树、导航和运动规划技术，哪些能在提交截止前转化为收益？

## 检索渠道

- 比赛官网、官方 GitHub、GitHub commits/forks/code search。
- 主办方/合作方公告与官方技术直播。
- arXiv 原文及项目官方仓库/项目页。
- Nav2、智谱、NVIDIA、Hugging Face 等官方文档。

## 代表性查询词

```text
JCIIOT2026 FactorySorting SOP-MapGuard
JCIIOT 2026 工业具身智能 赛程 baseline 直播
FactorySorting9_3FO3ERT2C5FP
line_5_container_h01_near
2026 vision language action mobile manipulation
2026 behavior tree LLM robot planning
2026 VLA distribution shift benchmark
payload aware navigation footprint inflation collision
```

GitHub 还逐一比较了 8 个公开 fork 相对官方仓库的 ahead/behind 状态，并审计了有新增提交的两个 fork。

## 核验方法

- 赛程：Biendata 页面与主办方 2026-07-15 公告交叉核对。
- 规则：比赛 DOCX、FAQ、选手手册、Erratum 与 `app.py` 实现交叉核对。
- 任务映射：`task_config.json`、`sop_main.md` 和生成的 semantic map 交叉核对。
- 最新技术：论文原文与作者/机构官方代码、文档或项目页交叉核对。
- 公开成绩：要求同时具备方法、代码/轨迹和未修改官方 scorer；不满足则标为未验证。

## 结果边界

- Biendata 公开抓取视图能确认比赛状态和 Team Merger Deadline，但未返回动态加载的详细规则正文；2026-07-22 用户提供的页面规则文本单独保存为规则快照，并用于更新合规清单。
- 官方技术直播于 2026-07-22 发布，时长 59:16，但没有公开字幕；本轮只索引，尚未逐帧人工转录 Q&A。
- QQ 群是私域，当前无法核对群内临时通知、免费额度细则和评测服务器信息。
- 公开 fork 的代码可审计，但其自报运行未在本机复现，且本机没有 Git LFS 权重。
- 本轮没有发现带可复现代码、原始轨迹和当前官方评分器证明的完整参赛方案。
- 2026-07 的新预印本可能快速变化，代码尚未发布的项目只列入观察清单。

## 后续更新规则

每次官方仓库更新或组委会答复后：

1. 记录新提交 SHA 和日期；
2. 比较 Erratum、手册、scorer、task config 和允许修改边界；
3. 更新 `docs/05-open-questions.md`；
4. 只有通过固定实验协议复现后，才更新技术路线中的优先级。

## 2026-08-01 增量复核

- 官方 `master` 从实验锁定提交 `0dcdddf` 更新到 `129e94a`，差异只有根
  `README.md` 的参赛者自报榜单；评分器、任务配置、场景和模型接口未变。
- 用固定清单重新校验 12 个本地参考 checkout，origin、commit 和干净状态
  全部通过。
- 在线复核 robomimic、MimicLabs、CP-Gen、ACT++、MobileManiBench 和
  `QiShengZhao/JCIIOT2026` 的分支 HEAD，均与 2026-07-22 固定 commit 一致。
- 没有发现符合当前官方修改边界、覆盖五关且带未修改评分器复现证据的完整
  方案。同赛题 fork 继续只用于 DAgger/HDF5/headless 诊断参考。
- 实验结果改变了采用优先级：BC-RNN 虽通过 held-out 动作误差门槛，但闭环
  0/5。Diffusion Policy 是最后一个同数据模型对照；若闭环失败，下一步固定为
  补采 approach drift 和单臂接触恢复数据，不继续无数据覆盖的模型搜索或从零 RL。
