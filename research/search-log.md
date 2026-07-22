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
