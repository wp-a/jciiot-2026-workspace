# SOP DOCX 原始来源审计

审计日期：2026-07-28（Asia/Shanghai）

## 目的

确认五份原始 SOP Word 文档中可直接提取的任务事实、图片资产和内部冲突，为原创 SOP 解析模块提供证据边界。本审计只读取 `sop+prompt/*.docx`，没有以官方手写 `knowledge/sop*.md` 作为生成输入。

## 方法

- 用 Pandoc `--track-changes=all` 提取段落、列表和图片引用。
- 用 DOCX ZIP 清单核对 `word/media/` 中的图片数量。
- 对每份原始 DOCX 记录 SHA-256。
- 将文档顶部 Prompt 与正文模板、官方 `task_config.json` 和已锁定任务事实交叉检查。

## 结果

| Case | Prompt 任务事实 | 图片 | DOCX SHA-256 | 主要观察 |
|---|---|---:|---|---|
| 1 / L1 | 蓝色中空塑料箱；Pick Station 2 -> Place Station 3；1 件 | 5 | `32a446a8395b03b46c9581d3e4978bd84bd90f31096e675da77366fd1cdc9c1c` | Prompt 与官方任务映射一致 |
| 3 / L2 | 绿色边框储物箱；Pick Station 1 -> Place Station 3；1 件 | 5 | `8eb46479564acfefe6abc6929eb749ea9b96abd458b5066e968424aa88080e00` | 正文模板错误残留 Pick Station 2 和蓝色中空箱，不能整篇等权摘要 |
| 5 / L3 | 蓝色周转箱；Pick Station 1 -> Place Station 2；1 件 | 5 | `e193fc7d16436d07a8091ecabee712cd9461066603df4d1efa09e9d6eb363927` | 具体辅助上料点需结合图片、Erratum 和语义地图确认 |
| 7 / L4 | 蓝色中空塑料箱；Pick Station 5 -> Place Station 2；1 件 | 5 | `df637fcb1e558cbcacfc895d38b435008aa9eb15e1f85744631cea12b137e568` | Prompt 与官方任务映射一致 |
| 9 / L5 | 三个白色边框储物箱；Pick Station 6 -> Place Station 1；3 件 | 5 | `de1a3779d119a17031a17d4ca7812366b5bc1f6c66982617272aa66006b7e5ba` | 数量 3 是任务编排的强约束，不能被通用“如有多个则重复”弱化 |

## 证据优先级

1. 文档顶部当前任务 Prompt：物体、起点、终点和数量的首要来源。
2. 原始图片与 VLM 描述：用于识别具体台面、摆放点和物体外观，并保留图片 SHA-256。
3. 官方 Erratum、`task_config.json` 和语义地图：用于解析命名并报告冲突，不能静默改写 Prompt。
4. SOP 通用正文：只提供安全、抓取、运输、放置和异常处理流程，不覆盖 Prompt 中的任务事实。
5. 官方手写 `knowledge/sop*.md`：仅作为独立验收参考，不进入生成输入。

## 对实现的约束

- 输出必须逐条保存 `value / source / evidence locator / confidence / conflict`，不能只输出无来源摘要。
- 对 Case 3 这类 Prompt 与正文冲突，默认采用 Prompt，并在生成报告中显式列出冲突。
- VLM 只处理图片；正文和表格使用确定性 DOCX 解析，减少模型幻觉和运行成本。
- 语义结果只生成任务约束和检查项，不直接生成底盘或关节动作。
- 若物体、起点、终点或数量仍不一致，解析状态应为 `needs_review`，不得把低置信度结果直接送入执行层。

## 当前边界

该审计证明了原创解析的输入事实和必要性，但尚未证明解析器实现、VLM 输出稳定性或自动生成知识文件的功能等价性。这些必须通过单元测试、五文档生成结果和冲突消融另行验证。
