# 深度阅读笔记

本目录保存对重要来源的代码级或论文级审计。文件名应包含 `source_id` 或明确主题，并遵循以下结构：

1. 来源、访问日期和固定版本；
2. 作者声称的能力与结果；
3. 本地实际看到的文件、接口和依赖；
4. 与 JCIIOT 的对应模块；
5. 可采用部分、禁止采用部分和许可证风险；
6. 是否已在本比赛环境复现；
7. 后续实验或复审条件。

证据状态统一使用：

- `remote-verified`：已由官方页面、论文或远程代码确认；
- `local-code-audited`：已下载固定 commit 并检查关键文件；
- `competition-reproduced`：已在未修改官方环境和 scorer 中复现；
- `rejected`：因许可证、边界、依赖或性能证据不足而拒绝。

禁止把论文作者结果、其他参赛者自报分数或单次本地成功写成 `competition-reproduced`。

## 当前审计

- [五份原始 SOP DOCX 来源、冲突与证据优先级](sop-docx-source-audit-2026-07-28.md)
- [公开 GitHub 同类项目审计](github-project-audit-2026-07-22.md)
