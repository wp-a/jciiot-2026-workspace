# 2026-07-28 原创 SOP 知识生成实验

## 目的

验证能否在不读取官方手写 `knowledge/sop*.md` 的前提下，从五份原始
DOCX 独立生成可追溯知识文件，并显式处理任务 Prompt、通用正文模板、
官方 Erratum、任务配置和语义地图之间的冲突。

## 协议

- 官方提交：`0dcdddf18a9e694569aa1433cdfc04eb097fed78`
- 生成器提交：`2325147a54acc52b7f12608e865e5aceee81da27`
- 输入：官方 Case 1、3、5、7、9 原始 DOCX，共 25 张内嵌图片
- 文本解析：Python 标准库 ZIP/XML，禁止读取官方手写 SOP Markdown
- 图片模型：`Qwen/Qwen3-VL-2B-Instruct`
- 权重 SHA-256：`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`
- 推理：本地确定性生成，`do_sample=false`，最多 2 次 schema 修复
- 验收：五个任务合同与 `task_config.json`、semantic map 一致；每张图片
  输出固定五字段 JSON；所有生成文件和输入证据均有 SHA-256

## 结果

| 指标 | 结果 |
|---|---:|
| 成功生成的 SOP | 5/5 |
| 完成分析的图片 | 25/25 |
| 首次符合 schema | 20/25 |
| 第二次修复后符合 schema | 5/5 |
| 任务配置交叉检查 | 5/5 |
| 语义端口交叉检查 | 5/5 |
| 使用官方手写 SOP 作为输入 | 0 |

L2 的通用正文残留了另一个任务的 `Pick Station 2` 和蓝色箱描述。生成器
没有静默平均这些互相冲突的信息，而是记录两项冲突，并按任务专属 Prompt
与官方 Erratum 选择 `Pick Station 1` 和绿色储物箱。L3 则按官方 Erratum
将有效源标签解析为 `Placement Point 1`，与官方 `aux_input_1` 对齐。

## 独立复核

- 8 个生成器单元测试覆盖 DOCX XML、五种 Prompt 表达、冲突、Erratum、
  VLM schema、失败重试、本地模型参数和 provenance 输出。
- 完整工作区测试为 `106 passed`。
- 机器复核确认 25 个描述均为固定五字段 JSON，每字段最多 4 个去重字符串。
- `checksums.sha256`、manifest 内输出哈希和原始证据 ZIP 均校验通过。
- 生成 Markdown 不包含本地/服务器绝对路径，提交不包含模型权重。

## 证据边界

VLM 对图片的描述可能为空、泛化或不准确，因此不能单独确定物体、工位或
安全状态。任务合同只由 Prompt、官方 Erratum、`task_config.json` 和语义地图
交叉验证。生成知识是离线静态产物；评分执行不加载 VLM，也不由 VLM 输出
底盘或关节命令。

## 证据

- 生成结果：`sop_generated/`
- 模型锁：`config/sop-vlm-lock.json`
- 原始输入审计：`research/notes/sop-docx-source-audit-2026-07-28.md`
- 忽略目录原始证据 ZIP：
  `artifacts/remote-sop-qwen3vl-20260728/sop-qwen3vl-20260728-evidence.zip`
- ZIP SHA-256：
  `969b5c3af4555fb71b794dcb463958c558bf4771b7f97c7329a796879badbc9d`

