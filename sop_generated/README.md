# 原创 SOP 知识生成结果

本目录保存从官方五份原始 `sop+prompt/*.docx` 生成的可追溯知识文件。
生成器没有读取官方手写 `knowledge/sop*.md`；DOCX 文本使用标准库解析，
25 张内嵌图片由公开的 Qwen3-VL-2B-Instruct 离线分析。VLM 输出只作为
图片证据，不是任务真值，不生成底盘或关节动作。

## 结果摘要

- 五份 DOCX 全部生成 Markdown 和 provenance JSON。
- 25/25 张图片得到符合固定 JSON schema 的证据；20 张一次通过，5 张在
  格式修复提示后第二次通过。
- 每个任务都通过官方 `task_config.json` 和对应 semantic map 交叉检查。
- L2 检出并保留两项正文模板冲突：错误的 `Pick Station 2` 和蓝色箱描述；
  根据任务 Prompt 与官方 Erratum 选择 `Pick Station 1` 和绿色储物箱。
- L3 根据官方 Erratum 将有效源标签解析为 `Placement Point 1`，并与
  `aux_input_1` 对齐。

| Level | 原始 DOCX SHA-256 | 状态 |
|---|---|---|
| L1 | `32a446a8395b03b46c9581d3e4978bd84bd90f31096e675da77366fd1cdc9c1c` | `ready` |
| L2 | `8eb46479564acfefe6abc6929eb749ea9b96abd458b5066e968424aa88080e00` | `ready_with_resolved_conflicts` |
| L3 | `e193fc7d16436d07a8091ecabee712cd9461066603df4d1efa09e9d6eb363927` | `ready` |
| L4 | `df637fcb1e558cbcacfc895d38b435008aa9eb15e1f85744631cea12b137e568` | `ready` |
| L5 | `de1a3779d119a17031a17d4ca7812366b5bc1f6c66982617272aa66006b7e5ba` | `ready` |

## 文件说明

- `generated_sop_l*.md`：可读知识文件，也是提交 overlay 中的静态产物。
- `generated_sop_l*.provenance.json`：源文档、任务合同、冲突、图片输入和
  VLM 响应哈希。
- `generated_sop_manifest.json`：五份结果的机器可读总索引。
- `checksums.sha256`：上述 11 个生成文件的内容校验。

## 复现

先按 `submission/README.md` 物化官方候选，并根据
`config/sop-vlm-lock.json` 准备公开模型。模型权重不随提交分发。

```bash
python candidate/JCIIOT/src/robot_agent/skills/sop_generator.py \
  --app-root candidate/JCIIOT \
  --output-dir results/sop_generated \
  --use-vision \
  --require-vision \
  --local-vlm-model /path/to/Qwen3-VL-2B-Instruct \
  --vision-model-id 'Qwen/Qwen3-VL-2B-Instruct@master+sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0' \
  --local-vlm-device cuda:0
```

历史原始证据 ZIP 位于忽略目录
`artifacts/remote-sop-qwen3vl-20260728/sop-qwen3vl-20260728-evidence.zip`，
SHA-256 为
`969b5c3af4555fb71b794dcb463958c558bf4771b7f97c7329a796879badbc9d`。
正式代码包保留生成结果和 provenance，不携带模型权重或服务器路径。

