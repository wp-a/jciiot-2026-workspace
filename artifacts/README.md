# Artifacts

放置模型权重、轨迹 JSON、评分结果、回放视频和最终提交包。大文件默认不进入版本控制。

每个正式实验产物至少记录：上游提交、方案提交、场景、随机种子、参数摘要、客观分、碰撞、耗时和轨迹路径。

## 五场景无权重冒烟

`scripts/smoke_official_scenes.py` 只验证官方场景能否构造、reset 和执行零动作 step，不加载 BC 权重、LLM 或 VLM。示例：

```bash
python scripts/smoke_official_scenes.py \
  --official-root /path/to/JCIIOT2026 \
  --steps 1 \
  --seed 20260727 \
  --output artifacts/scene-smoke.json
```

输出包含官方 commit、Python/平台、随机种子、每个场景的模型维度、耗时和失败阶段。`artifacts/` 下的运行结果默认不进入 Git；经过复核的摘要写入 `STATUS.md` 和 `CHANGELOG.md`。
