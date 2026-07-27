# 2026-07-27 五场景无权重冒烟

## 目的

验证官方提交 `0dcdddf18a9e694569aa1433cdfc04eb097fed78` 的五个 FactorySorting 场景是否能在 Linux 上构造、reset 并执行一个零动作 step。此实验隔离场景资产和基础动力学，不加载 BC checkpoint、LLM 或 VLM。

## 环境

- Ubuntu 24.04.3 LTS，Linux 7.0.0-28-generic，x86_64；
- 4 x NVIDIA L40S 46068 MiB，驱动 580.142；
- Python 3.11.15；
- NumPy 2.4.6；
- MuJoCo 3.9.0；
- 官方内嵌 robosuite 1.5.2；
- seed `20260727`，每场景 1 个零动作 step，renderer/offscreen renderer 均关闭。

服务器端实验根目录与登录信息不进入版本库。原始结果同步到本地忽略目录 `artifacts/remote-smoke-20260727/`。

## 依赖诊断

第一次按 robosuite `setup.py` 的 `mujoco>=3.3.0` 安装得到 MuJoCo 3.10.0。L1 在环境构造阶段失败：3.10.0 的 `mj_fullM` Python 签名为 `(model, data, dst)`，而官方控制器按 3.9.0 的 `(model, dst, M)` 调用。

根目录 `requirements.txt` 明确固定 `mujoco==3.9.0`。只把 MuJoCo 降回 3.9.0 后，用完全相同命令重跑 L1 即通过，因此确认根因是依赖版本漂移，不是场景资产损坏。

## 结果

| 场景 | 结果 | action shape | nq | nv | ngeom | ncam | 耗时（秒） |
|---|---|---:|---:|---:|---:|---:|---:|
| FactorySorting1 | 通过 | 20 | 44 | 42 | 724 | 6 | 3.023466 |
| FactorySorting3 | 通过 | 20 | 44 | 42 | 742 | 6 | 2.777289 |
| FactorySorting5 | 通过 | 20 | 79 | 72 | 770 | 6 | 2.838894 |
| FactorySorting7 | 通过 | 20 | 100 | 90 | 800 | 6 | 3.026798 |
| FactorySorting9 | 通过 | 20 | 100 | 90 | 808 | 6 | 3.252990 |

汇总：5/5 场景通过，失败 0。远端 `pip check` 报告 `No broken requirements found`。

## 产物校验

| 产物 | SHA-256 |
|---|---|
| `scene-smoke-all-0dcdddf.json` | `dd400a3e035de43dbd8971d736da9f40a0b939bdc59de79bc30af5180ae3f1b3` |
| `environment-0dcdddf.txt` | `b2d3ac83b1ca4a7663fa7ddf1915dfdc03e977ccde6b542cdeb9e36498a69017` |
| `pip-freeze-0dcdddf.txt` | `899a297ae716d7209a53656f1a14b0d9ef541033d034dffadea0f6c15bd30f04` |

## 结论边界

本实验证明五个场景的普通 Git 资产、Tiago 模型、控制器初始化和基础 MuJoCo step 可用。它没有验证：

- robomimic checkpoint 加载或抓取成功；
- `move -> pick -> move -> place` 完整 workflow；
- 碰撞、离开、到达和官方得分；
- renderer、相机观测或无头图像一致性；
- 多随机种子稳定性。

下一实验必须取得或训练 BC 权重，并先完成 L1 固定计划闭环。
