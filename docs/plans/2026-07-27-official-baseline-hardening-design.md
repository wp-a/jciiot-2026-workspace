# 官方基线加固与远程实验设计

日期：2026-07-27

## 目标

在不修改官方禁止目录的前提下，把工作空间同步到已审计的最新官方提交，自动发现任务事实漂移，并在独立 Linux GPU 服务器上证明五个 MuJoCo 场景能够无权重加载。任何“可运行”结论必须对应可重复命令和保存的结果。

## 已知问题

- 官方仓库在 2026-07-27 从 `f948609` 更新到 `0dcdddf`，L3 源工位改为 `aux_input_1`，L5 目标工位改为 `aux_output_1`。
- 当前 `config/tasks.json`、简报和上游锁仍固定在 `f948609`。
- 工作空间校验只检查任务 JSON 的结构，没有比较官方 `task_config.json` 和语义地图。
- 本地 `jciiot` Conda 环境尚未安装仿真依赖，计划中的五场景冒烟程序也不存在。
- Git LFS 权重和示例数据受上游配额限制；场景加载验证必须与 BC 策略验证分开。

## 方案

### 1. 固定最新上游

将忽略的 `vendor/JCIIOT2026` 快进到 `0dcdddf`，重新计算锁文件中的关键文件哈希。顶层仓库只记录 commit、哈希和审计结论，不提交 vendor 内容。

### 2. 建立任务事实一致性检查

增加一个只依赖 Python 标准库的检查器，比较：

- `config/tasks.json.official_commit` 与 `config/upstream-lock.json.repository.commit`；
- L1-L5 的场景、源工位、目标工位、候选对象和满分值与官方 `knowledge/task_config.json`；
- 工作空间中的源/目标中心与对应官方语义地图站点中心。

检查器接入 `scripts/check_workspace.sh`。这样官方任务事实再次变化时，现有测试会失败，而不是继续报告工作空间正常。

### 3. 五场景无权重冒烟

增加独立脚本，逐一创建五个官方 FactorySorting 环境，执行 reset 和少量零动作 step，记录：

- 场景名和随机种子；
- action shape；
- 模型关节、几何体和相机数量；
- reset/step 是否成功及耗时；
- 失败阶段和异常信息。

脚本不得加载 robomimic checkpoint、LLM 或 VLM，也不得修改官方代码。输出 JSON 可写入 `artifacts/`，大文件和运行产物保持 Git 忽略。

### 4. 远程实验隔离

服务器使用 `/home/user/jciiot-2026` 作为独立实验根目录，Micromamba 创建 Python 3.11 环境。只部署固定版本官方源码、工作空间脚本和必要依赖；不复用其他比赛目录，不写入登录凭据，不下载当前不可用的 LFS 大文件。

依赖安装与场景加载分阶段执行：先验证 Python/MuJoCo/robosuite 导入，再验证 L1，最后验证 L1-L5。任何失败都保留完整日志并按依赖、渲染后端、资源路径和场景构造边界定位。

## 错误处理

- 上游 commit、任务字段或坐标不一致时，校验立即失败并指出关卡和字段。
- 缺少依赖时，冒烟程序报告导入阶段失败，不误报为场景资源问题。
- 单个场景失败时继续保存已完成场景结果，但进程以非零状态退出。
- 权重缺失只影响抓取策略测试，不得阻塞或污染无权重场景加载结论。

## 验证标准

1. 任务一致性测试能在旧 L3/L5 配置上失败，在同步后通过。
2. 顶层现有测试与 `check_workspace.sh --require-private-remote` 全部通过。
3. 服务器上五个场景都能 reset、执行至少一个零动作 step 并关闭环境。
4. 实验结果记录固定官方 commit、Python、MuJoCo、GPU 驱动和运行时间。
5. 文档只把已经验证的能力写成“通过”，未获得的 BC 权重继续列为阻塞。

## 非目标

- 本阶段不训练 BC、VLA 或 Diffusion Policy。
- 不修改 `core/`、`environments/`、`app.py` 或官方 `task_config.json`。
- 不下载 USD、示例 HDF5 等与场景冒烟无关的大文件。
- 不把服务器地址、用户名、密码或密钥提交到 Git。
