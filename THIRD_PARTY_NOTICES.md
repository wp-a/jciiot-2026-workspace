# 第三方组件与许可台账

核对日期：2026-07-22。本文件是工程合规记录，不替代正式法律意见。任何组件进入最终提交前都必须再次核对固定版本的许可证全文、依赖链、模型权重和数据条款。

## 官方基线

| 组件 | 用途 | 来源 | 固定版本 | 状态 |
|---|---|---|---|---|
| JCIIOT2026 | 官方环境、评分器和基线 | <https://github.com/JCIIOT2026/JCIIOT2026> | `f4ab8fd...` | 必须保留来源；只在官方允许目录内提交改动 |
| robosuite | MuJoCo 机器人仿真 | 官方仓库内嵌版本 | 由 baseline 锁定 | 不单独替换，除非完成兼容性复现 |
| robomimic | BC 抓取训练与推理 | 官方仓库内嵌版本；上游 MIT | 由 baseline 锁定 | 首选学习框架 |

## 可采用候选

| 组件 | 许可证 | 计划用途 | 当前状态 |
|---|---|---|---|
| PythonRobotics | MIT | A*、Theta*、D* Lite 和路径平滑算法参考 | 可采用；仅移植必要算法并保留声明 |
| py_trees | BSD-3-Clause | 行为树、blackboard 和恢复语义 | 可采用；也可按接口成本实现更小状态机 |
| multimodal-bt-generation | MIT | 多模态任务到行为树 schema、grounding 和重试设计 | 研究后选择性采用 |
| KIOS | MIT | 世界状态、前置/后置条件和计划修复设计 | 仅采用设计；不引入重型服务依赖 |
| ACT++ | MIT | action chunking 和 temporal ensembling | 仅在 BC-RNN 达到瓶颈后评估 |
| MobileManiBench | BSD-3-Clause | 移动操作数据与评测结构参考 | 研究用途；不引入 Isaac Sim 栈 |
| RoboMonkey | MIT | test-time verifier 思路 | 研究用途；只考虑确定性轻量 verifier |

## 受限或待确认

| 组件 | 风险 | 赛事策略 |
|---|---|---|
| MimicGen | NVIDIA 许可证限制为非商业研究/评估 | 不复制或绑定其代码；只研究方法，待组委会和许可复核 |
| MimicLabs | 自身 MIT，但数据生成流程依赖 MimicGen 等组件 | 可研究采集结构；依赖链核清前不进入提交 |
| CP-Gen | 自身 MIT，但依赖 MimicGen、cuRobo 和特定 robosuite 分支 | 只研究约束保持增强方法 |
| OK-Robot AnyGrasp | 需要单独注册/授权，仓库含编译二进制 | 明确禁止用于比赛实现；仅参考模块化系统架构 |
| QiShengZhao/JCIIOT2026 | 其他参赛者公开 fork，且修改了官方禁止目录和评分代码 | 只做诊断对照；禁止复制实现或引用其自报分数作为证据 |

## 使用要求

- 每个实际采用的第三方文件保留版权声明和许可证要求。
- 报告中注明库、算法、模型、数据和外部服务，不把开源组件描述为自研贡献。
- 许可证不明、需要私下授权、权重无法公开获取或数据来源不清的组件不得进入最终方案。
- 私有工作仓库不等于可以忽略再分发条件；按获奖后全部代码公开的最严格场景管理依赖。
