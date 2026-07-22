# 外部参考仓库

`references/repos/` 保存调研用的第三方源码 checkout，不进入顶层 Git 历史。固定 URL、commit、许可证初判和用途记录在 [`repositories.json`](repositories.json)。

## 下载

```bash
bash scripts/fetch_references.sh
```

只下载一个项目：

```bash
bash scripts/fetch_references.sh --only robomimic
```

下载采用浅克隆、跳过 Git LFS，并固定到 detached commit。配置了 `sparse_paths` 的大型仓库只 checkout 相关源码目录和顶层说明文件。脚本不会初始化 submodule，也不会下载模型或数据集。

## 校验

```bash
bash scripts/check_references.sh
```

校验包括仓库是否存在、origin URL、HEAD commit 和工作区是否干净。任何参考仓库存在本地修改时，下载脚本会拒绝覆盖。

## 使用边界

- 本目录用于阅读和对照，不是参赛代码来源目录。
- 采用代码或算法前，先在 `THIRD_PARTY_NOTICES.md` 更新许可证和采用状态。
- 其他参赛者 fork 只允许诊断对照，禁止复制。
- AnyGrasp、MimicGen 等受限组件不因已下载公开仓库而自动变成比赛可用组件。
- 需要保留阅读批注时，写入 `research/notes/`，不要修改第三方 checkout。
