# 2026-02-14 ce7d358 — WSL+uv+MuJoCo 安装文档与脚本

## 背景

在 WSL2（Win11 + WSLg）新环境中，使用 uv 搭建可运行的开发环境，并把踩坑过程与可复用脚本固化到仓库，方便后续重复安装与排障。

## 改动

- 新增 WSL + uv 环境搭建文档：`docs/setup_wsl_uv.md`
  - 记录 apt 依赖、uv 用法、PyTorch 安装方式
  - 记录 MuJoCo 2.0 + mujoco-py 的典型坑与解决办法
- 新增非 MuJoCo 版本依赖清单：`requirements.no_mujoco.txt`
  - 用于在 MuJoCo 未就绪时先装其他依赖
  - 处理了 uv 对 requirements 中 VCS 依赖的兼容写法
- 新增 MuJoCo 2.0 安装脚本：`scripts/setup_mujoco200_wsl.sh`
  - 自动解压到 `~/.mujoco/`、安装 `mjkey.txt`
  - 处理 `~/.mujoco/mujoco200` 的目录命名期望（通过软链接）
  - 追加必要环境变量到 `~/.bashrc`
- 更新 `.gitignore`
  - 忽略 `uv.lock`（本仓库当前未采用 lockfile 作为标准交付物）

## 验证

- 在 WSL 中通过 `python -m mbrl.main <settings.yaml>` 成功跑通最小 smoke（1 iteration, 1 rollout）。
- `mujoco_py` 可在 headless（`MUJOCO_GL=osmesa`）模式下创建最小 `MjSim`。
- 在 WSLg 下 `MUJOCO_GL=glfw` 可创建窗口并渲染若干帧（验证 GUI 链路）。

## 注意/回滚

- 如果将来决定“正式采用 uv lockfile 作为可复现交付物”，需要重新评估是否提交 `uv.lock`，并相应调整 `.gitignore`。
- `~/.bashrc` 中追加的 MuJoCo 环境变量如果与其他项目冲突，可以手动删除由脚本标记的区块。
