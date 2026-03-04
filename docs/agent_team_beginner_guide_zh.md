# Agent Team 新手上手指南（小白版）

这篇文档面向第一次使用本仓库 `agent team` 的同学。你不需要先懂多代理，只要按步骤执行即可。

## 1. 你只需要说一句话
在 Codex 主线程里直接说：

```text
用agent team，目标是：<你的任务目标>
```

例如：

```text
用agent team，目标是：给训练脚本加一个新的配置开关，并补测试
```

## 2. 第一次使用前（只做一次）

1) 开启 Codex 多代理功能：

```bash
codex features enable multi_agent
```

2) 重启 Codex/TUI。  
3) 在仓库根目录执行预检查：

```bash
bash agent_team/scripts/codex_multi_agent_preflight.sh
```

如果看到 `multi_agent ... true`，说明环境已准备好。

## 3. 一次任务的标准流程

### 第一步：初始化 run

```bash
bash agent_team/scripts/bootstrap_agents.sh
bash agent_team/scripts/init_run.sh <run_id>
bash agent_team/scripts/setup_run_worktrees.sh <run_id>
```

`run_id` 建议格式：`run_YYYY_MM_DD_任务名`，例如 `run_2026_03_04_docs`。

### 第二步：让主 agent 分派子线程
使用模板：

`agent_team/templates/codex_spawn_prompt_template.md`

并在 TUI 里用 `/agent` 查看和切换子线程。

### 第二步半：主 agent 如何“慢慢等”
这是你最关心的规则：**主 agent 默认慢慢等，不设硬超时**。  
子代理即使慢，只要还在推进，就继续等待。

建议主 agent 周期巡检：

```bash
bash agent_team/scripts/monitor_subagents.sh <run_id> --interval-min 10 --stuck-min 45
```

- `interval-min 10`：建议心跳间隔 10 分钟
- `stuck-min 45`：超过 45 分钟无心跳会标记为 `stuck-suspected`
- 注意：`stuck-suspected` 只是告警，不是自动终止

只有人工确认卡死（`stuck-confirmed`）后，才允许重启同角色线程：

```bash
bash agent_team/scripts/restart_stuck_subagent.sh <run_id> <agent_id> <new_thread_id>
```

人工确认卡死的最小操作：
- 打开 `runs/<run_id>/threads/registry.md`
- 把该 agent 行改为：
  - `status = stuck-confirmed`
  - `stuck_candidate = confirmed`
  - 在 `notes` 写清原因（例如：`manual confirm: no progress for 90m`）

### 第三步：收尾检查与记忆合并

```bash
bash agent_team/scripts/check_run_logs.sh <run_id>
bash agent_team/scripts/update_agent_memory.sh <run_id>
```

可选清理 worktree：

```bash
bash agent_team/scripts/teardown_run_worktrees.sh <run_id> --delete-branches
```

## 4. 这些文件分别是什么

- `runs/<run_id>/logs/*.md`：每个 agent 的执行日志（调研/改动/测试/风险）
- `runs/<run_id>/memory/*.delta.md`：本次 run 的经验增量
- `runs/<run_id>/threads/registry.md`：仓库角色和 Codex 线程映射
- `runs/<run_id>/worktrees/registry.md`：可编辑角色的隔离分支和路径

## 5. 常见问题（新手必看）

### Q1: 我只说“用agent team”就够了吗？
够。主 agent 会按流程自动落地 run。  
如果你想更稳，可以附上目标和约束（例如“不能改公共 API”）。

### Q2: 为什么要 worktree？
因为并行改代码时容易互相覆盖。  
`setup_run_worktrees.sh` 会给可编辑角色分配独立分支和目录，减少冲突。

### Q3: 非编辑角色也要写日志吗？
要。`can_edit_code=false` 的角色要先写 handoff，再由 `doc-writer` 代写正式日志。

### Q4: 哪些文件不该 commit？
运行产生的临时 run 目录（`agent_team/runs/<run_id>/...`）通常不进版本库。  
本仓库已通过 `.gitignore` 规则默认忽略，只保留 `agent_team/runs/.gitkeep`。

### Q5: 子代理很慢怎么办？
默认继续等，不要急着终止。  
先跑 `monitor_subagents.sh` 看是否只是慢、还是疑似卡死。  
除非人工确认卡死，否则不要重启线程。
