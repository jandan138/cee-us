# 2026-03-05 backend-diagnostics-phase8-unified-switch-readiness

## 背景

phase7 已提供物理后端 readiness 与真实切换测试 skip 诊断，但排查“指定 env + physics backend + render backend 能否切换”仍需要手工拼多条信息：

1. 物理后端是否就绪；
2. 渲染后端是否已实现/可用；
3. 目标 env 是否存在对应 physics backend 映射。

phase8 新增统一 tuple 级诊断，降低切换前排查成本。

## 改动

- 更新 `mbrl/environments/backends/diagnostics.py`
  - 新增 `render_backend_readiness(...)`
    - 返回渲染后端 `implemented/ready/error_type/reason`。
  - 新增 `diagnose_unified_switch_readiness(...)`
    - 输入：`env_name + physics_backend_name + render_backend_name`
    - 输出至少覆盖三项：
      - physics backend readiness
      - render backend readiness/implemented
      - env mapping 是否存在（针对解析后的 physics backend）
    - 提供 `ready` 总结与 `next_actions` 建议。
  - 扩展 `collect_backend_diagnostics(...)`
    - 支持可选 tuple 诊断入口（不影响原有 phase7 结果结构）。
  - CLI 新增参数：
    - `--env`
    - `--physics-backend`
    - `--render-backend`
    - 仍兼容原有 `--backend`、`--json`、`--skip-genesis-dependency-check`。

- 更新 `mbrl/environments/backends/__init__.py`
  - 导出：
    - `render_backend_readiness`
    - `diagnose_unified_switch_readiness`

## 使用示例

```bash
# 只看 phase7 既有诊断（兼容旧用法）
python -m mbrl.environments.backends.diagnostics

# 查询指定 tuple 的统一 readiness（默认 physics=mujoco, render=native）
python -m mbrl.environments.backends.diagnostics --env PlaygroundwGoals

# 显式查询 env/backend/render tuple
python -m mbrl.environments.backends.diagnostics \
  --env PlaygroundwGoals \
  --physics-backend mujoco \
  --render-backend headless

# genesis synthetic 排查 + tuple 诊断
python -m mbrl.environments.backends.diagnostics \
  --env PlaygroundwGoals \
  --physics-backend genesis \
  --render-backend none \
  --skip-genesis-dependency-check
```

## 兼容性说明

- 未修改 `env_from_string(...)` 默认参数解析逻辑，默认 `mujoco` 路径行为保持不变；
- phase7 CLI/JSON 输出字段保持向后兼容，tuple 诊断为可选扩展字段；
- 对未实现渲染后端（如 `genesis/isaacsim/newton`）会明确报告 `implemented=false` 与原因。
