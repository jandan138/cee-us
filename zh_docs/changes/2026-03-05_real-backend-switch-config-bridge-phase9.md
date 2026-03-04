# 2026-03-05 real-backend-switch-config-bridge-phase9

## 背景

phase5 已经引入受 `ENABLE_REAL_BACKEND_TESTS=1` 控制的真实后端切换测试入口，phase8 补齐了统一诊断 API/CLI。  
但在“真实切换候选发现”与“候选后端构造参数”上，仍存在两个缺口：

1. 候选发现仅依赖当前进程内已注册后端，无法通过配置动态加载插件模块；
2. 候选 readiness 与真实 `env_from_string(...)` 构造没有共享可配置的 backend options（例如 Genesis 的 `skip_dependency_check`）。

## 改动

- 更新 `mbrl/environments/backends/diagnostics.py`
  - 新增 real-switch 配置桥接常量：
    - `REAL_BACKEND_SWITCH_PLUGIN_MODULES`
    - `REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON`
  - 新增 `resolve_real_backend_switch_configuration(...)`
    - 支持两种输入来源：
      - 显式参数（可编程调用）
      - 环境变量（CI / shell 控制）
    - 对插件模块与 backend options 做归一化与校验。
  - `diagnose_real_backend_switch_test(...)` 增强：
    - 在候选发现前按配置加载插件模块；
    - 候选选择改为“首个 **ready 且有 ENV_REGISTRY 映射** 的非 MuJoCo backend”；
    - 返回 `candidate_physics_backend_options`，供真实构造路径直接复用；
    - 在 skip 原因中提供 readiness failure / mapping 缺失细节。
  - `collect_backend_diagnostics(...)` 支持 real-switch 独立输入：
    - `real_switch_backend_plugin_modules`
    - `real_switch_options_by_backend`
  - CLI 新增：
    - `--real-switch-plugin-module`（可重复）
    - `--real-switch-physics-options-json`

- 更新 `mbrl/environments/backends/__init__.py`
  - 导出 `resolve_real_backend_switch_configuration`。

- 更新测试
  - `tests/test_environment_backends.py`
    - 真实切换 gated 测试改为复用 `diagnose_real_backend_switch_test(...)`；
    - 新增覆盖：
      - 通过 `REAL_BACKEND_SWITCH_PLUGIN_MODULES` 发现插件候选；
      - 通过显式 `backend_plugin_modules` 入参发现插件候选；
      - 通过 `REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON` 把 backend options 传入真实 env 构造路径。
  - `tests/test_backend_diagnostics.py`
    - 新增 real-switch 配置解析与参数覆盖测试。

## 兼容性说明

- 当新增环境变量未设置时，默认行为保持不变：
  - 不加载额外 real-switch 插件模块；
  - 不注入额外 backend options。
- 现有 MuJoCo 默认路径与原有 API 调用方式不受影响。
- 新字段为增量输出，不移除既有诊断字段。

## 使用示例

```bash
# 1) 启用真实切换测试并通过插件注册候选
export ENABLE_REAL_BACKEND_TESTS=1
export REAL_BACKEND_SWITCH_PLUGIN_MODULES=tests.backend_plugin_example

# 2) 对特定后端注入 readiness / 构造参数（JSON）
export REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON='{"genesis":{"skip_dependency_check":true}}'

# 3) 诊断同一路径
python -m mbrl.environments.backends.diagnostics --json
```
