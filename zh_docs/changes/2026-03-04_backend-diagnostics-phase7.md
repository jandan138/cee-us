# 2026-03-04 backend-diagnostics-phase7

## 背景

phase6 已有 `physics_backend_readiness(...)` API，但一线排查时仍有两个痛点：

1. 缺少一个可直接运行的开发者诊断命令；
2. 无法一次性看到“为什么 `test_real_backend_switch_when_enabled` 被 skip”。

本阶段提供最小可用诊断入口，并保持 MuJoCo 默认行为不变。

## 改动

- 新增 `mbrl/environments/backends/diagnostics.py`
  - API：
    - `collect_physics_backend_readiness(...)`：批量查看物理后端 readiness
    - `diagnose_real_backend_switch_test(...)`：给出真实切换测试的 skip 诊断（与现有测试 gate 顺序一致）
    - `collect_backend_diagnostics(...)`：合并 readiness 与 skip 诊断
  - CLI（最小入口）：
    - `python -m mbrl.environments.backends.diagnostics`
    - `python -m mbrl.environments.backends.diagnostics --json`
    - `python -m mbrl.environments.backends.diagnostics --backend genesis`
    - `python -m mbrl.environments.backends.diagnostics --backend genesis --skip-genesis-dependency-check`
- 更新 `mbrl/environments/backends/__init__.py`
  - 导出上述诊断 API，供外部脚本或 notebook 直接调用。

## 使用示例

```bash
# 文本报告：快速看是否会 skip，以及具体原因
python -m mbrl.environments.backends.diagnostics

# JSON 报告：便于 CI 或脚本自动解析
python -m mbrl.environments.backends.diagnostics --json

# 针对 genesis 查看 readiness（默认会检查 genesis 依赖）
python -m mbrl.environments.backends.diagnostics --backend genesis

# 在未安装 genesis 时做 synthetic 就绪性排查
python -m mbrl.environments.backends.diagnostics --backend genesis --skip-genesis-dependency-check
```

## 兼容性说明

- 未改动 `env_from_string(...)` 与默认 `mujoco` 路径；
- 未改动现有测试逻辑，仅新增可观测诊断能力；
- 当 `genesis` 缺失时，诊断命令仍可运行并给出明确原因。
