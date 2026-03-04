# 2026-03-05 real-backend-switch-runtime-policy-phase10

## 背景

phase9 已打通 real-switch 配置桥接（插件模块 + backend options），但诊断里还缺少“候选是否为真实 runtime”的显式信息。  
在 `skip_dependency_check=true` 等 synthetic 配置下，候选可以被发现并通过 readiness；这对排查很有价值，但在“必须验证真实 runtime”的 CI 场景里需要可选的严格门禁。

## 改动

- 更新 `mbrl/environments/backends/diagnostics.py`
  - `diagnose_real_backend_switch_test(...)` 新增 `candidate_runtime_mode` 诊断字段：
    - `true-runtime`
    - `synthetic-runtime`
  - 新增 `candidate_runtime_mode_reason`，当前对 `skip_dependency_check=true` 标记为 synthetic。
  - 新增严格策略配置入口（默认关闭）：
    - 环境变量：`REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1`
    - API 参数：`require_true_runtime=True`
    - CLI 参数：`--real-switch-require-true-runtime`
  - 当 strict 策略开启且候选为 synthetic-runtime 时：
    - `would_skip=true`
    - `first_skip_reason` 给出明确原因（strict 与 synthetic 冲突）
    - `require_true_runtime_flag.violated=true`
  - `collect_backend_diagnostics(...)` 增加 `real_switch_require_true_runtime` 并透传到 real-switch 诊断。
  - CLI 文本报告增加 strict 状态与 candidate runtime mode 显示。

- 更新 `tests/test_environment_backends.py`
  - gated helper `_require_real_backend_switch_candidate(...)` 支持并透传 `require_true_runtime`。
  - 新增 strict 模式下 synthetic 候选触发 `SkipTest` 的覆盖。

- 更新 `tests/test_backend_diagnostics.py`
  - 覆盖 strict 配置解析（含非法值校验）。
  - 覆盖 candidate runtime mode 输出。
  - 覆盖 strict API 在 synthetic 候选下触发 skip 原因。
  - 覆盖 CLI `--real-switch-require-true-runtime` 参数透传。

## 兼容性说明

- 默认行为不变：未开启 strict 策略时，real-switch 仍按原有逻辑执行/skip。
- 新增字段为增量输出，不移除既有字段。
- 仅当显式启用 strict 策略时，synthetic 候选会被门禁拦截。

## 使用示例

```bash
# 启用真实切换测试并要求 true-runtime 候选
export ENABLE_REAL_BACKEND_TESTS=1
export REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1

python -m mbrl.environments.backends.diagnostics \
  --real-switch-physics-options-json '{"genesis":{"skip_dependency_check":true}}' \
  --real-switch-require-true-runtime \
  --json
```

