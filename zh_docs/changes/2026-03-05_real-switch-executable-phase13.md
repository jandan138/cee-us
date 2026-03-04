# 2026-03-05 real-switch-executable-phase13

## 背景

phase12 已经支持 Genesis external runtime probe，但“真实切换后端测试”仍主要依赖 mock。
本阶段目标是补上可执行入口和可观测元数据，让真实切换链路可被稳定验证。

## 变更

- 更新 `mbrl/environments/backends/physics.py`
  - `physics_backend_readiness(...)` 新增：
    - `dependency_source`: `local` / `external` / `synthetic`
    - `runtime_mode`: `true-runtime` / `external-runtime` / `synthetic-runtime`
  - Genesis 路径按实际依赖来源设置上述元数据。

- 更新 `mbrl/environments/backends/diagnostics.py`
  - `candidate_runtime_mode` 新增 `external-runtime` 分类（基于 `dependency_source=external`）。
  - strict true-runtime 语义保持不变：仅阻断 `synthetic-runtime`。
  - 新增可复用入口：
    - `resolve_real_backend_switch_execution_target(...)`
    - 返回 `selected/skip_reason/env_name/backend_name/physics_backend_options/report`。

- 更新 `mbrl/environments/backends/__init__.py`
  - 导出 `resolve_real_backend_switch_execution_target`。

- 更新测试
  - `tests/test_backend_diagnostics.py`
    - 覆盖 `external-runtime` 分类与 execution target 入口行为。
  - `tests/test_environment_backends.py`
    - real switch helper 改为复用 execution target API。
    - 覆盖 `dependency_source` 新字段。
  - `tests/test_genesis_external_runtime_bridge.py`
    - 新增 phase13 可执行测试：
      - `test_phase13_real_switch_executes_external_probe_and_env_from_string`
      - 通过 `RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST=1` opt-in 执行。

## 测试结果

```bash
python3 -m unittest -v tests.test_backend_diagnostics tests.test_environment_backends tests.test_genesis_external_runtime_bridge
# Ran 49 tests ... OK (skipped=2)
```

```bash
RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST=1 \
python3 -m unittest -v \
  tests.test_genesis_external_runtime_bridge.GenesisExternalRuntimeBridgeTestCase.test_phase13_real_switch_executes_external_probe_and_env_from_string
# Ran 1 test ... OK
```

## 实机验证（Genesis external runtime）

在本机（主进程 Python 3.8 + `.venv-genesis` Python 3.10）下，已验证：

- `diagnose_real_backend_switch_test(...)` 在注册 Genesis 映射后可得到：
  - `would_skip=False`
  - `candidate.backend_name='genesis'`
  - `candidate_runtime_mode='external-runtime'`

这意味着“真实 external-runtime 切换路径”可执行，且与 strict true-runtime 策略兼容。
