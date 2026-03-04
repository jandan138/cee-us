# 2026-03-04 backend-plugin-idempotence-phase4

## 背景

phase3 已支持运行时插件加载，但在重复调用或并发调用插件加载入口时，仍存在重复注册风险（例如 `register_*` 在 `override=False` 时冲突）。

## 改动

- 更新 `mbrl/environments/backends/plugins.py`
  - 增加模块级锁，保证插件注册过程线程安全。
  - 对同一模块实现“每进程默认只注册一次”的幂等保护。
  - 保留 `force_reload` 行为，用于显式重载并重新注册。
  - 新增测试辅助函数 `reset_loaded_backend_plugins()`。
- 更新 `mbrl/environments/backends/__init__.py`
  - 导出 `reset_loaded_backend_plugins`。
- 更新插件示例 `tests/backend_plugin_example.py`
  - 改为通过 `register_backends()` 执行注册。
  - 增加 `REGISTER_CALL_COUNT` 计数器用于验证幂等性。
- 更新 `tests/test_environment_backends.py`
  - 新增并发场景测试：`test_plugin_loader_thread_safe_single_registration`。
  - 新增插件导入失败测试：`test_plugin_loader_import_error`。
  - 通过快照恢复 backend registry，避免测试间污染。

## 验证

- `python3 -m compileall mbrl/environments/backends/plugins.py tests/test_environment_backends.py`
- `python3 -m unittest tests.test_environment_backends -v`
  - `Ran 9 tests ... OK`

## 注意 / 后续

- 当前线程安全保证是“单进程”级别；多进程并发启动仍需要进程间协调策略。
- 后续可考虑增加插件依赖诊断/健康检查接口，提前暴露 SDK 缺失问题。
