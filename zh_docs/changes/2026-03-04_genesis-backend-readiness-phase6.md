# 2026-03-04 genesis-backend-readiness-phase6

## 背景

此前虽然框架支持多后端注册，但 `genesis` 在核心后端实现中仍是“不可执行占位”。即使外部完成 env 映射，运行时仍会直接报未实现，无法进入真实切换路径。

## 改动

- 更新 `mbrl/environments/backends/physics.py`
  - 将 `GenesisPhysicsBackend` 从不可执行占位改为可执行后端实现。
  - 增加 Genesis 依赖探测（默认检查 `genesis` 包可用性）。
  - 支持 `physics_backend_options.skip_dependency_check=true`，用于纯框架测试或 CI 环境的 synthetic 验证。
- 更新 `tests/test_environment_backends.py`
  - 新增 `test_genesis_backend_can_switch_when_mapped_and_dependency_check_skipped`
    - 验证：当 env 映射已注册且跳过依赖检查时，`physics_backend='genesis'` 可以完成切换构造。
  - 新增 `test_genesis_backend_requires_dependency_by_default`
    - 验证：默认情况下缺少 genesis 依赖会抛出明确 `ImportError`。

## 验证

- `python3 -m unittest tests.test_environment_backends -v`

## “什么时候能引入真的切换后端测试”（Genesis）

- **现在就能在代码里挂上自动激活入口**（已完成）。
- **真正执行 Genesis 切换测试** 的触发条件：
  1) 安装 Genesis 运行时依赖（`import genesis` 成功）；
  2) 至少一个 env 在 `ENV_REGISTRY` 有 `genesis` 映射；
  3) 设置 `ENABLE_REAL_BACKEND_TESTS=1`（以及你们 CI 的对应开关策略）。
- 条件满足后，同一天可从 skip 切到真实执行。
