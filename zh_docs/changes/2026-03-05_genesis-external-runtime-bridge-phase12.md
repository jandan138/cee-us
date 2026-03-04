# 2026-03-05 genesis-external-runtime-bridge-phase12

## 背景

phase11 的 strict true-runtime 门禁已经可用，但在很多 CI/开发机上会出现一个现实问题：  
主进程 Python（例如项目 `.venv` 的 3.8）无法直接 `import genesis`，而 Genesis 实际运行时在另一个解释器（例如 `.venv-genesis`）。

这会导致当前 readiness 只能判定为缺依赖，无法表达“外部 Genesis runtime 已就绪”的情况。

## 改动

- 更新 `mbrl/environments/backends/physics.py`
  - `GenesisPhysicsBackend.prepare_backend(...)` 增加可选 external runtime 探测路径：
    - 当本地 `find_spec("genesis")` 失败时，如果配置了 `external_python`，会在子进程执行 import probe；
    - 支持 `external_probe_timeout_sec`；
    - 支持注入 `PYOPENGL_PLATFORM`（以及 `pyopengl_platform` 等价键）；
    - 支持附加环境 `external_probe_env` / `external_env`。
  - 默认行为保持不变：未配置 external probe 时，仍按原逻辑报 `genesis` 缺失。

- 更新 `mbrl/environments/backends/diagnostics.py`
  - 新增 real-switch 环境变量桥接（自动注入到 `options_by_backend.genesis`）：
    - `REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON`
    - `REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC`
    - `REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM`
  - 新增对应 CLI 参数（可选）：
    - `--real-switch-genesis-external-python`
    - `--real-switch-genesis-probe-timeout-sec`
    - `--real-switch-genesis-pyopengl-platform`
  - 文本诊断报告增加 external runtime bridge 的 env 显示。

- 更新测试
  - `tests/test_environment_backends.py`
    - 覆盖 external probe 成功/失败行为；
    - 覆盖 real-switch 候选构造复用 external probe 选项。
  - `tests/test_backend_diagnostics.py`
    - 覆盖 external probe env 配置桥接与 timeout 参数校验。

## strict true-runtime 语义

- 本次不放松 phase11 门禁语义：
  - `skip_dependency_check=true` 仍被判定为 `synthetic-runtime`；
  - 开启 strict（`REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1`）时，synthetic 候选仍会被阻断。
- external runtime probe 不是 synthetic 跳过逻辑，不会把 `skip_dependency_check` 变成 true。

## 使用示例（`.venv-genesis`）

```bash
# 1) 启用 real-switch 检查 + strict true-runtime
export ENABLE_REAL_BACKEND_TESTS=1
export REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1

# 2) 把 Genesis probe 指向外部解释器
export REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON=.venv-genesis/bin/python
export REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC=15
export REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM=egl

# 3) 运行诊断（会把 external probe 选项桥接到 genesis backend options）
python -m mbrl.environments.backends.diagnostics --backend genesis --json
```

