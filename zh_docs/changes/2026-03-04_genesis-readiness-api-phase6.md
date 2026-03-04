# 2026-03-04 genesis-readiness-api-phase6

## 背景

你明确指出当前机器的 uv 环境大概率未安装 Genesis。仅靠手工尝试后端切换会导致测试结果不确定，缺少程序化“是否就绪”的判断接口。

## 改动

- 更新 `mbrl/environments/backends/physics.py`
  - 新增 `physics_backend_readiness(backend_name, options=None)`：
    - 返回结构化结果：`backend / ready / error_type / reason`
    - 覆盖场景：未知后端、未实现后端、依赖缺失、就绪可用
  - 保留 Genesis 依赖检查与 `skip_dependency_check` 开关。
- 更新 `mbrl/environments/backends/__init__.py`
  - 导出 `physics_backend_readiness`。
- 更新 `tests/test_environment_backends.py`
  - 新增 `test_physics_backend_readiness_reports_genesis_missing_dependency`
  - 新增 `test_physics_backend_readiness_can_skip_genesis_dependency_check`
  - 通过 mock 固化行为，不依赖本机是否安装 genesis。

## 验证

- `python3 -m unittest tests.test_environment_backends -v`

## 结论（当前机器什么时候能跑“真 Genesis 切换测试”）

- 你现在就可以通过 `physics_backend_readiness("genesis")` 判断当前环境是否可跑。
- 当它返回 `ready=true`，并且存在至少一个 `genesis` env 映射，再设置 `ENABLE_REAL_BACKEND_TESTS=1`，同一天即可执行真实切换测试。
