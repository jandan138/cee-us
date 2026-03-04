# 2026-03-04 real-backend-switch-test-readiness-phase5

## 背景

当前已经有后端解耦、插件加载、幂等注册与并发安全测试，但“真实后端切换测试”还需要明确触发条件和执行入口。

## 改动

- 更新 `tests/test_environment_backends.py`
  - 新增 `test_real_backend_switch_when_enabled`：
    - 默认 `skip`（避免在没有真实后端实现和依赖时误报失败）
    - 当设置 `ENABLE_REAL_BACKEND_TESTS=1` 时启用
    - 自动检查：
      1) 是否存在 `implemented=True` 且非 `mujoco` 的 physics backend
      2) `ENV_REGISTRY` 是否有该 backend 的 env 映射
      3) 若满足条件，执行一次真实 `env_from_string(..., physics_backend=<non-mujoco>)` 切换构造

## 验证

- `python3 -m unittest tests.test_environment_backends -v`
  - 当前环境下该测试按预期 `skip`，其余测试通过。

## 什么时候能引入“真的切换后端测试”

- **现在就能引入测试入口**（已经加上，受环境变量控制）。
- **变成“真正执行”**的条件是三件事同时满足：
  1) 仓库内有至少一个非 MuJoCo backend 的可运行实现（`implemented=True`）；
  2) 至少一个 env 在 `ENV_REGISTRY` 完成该 backend 映射；
  3) 运行机装好该 backend 依赖（并设置 `ENABLE_REAL_BACKEND_TESTS=1`）。
- 一旦上述条件满足，同一天即可把该测试从“只 skip”切到“真实执行”。
