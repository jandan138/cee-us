# 2026-03-04 backend-plugin-loading-phase3

## 背景

上一阶段已完成物理/渲染后端解耦框架，但如果每接入一个新后端都要改核心代码（`mbrl/environments/backends/*`），扩展成本仍然偏高。

## 改动

- 新增运行时插件加载模块：`mbrl/environments/backends/plugins.py`
  - `normalize_plugin_modules(...)`：规范化插件模块列表
  - `load_backend_plugins(...)`：在运行时动态导入插件模块
- 更新 `mbrl/environments/backends/__init__.py`
  - 导出 `load_backend_plugins` 与 `normalize_plugin_modules`
- 更新 `mbrl/environments/__init__.py`
  - `env_from_string(...)` 新增 `backend_plugin_modules` 参数支持
  - 在选择物理/渲染后端前先加载插件模块
  - 将插件模块列表写回 `env.init_kwargs`，保证环境克隆时可复现注册
- 更新 `mbrl/main.py`
  - 顶层参数透传新增 `backend_plugin_modules`
- 新增测试支撑与用例
  - `tests/backend_plugin_example.py`：示例插件模块（导入时注册 physics/render/env）
  - `tests/test_environment_backends.py`：新增插件加载与规范化用例

## 验证

- `python3 -m compileall mbrl/environments/backends mbrl/environments/testsupport_dummy_env.py tests`
- `python3 -m unittest tests.test_environment_backends -v`

## 注意 / 后续

- 当前插件机制提供“注册入口”能力，不包含引擎 SDK 生命周期管理（如 Isaac Sim app loop）。
- 下一步建议：为插件后端补 `health_check` 与依赖诊断接口，支持启动前自检。
