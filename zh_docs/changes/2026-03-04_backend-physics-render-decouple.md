# 2026-03-04 backend-physics-render-decouple

## 背景

当前环境构建路径默认强绑定 MuJoCo：环境注册是单层映射，`env.render(...)` 直接调用具体环境实现，难以在不改主流程的情况下替换物理引擎或渲染后端。

## 改动

- 新增后端抽象目录：`mbrl/environments/backends/`
  - `physics.py`：定义 `PhysicsBackend`，并注册 `mujoco / isaacsim / genesis / newton`
  - `render.py`：定义 `RenderBackend`、`dispatch_render(...)`，并注册 `native / headless / none / isaacsim / genesis / newton`
  - `registry.py`：把环境注册表升级为 `env -> physics_backend -> (module, class)` 的结构
- 改造 `mbrl/environments/__init__.py`
  - `env_from_string(...)` 支持后端选择参数：
    - `physics_backend`（别名：`simulator_backend`）
    - `render_backend`（别名：`renderer_backend`）
    - `physics_backend_options`
    - `render_backend_options`
  - 保持向后兼容：默认仍是 `mujoco + native`
  - 将后端选择写回 `env.init_kwargs`，确保克隆环境时保持一致
- 改造渲染调用点：
  - `mbrl/rollout_utils.py`
  - `mbrl/controllers/abstract_controller.py`
  - 改为通过 `dispatch_render(env, ...)` 调度渲染后端
- 更新入口参数透传：`mbrl/main.py`
  - 支持在配置顶层直接设置后端键，并合并到 `env_params` 后再创建环境
- 配置示例更新：`experiments/cee_us/settings/construction/common/construction_env.yaml`
  - 增加 `physics_backend: "mujoco"` 与 `render_backend: "native"` 示例

## 验证

- 运行 `python3 -m compileall` 对改动模块进行编译检查，通过。
- 运行最小导入检查，确认后端工厂函数可正常解析默认后端并读取环境注册表。

## 注意 / 后续

- 目前 `isaacsim / genesis / newton` 仍为占位后端：已完成接口接入点，但尚未接入具体环境实现与真实渲染桥接。
- 现有环境资产（XML、`mujoco_py` API、gym robotics）仍是 MuJoCo 生态。真正切到新物理引擎需要逐环境迁移实现。
