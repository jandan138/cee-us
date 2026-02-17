# 2026-02-17 367d1ed global-context-world-model

## 背景

在阅读 `02_object_centric_dynamics_in_cee-us.md` 与 `01_gnn_crash_course.md` 时，很多新手会卡在：

- “全局变量（global/context）”到底是什么
- 文档里说“把 agent state 与 action 拼进 edge_mlp/node_mlp”，这里的“模型”到底指哪个模型
- 这在论文 *Curious Exploration via Structured World Models...* 的叙事里属于哪一块（为什么与 world model 主题强相关）

需要一篇独立文档把“论文主线（structured world model + planning）”与“代码实现（GNNForwardModel / GraphNeuralNetwork 的 global 分支）”对齐解释。

## 改动

- 新增文档 [zh_docs/gnn_guide/09_global_context_agent_action_in_world_model.md](../gnn_guide/09_global_context_agent_action_in_world_model.md)
  - 明确“模型”指世界模型/动力学模型/forward model（而非 policy/value/reward）
  - 从论文 world model 主题解释为什么必须 action-conditioned，以及为什么 agent state 是 central global node
  - 对齐到本仓库实现：`GNNForwardModel`、`GraphNeuralNetwork` 的 `edge_mlp/node_mlp/global_mlp` 与 `ignore_agent_object` 分支
- 更新导读索引 [zh_docs/gnn_guide/README.md](../gnn_guide/README.md)
  - 在阅读顺序中加入 09

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 对照代码注释与函数签名，确保 `global_context` 在实现中对应传入的 `agent_state`，并解释其命名易混点。

## 注意 / 回滚

- 仅文档变更：如需回滚，`git revert 367d1ed` 即可。
