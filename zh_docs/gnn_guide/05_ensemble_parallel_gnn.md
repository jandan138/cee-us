# 05 — 并行 Ensemble GNN：为什么要用、代码怎么读

目标：看懂本项目的 ensemble GNN dynamics（更常用于不确定性估计与 MPC）。

## 1. 为什么要 ensemble

在 model-based 控制里，单模型可能会“过度自信”。Ensemble 常用于：

- 估计 epistemic uncertainty（模型不确定性）
- 在 MPC 里用 ensemble 的均值/方差做更稳健的规划
- 缓解过拟合与数据分布漂移

本项目的 settings 里常见 `n: 5`（5 个模型并行）。

## 2. 本项目的 ensemble 是“并行计算”而不是 Python 循环

你会看到：

- wrapper：mbrl/models/gnn_ensemble.py（`GNNForwardEnsembleModel`）
- 核心网络：mbrl/models/torch_parallel_ensembles/gnn_modules.py

这里的 `GraphNeuralNetworkEnsemble` 不是 `for k in range(n): ...`，而是把 ensemble 维度当作张量的第 0 维一起算：

- agent: `[E, B, agent_dim]`
- objects_dyn: `[E, B, nObj, object_dyn_dim]`
- action: `[E, B, action_dim]`

## 3. 两种结构：agent 是否作为 global node

`GNNForwardEnsembleModel` 支持两种风格：

- `agent_as_global_node=True`（默认）：
  - agent 走 global MLP
  - object 走 node MLP
  - 与单模型的 GraphNeuralNetwork 更接近

- `agent_as_global_node=False`：
  - 把 agent 当成“一个和 object 同质的节点”（nObj+1）
  - 为了对齐维度，可能需要 embedding（`embedding_dim`）
  - 对应 `HomogeneousGraphNeuralNetworkEnsemble`

初学建议先只看 `agent_as_global_node=True` 那条线。

## 4. 并行实现的关键：unsorted_segment_*_ensemble

单模型用 `unsorted_segment_mean/sum` 聚合。

ensemble 版用：

- `unsorted_segment_mean_ensemble`
- `unsorted_segment_sum_ensemble`

它们的输入多一维：
- edge_attr: `[E, B*num_edges, hidden_dim]`
- segment_ids: `[B*num_edges]`

聚合输出：
- `[E, B*num_nodes, hidden_dim]`

## 5. 训练上的差异（高层理解）

训练 loss 的思想不变：
- 仍然是 MSE

但你需要注意两点：

1) 是否 bootstrapped
- settings 里有 `bootstrapped: false/true`

2) 推理时 ensemble 如何用
- 有的 controller 用 mean
- 有的 controller 用每个成员的 rollout（更昂贵）

如果你下一步想深入“控制器如何使用 ensemble”，我建议我们再加一篇：从 controller（iCEM/MPC）侧追踪 `forward_model.predict_n_steps()` 的调用链。
