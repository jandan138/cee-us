# 09 — 论文“Structured World Model”里 global/context 到底是什么：为什么必须喂给模型 agent state + action

你在 01/02 里看到的这句话：

> “agent state 与 action 会被拼到 edge_mlp/node_mlp 的输入里”

如果你是新手，很容易产生两个疑问：

1) **这里的“模型”到底指哪个模型？**（策略网络？价值网络？奖励模型？）
2) **这在论文里对应哪一块？为什么和 ‘world model’ 主题强相关？**

这篇专门把这两件事讲清楚，并且紧贴本文项目对应的论文：

- *Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation*（arXiv:2206.11403）

---

## 1. 先回答最关键的问题：这里的“模型”指的是哪一个模型？

这里的“模型”指的是：

- **世界模型（world model）/ 动力学模型（dynamics model）/ 前向模型（forward model）**

它的任务是学一个“环境下一步怎么变”的函数（或概率模型）。在最常见的表述里：

- 输入：当前状态/观测（本项目用 `obs_preproc(o_t)` 处理后的观测）和动作 `a_t`
- 输出：下一步观测 `o_{t+1}`，或输出增量 `Δo` 再还原出 `o_{t+1}`

在本仓库里，它对应的就是：

- 单模型版本：`GNNForwardModel`（见 `mbrl/models/gnn_model.py`）
- 并行 ensemble 版本：`GNNForwardEnsembleModel`（见 `mbrl/models/gnn_ensemble.py`）

而它们内部真正做 GNN 前向计算的网络模块是：

- `GraphNeuralNetwork`（见 `mbrl/models/torch_models/gnn_modules.py`）
- `GraphNeuralNetworkEnsemble`（见 `mbrl/models/torch_parallel_ensembles/gnn_modules.py`）

所以：

- 你在文档里看到的“把 agent state 与 action 拼到 edge_mlp/node_mlp”，指的就是 **世界模型内部的 GNN 动力学网络**。

它不是 policy/value/reward model；它是 “predict next state” 的那个模型。

---

## 2. 这在论文里对应哪一块：为什么它是“structured world model”的核心

论文的主旨是：

- 不只用世界模型算 intrinsic reward
- 而是把**结构化世界模型**放进控制环（planning/MPC）里，直接在模型里“规划未来的新奇（novelty）”

要做到这一点，世界模型必须能回答：

> “如果我现在做动作 a_t，未来几步（agent + objects）会怎么变化？”

这就决定了两件事：

1) 世界模型必须是 **action-conditioned**（以动作作为条件输入）
2) 世界模型必须显式处理 **agent-object / object-object 交互**（论文摘要强调信息主要在稀疏交互里）

这两个点，正是 `global/context` 设计出现的原因。

---

## 3. 为什么一定要看 action：不看 action 的世界模型会学成“平均世界”

对任何可控系统来说，同一个状态下，不同动作会导致不同下一步。

- 如果模型看不见动作 `a_t`，它只能把“所有可能动作导致的不同后果”平均掉
- 结果就是：预测看起来“还行”，但用来做规划会非常差（因为规划需要区分动作的因果后果）

把 action 喂给世界模型，本质是在学：

- “动作 → 状态变化” 的因果映射

在本项目的 GNN 里，action 不是某一个物体的属性，而是 **对整个系统同时生效的控制输入**。

所以它被当作 **全局输入（global context 的一部分）**，会被广播到：

- 每条边消息的计算（edge_mlp）
- 每个节点状态更新的计算（node_mlp）
- agent（全局节点）自身的更新（global_mlp）

---

## 4. 那 agent state 为什么也算 global/context：它是“中心节点（central agent node）”

在这个项目里，观测会被拆成两部分：

- `agent state`：机器人/夹爪/末端执行器自身状态（全局）
- `objects`：每个物体一个节点（对象中心表示）

你可以把它理解成一张“以机器人为中心”的交互图：

- 物体之间会互相影响（object-object）
- 机器人和物体也会互相影响（agent-object）

因此 agent state 在模型里常见两种角色：

- **作为全局节点的状态**（central agent node）
- **作为所有局部计算共享的条件**（broadcast 到每条边/每个节点）

这也是为什么代码里有 `global_dim` 这种参数：

- `global_dim = agent_dim`（见 `GraphNeuralNetwork.__init__`）

并且代码注释非常明确：

- `global_dim is for the central agent node`

---

## 5. 贴合代码：到底是哪一步把 agent/action 拼进 edge_mlp / node_mlp 的？

以单模型 `GraphNeuralNetwork` 为例（`mbrl/models/torch_models/gnn_modules.py`）：

### 5.1 先澄清一个容易误解的命名

在 `_forward_transition_gnn(node_attributes, global_context, action)` 里：

- `global_context` 实际上传的是 **agent_state**
- `action` 就是 **action**

也就是说：

- 在这份实现里，“global/context”这词更多是“全局输入”的意思，并不等价于“只有 action”。

### 5.2 edge_mlp（边消息）为什么要吃 agent_state + action

edge_mlp 的输入维度来自这句注释：

- “Edge MLP takes in the states of neighboring nodes and the agent state and action”

对应实现：对每条边 `(row, col)` 取：

- sender/receiver 的 node feature
- 再拼上当前 batch 的 `agent_state` 和 `action`

直觉上：

> 物体 A 对物体 B 的影响，不只取决于 A/B 自己的状态，也取决于机器人在干什么（agent_state）以及你现在施加的控制（action）。

### 5.3 node_mlp（节点更新）为什么也要吃 agent_state + action

node_mlp 更新的是每个物体节点的下一步动态量（或 Δ）。

它需要：

- 这个物体自己的当前状态
- 从邻居来的聚合消息（object-object interactions）
- 以及全局条件（agent_state + action）

这能让模型学到：

- “同样的物体间相对关系，在不同动作下会产生不同变化”

### 5.4 global_mlp（全局节点更新）到底在预测什么

代码里写得很直白：

- “Global MLP for the prediction of the agent's state!”

也就是：

- 世界模型不仅预测 objects 的下一步，也预测 agent 自身状态的下一步（或 Δ）

这对 planning/MPC 很重要：

- MPC rollout 过程中，agent state 也在演化；如果 agent state 不更新，后续预测会越来越不一致。

---

## 6. “agent 作为全局节点” vs “agent 作为普通节点”：本项目到底是哪种？

你在文档里看到的两种说法，其实对应的是两种常见建图方式：

### 6.1 方式 A：agent 是全局节点（central node）

- agent state 单独走 global 分支（`global_mlp`）
- objects 走 node 分支（`node_mlp`）

本项目就是这种：

- `GraphNeuralNetwork` 同时输出 `(global_attr_out, node_attr_out)`
  - `global_attr_out` 对应 agent 的预测
  - `node_attr_out` 对应每个 object 的预测

### 6.2 方式 B：把 agent 当成一个普通节点

这在一些其它论文/实现中会出现：

- 把 agent 也塞进节点集合里
- 让所有节点共享同一个 node 更新网络

这种方式要解决一个实际问题：

- agent 的状态维度和 object 的状态维度通常不同
- 需要 embedding/对齐维度（或者做 heterogeneous GNN）

本项目选择的是方式 A，因此你会在代码里看到：

- global_dim 独立于 node_dyn_dim
- 有单独的 global_mlp

---

## 7. 但你又看到“agent-object 交互”的分支：ignore_agent_object / ignore_global_v_node

你在 `GNNForwardModel` 里会看到：

- `ignore_agent_object`（配置项）
- 它被传给 `GraphNeuralNetwork(ignore_global_v_node=...)`

在 `GraphNeuralNetwork` 里：

- `ignore_global_v_node=False` 时，会额外启用 `edge_mlp_global`
- 它会计算一种“object-agent interaction”的特征，并聚合到 agent 更新里

直觉上：

- 打开它：世界模型更显式地建模 agent 与每个 object 的交互
- 关闭它：相当于做一种结构上的简化/消融（ablation），减少一条交互通道

这点和论文强调的“agent-object / object-object 交互信息稀疏但关键”是同一条主线：

- 世界模型要能抓住交互，探索/规划才会有效。

---

## 8. 一张总览图：把论文的“world model”主题和代码结构对上

你可以用下面这张“计算图”来把论文叙事和代码实现对齐：

1) 观测拆分（对象中心）

- `obs_preproc(o_t)`（可能去 goal）
- `o_t -> agent_state + object_dyn + object_stat`

2) 动力学预测（结构化 world model）

- 输入：`agent_state, objects, action`
- 输出：`agent_delta, obj_delta`（或 absolute）

3) 用在 MPC / planning

- rollout：用 world model 在想象里滚动多步
- intrinsic phase：规划未来 novelty（论文主线）
- extrinsic phase：在下游任务上零样本规划（论文主线）

---

## 9. 你接下来怎么读才不迷路（建议路径）

如果你现在的困惑是“这段话到底对应代码哪几行”，建议按这个顺序看：

1) `GNNForwardModel._get_model_input`：确认输入里确实有 `agent_state` 和 `action`
2) `GraphNeuralNetwork._forward_transition_gnn`：看 action/agent_state 如何被广播到每条边/每个节点
3) `ignore_agent_object / ignore_global_v_node`：理解 agent-object 交互分支是开还是关

如果你愿意，我也可以把 `02_object_centric...` 里相关段落补上“链接到本文”的提示，避免你来回跳。
