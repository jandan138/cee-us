# 02 — 本项目里的“对象中心动力学”：观测怎么拆成图

这一篇回答：**本项目的 GNN 到底在预测什么？输入输出如何对应环境？**

## 1. 本项目的 GNN 是“动力学模型（forward model）”

在 model-based RL / MPC 里，你会需要一个模型：

- 输入：当前观测 $o_t$ 和动作 $a_t$
- 输出：下一步观测 $o_{t+1}$（或增量 $\Delta o$）

本项目的 GNN 就是这种 forward model，只不过它把观测拆成：

- agent（全局）部分
- objects（节点）部分

并显式建模 objects 之间的交互（边）。

## 2. 环境提供的“拆分约定”

关键是：环境类会提供四个属性，告诉模型如何拆 observation：

- `agent_dim`
- `object_dyn_dim`
- `object_stat_dim`
- `nObj`

例子：

- Playground（mbrl/environments/playground_env_wgoals.py）：
  - `agent_dim = 4`
  - `object_dyn_dim = 6`
  - `object_stat_dim = 3`
  - `nObj = num_objs`

- Construction（mbrl/environments/fpp_construction_env.py）：
  - `agent_dim = 10`
  - `object_dyn_dim = 12`
  - `object_stat_dim = 0`
  - `nObj = num_blocks`

- Robodesk（mbrl/environments/robodesk_env.py）：
  - `agent_dim = 24`
  - `object_dyn_dim = 13`
  - `object_stat_dim = num_object_types`
  - `nObj = len(env_body_names)`

## 3. 一个 observation 在本项目里长什么样

以最常见的形式（忽略 goal 之前）：

```
obs = [ agent | obj1_dyn | obj2_dyn | ... | objN_dyn | obj1_stat | ... | objN_stat | (可能还有 goal) ]
```

注意：很多环境是 goal-conditioned 的（Gym robotics / playground with goals）。

因此你会看到 forward model 在进入 GNN 前先做：
- `obs = env.obs_preproc(obs)`

例如 playground 的 `obs_preproc` 会把 goal 从 observation 里“去掉”（只预测非 goal 部分）。

## 4. 本项目把 observation 变成图输入的方式

对应到 GNNForwardModel（mbrl/models/gnn_model.py）里的 `_obs_preprocessing`：

- `flat_agent_state`：`[B, agent_dim]`
- `batch_objects_dyn`：`[B, nObj, object_dyn_dim]`
- `batch_objects_stat`：`[B, nObj, object_stat_dim]`（如果 `object_stat_dim==0`，就是空/None）

这三块就是：
- 全局节点（agent）
- object 节点们（每个 object 一个节点）
- object 的静态属性（不会随时间变，训练时还会 assert 前后相等）

## 5. 预测目标：delta 还是 absolute

在 settings 里你会看到：

- `target_is_delta: true`

这意味着模型输出的是：

- `agent_delta = agent_{t+1} - agent_t`
- `obj_delta = obj_{t+1} - obj_t`

然后再通过 `env.obs_postproc` 把 delta 加回去得到 next_obs。

这种设计通常更容易学（数值尺度更稳定）。

## 6. 为什么要分“动态/静态”属性

- 动态（dyn）：位置/速度/关节等会随时间变化的量，是预测目标
- 静态（stat）：类别、形状、颜色、物体类型 one-hot 等，不预测，只作为条件输入

这就是 `object_stat_dim` 的意义。

下一篇我们开始真正读网络结构：GraphNeuralNetwork 是怎么把这些输入变成输出的。
