# 01 — GNN 最小速成：你需要掌握哪些概念

这一篇只讲“足够看懂本项目代码”的 GNN。

## 1. 图、节点、边：为什么叫 Graph Neural Network

- **节点（node）**：这里对应“一个对象”（object），例如一个方块/按钮/抽屉把手。
- **边（edge）**：表示两个对象之间可能存在交互（碰撞/接触/遮挡/影响）。
- **全连接图（fully-connected）**：本项目默认把对象之间建成全连接（除了自环），让网络自己学哪些交互重要。

直觉：如果世界里有多个物体，物体 A 的下一时刻可能会受到物体 B 的影响。GNN 给这种“多体交互”提供了结构化建模方式。

## 2. Message Passing（信息传递）

最常见的 GNN 结构是：

1) 对每条边计算“消息” $m_{i\leftarrow j}$（edge model）
2) 对每个节点把来自邻居的消息聚合（aggregate）
3) 用聚合后的信息更新节点状态（node model）

在代码里通常表现为：

- `edge_mlp(...)`：对 (source node, target node, context) 生成消息
- `unsorted_segment_mean/sum(...)`：把同一个目标节点的消息聚合起来
- `node_mlp(...)`：节点更新

## 3. 聚合函数（Aggregation）

聚合要满足一个关键性质：**对邻居顺序不敏感（permutation invariant）**。

常用：
- `mean`：平均聚合，稳定，尺度不随邻居数量爆炸
- `sum`：求和聚合，表达能力强，但节点数变化会影响尺度

本项目在配置里用 `aggr_fn: "mean"` 或 `"sum"` 选择。

## 4. 全局变量（Global / Context）

本项目里有两个“全局”的东西：

- **agent state**（例如机械臂/夹爪的状态）
- **action**（控制输入）

它们并不是某一个 object 的属性，但会影响所有 object 的下一步。

因此你会看到模型把 `agent_state` 和 `action` 当作全局 context，拼到边模型/节点模型的输入里。

## 5. 为什么 GNN 能做“对象数可变”的泛化

GNN 的参数是“共享”的：同一个 `edge_mlp` 用在所有边上，同一个 `node_mlp` 用在所有节点上。

这意味着：
- 训练时见过 3 个物体
- 测试时遇到 5 个物体

只要你仍然按同样的规则建图并做聚合，它仍然能工作。

这就是本项目一些 settings 里强调的：GNN 支持 zero-shot generalization（对对象数量变化的泛化）。

## 6. 你需要的 PyTorch 张量形状直觉

本项目有两种实现：

- 单模型：batch 维度是 `[B, ...]`
- 并行 ensemble：多了 ensemble 维度 `[E, B, ...]`

你会频繁看到这样的形状：
- 对象动态状态：`[B, nObj, object_dyn_dim]`
- 对象静态属性：`[B, nObj, object_stat_dim]`（可能为 0）
- agent 状态：`[B, agent_dim]`
- 动作：`[B, action_dim]`

下一篇我们就把这些形状和本项目的 observation 对齐。
