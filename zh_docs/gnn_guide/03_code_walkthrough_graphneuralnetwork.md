# 03 — 读代码：GraphNeuralNetwork（单模型核心网络）

目标：读完这一篇，你能看懂 mbrl/models/torch_models/gnn_modules.py 里的 `GraphNeuralNetwork`，并能在脑子里跑一遍 forward。

## 1. 总体结构：三块 MLP

`GraphNeuralNetwork` 里最关键的是三套 MLP：

1) `edge_mlp`：算边消息（object-object）
2) `node_mlp`：更新每个 object 的动态状态（输出维度是 `node_dyn_dim`）
3) `global_mlp`：更新 agent 的状态（输出维度是 `global_dim`）

你可以把它理解成：
- 先算“物体间交互”
- 再更新“每个物体”
- 同时更新“agent（全局）”

## 2. 边是怎么建出来的：全连接去掉自环

`_get_edge_list_fully_connected(batch_size, num_nodes)` 做了三件事：

- 创建 `num_nodes x num_nodes` 全 1 矩阵
- 减去单位阵去掉自环
- `nonzero()` 得到边列表，然后复制到每个 batch sample 并加 offset

结果是 COO 形式：
- `edge_list` 形状 `[2, num_edges_total]`
- 其中 `row, col = edge_index`

直觉：把 batch 中每个图都“拼接”成一个大图，用 offset 区分不同样本。

## 3. Edge model：一条边的消息怎么计算

在 `_forward_transition_gnn` 里：

- 把节点属性 `node_attributes` 展平为 `node_attr`（`[B*nObj, node_attr_dim]`）
- 对每条边 (row, col)：取 source 和 target 的 node_attr
- 再拼上 context（`agent_state` 和 `action`）

最后过 `edge_mlp` 得到每条边的 `edge_attr`：

- `edge_attr` 形状 `[B*num_edges, hidden_dim]`

## 4. Aggregation：把边消息聚合回节点

`_node_model` 里用：

- `unsorted_segment_mean(edge_attr, segment_ids=row, num_segments=node_attr.size(0))`

这里 `segment_ids` 传的是 `row`（来自 `edge_index` 的第一行）。

更精确地说：
- 代码会把 `edge_attr[k]` 按 `row[k]` 这个节点 id 分桶聚合
- 因此每个节点会聚合“它在 `row` 中出现的那些边”的消息

（如果你在意“入边/出边”的语义：在不同实现里 `edge_index` 的方向定义可能不同；这里最可靠的方式就是按代码实际传入的 `segment_ids` 来理解。）

然后拼接：

- `out = [node_attr, agg, agent_state, action]`

交给 `node_mlp` 输出下一步的 object_dyn。

## 5. Global update：agent 怎么更新

global 更新的输入是：

- `global_node_input = [agent_state, action]`

再加上（可选）从节点聚合来的 `global_node_agg`：

- 这个由 `ignore_global_v_node` 控制
- 如果 `ignore_global_v_node=False`，会额外建一套 `edge_mlp_global` 来做 object->agent 的交互建模

最终 global 更新也是通过 `_node_model(..., mode="global")`，落到 `global_mlp`。

## 6. Message passing 次数

`forward()` 里会循环 `num_message_passing` 次：

- 每一轮都用新的 node_attr/global_attr 继续算边、聚合、更新

直觉：
- 1 次 message passing：只能建模“一跳邻居”影响
- 多次 message passing：允许信息传播更远（但也更难训练/更慢）

## 7. 输出是什么

`forward()` 返回：

- `global_attr`：下一步的 agent（或 agent_delta，取决于外层 wrapper 怎么训练）
- `node_attr`：下一步每个 object 的 dyn 状态（或 dyn_delta）

下一篇我们把它接到 `GNNForwardModel` 上：如何做 obs 预处理、归一化、目标构造和训练。
