
# 01 — GNN 速成（超详细版）：从 0 到能读懂本项目代码

你是 GNN 新手也没关系：这一篇把后续会出现的概念都展开讲清楚，并尽量用“直觉 + 小例子 + 张量形状（shape）”的方式解释。

读完目标：

- 能用自己的话解释：什么是节点/边/聚合/message passing/global context。
- 能把这些概念对应到本项目：object、agent、action、forward model。
- 看到 `[B, nObj, d]`、`[E, B, ...]` 这类形状不再困惑。

本项目里 GNN 主要用于**动力学模型（forward model）**：输入 $(o_t, a_t)$，预测 $o_{t+1}$ 或者变化量 $\Delta o$。

---

## 0. 一个最直观的比喻：群聊

把环境里的多物体系统想成一个群聊：

- 每个物体是一个人（节点 node）
- 物体之间的潜在交互是“可以互相发消息”（边 edge）
- 每个时间步：先互相发消息（message passing），再各自更新状态（node update）
- 机器人（agent）的状态和你给机器人的动作（action）像群公告：会影响所有人的变化

GNN 做的就是用神经网络模拟这种“互相影响”的更新。

---

## 1. 图（Graph）、节点（Node）、边（Edge）：到底是什么

### 1.1 图是什么

图包含：

- 节点集合 $V$：本项目里通常是“物体集合”（object set）
- 边集合 $E$：表示哪些物体之间可能有交互

为什么要用图？因为多物体动力学里，下一步状态往往由“交互”决定，比如推、碰撞、接触、遮挡。

### 1.2 节点特征（node features）是什么

节点特征就是一个向量，描述某个物体“此刻是什么状态”。在本项目里常拆成两类：

- 动态特征 dyn：会随时间变化，模型要预测（位置、速度、角度……）
- 静态特征 stat：不会随时间变化，不预测，只作为条件输入（物体类型 one-hot、类别 id、固定尺寸等）

对应到维度：

- `object_dyn_dim`：每个物体动态特征维度
- `object_stat_dim`：每个物体静态特征维度（有的环境为 0）

通俗例子（仅用于理解）：

- dyn = `[x, y, vx, vy]`
- stat = `[is_cube, is_cylinder, mass]`

### 1.3 边（edge）是什么，边特征一定要手工给吗

不一定。

很多 GNN 会把“相对距离/相对位置”当作边特征输入。但本项目的核心实现更偏向：

- 边消息（message）由一个 MLP 直接从“两个端点的节点特征 + 全局 context”学出来

也就是说，本项目里你可以把“边特征”理解为“网络算出来的消息向量”。

### 1.4 为什么用全连接图（fully-connected）

本项目常见做法：对 $nObj$ 个物体建全连接图（去掉自环）。

优点：不需要你手工指定“谁影响谁”，网络会学。

代价：边数大约是 $nObj(nObj-1)$（有向实现），对象数多时计算变重。

---

## 2. 为什么不用大 MLP：GNN 的归纳偏置（inductive bias）

你当然可以把所有物体拼成一个大向量，再用 MLP 预测下一步。但会遇到三个典型问题：

1) 对顺序敏感：交换 obj1/obj2 的顺序，输入就完全变了
2) 难泛化到对象数变化：训练是 3 个物体，测试 5 个物体，输入维度直接变
3) 参数不共享：网络没有被强制“用同一种规则”处理每个物体

GNN 的关键设计是“共享 + 聚合”：

- 所有边共享一套 `edge_mlp`
- 所有节点共享一套 `node_mlp`
- 聚合用 `sum/mean` 等对顺序不敏感的操作

这让模型天然更适合“多物体 + 可变数量”的任务（这也是本项目强调 zero-shot generalization 的原因）。

---

## 3. Message Passing（信息传递）：一步一步讲清楚

### 3.1 先用 3 个物体的小故事

想象 3 个方块 A、B、C：

- A 可能推到 B
- B 可能再推到 C

如果你只看每个方块自己的位置速度，预测下一步很难，因为关键在“相互影响”。

GNN 的做法：

1) 对每对物体算“消息”
2) 每个物体把收到的消息聚合
3) 每个物体根据聚合消息更新自己的下一步

### 3.2 数学形式（只为建立直觉）

- 边消息：

$$
m_{ij} = \phi_e(x_i, x_j, c)
$$

- 聚合：

$$
\bar{m}_i = \rho(\{m_{ij}\}_{j \in \mathcal{N}(i)})
$$

- 节点更新：

$$
x'_i = \phi_v(x_i, \bar{m}_i, c)
$$

其中：

- $x_i$：节点特征（物体 dyn/stat）
- $c$：全局 context（本项目里常见是 agent state + action）
- $\phi_e$：edge model（MLP）
- $\phi_v$：node model（MLP）
- $\rho$：聚合（mean 或 sum）

### 3.3 对应到代码名词

你后面读代码会看到：

- `edge_mlp(...)`：就是 $\phi_e$
- `unsorted_segment_mean/sum(...)`：就是 $\rho$
- `node_mlp(...)`：就是 $\phi_v$

---

## 4. 聚合（Aggregation）：为什么必须对顺序不敏感

### 4.1 置换不变性（Permutation Invariance）

物体没有天然顺序。一个合理的模型应该满足：

- 只要物体集合相同（只是排列不同），预测本质不应改变

`sum/mean/max` 这类聚合满足“换顺序输出不变”，因此适合汇总邻居消息。

### 4.2 mean vs sum：用数字举例

假设某节点收到两条消息：10 和 0。

- `sum` 得到 10
- `mean` 得到 5

如果邻居数量变多（多了两个 0）：

- `sum` 还是 10
- `mean` 变成 2.5

这说明：聚合会受到“邻居数量”的影响。

在对象数可变的场景里，`mean` 往往更稳（尺度不随对象数爆炸）；`sum` 有时表达力更强，但对对象数变化更敏感。

本项目通过 `aggr_fn: "mean"/"sum"` 选择。

---

## 5. 全局变量（Global / Context）：agent state 与 action 为什么要进入模型

### 5.1 为什么一定要看 action

同一个状态 $(o_t)$ 下，不同动作 $(a_t)$ 会导致完全不同的下一步。

如果模型看不见 action，就只能学一个“平均效果”，预测会很差。

### 5.2 本项目的 global context 通常是什么

- agent state：机器人/夹爪自身状态（全局）
- action：控制输入（全局）

它们会被拼到 edge_mlp/node_mlp 的输入里。

### 5.3 agent 作为“全局节点”还是“普通节点”

在本项目里两种设计都存在：

- agent 作为 global：走 `global_mlp` 更新
- agent 作为普通节点：把 agent 也当成一个节点，需要 embedding 对齐维度

你在 ensemble 版本里会更明显地看到这条分支。

---

## 6. message passing 轮数：为什么有时要多轮

- 1 轮：节点只聚合“一跳邻居”的信息
- 多轮：允许信息在图上多跳传播

例如 A 影响 B，B 再影响 C，这种间接影响可能用多轮更容易表达。

但多轮会更慢、更难训练，所以初学建议先从 `num_message_passing = 1` 开始。

---

## 7. 有向/无向、自环：这些术语在实现里怎么对应

### 7.1 有向 vs 无向

- 无向：A-B 只算一次
- 有向：A->B 与 B->A 分开算

本项目常见的全连接边列表通常包含 (i,j) 和 (j,i)，更像“有向实现”。直觉上这允许学习非对称影响。

### 7.2 自环（self-loop）

自环是 i->i。

本项目常见做法是去掉自环（`ones - eye`），但节点更新时仍会直接拼上自身特征，所以自信息不会丢。

---

## 8. 张量形状（shape）详细直觉：你读代码最常卡住的点

### 8.1 单模型（无 ensemble）

你会频繁看到：

- agent：`[B, agent_dim]`
- obj_dyn：`[B, nObj, object_dyn_dim]`
- obj_stat：`[B, nObj, object_stat_dim]`（可能为 None/空）
- action：`[B, action_dim]`

其中 `B` 是 batch size。

### 8.2 为什么要 flatten

实现经常把 `[B, nObj, d]` flatten 成 `[B*nObj, d]`，为了：

- 用同一个 MLP 批量处理所有节点
- 配合 segment 聚合（按节点 id 分桶聚合边消息）

### 8.3 ensemble 并行版本（多一维 E）

并行 ensemble 会多一维：

- agent：`[E, B, agent_dim]`
- obj_dyn：`[E, B, nObj, object_dyn_dim]`
- action：`[E, B, action_dim]`

其中 `E` 是 ensemble size。

---

## 9. 训练稳定性常用技巧：归一化与 delta 目标

### 9.1 归一化（Normalization）

不同维度尺度不同（位置/速度/one-hot），不归一化会让训练不稳定。

常见做法是对输入与输出分别做标准化：$\hat{x}=(x-\mu)/\sigma$。

### 9.2 delta 目标（target_is_delta）

`target_is_delta=true` 表示模型预测变化量：$\Delta x = x_{t+1} - x_t$。

优点：变化量通常更小、更平稳，更容易学。

---

## 10. 一段最小伪代码：把概念和实现对齐

下面这段可以当作你读本项目核心 GNN 的“心智模型”（不必逐字对应代码）：

```python
# agent:   [B, agent_dim]
# obj_dyn: [B, nObj, object_dyn_dim]
# obj_stat:[B, nObj, object_stat_dim] or None
# action:  [B, action_dim]

node_attr = concat(obj_dyn, obj_stat)  # [B, nObj, node_attr_dim]

for _ in range(num_message_passing):
	row, col = edge_index  # 全连接去自环
	msg = edge_mlp(node_attr[row], node_attr[col], context=[agent, action])
	agg = segment_mean_or_sum(msg, segment_ids=row, num_segments=B*nObj)
	node_attr = node_mlp(concat(node_attr, agg, agent, action))

agent_out = global_mlp(concat(agent, action, graph_summary))
return agent_out, node_attr
```

下一篇我们会把这些概念和本项目的 observation 拆分方式对齐。
