# 08 — segment_sum/mean（scatter 分桶聚合）到底在干嘛：从 0 到能手写实现

你卡在的点非常典型：

- 为什么要 flatten？
- `target_index`（或 segment id）到底是什么？
- `segment_sum/mean` / `scatter_add` 这种东西为什么能代表“聚合”？

这篇不讲抽象概念，**只用一个最小、可运行的例子**把“边消息 -> 按目标节点分桶 -> 求和/求平均”这条链路讲清楚。

---

## 1. 先把问题说清楚：我们想实现什么

在 message passing 里，我们会对每条边算一条 message：

- 边：`j -> i`（j 是发送者 sender，i 是接收者 receiver）
- message：`m_{i<-j}` 是一个向量（比如长度 `d_msg`）

然后我们要为每个节点 i 得到一个聚合后的向量：

$$
\bar m_i = \sum_{j \in \mathcal{N}(i)} m_{i\leftarrow j} \quad \text{或} \quad \text{mean}
$$

**重点：**邻居数量 $|\mathcal{N}(i)|$ 可变，所以我们必须把“很多条 message”压缩成“一个向量”。

---

## 2. 用“收件箱分桶”理解 target_index

把每个节点 i 想象成一个收件箱：

- 每条边 message 就是一封信
- 这封信要投递到“接收者 i”的收件箱

那么我们需要一个“投递地址数组”，告诉每封信应该放到哪个收件箱。

这个“投递地址数组”就是：

- `target_index`（也常叫 `segment_ids` / `receiver_index`）

如果你有 `M` 条边 message，那 `target_index` 就是长度为 `M` 的整数数组：

- 第 k 条 message 应该投递到哪个节点 id

然后 `segment_sum` 就是在做：

- 对所有 k：把 `msg[k]` 加到 `out[target_index[k]]` 上

这就是“分桶求和”。

---

## 3. 为什么要 flatten：因为 batch 里其实有很多张图

你在代码里看到：

- 节点特征 `[B, nObj, d_node]`

这里的意思是：

- batch 里有 B 个样本
- 每个样本是一张“有 nObj 个节点的图”

这其实是 **B 张互不相干的图**。

为了用一次 tensor 运算同时处理这 B 张图，常见做法是把所有节点拼成一个大节点表：

- `nodes_flat` 形状变成 `[B*nObj, d_node]`

对应地，所有边也拼成一个大边表。

关键是：拼在一起之后，**节点 id 必须全局唯一**。

- 第 0 个样本的节点 id：`0..nObj-1`
- 第 1 个样本的节点 id：`nObj..2*nObj-1`
- ...

这就是所谓的 **offset（偏移量）**：第 b 个样本的节点整体加上 `b*nObj`。

---

## 4. 一个最小例子：手算一次 segment_sum

### 4.1 设定：B=1，nObj=3

有 3 个节点：0,1,2。

我们造 4 条 message（想象来自不同 sender），每条 message 是 1 维（`d_msg=1`）：

- message 值：`msg = [10, 1, 100, 7]`
- 它们要投递的 receiver：`target_index = [0, 0, 2, 1]`

意思是：

- 10 和 1 都投递给节点 0
- 100 投递给节点 2
- 7 投递给节点 1

那么聚合 `sum` 的结果应该是：

- out[0] = 10 + 1 = 11
- out[1] = 7
- out[2] = 100

也就是 `out = [11, 7, 100]`。

`segment_sum/scatter_add` 做的就是这件事（只不过 msg 通常是向量，逐元素加）。

---

## 5. 代码版：用 PyTorch 手写 scatter_add / segment_sum

下面这段代码你可以直接复制到 notebook 或 python 里跑（只依赖 torch）：

```python
import torch

# 4 条 message，每条维度 d_msg=1
msg = torch.tensor([[10.0], [1.0], [100.0], [7.0]])  # [M, d_msg]

# 每条 message 要投递到哪个 receiver 节点
target_index = torch.tensor([0, 0, 2, 1], dtype=torch.long)  # [M]

num_nodes = 3
out = torch.zeros((num_nodes, msg.shape[1]))  # [num_nodes, d_msg]

# scatter_add_: out[target_index[k]] += msg[k]
out.index_add_(dim=0, index=target_index, source=msg)
print(out.squeeze(-1))  # tensor([ 11.,   7., 100.])

# 如果要 mean，你需要每个桶的计数
counts = torch.zeros((num_nodes, 1))
counts.index_add_(dim=0, index=target_index, source=torch.ones((msg.shape[0], 1)))
mean_out = out / counts.clamp_min(1.0)
print(mean_out.squeeze(-1))
```

你会看到：

- `sum` 是“分桶相加”
- `mean` = `sum / count`

很多库里的 `segment_mean` 本质就是做了计数（或者同时做 sum 和 count）。

---

## 6. 连接回 GNN：message 是怎么来的？target_index 是怎么来的？

### 6.1 message 是怎么来的

在 GNN 里，message 通常来自 edge MLP：

- 输入：sender 节点特征、receiver 节点特征、以及 global context
- 输出：一条 message 向量

你可以把它理解成：

> “sender 这个物体以当前状态，对 receiver 的下一步造成什么影响？”

### 6.2 target_index 来自 edge_index（谁发给谁）

通常你会有一个 `edge_index`，表示边列表。

常见约定：

- `senders`：每条边的 sender 节点 id
- `receivers`：每条边的 receiver 节点 id

那么：

- 如果你要把消息聚合到 receiver 上（最常见）
  - `target_index = receivers`

反过来，如果你的实现是“聚合到 sender”（少见，但也可能），那就是：

- `target_index = senders`

所以你读代码时只要盯住：

- message 算的是 `sender -> receiver` 还是 `receiver -> sender`
- segment/scatter 用的是哪一个 index

就不会迷路。

---

## 7. 最容易犯错的点：batch 的 offset 没加对

当 B>1 时，你不能让不同样本共享同一套节点 id（否则会把不同图的消息混在一起）。

正确做法是给每个样本的节点 id 加 offset：

- 样本 b 的节点整体加 `b*nObj`

同理，边里的 sender/receiver id 也要加相同 offset。

如果你看到代码里有类似：

- `base = torch.arange(B) * nObj`
- 然后 `senders + base` / `receivers + base`

这就是在做 offset。

---

## 8. 你可以用一句话记住这件事

- `segment_sum/mean/scatter_add` 不是“神秘 GNN 操作”
- 它只是：**把边消息按 receiver 节点 id 分桶，然后在每个桶里做 sum 或 mean**

如果你愿意，我可以再补一个“B=2、nObj=3 的完整 edge_index + offset + 聚合”的手算例子（会更彻底）。
