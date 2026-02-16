# 2026-02-16 e2ad057 segment-sum-mean-scatter

## 背景

在阅读 GNN 的实现时，`segment_sum/segment_mean`、`scatter_add` 这类操作经常是“看起来像魔法”的地方：

- 为什么要把节点 flatten 成 `[B*nObj, d]`
- `target_index/receiver_index` 到底是什么
- 为什么按 index 分桶相加就等价于“聚合邻居消息”

需要一篇独立文档把这些概念用最小例子讲透，方便之后读核心 GNN 模块。

## 改动

- 新增文档 [zh_docs/gnn_guide/08_segment_sum_mean_scatter_explained.md](../gnn_guide/08_segment_sum_mean_scatter_explained.md)
  - 用“收件箱分桶”直觉解释 `target_index` 与聚合
  - 给出可运行的最小 PyTorch 示例（`index_add_` 实现 sum/mean）
  - 解释 batch 下的 offset（避免不同样本的图消息混桶）
- 更新导读索引 [zh_docs/gnn_guide/README.md](../gnn_guide/README.md)
  - 在阅读顺序中加入 08

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 本地检查示例代码逻辑完整，概念与术语能对应到 `segment_* / scatter_*` 的实现。

## 注意 / 回滚

- 仅文档变更：如需回滚，`git revert e2ad057` 即可。
