# 2026-02-16 b97d6b2 beginner-friendly-segment-sum-mean

## 背景

之前的 [zh_docs/gnn_guide/08_segment_sum_mean_scatter_explained.md](../gnn_guide/08_segment_sum_mean_scatter_explained.md) 对纯新手仍然跳出了太多新概念（`batch/flatten/target_index`），阅读容易“看蒙”。需要把关键概念的引入顺序改成更循序渐进的版本。

## 改动

- 重写/增强 [zh_docs/gnn_guide/08_segment_sum_mean_scatter_explained.md](../gnn_guide/08_segment_sum_mean_scatter_explained.md) 的前半部分，使其更适合零基础读者：
  - 增加“词汇表”：用一句话解释 `batch/flatten/target_index` 等术语
  - 增加 `B=2, nObj=3` 的映射表：把 `(b, i)` 映射到全局节点 id `g=b*nObj+i`
  - 增加“混桶 vs 不混桶”的手算演示：直观看到为什么 batch 下必须做 offset
  - 将公式/较抽象的表述改为可选阅读（先建立直觉再看符号）

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 本地通读确认：新手能先靠例子理解，再回看代码与实现。

## 注意 / 回滚

- 仅文档变更：如需回滚，`git revert b97d6b2` 即可。
