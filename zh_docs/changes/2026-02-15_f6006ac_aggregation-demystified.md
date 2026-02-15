# 2026-02-15 f6006ac aggregation-demystified

## 背景

在阅读 GNN 时，“聚合（aggregation）/置换不变性/mean vs sum”通常是第一道门槛：概念抽象、实现又经常藏在 `segment_*` 之类的函数里，导致新手很难建立直觉。

## 改动

- 新增聚合专讲文档 [zh_docs/gnn_guide/07_aggregation_demystified.md](../gnn_guide/07_aggregation_demystified.md)
  - 用直觉比喻解释为什么必须聚合（邻居数量可变 + 对象无序）
  - 解释“边上算消息、点上收消息”的整体流程，以及 `segment_sum/mean` 的分桶含义
  - 用数字例子讲清 `mean` vs `sum` 在对象数变化时的尺度差异
  - 简要介绍 attention 聚合作为扩展概念（不影响读本项目代码）
- 更新导读索引 [zh_docs/gnn_guide/README.md](../gnn_guide/README.md)
  - 在推荐阅读顺序里加入 07，并在 01 下加提示入口

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 本地检查章节结构与术语一致性，确保与本项目 `aggr_fn` 配置项对齐。

## 注意 / 回滚

- 仅文档变更：如需要回滚，`git revert <commit>` 或回退对应文件即可。
