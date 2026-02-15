# 2026-02-15 18e43c6 expand-gnn-crash-course

## 背景

之前的 [zh_docs/gnn_guide/01_gnn_crash_course.md](../gnn_guide/01_gnn_crash_course.md) 更像提纲，概念密度高但对新手不够友好；需要把每个概念补齐直觉解释、例子和与本项目实现的对应关系，方便后续阅读代码。

## 改动

- 大幅扩写 [zh_docs/gnn_guide/01_gnn_crash_course.md](../gnn_guide/01_gnn_crash_course.md)
  - 增加“直觉比喻 + 小例子 + 常见术语对照”
  - 细化 message passing（edge / aggregate / node）流程，并给出最小伪代码
  - 补充 `mean` vs `sum` 聚合在“对象数可变”场景下的尺度差异
  - 解释 global/context（`agent state`、`action`）为什么必须作为条件输入
  - 增加常见图概念：多轮传递、有向/无向、自环
  - 增加单模型 vs 并行 ensemble 的常见张量形状直觉（`[B, nObj, d]` 与 `[E, B, ...]`）
  - 简述 normalization 与 delta target（`target_is_delta`）的训练稳定性动机

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 通过本地 diff 检查：章节结构完整、关键术语均有解释与例子、并与项目术语保持一致。

## 注意 / 回滚

- 仅文档变更：如需要回滚，直接 `git revert <commit>` 或回退该文件到上一版本即可。
