# 2026-02-18 36e2a0a paper-two-phase-pipeline-mapping

## 背景

虽然前面已经在各篇导读里补了“与论文 structured world model 主线的对应”，但对新手来说还缺一个更具体的桥梁：

- 论文两阶段（Intrinsic free-play → Extrinsic zero-shot）到底对应仓库里的哪些 settings 路径
- world model（GNN forward model）在这两阶段分别扮演什么角色

因此需要把“论文流程 → 本仓库 settings/运行方式”的对照写进导读关键章节。

## 改动

- 更新 [zh_docs/gnn_guide/02_object_centric_dynamics_in_cee-us.md](../gnn_guide/02_object_centric_dynamics_in_cee-us.md)
  - 新增“两阶段主线与 settings 路径对应”说明，并列出典型文件名示例
- 更新 [zh_docs/gnn_guide/04_code_walkthrough_gnn_forward_model_training.md](../gnn_guide/04_code_walkthrough_gnn_forward_model_training.md)
  - 明确 world model 学习发生在 intrinsic 阶段（curious_exploration），zero-shot 阶段复用/加载模型用于规划
- 更新 [zh_docs/gnn_guide/06_how_to_run_and_tune_gnn_experiments.md](../gnn_guide/06_how_to_run_and_tune_gnn_experiments.md)
  - 增加“最短路径”：各给一条 intrinsic 与 extrinsic 的 settings 示例，并强调统一入口 `python -m mbrl.main`

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 对照仓库实际 settings 文件路径（`curious_exploration/` 与 `zero_shot_generalization/`）确保示例存在。

## 注意 / 回滚

- 仅文档变更：如需回滚，`git revert 36e2a0a` 即可。
