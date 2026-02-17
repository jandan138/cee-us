# 2026-02-18 0cc964c world-model-mapping-across-gnn-guide

## 背景

前面的 GNN 导读更多是在解释“代码与实现细节”，但对于新手来说还缺一层关键连接：

- 论文的主题是 **structured world model + planning（MPC）**
- 代码里的每个模块（对象中心拆分、GNN、聚合、ensemble、segment/scatter）到底如何服务于“world model 能用于规划与 zero-shot 泛化”

为了让阅读路径更贴近论文叙事，需要在每篇导读里补一段“与论文主线的对应”。

## 改动

- 在以下导读文档中新增/补强“与论文 Structured World Model 主线的对应”小节（保持原文结构，仅补对应关系与动机）：
  - `01_gnn_crash_course.md`
  - `02_object_centric_dynamics_in_cee-us.md`
  - `03_code_walkthrough_graphneuralnetwork.md`
  - `04_code_walkthrough_gnn_forward_model_training.md`
  - `05_ensemble_parallel_gnn.md`
  - `06_how_to_run_and_tune_gnn_experiments.md`
  - `07_aggregation_demystified.md`
  - `08_segment_sum_mean_scatter_explained.md`
- 更新 `zh_docs/gnn_guide/README.md`：说明每篇会补充论文主线对齐信息，便于将 world model / planning 叙事与代码类名对应起来。

## 验证

- 文档为纯 Markdown 更新，无运行时验证需求。
- 本地通读确认：新增段落不引入与代码不一致的断言，避免“凭空脑补论文细节”，仅做主题对齐与动机解释。

## 注意 / 回滚

- 仅文档变更：如需回滚，`git revert 0cc964c` 即可。
