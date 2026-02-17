# GNN 导读（面向新手 + 面向本项目）

目标：读完本导读后，你应该能同时回答两类问题：

1) **GNN 是什么、怎么工作的**（message passing、聚合、归纳偏置）
2) **本项目的 GNN 在哪里、输入输出是什么、怎么训练/怎么用在 MPC 里**

本导读默认你会：Python、PyTorch 基础张量操作；不要求你有 GNN 经验。

补充说明：本导读会尽量在每篇开头加一小段“与论文 Structured World Model 主线的对应”，帮助你把：

- 论文里的 world model / planning 叙事
- 与本仓库里的具体类（`GNNForwardModel` / `GraphNeuralNetwork` 等）

对齐起来。

## 阅读顺序（建议）

1. 01_gnn_crash_course.md：GNN 最小必要知识（不讲花活）
  - 如果你对“聚合/为什么要 mean/sum”没概念，先看：07_aggregation_demystified.md
2. 02_object_centric_dynamics_in_cee-us.md：为什么本项目需要 GNN；观测如何拆成“agent + objects”
3. 03_code_walkthrough_graphneuralnetwork.md：看懂核心网络 GraphNeuralNetwork（单模型版）
4. 04_code_walkthrough_gnn_forward_model_training.md：看懂 GNNForwardModel（预处理/归一化/损失/训练）
5. 05_ensemble_parallel_gnn.md：看懂并行 ensemble 版（更贴近论文/实验）
6. 06_how_to_run_and_tune_gnn_experiments.md：怎么跑、怎么改超参、常见坑怎么排
7. 07_aggregation_demystified.md：把“聚合”讲到完全不懵（为什么需要、怎么实现、怎么选 mean/sum）
8. 08_segment_sum_mean_scatter_explained.md：把 `segment_sum/mean` / `scatter_add` 讲到能自己写出来（含最小可运行例子）
9. 09_global_context_agent_action_in_world_model.md：从论文 world model 主线解释为什么必须输入 agent state + action、以及代码里对应哪条分支

## 关键代码索引

- 单模型 GNN dynamics wrapper：mbrl/models/gnn_model.py（类 `GNNForwardModel`）
- 并行 ensemble GNN dynamics wrapper：mbrl/models/gnn_ensemble.py（类 `GNNForwardEnsembleModel`）
- GNN 核心模块（单模型）：mbrl/models/torch_models/gnn_modules.py（类 `GraphNeuralNetwork`）
- GNN 核心模块（并行 ensemble）：mbrl/models/torch_parallel_ensembles/gnn_modules.py（类 `GraphNeuralNetworkEnsemble` 等）
- 环境提供的“对象中心”维度约定（agent_dim/object_dyn_dim/object_stat_dim/nObj）：
  - mbrl/environments/playground_env_wgoals.py
  - mbrl/environments/fpp_construction_env.py
  - mbrl/environments/robodesk_env.py
- 配置（YAML settings）：experiments/cee_us/settings/**/gnn*.yaml

## 你读完后应该能做到

- 画出本项目 GNN dynamics 的数据流：`obs, action -> (agent_in, obj_in) -> (agent_delta, obj_delta) -> next_obs`
- 理解为什么它能做“对象数可变”的泛化（zero-shot generalization）
- 能定位并修改：message passing 次数、聚合函数、是否建模 agent-object 交互、训练超参

如果你希望更“手把手”，我也可以再加一篇：用最小脚本把一个 batch 送入 GNN 并打印中间张量形状。
