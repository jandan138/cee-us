# 06 — 怎么跑 GNN 实验、怎么改超参、常见坑

目标：你能从 settings 出发，跑一个 GNN/Ensemble GNN 实验，并知道改哪些 key 会影响什么。

## 与论文主线的对应：两阶段（intrinsic free-play → extrinsic zero-shot）如何落在 settings 上

论文的叙事主线是：

- **Intrinsic 阶段**：用 curiosity 驱动的 planning，在 structured world model 里规划“未来新奇”，得到高交互的 free-play 数据
- **Extrinsic 阶段**：用同一个学到的 world model，在下游任务上做 model-based planning，实现 zero-shot

在这个仓库里，这两段通常分别体现在 settings 路径/命名上（例如 `curious_exploration/` vs `zero_shot_generalization/`）。

所以这篇“怎么跑/怎么调参”本质上是在教你：

- 如何复现实验中“训练 world model + 用 world model 做规划”的 pipeline
- 哪些超参会直接影响 world model 的表达能力与泛化，从而影响论文关心的 zero-shot 效果

### 最短路径：分别跑一条 intrinsic 和一条 extrinsic

如果你想“按论文两阶段”快速建立直觉，可以先各跑一个：

- Intrinsic/free-play（curious exploration）：
  - `experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml`
  - `experiments/cee_us/settings/playground/curious_exploration/gnn_ensemble_cee_us.yaml`
- Extrinsic/zero-shot（downstream tasks）：
  - `experiments/cee_us/settings/construction/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_stack.yaml`
  - `experiments/cee_us/settings/playground/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_push4.yaml`

它们都用同一个入口命令：

- `python -m mbrl.main <settings.yaml>`

区别主要在 settings 中：

- 是否处在 curious exploration（以 intrinsic 目标驱动采样/训练）
- 是否处在 zero-shot generalization（加载训练好的 world model 直接规划）

## 1. 从 settings 选一个 GNN 实验

常见入口在：

- experiments/cee_us/settings/*/common/gnn_model.yaml
- experiments/cee_us/settings/*/common/gnn_ensemble.yaml
- experiments/cee_us/settings/*/curious_exploration/gnn_*.yaml
- experiments/cee_us/settings/*/zero_shot_generalization/gnn_*.yaml

例子（construction）：
- `experiments/cee_us/settings/construction/common/gnn_model.yaml`
- `experiments/cee_us/settings/construction/common/gnn_ensemble.yaml`

## 2. 运行方式（避免 import 路径坑）

建议用模块方式启动：

- `python -m mbrl.main <settings.yaml> ...覆盖参数...`

（我们之前遇到过 `python mbrl/main.py` 导致找不到 `experiments.*` 的问题。）

## 3. 你最常改的超参：model_params

以 gnn_model.yaml 为例：

- `hidden_dim`：消息/隐藏层宽度
- `num_layers`：MLP 深度
- `num_message_passing`：message passing 轮数（更大更慢，可能更强）
- `aggr_fn`：mean/sum
- `layer_norm`：是否 layer norm（训练稳定性）
- `ignore_agent_object`（在单模型里叫 `ignore_global_v_node`）：是否显式建模 agent-object 交互

建议调参顺序：
1) 先固定 message passing=1，调 hidden_dim
2) 再决定是否加 message passing

## 4. 训练超参：train_params

- `batch_size`：越大越稳但更吃显存
- `epochs` vs `iterations`：通常只用一个
- `learning_rate`：最敏感的超参之一
- `weight_decay`：正则
- `train_epochs_only_with_latest_data`：是否只用最新 rollout 的数据训练（在线学习场景）

## 5. 两个非常关键的开关：normalization 与 delta

- `target_is_delta: true`：通常推荐
- `use_input_normalization/use_output_normalization`：通常推荐
- `normalize_w_running_stats`：
  - true：更像 online（只看最新）
  - false：更像 offline（看全量）

如果你出现 loss 波动巨大、预测发散，优先检查这些。

## 6. 常见坑与排查

- **维度 assert 失败**：通常是 env 的 `agent_dim/object_*_dim/nObj` 与实际 observation 不一致
- **object_stat_dim=0 的情况**：确认代码路径处理 None/空 tensor
- **训练很慢**：GNN 的全连接边数是 $nObj(nObj-1)$，对象多时计算会变重
- **zero-shot（对象数变化）效果差**：可能是用 sum 聚合导致尺度随对象数变化；尝试 mean

## 7. 建议的“最小复现”运行策略

在你调参/改代码时，先跑一个很小的 smoke：

- `training_iterations=1`
- `number_of_rollouts=1`
- `rollout_params.task_horizon=20`
- `device='cpu'` 或小显卡先用 cpu

确认 pipeline 没问题后再放大。

如果你希望我把“最小 smoke 命令 + 常用覆盖参数表”也写成一页 cheat sheet，我可以补一篇 07。
