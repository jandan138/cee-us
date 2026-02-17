# 04 — 读代码：GNNForwardModel 如何训练与预测（单模型版）

目标：读完你能顺着 mbrl/models/gnn_model.py 的 `GNNForwardModel` 走通：

- 输入如何从环境 observation 拆出来
- 训练时目标怎么构造（delta/absolute）
- 归一化器怎么更新、什么时候更新
- loss 是怎么计算的

## 与论文主线的对应：这是“学 world model”的训练工程化落地

论文里讲“structured world model”听起来像一个高层概念，但最终必须落到：

- 从真实环境 rollout 收集数据（$o_t, a_t, o_{t+1}$）
- 用监督学习训练动力学模型（世界模型）去预测下一步
- 让这个模型足够稳定/可泛化，才能在 MPC/planning 里做多步 rollout（用于 curiosity free-play 或下游 zero-shot）

这篇文档解释的这些工程细节（`target_is_delta`、输入/输出归一化、如何构造训练 batch、MSE loss）就是把“世界模型可训练、可用于规划”这件事落到代码层面。

### 论文两阶段里“world model 学习”在哪发生

把论文流程翻译成仓库动作，大致是：

1) 在 **Intrinsic/free-play** settings 下跑训练：边探索边收集数据边更新 world model
	- `experiments/cee_us/settings/<env>/curious_exploration/*.yaml`
2) 进入 **Extrinsic/zero-shot** settings：加载训练好的 world model，用 MPC/planning 做下游任务
	- `experiments/cee_us/settings/<env>/zero_shot_generalization/*.yaml`

这两阶段共享的关键资产就是你在这一篇里读到的 `GNNForwardModel` / `GNNForwardEnsembleModel`：

- 它们学到的动力学预测能力决定了“在 world model 里 rollout 未来”是否可信
- 也就直接影响论文关心的：free-play 的交互丰富程度、以及 zero-shot 下游规划是否成功

## 1. GNNForwardModel 在项目中的位置

它实现了一个 forward model（动力学模型）。

配置里写：

- `forward_model: GNNForwardModel`

运行时会通过 `forward_model_from_string()` 找到这个类。

## 2. 初始化：先从 env 拿维度约定

`__init__` 里有一个关键 assert：env 必须有

- `agent_dim / object_dyn_dim / object_stat_dim / nObj`

然后用这些值构建 `GraphNeuralNetwork`：

- `global_dim = agent_dim`
- `node_dyn_dim = object_dyn_dim`
- `node_stat_dim = object_stat_dim`
- `global_context_dim = action_dim`

## 3. 输入预处理：_get_model_input / _obs_preprocessing

核心流程（忽略归一化细节）：

1) `obs, action` -> tensor
2) `obs = env.obs_preproc(obs)`（很多环境会去掉 goal）
3) `_obs_preprocessing(obs)` -> `(agent, objects_dyn, objects_stat)`

形状：
- agent: `[B, agent_dim]`
- objects_dyn: `[B, nObj, object_dyn_dim]`
- objects_stat: `[B, nObj, object_stat_dim]` 或 None
- action: `[B, action_dim]`

## 4. 训练样本如何构造：_process_batch

训练 batch 输入：
- `observations, actions, next_observations`

处理步骤：

- 对 obs 和 next_obs 都做 `env.obs_preproc`
- 拆出 agent/object（当前与下一步）
- assert 静态属性前后完全一致（`batch_objects_stat == batch_objects_stat_next`）

### delta target

如果 `target_is_delta=True`：
- `target_agent = agent_next - agent`
- `target_object = obj_dyn_next - obj_dyn`

否则就是 absolute target。

## 5. Normalizer（归一化器）

这个模型有多组 normalizer：

- 输入：agent、obj_dyn、obj_stat、action
- 输出：agent_target、obj_target

为什么这么做：
- 减少不同维度量纲差异
- 训练更稳定

它还支持两种统计方式：
- `normalize_w_running_stats=True`：只用最新 rollout 更新统计
- 否则用整个 buffer（开销更大）

## 6. loss：MSE over concatenated outputs

`loss()` 里：

- 调用底层 `self.model.forward(...)` 得到 `(agent_out, obj_out)`
- 把 object 输出展平并与 agent 输出拼接
- 与 target 拼接后做 MSE

注意：这里 loss 是对“所有 object 的 dyn”一起做的，这也会促使模型学会 object-level 预测。

## 7. 训练循环：train() -> _train()

训练参数来自 YAML：

- `epochs / iterations / batch_size / learning_rate / weight_decay / optimizer`

内部用 `TrainingIterator` 把 numpy array 包成 batch。

每个 batch：
- `loss.backward()`
- `optimizer.step()`

并通过 allogger 记录 train/test loss 到 tensorboard。

## 8. 预测（推理）时你应该关注什么

你在 MPC 中用 forward model 的时候，关键是：

- 输入必须先 `env.obs_preproc`
- 输出如果是 delta，需要再 `env.obs_postproc` 加回去

本项目里这些通常由调用方（controller / rollout utils）按约定完成。

下一篇我们看并行 ensemble 版，它与单模型版的最大区别是“多了一维 ensemble 维度”以及并行实现方式。
