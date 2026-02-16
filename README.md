# Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation

## 中文快速导读（Python 新手版）

如果你是第一次接触这个项目，可以先按下面顺序理解：

1. 先看“这是什么”
  - 这是一个基于世界模型（World Model）的强化学习项目，目标是先进行“好奇探索”，再在下游任务上做零样本泛化。

2. 再看“怎么安装”
  - 推荐 Python 3.8 + 虚拟环境。
  - 如果你要跑完整环境（construction/playground/robodesk），通常需要 MuJoCo。
  - 如果先做代码阅读或离线分析，可以先安装不含 MuJoCo 的依赖（见 requirements.no_mujoco.txt）。

3. 再看“怎么运行”
  - 主入口是 mbrl/main.py。
  - 运行时必须给一个 YAML 配置文件（位于 experiments/cee_us/settings）。
  - 你可以把 YAML 理解成“实验配方”：环境、模型、控制器、训练轮数都写在里面。

4. 再看“输出在哪里”
  - 日志与指标会写到 working_dir（由配置控制），可用 TensorBoard 查看。

5. 常见新手坑
  - Python 版本不匹配（建议按文档使用 3.8）。
  - MuJoCo 环境变量没配好。
  - 配置文件里需要你手动填写已训练模型路径（做 zero-shot 时尤其常见）。

如果你只想先跑通一次：
- 先按 Installation 安装；
- 再复制 How to run 的示例命令直接运行；
- 成功后再改 YAML 做自己的实验。

<p align="center">
<img src="docs/images/cee_us_summary.gif" width="500"/>
</p>

This repository contains the code release for the paper [Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation](https://arxiv.org/abs/2206.11403) by Cansu Sancaktar, Sebastian Blaes, and Georg Martius, published as a poster at [*NeurIPS 2022*](https://neurips.cc/virtual/2022/poster/53198). Please use the [provided citation](#citation) when making use of our code or ideas.
## Installation

1. Install and activate a new python3.8 virtualenv.
```bash
virtualenv mbrl_venv --python=python3.8
```

```bash
source mbrl_venv/bin/activate
```

For the following steps, make sure you are sourced inside the `mbrl_venv` virtualenv.

2. Install torch with CUDA. Here is an example for CUDA version 11.3.
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
You can change the CUDA version according to your system requirements, however we only tested for the versions specified here. 

3. Prepare for [mujoco-py](https://github.com/openai/mujoco-py) installation.
    1. Download [mujoco200](https://www.roboti.us/index.html)
    2. `cd ~`
    3. `mkdir .mujoco`
    4. Move mujoco200 folder to `.mujoco`
    5. Move mujoco license key `mjkey.txt` to `~/.mujoco/mjkey.txt`
    6. Set LD_LIBRARY_PATH (add to your .bashrc (or .zshrc) ):
    
    `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin"`

    7. For Ubuntu, run:
    
    `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
    
    `sudo apt install -y patchelf`

4. Install supporting packages
```bash
pip3 install -r requirements.txt
```

5. From the project root:
```bash
pip install -e .
```

6. Set PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path/to/repository>
```

Note: These settings have only been tested on Ubuntu 20. It is recommended to use Ubuntu 20. 

## How to run

```bash
python mbrl/main.py experiments/cee_us/settings/[env]/curious_exploration/[settings_file].yaml
```

The settings files are stored in the experiments folder. Parameters for models, environments, controllers, free play vs. zero-shot downstream task generalization are all specified in these files. In the corresponding folders, you will also find the settings files for the baselines.

For example, in order to run CEE-US free play in the construction environment run:
```bash
python mbrl/main.py experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml
```

After the free play phase to perform zero-shot dowmstream task generalization on stacking with 2 objects, run:
```bash
python mbrl/main.py experiments/cee_us/settings/construction/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_stack.yaml
```
You need to add the path to the trained model in this settings file! (e.g. see [`gnn_ensemble_cee_us_zero_shot_stack`](/./experiments/cee_us/settings/construction/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_stack.yaml))
## Usage Examples

Our method CEE-US as well as the baselines can be run using the settings files in  in [`experiments/cee_us/settings`](/./experiments/cee_us/settings/). E.g. for free play in the construction environment:
- [`gnn_ensemble_cee_us.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): (CEE-US) Uses disagreement of GNN ensemble as intrinsic reward, MPC with iCEM
- [`mlp_ensemble_cee_us.yaml`](./experiments/cee_us/settings/construction/curious_exploration/mlp_ensemble_cee_us.yaml): Uses disagreement of MLP ensemble as intrinsic reward, MPC with iCEM
- [`gnn_rnd_icem.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): Uses GNN model with Random Network Distillation as intrinsic reward, MPC with iCEM
- [`mlp_rnd_icem.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): Uses MLP model with Random Network Distillation as intrinsic reward, MPC with iCEM

See the [full paper](https://arxiv.org/abs/2206.11403) for more details.

## Code style
Run to set up the git hook scripts
```bash
pre-commit install
```

This command will install a number of git hooks that will check your code quality before you can commit.

The main configuration file is located in

`/.pre-commit-config`

Individual config files for the different hooks are located in the base directory of the rep. For instance, the configuration file of `flake8` is `/.flake8`.  

## Citation 

Please use the following bibtex entry to cite us:

    @inproceedings{sancaktar22curious,
      Author = {Sancaktar, Cansu and
      Blaes, Sebastian and Martius, Georg},
      Title = {Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation},
      Booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
      Year = {2022}
    }

## Credits

We adapted [C-SWM](https://github.com/tkipf/c-swm) by Thomas Kipf for the GNN implementation and [fetch-block-construction](https://github.com/richardrl/fetch-block-construction) by Richard Li for the construction environment, both under MIT license. The RoboDesk environment was taken from [RoboDesk](https://github.com/google-research/robodesk) and adapted to mujoco-py and to be object-centric.
