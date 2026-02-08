# WSL (Ubuntu 20.04) + uv 安装 CEE-US（本仓库）

> 目标：在 WSL2 / Ubuntu 20.04 上，用 `uv` 管理 Python 虚拟环境与依赖，跑通本仓库 `mbrl/main.py`。
>
> 重要说明：本仓库依赖 `mujoco-py==2.0.2.0`，它需要你**手动**下载并安装 MuJoCo 2.0（mujoco200）和 license（这是上游的授权要求，无法自动化下载）。

## 0. 推荐目录与约定

- 仓库目录：`/home/zhuzihou/dev/cee-us`（你当前已有）
- Python 虚拟环境：放在仓库内 `.venv/`（项目级隔离，最不容易迷路）
- MuJoCo 2.0：最终需要能在 `~/.mujoco/mujoco200/` 找到（`mujoco-py` 的约定；脚本会自动创建兼容 symlink）

## 1. 安装系统依赖（apt）

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates pkg-config patchelf \
  python3.8-venv python3.8-dev python3-pip \
  libosmesa6-dev libgl1-mesa-glx \
  libglfw3 libglfw3-dev libglew-dev \
  libxrender1 libxext6 libsm6 \
  libxrandr2 libxinerama1 libxcursor1 libxi6
```

踩坑点：
- 缺少 `build-essential/python3.8-dev` 时，很多包会在编译阶段失败。
- 缺少 `libosmesa6-dev/libglfw3-dev` 时，`mujoco-py` 常见编译/链接错误。

## 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

通常会装到：`~/.local/bin/uv`

如果新开终端找不到 `uv`，把下面这一行加到 `~/.bashrc`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

验证：

```bash
uv --version
```

## 3. 创建虚拟环境（Python 3.8）

在仓库根目录：

```bash
cd /home/zhuzihou/dev/cee-us
uv venv --python python3.8 .venv
source .venv/bin/activate
python --version  # 应该是 3.8.x
```

## 4. 安装 PyTorch（按仓库 README，CUDA 11.3 轮子）

> 这一部分下载很大：`torch` 大约 1.7GiB。

```bash
source /home/zhuzihou/dev/cee-us/.venv/bin/activate
export UV_NO_PROGRESS=1
uv pip install --no-progress \
  torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

踩坑点：
- WSL2 是否有 GPU：用 `nvidia-smi` 检查。即使是 CUDA 轮子，也主要依赖 Windows 侧驱动；一般不需要你在 WSL 内安装完整 CUDA Toolkit。
- 速度慢/卡住：这是正常的（大文件）。可以先不装 CUDA 版，改装 CPU 版（更小），确认流程无误后再切换。

## 5. 安装其余 Python 依赖（先跳过 mujoco-py）

由于 `mujoco-py` 需要你先把 MuJoCo 2.0 放到 `~/.mujoco`，所以建议分两步：

1) 先装除 `mujoco-py` 外的依赖：

```bash
cd /home/zhuzihou/dev/cee-us
source .venv/bin/activate

# 注意：uv 不支持 requirements 里 `-e git+...` 这种“可编辑的 Git 依赖”。
# 仓库内已提供 uv 兼容版本（把该依赖改成 PEP 508 形式）。
uv pip install --no-progress -r requirements.no_mujoco.txt

# 开发模式安装本项目
uv pip install --no-progress -e .
```

2) 等你完成 MuJoCo 2.0 安装后，再装 `mujoco-py`：见下一节。

踩坑点：
- 本仓库的 `requirements.txt` **不包含 torch**，因为作者在 README 单独让你先装 torch。
- `smart-settings` 是 git 依赖（requirements 里 `-e git+...`），第一次安装会略慢。
 - `smart-settings` 是 git 依赖；在 uv 下需要用 PEP 508 写法（本仓库的 `requirements.no_mujoco.txt` 已处理）。

## 6. 安装 MuJoCo 2.0（mujoco200）与 mujoco-py

### 6.1 手动安装 MuJoCo 2.0（必须手动）

由于授权限制，MuJoCo 2.0（mujoco200）和 `mjkey.txt` 需要你手动从 roboti.us 获取。

推荐做法（方便在 WSL 里操作/记录）：把这两个文件先放到本仓库的 `downloads/` 目录（该目录已在 `.gitignore` 中忽略，不会被提交）。例如：

- `downloads/mujoco200_linux.zip`（或 `.tar.gz` / `.tgz`）
- `downloads/mjkey.txt`

然后运行仓库自带脚本，一键解压/复制到正确位置：

```bash
cd /home/zhuzihou/dev/cee-us
bash scripts/setup_mujoco200_wsl.sh \
  --archive downloads/mujoco200_linux.zip \
  --mjkey downloads/mjkey.txt
```

脚本会把 MuJoCo 放到：`~/.mujoco/mujoco200_linux/`，并创建 `~/.mujoco/mujoco200 -> mujoco200_linux` 的 symlink（因为 `mujoco-py` 期望路径是 `~/.mujoco/mujoco200/`）。key 会放到：`~/.mujoco/mjkey.txt`。同时会把必要环境变量追加到 `~/.bashrc`（可重复运行，不会重复写入）。

### 6.2 配置环境变量

建议加到 `~/.bashrc`：

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"
```

可选（WSL/无显示器时更稳）：

```bash
export MUJOCO_GL=osmesa
```

### 6.3 安装 mujoco-py

```bash
cd /home/zhuzihou/dev/cee-us
source .venv/bin/activate
uv pip install --no-progress "Cython==0.29.36"

# 关键：mujoco-py 的构建脚本会 import numpy/Cython，但它没有把它们声明成 build dependency。
# 所以需要关闭 build isolation，让它使用当前 venv 里的 numpy/Cython。
uv pip install --no-progress --no-build-isolation mujoco-py==2.0.2.0
```

踩坑点（最常见）：
- `fatal error: GL/osmesa.h: No such file or directory`：说明 `libosmesa6-dev` 没装。
- `Could not load library ... libmujoco200.so`：说明 `LD_LIBRARY_PATH` 没指到 `.../mujoco200_linux/bin`。
- WSL 图形相关报错：优先用 `MUJOCO_GL=osmesa` 跑 headless，等确认能跑再切换到 glfw。

## 7. 最小自检（不跑训练，只验证 import/入口）

```bash
cd /home/zhuzihou/dev/cee-us
source .venv/bin/activate
python -c "import mbrl; import torch; print('ok')"
```

真正跑实验（示例）：

```bash
python mbrl/main.py experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml
```

## 8. 实际执行记录（2026-02-08，WSL2 + Ubuntu 20.04）

> 这一节记录一次真实安装过程里遇到的坑与最终可复现命令。

### 8.1 安装 MuJoCo 2.0（从 downloads/）

```bash
cd /home/zhuzihou/dev/cee-us
bash scripts/setup_mujoco200_wsl.sh \
  --archive downloads/mujoco200_linux.zip \
  --mjkey downloads/mjkey.txt
```

坑：`mujoco-py` 默认找 `~/.mujoco/mujoco200`，而下载包常见目录名是 `mujoco200_linux`。脚本已自动创建 symlink 解决。

### 8.2 安装 mujoco-py（uv + no-build-isolation）

```bash
cd /home/zhuzihou/dev/cee-us
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"

uv pip install --no-progress "Cython==0.29.36"
uv pip install --no-progress --no-build-isolation mujoco-py==2.0.2.0
```

坑：
- 不加 `--no-build-isolation` 时，build env 里缺 `numpy` 会报 `ModuleNotFoundError: No module named 'numpy'`。
- 关闭 build isolation 后，如果 venv 里缺 `Cython` 会报 `ModuleNotFoundError: No module named 'Cython'`。
- `uv` 会对 `.venv` 加锁；不要在另一个终端同时跑 `uv pip install`，否则会卡在 `.venv/.lock`。

### 8.3 运行验证（headless）

```bash
python -c "import mujoco_py; print('mujoco_py import ok')"

python - <<'PY'
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
from mujoco_py import load_model_from_xml, MjSim
xml = """<mujoco model='min'><worldbody></worldbody></mujoco>"""
sim = MjSim(load_model_from_xml(xml))
print('MjSim ok', sim.data.time)
PY
```

---

如果你希望我把 `requirements.no_mujoco.txt` 也正式加入仓库（避免每次手动 grep 生成），我可以帮你加一个更清晰的 `requirements.wsl.txt`。