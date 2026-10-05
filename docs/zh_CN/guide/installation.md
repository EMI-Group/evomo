# 安装与环境

## 基本要求

当前包声明的最低要求为 Python 3.10、PyTorch 2.6.0 和 EvoX 1.2.1。
数值问题可以在 CPU 上运行；使用 CUDA 时，需要与驱动和系统匹配的 PyTorch 安装。
先检查现有环境，保留已有的 CPU/CUDA 配置：

```sh
python --version
python -m pip show torch evox evomo
```

如需安装或更换 PyTorch，按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)
选择适合当前系统的命令。

## 安装发行版

```sh
python -m pip install evomo
```

包管理器会根据声明安装或检查核心依赖。已满足版本要求的 PyTorch 无需为了使用 EvoMO
重新安装。线上发行版与当前开发分支可能存在差异；需要本文对应的最新源码时采用下面的方法。

## 从源码安装

```sh
git clone https://github.com/EMI-Group/evomo.git
cd evomo
python -m pip install -e .
```

已有仓库副本时直接在仓库根目录执行最后一行。可编辑安装使本地源码修改立即生效。

## 可选依赖

| Extra | 安装命令 | 用途 |
| --- | --- | --- |
| `vis` | `python -m pip install "evomo[vis]"` | Plotly、pandas 可视化依赖 |
| `neuroevolution` | `python -m pip install "evomo[neuroevolution]"` | Brax/JAX 与 torchvision |
| `playground` | `python -m pip install "evomo[playground]"` | Playground/MJX 控制任务 |

源码安装时可以写成 `python -m pip install -e ".[vis]"` 等形式。
Playground 工作流请使用 Python 3.11 或更新版本，并遵循包中的 JAX 与 MuJoCo 版本范围。
安装 extra 不保证 JAX 能使用 GPU；用 `jax.devices()` 单独确认模拟器设备。
普通数值优化不需要这些可选依赖。

## 验证安装

```sh
python -c "import torch; from evomo.algorithms import NSGA2; from evomo.problems.numerical import DTLZ2; print(torch.__version__, torch.cuda.is_available(), NSGA2.__name__, DTLZ2.__name__)"
python -m pip check
```

第一条命令检查核心导入以及 CUDA 可见性。然后运行[快速入门](quickstart.md)验证完整优化流程。

## 开发与文档依赖

```sh
python -m pip install pytest ruff
python -m pip install -r docs/requirements.txt
```

`test` extra 包含部分可选运行依赖，不会安装 pytest 或 Ruff。仅构建文档时，可以在独立环境中
只安装 `docs/requirements.txt`；API 文档静态解析源码，无需安装优化或物理模拟依赖。
