<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="./assets/evox_logo_light.png">
  </picture>
  </a>
  <br>
</h1>

<h2 align="center">
🌟 EvoMO: Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization 🌟
</h2>

<div align="center">
  <a href="http://arxiv.org/abs/2503.20286">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoMO Paper on arXiv">
  </a>
</div>

## Table of Contents

1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Installation Guide](#installation-guide)
4. [Examples](#examples)
6. [Community & Support](#community--support)


## Overview  

EvoMO is a GPU-accelerated library for evolutionary multiobjective optimization (EMO) that leverages advanced tensorization techniques. By transforming key data structures and operations into tensor representations, EvoMO enables more efficient mathematical modeling and delivers significant performance improvements. Designed with scalability in mind, EvoMO can efficiently handle large populations and complex optimization tasks. Additionally, EvoMO includes MoRobtrol, a multiobjective robot control benchmark suite, providing a platform for testing tensorized EMO algorithms in real-world, black-box environments. EvoMO is a sister project of [EvoX](https://github.com/EMI-Group/evox).  

> [!NOTE]
> To use the JAX version of EvoMO, you can switch to the `v0.0.1-dev` branch. This branch is fully compatible with EvoX version 0.9.0.
> 
## Key Features

### 💻 High-Performance Computing

#### 🚀 General Tensorization Methodology
- **EvoMO** adopts a unified tensorization approach, restructuring EMO algorithms into tensor representations, enabling efficient GPU acceleration.

#### ⚡ Ultra Performance
- Supports tensorized implementations of **NSGA-II**, **NSGA-III**, **MOEA/D**, **RVEA**, **HypE**, and more, achieving up to **1113× speedup** while preserving solution quality.

#### 📈 Scalability
- Handles large populations, scaling to hundreds of thousands for complex optimization tasks, ensuring scalability for real-world applications.


### 📊 Benchmarking

#### 🤖 MoRobtrol Benchmark
- Includes **MoRobtrol**, a multiobjective robot control benchmark, for testing tensorized EMO algorithms in challenging black-box environments.

### 🔧 Easy-to-Use Integration

#### 🔄 Shared Name with EvoX
- After installation, you can directly import EvoMO algorithms using `import evox`, making it seamless to access both EvoX and EvoMO algorithms with a unified interface.

## Installation Guide


To install EvoMO, you need to install EvoX first. 


1. Install EvoX:

```bash
pip install evox
```

   
2. Install EvoMO:

```bash
pip install evomo
```


For the latest development version, you can install from the source:

```bash
git clone https://github.com/EMI-Group/evomo.git
cd evomo
pip install -e.
```

## Examples

### Numerical optimization problem

Solve the DTLZ2 problem using the TensorMOEA/D algorithm:

```python
import time
import torch
from evox.workflows import StdWorkflow
from evox.algorithms import TensorMOEAD
from evox.problems.numerical import DTLZ2
from evox.metrics import igd

if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    algo = TensorMOEAD(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    t = time.time()
    for i in range(100):
        print(i)
        jit_state_step()
        fit = workflow.algorithm.fit
        fit = fit[~torch.any(torch.isnan(fit), dim=1)]
        print(f"Generation {i + 1} IGD: {igd(fit, pf)}")

    print(f"Total time: {time.time() - t} seconds")
```

### MoRobtrol

Solve the MoSwimmer problem in MoRobtrol using the TensorMOEA/D algorithm:

```python
import time

import torch
import torch.nn as nn
from evox.algorithms import TensorMOEAD
from evox.problems import MoRobtrol
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        return torch.tanh(self.features(x))


def setup_workflow(model, pop_size, max_episode_length, num_episodes, device):
    adapter = ParamsAndVector(dummy_model=model)
    model_params = dict(model.named_parameters())
    pop_center = adapter.to_vector(model_params)
    lower_bound = torch.full_like(pop_center, -5)
    upper_bound = torch.full_like(pop_center, 5)

    problem = MoRobtrol(
        policy=model,
        env_name="mo_swimmer",
        max_episode_length=max_episode_length,
        num_episodes=num_episodes,
        pop_size=pop_size,
        device=device,
        num_obj=2,
        observation_shape=8,
        obs_norm={"clip_val": 5.0, "std_min": 1e-6, "std_max": 1e6},
    )

    algorithm = TensorMOEAD(
        pop_size=pop_size, lb=lower_bound, ub=upper_bound, n_objs=2, device=device
    )
    monitor = EvalMonitor(device=device)

    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        opt_direction="max",
        solution_transform=adapter,
        device=device,
    )
    return workflow


def run_workflow(workflow, compiled=True, generations=10):
    workflow.init_step()
    step_function = torch.compile(workflow.step) if compiled else workflow.step
    for index in range(generations):
        print(f"In generation {index}:")
        t = time.time()
        step_function()
        print(f"\tFitness: {-workflow.algorithm.fit}.")
    print(f"\tTime elapsed: {time.time() - t: .4f}(s).")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleMLP().to(device)
    workflow = setup_workflow(model, 12, 100, 2, device)
    run_workflow(workflow)

```

## Community & Support

We welcome contributions and look forward to your feedback!
- Engage in discussions and share your experiences on [GitHub Issues](https://github.com/EMI-Group/evomo/issues).
- Join our QQ group (ID: 297969717).

## Citing EvoMO

If you use EvoMO in your research and want to cite it in your work, please use:
```
@article{evomo,
  title = {Bridging Evolutionary Multiobjective Optimization and {GPU} Acceleration via Tensorization},
  author = {Liang, Zhenyu and Li, Hao and Yu, Naiwei and Sun, Kebin and Cheng, Ran},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2025,
  doi = {10.1109/TEVC.2025.3555605}
}
```
