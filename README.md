<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="docs/evox_logo_light.png">
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
3. [Requirements](#requirements)
4. [Examples](#examples)
5. [Community & Support](#community--support)


## Overview  

EvoMO is a GPU-accelerated library for evolutionary multiobjective optimization (EMO) that leverages advanced tensorization techniques. By transforming key data structures and operations into tensor representations, EvoMO enables more efficient mathematical modeling and delivers significant performance improvements. Designed with scalability in mind, EvoMO can efficiently handle large populations and complex optimization tasks. Additionally, EvoMO includes MoRobtrol, a multiobjective robot control benchmark suite, providing a platform for testing tensorized EMO algorithms in real-world, black-box environments. EvoMO is a sister project of [EvoX](https://github.com/EMI-Group/evox).  

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

## Requirements

- Python 3.12 or later
- evox (version == 0.9.0)
- jax (version >= 0.4.16)
- jaxlib (version >= 0.3.0)
- brax (version == 0.10.3)
- flax
- Visualization tools: plotly, pandas

## Examples

Below is a basic example demonstrating how one might utilize the tensorized HypE algorithm within EvoMO. 

```python
from evox import workflows, problems
from evox.metrics import IGD
from algorithms import TensorHypE
import jax
import jax.numpy as jnp
import time


def run_moea(algorithm, key):
    problem = problems.numerical.DTLZ2(m=3)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
    )

    state = workflow.init(key)

    true_pf = problem.pf()
    igd = IGD(true_pf)

    for i in range(100):
        print("Generation", i + 1)
        key, subkey = jax.random.split(key)
        state = workflow.step(state)
        fit = state.get_child_state("algorithm").fitness
        print("IGD:", igd(fit))


if __name__ == "__main__":
    print("TensorHypE")

    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    algorithm = TensorHypE(
        lb=lb,
        ub=ub,
        n_objs=3,
        pop_size=100,
    )

    key = jax.random.PRNGKey(42)

    for i in range(1):
        start = time.time()
        run_moea(algorithm, key)
        end = time.time()
        print("Time:", end - start)
        key, subkey = jax.random.split(key)
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