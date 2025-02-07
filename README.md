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

<p align="center">
🌟 EvoMO: Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization 🌟
</p>

<p align="center">
  <a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoMO Paper on arXiv">
  </a>
</p>

## Table of Contents

1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Requirements](#requirements)
4. [Examples](#examples)
6. [Community & Support](#community--support)


## Overview  

EvoMO is a high-performance library that integrates evolutionary multiobjective optimization (EMO) with GPU acceleration via tensorization. By converting key data structures and operations into tensor representations, it enables concise mathematical modeling and significant speedups. Designed for scalability, EvoMO efficiently handles large populations and complex optimization tasks. EvoMO is a sister project of [EvoX](https://github.com/EMI-Group/evox).  

## Key Features  

- **General Tensorization Methodology ⚡:**  
  EvoMO adopts a unified tensorization approach that restructures EMO algorithms into tensor-based representations, enabling efficient GPU acceleration.  

- **Optimized State-of-the-Art Algorithms 🚀:**  
  Supports **NSGA-II**, **NSGA-III**, **MOEA/D**, **RVEA**, and **HypE**, achieving up to **1113× speedup** while preserving solution quality.  

- **Scalability 📈:**  
  Handles large populations, scaling to hundreds of thousands for complex optimization tasks.  

- **MoRobtrol Benchmark 🤖:**  
  Includes **MoRobtrol**, a multiobjective robot control benchmark for testing tensorized EMO algorithms in challenging black-box environments.  

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
  title = {Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization},
  author = {Zhenyu Liang, Hao Li, Naiwei Yu, Kebin Sun, and Ran Cheng},
  year = {2025},
}
```