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

EvoMO is a high-performance framework that pioneers the integration of evolutionary multiobjective optimization (EMO) with GPU acceleration through advanced tensorization techniques. By transforming key data structures (e.g., candidate solutions and objective values) and operations (e.g., crossover, mutation, and selection) into tensor representations, EvoMO establishes concise yet versatile mathematical models that unlock significant computational speedups. This framework is designed to efficiently scale to large population sizes and tackle complex, computationally intensive problems.

## Key Features

- **General Tensorization Methodology:**  
  EvoMO introduces a comprehensive tensorization approach that converts key data structures and operations of EMO algorithms into tensor representations. This concise and versatile mathematical model enables efficient GPU acceleration.

- **Application to Representative Algorithms:**  
  The tensorization methodology is applied to three state-of-the-art EMO algorithms: **NSGA-III**, **MOEA/D**, and **HypE**. These tensorized implementations maintain solution quality while achieving up to a **1113× speedup** compared to their CPU-based counterparts.

- **Scalability:**  
  Designed to handle large-scale problems, EvoMO efficiently scales population sizes to hundreds of thousands, making it suitable for tackling computationally intensive optimization tasks.

- **Challenging Benchmark – MoRobtrol:**  
  EvoMO includes a novel multiobjective robot control benchmark called **MoRobtrol**. This benchmark simulates a complex black-box environment, demonstrating the ability of tensorized EMO algorithms to generate diverse, high-quality solutions under demanding conditions.

## Requirements

- Python 3.12 or later
- evox (version == 0.8.1)
- jax (version >= 0.4.16)
- jaxlib (version >= 0.3.0)
- brax (version == 0.10.3)
- flax
- Visualization tools: plotly, pandas

## Examples

Below is a basic example demonstrating how one might utilize the tensorized NSGA-III algorithm within EvoMO. (Note: The import paths and API details are placeholders and will be updated once the package is officially released.)

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