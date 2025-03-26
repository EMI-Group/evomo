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
3. [Installation Guide](#installation-guide)
4. [Examples](#examples)
6. [Community & Support](#community--support)


## Overview  

EvoMO is a GPU-accelerated library for evolutionary multiobjective optimization (EMO) via advanced tensorization. By converting key data structures and operations into tensor representations, it enables concise mathematical modeling and significant speedups. Designed for scalability, EvoMO efficiently handles large populations and complex optimization tasks. EvoMO is a sister project of [EvoX](https://github.com/EMI-Group/evox).  

## Key Features  

- **General Tensorization Methodology ⚡:**  
  EvoMO adopts a unified tensorization approach that restructures EMO algorithms into tensor-based representations, enabling efficient GPU acceleration.  

- **Optimized State-of-the-Art Algorithms 🚀:**  
  Supports **NSGA-II**, **NSGA-III**, **MOEA/D**, **RVEA**, and **HypE**, achieving up to **1113× speedup** while preserving solution quality.  

- **Scalability 📈:**  
  Handles large populations, scaling to hundreds of thousands for complex optimization tasks.  

- **MoRobtrol Benchmark 🤖:**  
  Includes **MoRobtrol**, a multiobjective robot control benchmark for testing tensorized EMO algorithms in challenging black-box environments.  

## Installation Guide


## Examples

Below is a basic example demonstrating how one might utilize the tensorized HypE algorithm within EvoMO.

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
