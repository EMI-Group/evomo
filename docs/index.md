# EvoMO documentation

EvoMO is a PyTorch library for tensorized, GPU-accelerated multiobjective evolutionary
optimization, built on EvoX. It represents populations, objective values, and evolutionary
operations as tensors for numerical optimization, constrained optimization, and policy search
in robot control tasks.

EvoMO is a sister project in the [EvoX family](https://github.com/EMI-Group/evox),
and shares its logo with EvoX.

This guide covers the current PyTorch implementation. Start with
[installation](guide/installation.md) and the [quickstart](guide/quickstart.md).
If you already use EvoX, see [workflows](guide/workflows.md) and the
[API reference](reference/index.md).

English is the primary documentation language. A complete
[简体中文 guide](zh_CN/index.md) is also available. Use the language switch on each page
to open the corresponding chapter; generated API documentation is shared in English.

## What you can do

- Search for trade-off solutions with NSGA-II, NSGA-III, MOEA/D, RVEA, and other algorithms.
- Evaluate algorithms on DTLZ, ZDT, WFG, MaF, and LSMOP numerical benchmarks.
- Use constraint violations returned by constrained problems in constraint-aware selection.
- Assess solution sets with GD, IGD, and Monte Carlo hypervolume estimates.
- Integrate EvoX monitors and model parameter adapters into EvoMO workflows.
- Search for robot control policies with MoRobtrol after installing optional dependencies.

GPU performance depends on population size, objective count, algorithm, evaluation cost,
and the runtime environment. Algorithms do not all support the same combinations of constraints,
compilation, and `vmap`; check their interfaces and the relevant guides.

```{toctree}
:caption: Getting started
:maxdepth: 2

guide/introduction
guide/installation
guide/quickstart
```

```{toctree}
:caption: User guides
:maxdepth: 2

guide/algorithms
guide/problems
guide/workflows
guide/constraints
guide/metrics
guide/performance
guide/neuroevolution
guide/troubleshooting
```

```{toctree}
:caption: Development and reference
:maxdepth: 2

development
reference/index
non_dominate_backends
```

```{toctree}
:caption: Languages
:hidden:
:maxdepth: 1

简体中文 <zh_CN/index>
```

## Project and citation

[Source code](https://github.com/EMI-Group/evomo) ·
[Issue tracker](https://github.com/EMI-Group/evomo/issues) ·
[EvoX documentation](https://evox.readthedocs.io/en/latest/index.html)

For related algorithm papers and downloadable BibTeX entries, see
[Publications and citation](reference/citation.md).

When using EvoMO in research, please cite:

```bibtex
@article{evomo,
  title = {Bridging Evolutionary Multiobjective Optimization and {GPU} Acceleration via Tensorization},
  author = {Liang, Zhenyu and Li, Hao and Yu, Naiwei and Sun, Kebin and Cheng, Ran},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2025,
  doi = {10.1109/TEVC.2025.3555605}
}
```
