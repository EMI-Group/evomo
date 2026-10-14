# EvoMO 文档

[English documentation](../index.md) · 简体中文

英文为主文档；本目录提供完整中文版。页面上方可切换到对应章节，两种语言共用英文 API 参考。

EvoMO 是基于 PyTorch 和 EvoX 的多目标进化优化库。它将种群、目标值和进化算子组织为
张量运算，用于数值多目标优化、约束优化以及机器人控制中的策略搜索。

EvoMO 是 [EvoX 系列](https://github.com/EMI-Group/evox)的姊妹项目，并与 EvoX 共用 logo。

本指南面向当前 PyTorch 版本。首次使用请从[安装](guide/installation.md)和
[快速入门](guide/quickstart.md)开始；已有 EvoX 使用经验的读者可以直接查看
[工作流](guide/workflows.md)与 [API 参考](reference/index.md)。

## 能做什么

- 使用 NSGA-II、NSGA-III、MOEA/D、RVEA 等算法搜索多个目标之间的折中解。
- 在 DTLZ、ZDT、WFG、MaF、LSMOP 等数值测试问题上评估算法。
- 通过约束问题返回的违反量执行约束感知选择。
- 用 GD、IGD 和蒙特卡洛 HV 评估解集。
- 将 EvoX 的监视器和模型参数转换工具接入 EvoMO 工作流。
- 在安装可选依赖后，通过 MoRobtrol 搜索机器人控制策略。

GPU 加速效果取决于种群规模、目标数、算法、评估成本和运行环境。可用的类不代表所有算法
都支持相同的约束、编译或 `vmap` 组合；具体接口与限制见各章节。

```{toctree}
:caption: 入门
:maxdepth: 2

guide/introduction
guide/installation
guide/quickstart
```

```{toctree}
:caption: 使用指南
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
:caption: 开发与参考
:maxdepth: 2

development
reference/index
non_dominate_backends
```

## 项目与引用

[源代码](https://github.com/EMI-Group/evomo) ·
[问题反馈](https://github.com/EMI-Group/evomo/issues) ·
[EvoX 文档](https://evox.readthedocs.io/en/latest/index.html)

相关算法论文和 BibTeX 下载见[论文与引用](reference/citation.md)。

在研究中使用 EvoMO 时，请引用：

```bibtex
@article{evomo,
  title = {Bridging Evolutionary Multiobjective Optimization and {GPU} Acceleration via Tensorization},
  author = {Liang, Zhenyu and Li, Hao and Yu, Naiwei and Sun, Kebin and Cheng, Ran},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = 2025,
  doi = {10.1109/TEVC.2025.3555605}
}
```
