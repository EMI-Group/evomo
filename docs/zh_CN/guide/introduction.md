# 认识 EvoMO

## 多目标优化的输入与输出

多目标问题需要同时优化多个通常相互冲突的目标。例如，更快的机器人运动可能需要更大的
能量消耗。进化算法维护一组候选解，通过评估、选择、交叉或变异逐步更新这组解。

在最小化约定下，若解 A 的每个目标都不差于 B，且至少一个目标严格更小，则 A 支配 B。
不被当前解集中其他解支配的点组成第一非支配前沿。得到这个前沿不意味着已经找到了问题的
真实 Pareto 前沿；仍需参考前沿、指标和多次运行来衡量质量。

EvoMO 使用以下批量张量约定：

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| `population` / `pop` | `(N, D)` | N 个候选解，每个有 D 个决策变量 |
| `fitness` / `fit` | `(N, M)` | 每个候选解的 M 个目标值 |
| `lb`, `ub` | `(D,)` | 每个变量的下界与上界 |
| `cv` | `(N, C)` | 各约束的非负违反量；部分算子也接收 `(N,)` 总违反量 |
| `rank` | `(N,)` | 从 0 开始的非支配等级 |

输入张量、算法状态、问题和参考前沿应放在一致的设备上，并使用兼容的 dtype。

## 与 EvoX 的关系

EvoMO 使用 EvoX 的 `Algorithm`、`Problem`、`Workflow` 和状态管理机制。
EvoX 提供通用算子、监视器和神经网络参数适配工具；EvoMO 提供自己的多目标算法、基准问题、
选择算子、指标，以及保留约束违反量的 `UnifiedWorkflow`。

```python
from evomo.algorithms import NSGA2
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow

from evox.workflows import EvalMonitor
```

名称相同的 `evomo.algorithms.NSGA2` 与 `evox.algorithms.NSGA2` 属于不同实现。
复现实验时记录实际导入路径，避免无意切换实现。

## 一次运行的组成

1. **问题**：实现批量 `evaluate`，输出目标值，约束问题同时输出违反量。
2. **算法**：保存种群和目标值，根据评估结果生成和选择下一代。
3. **工作流**：连接算法与问题，处理优化方向、转换函数和监视器回调。
4. **结果分析**：提取当前种群或监视器记录，筛选可行且非支配的解，再计算指标。

调用 `workflow.init_step()` 完成初始化，之后重复调用 `workflow.step()`。
直接使用尚未接入工作流的算法进行评估不符合这里的调用方式。

## 版本范围

本套文档对应当前仓库的 PyTorch 实现，构建时从 `pyproject.toml` 读取版本号。
历史 JAX 实现在 `v0.0.1-dev` 分支，与 EvoX 0.9.0 配套；其状态与调用方式不适用于本文示例。
