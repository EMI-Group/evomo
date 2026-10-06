# 工作流与监视器

## UnifiedWorkflow

`evomo.workflows.UnifiedWorkflow` 连接算法、问题和可选的监视器：

```python
workflow = UnifiedWorkflow(algorithm, problem, device=device)
workflow.init_step()
for _ in range(100):
    workflow.step()
```

它将算法的 `evaluate` 接入问题，并依次调用监视器回调、解转换与适应度转换。
如果问题返回 `(fitness, cv)`，工作流将两个张量都传回算法，优化方向和
`fitness_transform` 仅作用于目标值。

| 参数 | 用途 |
| --- | --- |
| `algorithm`, `problem` | 算法与批量问题 |
| `monitor` | 评估记录；默认是基础 `Monitor`，不提供完整历史分析 |
| `opt_direction` | `"min"`、`"max"`，或每个目标对应的方向列表 |
| `solution_transform` | 在评估之前将算法决策向量转换成问题输入 |
| `fitness_transform` | 在优化方向转换之后变换目标值 |
| `device` | 工作流组件的目标设备 |
| `enable_distributed`, `group` | 已初始化的 PyTorch 分布式评估配置 |

工作流通过内部子类连接算法，因此读取结果时统一使用 `workflow.algorithm`。
算法的初始化发生在 `init_step()`，某些状态（如 NSGA-II 的 `cv`）要到首次评估后才确定形状。

## 最小化、最大化与混合方向

算法内部使用最小化约定。`opt_direction="max"` 会在目标进入算法之前取负。
例如 `opt_direction=["min", "max"]` 保留第一列、取负第二列，列表长度应与目标数一致。

没有额外 `fitness_transform` 时，要恢复原始目标值，可以在结果分析阶段使用：

```python
raw_fitness = workflow.algorithm.fit * workflow.opt_direction
```

如果另有缩放、归一化或其他转换，需要按其定义恢复或重新评估原始问题，不能只乘方向因子。

## 记录无约束问题的历史前沿

EvoX 的 `EvalMonitor` 可接入工作流。以下代码接在快速入门中问题和算法构造之后：

```python
from evox.workflows import EvalMonitor

monitor = EvalMonitor(multi_obj=True, full_fit_history=True, full_sol_history=True, device=device)
workflow = UnifiedWorkflow(algorithm, problem, monitor=monitor, device=device)
workflow.init_step()
for _ in range(50):
    workflow.step()

historical_solutions, historical_fitness = monitor.get_pf()
```

`get_pf()` 返回历史评估中的非支配解和目标值，与最终种群的前沿范围不同。
只需要目标值时可以关闭 `full_sol_history`，用 `get_pf_fitness()`。
长时间运行或大种群记录完整历史会占用较多内存，按需要开启。

`UnifiedWorkflow` 的监视器回调仅传入目标值，不传入违反量。因此 `EvalMonitor` 的历史前沿
不能作为约束问题的可行前沿。约束示例直接读取最终种群的 `fit` 与 `cv` 并配对筛选。

## 与 StdWorkflow 配合

返回纯目标张量的问题也可以配合 `evox.workflows.StdWorkflow` 使用。
约束问题应采用 `UnifiedWorkflow` 保留违反量；工作流能转发 tuple 仍不保证算法会使用约束。

## 分布式评估

`enable_distributed=True` 将种群按进程分块评估，再收集目标值与约束违反量。
用户需先初始化 `torch.distributed` 进程组，确保问题能够独立评估每个分块，
并验证所用后端对分块大小、张量形状与设备的要求。本文没有提供跨设备或多进程性能保证。
