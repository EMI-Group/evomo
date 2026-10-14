# Workflows and monitors

## UnifiedWorkflow

`evomo.workflows.UnifiedWorkflow` connects an algorithm, a problem, and an optional monitor:

```python
workflow = UnifiedWorkflow(algorithm, problem, device=device)
workflow.init_step()
for _ in range(100):
    workflow.step()
```

It connects the algorithm's `evaluate` to the problem, invoking monitor callbacks, solution
transforms, and fitness transforms in sequence. If the problem returns `(fitness, cv)`, both
tensors reach the algorithm. Objective direction and `fitness_transform` affect only objectives.

| Parameter | Purpose |
| --- | --- |
| `algorithm`, `problem` | Algorithm and batched problem |
| `monitor` | Evaluation recording; the default base `Monitor` does not provide full history analysis |
| `opt_direction` | `"min"`, `"max"`, or a direction list with one entry per objective |
| `solution_transform` | Converts decision vectors to problem inputs before evaluation |
| `fitness_transform` | Transforms objectives after the direction conversion |
| `device` | Target device for workflow components |
| `enable_distributed`, `group` | Distributed evaluation using an initialized PyTorch process group |

The workflow connects an algorithm through an internal subclass. Read results consistently
from `workflow.algorithm`. Initialization takes place in `init_step()`. Some state, such as
NSGA-II's `cv`, gets its shape only after the first evaluation.

## Minimization, maximization, and mixed directions

Algorithms use minimization internally. `opt_direction="max"` negates objectives before they
reach the algorithm. For example, `opt_direction=["min", "max"]` preserves the first column
and negates the second. The list length must match the objective count.

When no additional `fitness_transform` is applied, recover original objective values during analysis:

```python
raw_fitness = workflow.algorithm.fit * workflow.opt_direction
```

If you also apply scaling, normalization, or another transform, invert that transform according
to its definition or re-evaluate the original problem. Multiplying by the direction alone is insufficient.

## Recording an unconstrained historical front

EvoX's `EvalMonitor` can be attached to the workflow. Insert this code after constructing
the problem and algorithm in the quickstart:

```python
from evox.workflows import EvalMonitor

monitor = EvalMonitor(multi_obj=True, full_fit_history=True, full_sol_history=True, device=device)
workflow = UnifiedWorkflow(algorithm, problem, monitor=monitor, device=device)
workflow.init_step()
for _ in range(50):
    workflow.step()

historical_solutions, historical_fitness = monitor.get_pf()
```

`get_pf()` returns non-dominated solutions and objectives from historical evaluations, which
has a different scope from the final population's front. If you need objectives only, disable
`full_sol_history` and use `get_pf_fitness()`. Full history can consume substantial memory
for long runs or large populations; enable only what you need.

`UnifiedWorkflow` passes objectives, but not violations, to monitor callbacks. Therefore,
`EvalMonitor`'s historical front cannot be treated as a feasible front for constrained problems.
The constrained example reads paired `fit` and `cv` from the final population and filters them together.

## Using StdWorkflow

Problems returning only an objective tensor can also use `evox.workflows.StdWorkflow`.
For constrained problems, use `UnifiedWorkflow` to preserve violations. A workflow's ability
to forward a tuple does not establish that the algorithm uses its constraints.

## Distributed evaluation

`enable_distributed=True` splits a population across processes for evaluation and gathers
objectives and violations. Initialize a `torch.distributed` process group first. Verify that
the problem can evaluate each partition independently and that the chosen backend supports
the partition sizes, tensor shapes, and devices. This guide does not establish multi-device
or multi-process performance guarantees.
