# 基准问题与自定义问题

## 数值基准

从 `evomo.problems.numerical` 导入具体问题：

| 问题族 | 公开类 | 参数与边界注意事项 |
| --- | --- | --- |
| ZDT | `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6` | 决策维度参数是 `n`；通常为双目标；ZDT4 的后续变量使用 `[-5, 5]` |
| DTLZ | `DTLZ1`–`DTLZ7` | 使用 `d`、`m`、`ref_num`；常用变量范围为 `[0, 1]` |
| WFG | `WFG1`–`WFG9` | 使用 `d`、`m`、`k`、`ref_num`；第 i 个变量范围为 `[0, 2i]`，i 从 1 开始 |
| MaF | `MAF1`–`MAF15` | 使用 `d`、`m` 等参数；不同变体的维度与边界约定需逐一核对 |
| LSMOP | `LSMOP1`–`LSMOP9` | 显式指定 `d`、`m`；前 `m-1` 个变量通常为 `[0, 1]`，后续为 `[0, 10]` |

数值问题通常返回形状为 `(N, M)` 的目标张量。与约束问题不同，不应假设每个数值问题
都提供 `lb`、`ub`；必要时按问题定义自行构造。

`pf()` 返回参考前沿或其采样点，用于结果分析。`ref_num` 不总是返回点数的严格上限或精确值，
应读取实际形状。某些参考前沿通过数值采样或附带数据构造，不应把全部 `pf()` 结果都称作
精确解析前沿。

## 约束基准

`evomo.problems.constrained` 提供：

| 问题族 | 变体范围 |
| --- | --- |
| LIRCMOP | `LIRCMOP1`–`LIRCMOP14` |
| DASCMOP | `DASCMOP1`–`DASCMOP9` |
| CTP | `CTP1`–`CTP8` |
| MW | `MW1`–`MW14` |
| FCP | `FCP1`–`FCP5` |
| DOC | `DOC1`–`DOC9` |
| SDC | `SDC1`–`SDC15` |
| LSCM | `LSCM1`–`LSCM12` |

这些问题基于 `evomo.problems.constrained.base.CMOP`，提供维度 `d`、目标数 `m` 和
`lb`、`ub`。部分变体有固定维度，不能统一假设传入 `d`、`m` 就能改变问题。
其 `evaluate` 返回 `(fitness, cv)`；完整用法见[约束优化](constraints.md)。

## 自定义批量问题

继承 EvoX 的 `Problem`，实现一次评估整个种群的 `evaluate`。
下面同时最小化到两个中心的平方距离：

```{literalinclude} ../../examples/custom_problem.py
:language: python
```

```sh
python docs/examples/custom_problem.py
```

返回张量的第一维必须对应输入候选解，第二维对应算法的 `n_objs`。
函数内使用输入张量衍生计算，避免将结果转换成 NumPy 或逐个解循环评估。
需要常量时在初始化阶段注册为 buffer，或使用输入张量的设备与 dtype 创建。

自定义约束问题可以返回 `(fitness, cv)`，其中 `cv` 应为非负违反量，而不是未经转换的
原始约束函数。使用 `UnifiedWorkflow` 和支持约束选择的算法接入。
