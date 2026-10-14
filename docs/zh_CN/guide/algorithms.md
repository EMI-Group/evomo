# 算法

算法从 `evomo.algorithms` 导入。以下名称对应当前公开导出；构造签名、默认值及论文引用
由 [API 页面](../reference/index.md)从源码生成。

## 常用入口

| 类 | 主要机制 | 使用时关注 |
| --- | --- | --- |
| `NSGA2` | 非支配排序与拥挤距离 | 适合作为数值与约束示例的起点 |
| `NSGA3` | 非支配排序与参考点 | 参考点数量与目标数、种群数的关系 |
| `MOEAD` | 分解与邻域更新 | 分解方式和邻域参数 |
| `TensorMOEAD` | 张量化分解与邻域更新 | `aggregate_op`、实际种群数及邻域大小 |
| `RVEA`, `RVEAa` | 参考向量引导选择 | `max_gen` 与实际迭代预算保持一致 |
| `IBEA` | 指标驱动选择 | 指标与尺度对选择的影响 |
| `HypE` | 基于超体积估计的选择 | 采样与参考值相关参数 |
| `LMOCSO` | 大规模多目标竞争群优化 | 决策维度与算法参数 |

这张表介绍接口用途，不代表基于实验的算法排名。比较算法需要相同评估预算和多个随机种子。

## 构造参数与状态

算法接受 `pop_size`、`n_objs`、`lb` 和 `ub`，部分算法还显式提供 `device`。
`lb`、`ub` 必须为等形状的一维张量，设备和 dtype 一致。设置 `n_objs` 时应与问题
`evaluate` 返回的列数一致。设备规则、算子调用约定与算法专有设置见
[参数指南](../reference/parameters.md)。

常见状态为 `pop` 和 `fit`。某些算法还维护参考向量、理想点、档案、速度、等级或违反量，
不能将某个算法的特有属性当成所有算法的统一接口。

使用参考向量或权重采样的算法可能调整 `pop_size`。创建完成后以
`workflow.algorithm.pop.shape[0]` 为实际种群数。`TensorMOEAD` 要求采样后的种群数大于 10。

例如，将快速入门的构造部分替换为：

```python
from evomo.algorithms import TensorMOEAD

algorithm = TensorMOEAD(
    pop_size=100,
    n_objs=problem.m,
    lb=torch.zeros(problem.d, device=device),
    ub=torch.ones(problem.d, device=device),
    aggregate_op=("pbi", "pbi"),
    device=device,
)
```

`TensorMOEAD` 支持的聚合名称为 `pbi`、`tchebycheff`、`tchebycheff_norm`、
`modified_tchebycheff` 和 `weighted_sum`。

## 算子替换

部分算法允许传入 `selection_op`、`mutation_op`、`crossover_op`，但调用契约由算法实现决定。
先查该算法的 `step`，确认算子的输入、输出形状和语义，再进行替换。

例如，NSGA-II 的 `selection_op` 用于选择交配父代，默认是 EvoX 的
`tournament_selection_multifit`。它不负责环境选择，也不会切换非支配排序后端。
后端选择说明见[现有后端指南](../non_dominate_backends.md)。

## 其他公开算法

为便于检索，下面保留源码中的类名拼写：

| 分组 | 导出名称 |
| --- | --- |
| MOEA/D 及相关实现 | `BCEMOEAD`, `MOEADAWA`, `MOEADDE`, `MOEADDU`, `MOEADDYTS`, `MOEADFRRMAB`, `MOEAURAW`, `MOEAD_DCWV`, `MOEAD_DRA`, `MOEAD_PaS` |
| 稀疏与大规模实现 | `LSMOF`, `SparseEA`, `SparseEA2`, `TSSparseEA`, `WOF` |
| 其他实现（一） | `AGEMOEA`, `BCE_IBEA`, `BiGE`, `CLIA`, `CMOEA_MS`, `CMOPSO`, `CoMMEA`, `DMMOEA`, `EFRRR`, `GDE3`, `GrEA`, `GWASFGA`, `KnEA` |
| 其他实现（二） | `MaOEACSS`, `NSBiDiCo`, `NSGAII_SDR`, `OSP_NSDE`, `PESA2`, `PICEAg`, `PREA`, `SIBEA`, `SMPSO`, `SNSGA2`, `SPEAR`, `SSCEA` |
| 其他实现（三） | `TELSO`, `TSNSGAII`, `ThetaDEA`, `Two_Arch2`, `VaEA`, `WASFGA`, `eMOEA`, `tDEA_CPBI` |

算法名称含有 “constrained” 或相关缩写不构成对任意约束问题的兼容保证。
约束任务优先使用本文已经给出的 NSGA-II 流程；其他算法需核实违反量是否参与选择。
