# 算法参数说明

每个公开算法的英文 API 页面都说明构造参数的类型、形状、默认值及当前实现中的用途。
本页用中文解释共同约定与容易混淆的参数。完整类名见[算法目录](../guide/algorithms.md)。
“正数预算”等要求属于使用约定，不表示每个构造函数都会主动校验所有输入。
具体签名应以相应类的 API 页面为准。

## 种群、目标、边界与设备

| 参数 | 类型与形状 | 含义 |
| --- | --- | --- |
| `pop_size` | 正整数 | 请求的候选解数量。部分参考向量算法会按实际采样数调整，请读取 `algorithm.pop.shape[0]`。`GDE3` 至少需要 4 个候选解；`MOEAD`、`TensorMOEAD` 的实际采样数必须大于 10。 |
| `n_objs` | 整数，通常至少为 2 | 最小化目标的个数，须匹配问题返回目标张量的列数，与决策维数不同。 |
| `lb`、`ub` | 浮点张量，`(D,)` | 每个决策变量的下界、上界。形状、dtype、设备须一致，并逐元素满足 `lb <= ub`。`D` 由边界推断。 |
| `device` | 显式支持时为 `torch.device` 或 `None` | 暴露该参数的构造函数中，`None` 表示 `torch.get_default_device()`，并将边界移动到指定设备。大多数其他算法在 `lb.device` 上分配张量。 |
| `**kwargs` | 支持时为额外命名设置 | 通常不会读取，包括未识别的 `device` 关键字。`LSMOF` 读取下文所列的四项设置。能接受关键字不表示对应功能生效。 |

显式支持 `device` 的构造函数是 `HypE`、`IBEA`、`LMOCSO`、`MOEAD`、`NSGA2`、`NSGA3`、
`RVEA`、`RVEAa` 和 `TensorMOEAD`。其他算法需要先将两个边界张量放到目标设备，再构造算法。

初始化后，常见状态 `pop` 的决策维数为 `D`，`fit` 的目标维数为 `n_objs`。
算法还可能保存档案、稀疏掩码、速度、参考向量或 NaN 占位槽；读取状态时按算法语义筛选有效结果。
`step()` 原地更新状态并返回 `None`，调用前需要先执行 `workflow.init_step()`。

当前这组算法中，文档说明的 `NSGA2` 评估路径支持 `(fitness, constraint_violation)`。
其他当前实现要求返回目标张量。接入约束问题前请阅读[约束指南](../guide/constraints.md)。

## 分解与差分进化

以下 `N` 表示权重采样后的实际种群数量。

| 参数 | 使用算法及默认值 | 用途与限制 |
| --- | --- | --- |
| `T` | `MOEAD_DCWV`、`MOEAD_DRA`、`MOEADDU`、`MOEADDYTS`、`MOEADFRRMAB`：`20` | 最近权重向量构成的交配邻域大小，包含自身；应为正数，超过 `N` 时截断为 `N`。 |
| `T` | `BCEMOEAD`、`MOEAD_PaS`：`None` | 自动取 `ceil(N / 10)`，随后按 `min(max(2, T), N)` 限制。 |
| `T` | `MOEADAWA`：`None` | 自动取 `ceil(N / 10)`；显式值须满足 `1 <= T <= N`，此实现不会截断过大的值。 |
| `delta` | `MOEADDE`、`MOEADDU`、`MOEADFRRMAB`、`MOEAURAW`：`0.9` | 在局部邻域中抽取交配父代的概率，范围为 `[0, 1]`；其余抽样来自全种群。 |
| `nr` | `MOEADDE`、`MOEAD_DRA`、`MOEADDYTS`、`MOEADFRRMAB`、`MOEAURAW`：`2` | 单个子代最多替换的已有解数，与交配邻域大小是两个设置。 |
| `nr` | `BCEMOEAD`、`MOEADAWA`：`None` | 自动取 `ceil(N / 100)`；`BCEMOEAD` 还将结果限制到 `[1, N]`。 |
| `nEP` | `MOEAURAW`：`200`；`MOEADAWA`：`None` | 外部档案容量；`MOEADAWA` 的自动值为 `ceil(1.5 * N)`。 |
| `F` | `GDE3`、`MOEADDE`：`0.5` | 正的差分步长系数，乘在父代决策向量的差上。 |
| `CR` | `GDE3`：`0.5` | 二项交叉中选择变异向量分量的概率，范围为 `[0, 1]`；每个试验解还会强制使用一个变异分量。 |
| `CR` | `MOEADDE`：`1.0` | 构造函数接受并保存，但当前未使用。该实现直接生成差分子代，再施加多项式变异，没有二项交叉。 |
| `K` | `EFRRR`：`2`；`MOEADDU`：`5` | 分别表示受限排名允许的最近参考权重数，或更新时按余弦相似度考虑的子问题数。应为正数，且不能超过可用向量数量。 |
| `window_size` | `MOEADFRRMAB`：`None` | 奖励窗口的记录条数，自动值为 `ceil(N / 2)`，须为正数。一个工作流步骤含五次内部子迭代，因此窗口大小并非工作流步数。 |
| `p` | `MOEAD_DCWV`：`-1.0` | `-1.0` 启用自适应权重变换；其他值关闭该分支，当前代码不会将其数值用作固定变换参数。 |
| `aggregate_op` | `TensorMOEAD`：`("pbi", "pbi")` | 分别用于已有解比较和合格子代选择的两个聚合函数名称。可选 `pbi`、`tchebycheff`、`tchebycheff_norm`、`modified_tchebycheff`、`weighted_sum`。 |

## 调度、选择与稀疏性

| 参数 | 使用算法及默认值 | 用途与限制 |
| --- | --- | --- |
| `max_gen` | `LMOCSO`、`RVEA`、`RVEAa`、`MOEAD_PaS`、`SSCEA`：`100` | 正的代数调度上限，用于自适应或惩罚系数；终止循环由调用者控制。 |
| `max_fe` | `TELSO`、`TSNSGAII`：`10000`；也可通过 `LSMOF` 关键字传入 | 正的评估量调度上限。`TSNSGAII`、`LSMOF` 在一半处切换阶段，`TELSO` 用它缩放学习和惩罚进度；不会自动结束工作流。 |
| `alpha` | `LMOCSO`、`RVEA`、`RVEAa`：`2.0` | `(gen / max_gen) ** alpha` 的指数；较大的正值延后角度惩罚增长。 |
| `fr` | `RVEA`、`RVEAa`：`0.1` | 正的参考向量调整频率，实际间隔为 `max(round(1 / fr), 1)` 步；默认间隔为 10 步，与 `max_gen` 无关。 |
| `t_max` | `OSP_NSDE`：`100` | 轨迹缓存容量，含初始条目。后续步骤数应小于该容量，因为缓存不会循环覆盖。 |
| `kappa` | `BCE_IBEA`、`IBEA`：`0.05` | 指数型指标适应度的正缩放系数；更小的值放大指标差异的影响。 |
| `n_sample` | `HypE`：`10000` | 每次超体积估计的正蒙特卡洛样本数；增加样本降低估计噪声，也增加计算成本。 |
| `div` | `GrEA`、`PESA2`：`10` | 每个目标上的正网格划分数；更大的值产生更细的目标空间网格。 |
| `type` | `CMOEA_MS`：`1` | `1` 使用 SBX 与多项式变异；其他值进入固定尺度的差分变异分支。 |
| `t` | `MaOEACSS`：`0.0` | 对角度接近的一对解进行淘汰时，归一化收敛分数差的阈值。 |
| `eps` | `CoMMEA`：`0.2` | 采用 `(1 + eps) * front1_fit` 的乘法容差；其行为受目标尺度和符号影响。 |
| `epsilon` | `eMOEA`：`0.05` | 所有目标共用的正加法网格宽度，以目标值的单位度量。 |
| `sLower`、`sUpper` | `SNSGA2`：`0.1`、`0.9` | 初始决策中零分量比例的两个端点，满足 `0 <= sLower <= sUpper <= 1`；只控制初始化，不保证后续永久保持该稀疏度。 |
| `gamma` | `WOF`：`10` | 正的决策变量分组数，也决定权重向量维数；通常不大于 `D`，以避免空组。 |
| `point` | `WASFGA`：`None` | 目标空间偏好点，形状为 `(n_objs,)` 或 `(1, n_objs)`；`None` 创建零点。显式点不会自动移动到边界所在设备。 |
| `data_type` | `NSGA3`：`None` | `torch.bool` 选择布尔初始化，其他值使用连续初始化；不会转换到任意指定 dtype。布尔运行需要兼容的交叉、变异算子。 |

`LSMOF` 还会从 `**kwargs` 读取参考解数 `wD=5`、权重候选数 `SubN=20`、
决策重构尺度 `wmax=0.1` 和评估调度上限 `max_fe=10000`。
第一阶段每步评估 `2 * wD * SubN` 个决策向量。数量参数应为正数，且须有足够的不同参考解满足 `wD`。
`wmax` 缩放沿单位方向的偏移，并非评估预算比例；初始评估计入阶段切换计数。

## 自定义算子接口

| 参数 | 调用约定 | 默认值与例外 |
| --- | --- | --- |
| `mutation_op` | `(offspring, lb, ub) -> offspring`，形状为 `(B, D)` | 暴露该参数时，`None` 使用 EvoX 的 `polynomial_mutation`；应保持决策维数和设备。 |
| `crossover_op` | `(parents) -> offspring`，输入输出为二维决策张量 | 默认使用 `simulated_binary`；`MOEAD`、`TensorMOEAD` 使用 `simulated_binary_half`，每对父代产生一个子代。 |
| `NSGA2`、`NSGA3` 的 `selection_op` | `(pop_size, sort_keys) -> parent_indices` | 交配选择，默认 `tournament_selection_multifit`；不会替换环境选择。 |
| `RVEA`、`RVEAa`、`LMOCSO` 的 `selection_op` | `(pop, fit, vectors, theta) -> (pop, fit)` | 环境选择。RVEA 两版默认使用 EvoMO 的 `ref_vec_guided`，LMOCSO 使用 EvoX 版本；应保持参考向量对应的输出槽位。 |
| `HypE` 的 `selection_op` | 构造签名接受该参数 | 当前实现将其覆盖为 `tournament_selection`，自定义值不生效。 |
| `MOEAD`、`TensorMOEAD` 的 `selection_op` | 构造函数保存该参数 | 当前 `step()` 未调用它，邻域交配由内部逻辑完成。 |

## MOEADFRRMAB 示例

下面的示例将边界显式放到所选设备，使用自动窗口大小，执行三个工作流步骤，
并输出实际种群数与窗口大小。这些步骤不构成相同评估预算的性能对比。

```{literalinclude} ../../examples/moead_frrmab.py
:language: python
```

完整构造字段与生命周期方法见 [MOEADFRRMAB API](../../autoapi/evomo/algorithms/moead_frrmab/index.rst)。
