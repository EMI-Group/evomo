# 解集质量指标

从 `evomo.metrics` 导入 `gd`、`igd`、`hv`。它们评估的是目标空间中的解集，而不是决策变量。
调用之前排除非有限值；约束问题还需先筛选可行解。比较不同运行时采用相同目标方向、
尺度、参考前沿和参考点。

## GD 与 IGD

设 A 为求得的解集，P 为参考前沿，`d(a, P)` 表示点 a 到 P 的最小欧氏距离。
当前实现使用：

$$
\operatorname{GD}(A,P)=\frac{\sqrt{\sum_{a\in A}d(a,P)^2}}{|A|}
$$

$$
\operatorname{IGD}_p(A,P)=\left(\frac{1}{|P|}\sum_{q\in P}d(q,A)^p\right)^{1/p}
$$

`gd(objs, pf)` 使用上面的固定定义，不是最小距离的算术平均或均方根。
`igd(objs, pf, p=1)` 默认是参考前沿各点到解集的最小距离的算术平均。
二者越小越好，输入必须非空；IGD 的 `p` 应为正数。

```python
from evomo.metrics import gd, igd

reference_front = problem.pf().to(device=pareto_fitness.device, dtype=pareto_fitness.dtype)
gd_score = gd(pareto_fitness, reference_front)
igd_score = igd(pareto_fitness, reference_front)
```

代码接在快速入门的结果筛选之后使用。`torch.cdist` 会形成两组点之间的距离矩阵；
超大解集和参考前沿需要考虑显存开销。目标量纲相差很大时距离会被较大尺度的目标主导，
若做归一化，应对解集和参考前沿使用相同变换并记录规则。

## HV：蒙特卡洛估计

`hv(objs, ref, num_sample=100000)` 按最小化约定估计超体积。`ref` 的形状为 `(M,)`，
应选择比关注的解更差的参考点，并在所有比较中固定它。不在参考点以内的解不贡献体积。
在固定参考点与尺度下，HV 越大越好。

```python
import torch
from evomo.metrics import hv

objectives = torch.tensor([[0.2, 0.8], [0.5, 0.4], [0.8, 0.2]])
reference = objectives.new_tensor([1.1, 1.1])
torch.manual_seed(42)
print(hv(objectives, reference, num_sample=10000))
```

这是随机估计，不是精确的超体积计算。增大 `num_sample` 会增加时间和内存成本；
内部比较规模随样本数、解集大小和目标数增长。所有样本一次性通过广播张量运算计算。
float16、bfloat16 和整数输入提升到至少 float32，
float64 输入保留 float64 运算。空解集返回浮点标量零；非有限目标行和参考点边界上的解
不贡献体积，非有限参考点返回 NaN。形状不匹配、非正或非整数采样数会触发错误。

高维解集的支配体积可能只占采样盒的极小一部分，有限次数采样可能完全未命中而返回零，
因此估计值为零不能证明精确 HV 为零。

这里返回输入目标尺度下的体积。[PlatEMO 的 `HV`](https://github.com/BIMK/PlatEMO/blob/master/PlatEMO/Metrics/HV.m) 会先按参考前沿归一化，目标数少于 4 时
采用精确计算，因此比较两者的报告值前应统一归一化规则。

最大化问题需同时取负目标与参考点。例如 `hv(-rewards, -reference_rewards)`。
混合目标方向则按每列方向同时转换两个输入。不要将恢复后的最大化目标直接按最小化调用 HV。

## 可复现比较

记录算法种子、评估预算、解集提取范围（最终种群或历史档案）、参考前沿采样方式、
HV 参考点与采样数。多个种子报告分布或均值与离散程度，避免把一次运行的指标当作稳定结论。
