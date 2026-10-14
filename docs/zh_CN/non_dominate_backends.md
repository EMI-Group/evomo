# 非支配排序后端

现有 PyTorch 实现继续作为默认后端。可选的实验性 Triton 后端加速 CUDA 上的支配关系
构建和排名，同时保留 PyTorch 的拥挤距离及稳定环境选择。
扩展的 Triton 拥挤距离/排序实验不属于这个后端。

在初始化阶段选择后端，先完成选择再编译函数或算法：

```python
from evomo.operators.selection import get_non_dominate_backend

ops = get_non_dominate_backend("torch")  # 默认后端，无需 Triton
# ops = get_non_dominate_backend("triton")

rank = ops.non_dominate_rank(fitness, cv=None)
survivors = ops.nd_environmental_selection(population, fitness, topk=100, cv=None)
```

两个模块都提供 `non_dominate_rank`、`dominate_relation`、`crowding_distance` 和
`nd_environmental_selection`，参数及返回值约定一致。
`non_dominate_rank` 返回全部个体的完整等级；环境选择排名到完整的截止前沿。
现有导入继续使用 PyTorch，不会加载 Triton 后端。

NSGA2 直接调用默认的 PyTorch 环境选择。其 `selection_op` 用于选择交配父代，
不会切换环境选择后端。上面的后端解析仅选择独立算子，不改变 NSGA2 或其他算法内部的算子。

## 要求与限制

- Triton 是可选依赖，不属于 EvoMO 的必需依赖。显式选择前，安装与 PyTorch、CUDA 和
  操作系统兼容的版本。PyTorch 需提供 `torch.library.triton_op` 和
  `torch.library.wrap_triton`；缺少支持会产生导入错误。
- 原后端指南记录的测试环境为 PyTorch 2.11.0+cu128、Triton 3.6.0 和 RTX 5060 Ti。
  支持 CUDA float32/float64 目标和单个 `[N, M]` 种群。
  加载可选后端后，CPU 输入回退到 PyTorch。
- 没有提供 Triton `vmap` 规则。批量或 `vmap` 算法使用 PyTorch。
- 完整图编译可以执行该后端，但可能重新编译。这不意味着 CUDA Graph 支持或单个融合 GPU kernel。
- 历史规模测试在拥挤距离和算法状态中观察到细小浮点差异，包括仅替换排名的后端。
  最终 IGD 相同不保证中间状态完全相同；需要完全一致时使用 PyTorch。
  当前集成并未解决这些数值差异。
- 小种群使用 Triton 可能更慢。根据实际工作负载显式选择，系统不会按种群大小自动切换后端。
