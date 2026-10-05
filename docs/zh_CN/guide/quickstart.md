# 快速入门：NSGA-II 求解 DTLZ2

本例在 12 维决策空间内最小化 DTLZ2 的 3 个目标，决策变量范围为 `[0, 1]`。
程序自动选择可用的 CUDA 设备或 CPU，固定随机种子，运行初始化和 50 次迭代，
最后从当前种群中提取非支配解并计算 IGD。

## 完整代码

```{literalinclude} ../../examples/quickstart.py
:language: python
```

源码安装后，从仓库根目录运行：

```sh
python docs/examples/quickstart.py
```

## 理解结果

`workflow.algorithm.pop` 保存决策变量，`workflow.algorithm.fit` 保存对应目标值。
每个目标值行对应同一行的候选解。算法内部以最小化为约定；本例没有改变优化方向，
所以这些值也是 DTLZ2 的原始目标值。

`non_dominate_rank(fitness) == 0` 提取当前解集的第一非支配前沿。筛选在迭代结束之后执行，
不放进编译的 `step` 中。`problem.pf()` 提供问题的参考前沿采样；IGD 越小表示参考点
平均更接近求得的解集。不同硬件、版本和随机过程可能得到不同的数值，本文不规定固定分数。

本例的 50 次迭代用于展示接口，不构成收敛保证或算法性能结论。

## 下一步

- 替换算法时，查看[算法指南](algorithms.md)，核对构造参数与实际种群数。
- 替换问题时，查看[基准问题](problems.md)，重新设置维度、目标数和合法边界。
- 处理可行性时，使用[约束优化示例](constraints.md)。
- 启用编译之前，先阅读[编译与性能](performance.md)。
