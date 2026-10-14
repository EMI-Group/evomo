# API 参考

以下页面由当前源码静态生成，包含构造签名、方法与已有 docstring。
中英文教程共用英文 API 页。57 个公开算法的构造参数都包含含义、类型、默认值及实现说明。
导入接口优先使用各模块公开导出；内部辅助方法不代表稳定接口。

```{toctree}
:maxdepth: 1

parameters
citation
```

- [算法](../../autoapi/evomo/algorithms/index.rst)
- [数值问题](../../autoapi/evomo/problems/numerical/index.rst)
- [约束问题](../../autoapi/evomo/problems/constrained/index.rst)
- [神经进化](../../autoapi/evomo/problems/neuroevolution/index.rst)
- [工作流](../../autoapi/evomo/workflows/index.rst)
- [选择算子](../../autoapi/evomo/operators/selection/index.rst)
- [指标](../../autoapi/evomo/metrics/index.rst)
- [工具函数](../../autoapi/evomo/utils/index.rst)

## 选择算子速查

常用选择算子从 `evomo.operators.selection` 导入，目标值采用最小化约定。

| 接口 | 输入 | 输出与用途 |
| --- | --- | --- |
| `non_dominate_rank(x, cv=None)` | `(N, M)` 目标和可选违反量 | `(N,)` 完整零起始等级 |
| `crowding_distance(costs, mask)` | 目标与 `(N,)` 布尔掩码，或 `None` | `(N,)` 拥挤距离 |
| `nd_environmental_selection(x, f, topk, cv=None)` | 种群、目标、存活数与可选违反量 | `(selected_pop, selected_fit, selected_rank, selected_distance, selected_cv)` |
| `ref_vec_guided(x, f, v, theta)` | 种群、目标、参考向量与角度惩罚参数 | `(next_pop, next_fit)`；未关联槽位可能为 NaN |
| `get_non_dominate_backend(backend="torch")` | `"torch"` 或 `"triton"` | 对应后端模块；不会自动修改算法 |

无约束环境选择的第五个返回值为 `None`。`topk` 应在有效种群大小范围内。
环境选择只需排名到截止前沿，而返回的已选择个体等级仍完整；
需要全部输入个体等级时使用 `non_dominate_rank`。

后端模块还提供 `dominate_relation(x, y, cv_x=None, cv_y=None)`，返回两组目标间的支配矩阵。
这个函数不在 `evomo.operators.selection` 的顶层公开导出中，需从后端模块访问。

## 评估结果与状态工具

- `parse_evaluate(eval_out)`：返回 `(fitness, cv)`；纯目标张量对应 `cv=None`。
- `at_least_2d`：处理单个解与批量输入，具体返回契约见生成页面。
- `get_pareto_front`、`unique_rows_sorted`：结果筛选与去重，注意动态输出大小对编译的影响。
- `load_pareto_front_from_file`：加载约束问题的包内参考前沿资源。
- `register_lazy_buffer`：首次评估才确定形状时的状态辅助工具。

工作流和约束处理的语义说明分别见[工作流指南](../guide/workflows.md)与
[约束指南](../guide/constraints.md)。
