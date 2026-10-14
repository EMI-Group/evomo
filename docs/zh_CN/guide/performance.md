# 设备、编译与性能

## 设置设备与 dtype

在创建算法、问题和张量之前设置默认设备，或为相关构造函数显式传入同一设备：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
```

默认设备会影响后续未显式指定设备的分配。在已有应用中需要控制这个全局设置的作用范围。
`device` 参数通常控制设备，不意味着算法自动保持任意输入的 dtype；不同算法的内部初始化
需要分别检查。使用非默认 dtype 时先核对问题、边界、状态与结果张量的 dtype，
必要时在构造前设置默认浮点类型并在使用后恢复。

## 编译工作流

先验证 eager 执行，再在初始化完成后编译 `step`：

```python
workflow.init_step()
compiled_step = torch.compile(workflow.step, fullgraph=True)
for _ in range(100):
    compiled_step()
```

EvoX 也提供 `evox.core.compile`；项目的部分算法测试使用
`compile(workflow.step, dynamic=False)`。选择方式应与实际算法和环境的验证结果一致。
编译时固定种群大小、决策维度、目标数及问题返回结构有助于减少重新编译。

首次调用包含图捕获与编译成本。`fullgraph=True` 成功只说明该调用完成完整图捕获，
不意味着单个 GPU kernel、CUDA Graph 支持或 `vmap` 兼容。
特定算法、算子后端和编译器组合可能不支持完整图捕获，需要在本机分别验证。

## 结果分析放在迭代之后

动态布尔筛选、`.item()`、打印、转 NumPy 和写文件适合放在迭代之外。
在 GPU 热路径中，读取 Python 标量会带来同步，数据依赖的分支和动态形状操作也可能影响编译。
自定义问题优先使用固定形状的掩码、`torch.where` 和批量运算。

## 计时方法

下面代码假定工作流已初始化，`step_function` 是 eager 或 compiled 的步函数。
预热会推进算法状态；要严格对比同一起点，分别创建工作流或恢复等价状态。

```python
import time

for _ in range(5):
    step_function()

cuda = workflow.algorithm.pop.device.type == "cuda"
if cuda:
    torch.cuda.synchronize(workflow.algorithm.pop.device)
start = time.perf_counter()
for _ in range(50):
    step_function()
if cuda:
    torch.cuda.synchronize(workflow.algorithm.pop.device)
seconds_per_step = (time.perf_counter() - start) / 50
print(seconds_per_step)
```

重复多轮并报告设备、PyTorch/EvoX 版本、dtype、实际种群大小、目标数、问题、编译选项、
预热次数与同步方法。该计时覆盖问题评估、算法更新和启用的监视器开销。
速度测试与解集质量测试分别报告；后者需要多个种子、相同评估预算与一致指标。

## 算子后端与向量化

非支配排序默认使用 PyTorch；可选 Triton 后端的要求和已知限制见
[后端说明](../non_dominate_backends.md)。选择独立算子后端不会自动切换算法内部实现。
Triton 后端没有 `vmap` 规则。其他算法的 `vmap` 使用也需验证具体状态和算子组合，
不能根据 compiled 单次运行成功推断支持批量独立实验。
