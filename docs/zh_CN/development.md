# 开发与贡献

## 目录与环境

| 路径 | 内容 |
| --- | --- |
| `src/evomo/algorithms/` | 算法实现与公开导出 |
| `src/evomo/operators/` | 选择与可复用算子 |
| `src/evomo/problems/` | 数值、约束与神经进化问题 |
| `src/evomo/workflows/` | 算法、问题与监视器的连接 |
| `src/evomo/metrics/` | 解集质量指标 |
| `src/evomo/utils/` | 张量、评估结果与状态辅助工具 |
| `unit_test/` | 算法、算子与问题测试 |
| `docs/` | 本文档、示例与构建配置 |

先检查已有 Python 环境和 CUDA 配置，再安装可编辑包和开发工具：

```sh
python -m pip install -e .
python -m pip install pytest ruff
```

## 扩展算法

继承 EvoX 的 `Algorithm`，按照现有实现管理可变状态。种群、适应度与其他逐代更新数据
遵循 `Mutable` 模式；首次评估后才确定形状的 buffer 可以参考
`evomo.utils.register_lazy_buffer`。在工作流提供的 `evaluate` 中调用问题，
在 `init_step` 初始化状态，在 `step` 实现每代更新。

新增公开算法时更新 `src/evomo/algorithms/__init__.py` 的导入及 `__all__`，
保持类名和构造接口清晰。算法的 `selection_op` 等参数应明确调用契约，不能只说明“可替换算子”。
文档和论文引用随实现一并更新。

## 张量实现约定

- 保留设备和 dtype；常量使用输入张量的设备，避免隐藏的 CPU 分配。
- 优先批量计算；编译热路径减少 `.item()`、张量驱动的 Python 分支和动态大小索引。
- 使用固定形状掩码及 scatter/gather 等操作时，保持原算法的排序、选择和更新语义。
- 非支配等级从 0 开始，不用布尔选择掩码替代等级。
- 约束问题的违反量应贯穿评估、比较、选择与状态更新。
- 新增可选后端之前阅读[非支配排序后端说明](non_dominate_backends.md)，保持 PyTorch 默认路径。

## 验证改动

在仓库根目录运行相关测试，例如：

```sh
python -m pytest unit_test/problems/test_dtlz.py -q
python -m pytest unit_test/operators/test_non_dominate.py -q
python -m pytest unit_test/algorithms/test_moea.py -q
```

选择与归一化改动应覆盖重复点、常量目标、退化目标范围、非有限值和约束处理。
涉及编译的代码同时验证 eager 与 compiled；先初始化工作流再编译 `step`。
区分工具链或设备不支持与算法错误，按实际验证范围报告结果。

只对修改的 Python 文件执行检查，示例如下：

```sh
python -m ruff check src/evomo/algorithms/nsga2.py
python -m ruff format --check src/evomo/algorithms/nsga2.py
```

项目使用 128 字符行长和 LF 换行。保持改动范围，保留无关用户修改。
实验脚本与产物独立存放，不提交本地环境、缓存或临时性能数据。

## 维护文档

安装独立的文档依赖并将警告视为构建失败：

```sh
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

API 参考通过源码静态解析生成，不需要导入 PyTorch、Brax 或 JAX。
英文主文档位于 `docs/`，中文版位于 `docs/zh_CN/`，章节文件名与相对路径一一对应。
新增或修改教程时同步更新两种语言的参数、公式、限制和示例，并更新两版首页导航。
页面的语言切换会按路径查找对应章节；自动生成的英文 API 页为两种语言共用。

教程使用 MyST Markdown；完整代码放在 `docs/examples/`，正文通过 `literalinclude` 引用。
英文指南使用 `../examples/`，中文指南使用 `../../examples/`，确保引用同一份代码。
修改示例后检查实际导入、形状和设备。
文档构建不执行示例，构建成功不代表运行或 GPU/编译验证成功。

更多构建文件说明见仓库中的 `docs/README.md`。`.readthedocs.yaml` 提供 Read the Docs
构建配置，线上发布需在服务中连接仓库。
