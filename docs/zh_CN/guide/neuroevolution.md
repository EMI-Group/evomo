# 神经进化与 MoRobtrol

MoRobtrol 使用 PyTorch 策略网络，在 Brax（默认）或 Playground/MJX 环境中评估多个控制目标。
优化变量为网络参数，输出为每个策略的目标奖励。
此章节需要[可选依赖](installation.md)，普通数值优化无需安装模拟器。

## 参数向量与策略输入

`evox.utils.ParamsAndVector` 将网络参数向量转换为带种群维度的参数字典。
算法负责优化参数向量，工作流通过 `solution_transform=adapter` 将它们传给 MoRobtrol。
奖励按最大化处理，因此使用 `opt_direction="max"`；内部算法的 `fit` 是取负后的奖励。

## Brax 示例

```{literalinclude} ../../examples/morobtrol.py
:language: python
```

```sh
python -m pip install "evomo[neuroevolution]"
python docs/examples/morobtrol.py
```

该脚本只运行短回合和少量代数以展示接口，不预期产生成熟控制策略。
如果 Torch 或 JAX 任一侧没有可用 GPU，示例使用 CPU。
PyTorch 的 CUDA 可见性不代表 JAX 使用相同设备；实际运行需分别检查。

MoRobtrol 的 `pop_size` 必须等于每次传入的策略数。
本例使用种群大小固定的 NSGA-II。切换到会调整种群数的算法时，应读取实际种群数后
创建问题，并验证每次评估的候选策略数仍然匹配。

## 切换 Playground

安装 `evomo[playground]` 后，可以查询环境元信息，再按其维度构造网络：

```python
from evomo.problems.neuroevolution import MoRobtrol

info = MoRobtrol.playground_task_info("CartpoleBalance")
policy = torch.nn.Sequential(
    torch.nn.Linear(info["observation_size"], info["action_size"]),
    torch.nn.Tanh(),
).to(device)
problem = MoRobtrol(
    policy=policy,
    env_name="CartpoleBalance",
    engine="playground",
    max_episode_length=100,
    num_episodes=2,
    pop_size=20,
    seed=42,
    device=device,
)
```

用 `MoRobtrol.available_playground_tasks()` 查看当前安装环境中的可用任务。
原生任务的模型资源可能在首次使用时下载。查询元信息与构造问题时，应使用相同的
`env_config`、`observation_key`、`objectives` 配置。
更换网络后也要重新创建参数适配器、边界和算法；`n_objs` 从 `problem.num_obj` 读取。

## 随机性与归一化

| 参数 | 当前行为 |
| --- | --- |
| `seed` | 初始化模拟器随机种子 |
| `rotate_key=True` | 每次评估更新随机键 |
| `rotate_key=False` | 重复使用回合种子；归一化状态变化仍可能改变结果 |
| `num_episodes` | 每个候选策略的回合数 |
| `reduce_fn` | 按回合聚合奖励，默认 `torch.mean` |
| `useless=True` | 历史命名，表示默认关闭观测归一化 |
| `useless=False` | 开启共享的运行观测统计与归一化 |
| `obs_norm` | `[clip, min_variance, max_variance]`，默认 `[5.0, 1e-6, 1e6]` |

`num_obj` 与 `observation_shape` 默认从环境推断；显式传入时必须匹配环境。
奖励包含终止时的最后一次有效转移，各候选策略共享回合种子。

不同物理引擎或 backend 的奖励不一定可直接互换，跨引擎比较需要单独设计验证。
MoRobtrol 的外层 `vmap` 仅支持策略参数批次，不支持映射独立问题状态或嵌套 HPO 重复；
参数批次对应的总策略数需与 `pop_size` 匹配。
