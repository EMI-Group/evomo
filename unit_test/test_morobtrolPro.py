import os
import sys

# 获取项目的根目录，并添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

import torch
import torch.nn as nn

from evox.algorithms import NSGA2
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from src.evox.problems import MoRobtrolPro


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)


class TestProTest(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        # 重置随机种子
        self.seed = 1234
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # 初始化 MLP 模型
        self.model = SimpleMLP().to(self.device)
        self.adapter = ParamsAndVector(dummy_model=self.model)

        # 设置种群大小和目标数
        self.POP_SIZE = 8
        self.OBJs = 2

        # 获取模型参数并设定搜索边界
        model_params = dict(self.model.named_parameters())
        self.pop_center = self.adapter.to_vector(model_params)
        self.lower_bound = torch.full_like(self.pop_center, -5)
        self.upper_bound = torch.full_like(self.pop_center, 5)

        # 初始化 NSGA2 算法
        self.algorithm = NSGA2(
            pop_size=self.POP_SIZE,
            n_objs=self.OBJs,
            lb=self.lower_bound,
            ub=self.upper_bound,
            device=self.device,
        )
        self.algorithm.setup()

        self.obs_norm = {"clip_val": 5.0, "std_min": 1e-6, "std_max": 1e6}

        # 初始化测试问题
        self.problem = MoRobtrolPro(
            policy=self.model,
            env_name="mo_swimmer",
            max_episode_length=1000,
            num_episodes=3,
            pop_size=self.POP_SIZE,
            device=self.device,
            backend="generalized",
            num_obj=self.OBJs,
            observation_shape=8,
            obs_norm=self.obs_norm,
            useless=False,
        )

        # 监控器
        self.monitor = EvalMonitor(device=self.device)
        self.monitor.setup()

        # 工作流
        self.workflow = StdWorkflow(opt_direction="max")
        self.workflow.setup(
            algorithm=self.algorithm,
            problem=self.problem,
            solution_transform=self.adapter,
            monitor=self.monitor,
            device=self.device,
        )

    def test_mlp_forward(self):
        """测试 MLP 网络前向传播"""
        x = torch.randn(1, 8).to(self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 2), "MLP输出的形状应为 (1,2)")

    def test_algorithm_initialization(self):
        """测试 NSGA2 算法初始化"""
        self.assertEqual(self.algorithm.pop_size, self.POP_SIZE, "种群大小应正确初始化")
        self.assertEqual(self.algorithm.n_objs, self.OBJs, "目标数量应正确初始化")

    # def test_workflow_step(self):
    #     """测试 workflow 的 step() 方法"""
    #     initial_pop = self.algorithm.pop.clone()

    #     self.workflow.step()
    #     new_pop = self.algorithm.pop.clone()

    #     self.assertFalse(torch.equal(initial_pop, new_pop), "种群应在演化过程中发生变化")

    def test_monitor_logging(self):
        """测试 monitor 是否正确记录数据"""
        self.workflow.step()
        self.monitor = self.workflow.get_submodule("monitor")
        fitness_history = self.monitor.get_fitness_history()

        # print(fitness_history)
        self.assertTrue(len(fitness_history) > 0, "应记录至少一个 fitness 值")

    # def test_time_logging(self):
    #     """测试时间记录功能"""
    #     max_generation = 5  # 限制测试代数，避免耗时
    #     times = []
    #     start_time = time.perf_counter()

    #     for i in range(max_generation):
    #         self.workflow.step()
    #         times.append(time.perf_counter() - start_time)

    #     self.assertEqual(len(times), max_generation, "时间记录应与迭代次数匹配")
    #     self.assertGreater(times[-1], times[0], "时间应随迭代增加")

if __name__ == "__main__":
    unittest.main()
