from typing import Any, Tuple, Union

import torch
from evox.core import Algorithm, Monitor, Problem, Workflow
from evox.workflows.std_workflow import OptDirectionTransform


class UnifiedWorkflow(Workflow):
    """
    A unified standard workflow for evomo that supports both constrained and unconstrained problems.
    If the problem's evaluate function returns a tuple (fitness, cv),
    both will be correctly handled, gathered and passed to the algorithm.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitor: Monitor | None = None,
        opt_direction: str | list[str] = "min",
        solution_transform: torch.nn.Module | None = None,
        fitness_transform: torch.nn.Module | None = None,
        device: str | torch.device | int | None = None,
        enable_distributed: bool = False,
        group: Any = None,
    ):
        super().__init__()
        if device is None:
            device = torch.get_default_device()

        if isinstance(opt_direction, str):
            assert opt_direction in [
                "min",
                "max",
            ], f"Expect optimization direction to be `min` or `max`, got {opt_direction}"
            self.opt_direction = torch.tensor(1 if opt_direction == "min" else -1, device=device)
        elif isinstance(opt_direction, list):
            assert all(d in ["min", "max"] for d in opt_direction), (
                f"Expect optimization direction to be `min` or `max`, got {opt_direction}"
            )
            self.opt_direction = torch.tensor([1 if d == "min" else -1 for d in opt_direction], device=device)

        if solution_transform is None:
            solution_transform = torch.nn.Identity()
        if fitness_transform is None:
            fitness_transform = torch.nn.Identity()

        fitness_transform = torch.nn.Sequential(OptDirectionTransform(self.opt_direction), fitness_transform)

        assert callable(solution_transform), f"Expect solution transform to be callable, got {solution_transform}"
        assert callable(fitness_transform), f"Expect fitness transform to be callable, got {fitness_transform}"

        if isinstance(solution_transform, torch.nn.Module):
            solution_transform.to(device=device)
        if isinstance(fitness_transform, torch.nn.Module):
            fitness_transform.to(device=device)

        if monitor is None:
            monitor = Monitor()
        else:
            monitor.set_config(opt_direction=self.opt_direction)
        algorithm.to(device=device)
        monitor.to(device=device)
        problem.to(device=device)

        self._has_init_ = type(algorithm).init_step != Algorithm.init_step
        self._has_final_ = type(algorithm).final_step != Algorithm.final_step

        class _SubAlgorithm(type(algorithm)):
            def __init__(self_algo):
                super(Algorithm, self_algo).__init__()
                self_algo.__dict__.update(algorithm.__dict__)

            def evaluate(self_algo, pop: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                return self._evaluate(pop)

        self.algorithm = _SubAlgorithm()
        self.monitor = monitor
        self.problem = problem
        self.solution_transform = solution_transform
        self.fitness_transform = fitness_transform
        self.enable_distributed = enable_distributed
        self.group = group

    def _evaluate(self, population: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.monitor.post_ask(population)

        if self.enable_distributed:
            rank = torch.distributed.get_rank(group=self.group)
            pop_size = population.size(0)
            world_size = torch.distributed.get_world_size(group=self.group)
            population = population.tensor_split(world_size, dim=0)[rank]

        population = self.solution_transform(population)
        self.monitor.pre_eval(population)

        if self.enable_distributed:
            with torch.random.fork_rng():
                eval_out = self.problem.evaluate(population)

            is_tuple = isinstance(eval_out, tuple)
            if is_tuple:
                fitness, cv = eval_out
            else:
                fitness, cv = eval_out, None

            # Gather fitness
            all_fitness = torch.zeros(pop_size, *fitness.shape[1:], device=fitness.device, dtype=fitness.dtype)
            all_fitness_list = list(all_fitness.tensor_split(world_size, dim=0))
            torch.distributed.all_gather(all_fitness_list, fitness, group=self.group)
            fitness = torch.cat(all_fitness_list, dim=0)

            # Gather cv if exists
            if cv is not None:
                all_cv = torch.zeros(pop_size, *cv.shape[1:], device=cv.device, dtype=cv.dtype)
                all_cv_list = list(all_cv.tensor_split(world_size, dim=0))
                torch.distributed.all_gather(all_cv_list, cv, group=self.group)
                cv = torch.cat(all_cv_list, dim=0)
        else:
            eval_out = self.problem.evaluate(population)
            if isinstance(eval_out, tuple):
                fitness, cv = eval_out
            else:
                fitness, cv = eval_out, None

        self.monitor.post_eval(fitness)
        fitness = self.fitness_transform(fitness)
        self.monitor.pre_tell(fitness)

        if cv is not None:
            return fitness, cv
        return fitness

    def _step(self, init: bool = False, final: bool = False):
        if init and self._has_init_:
            self.algorithm.init_step()
        elif final and self._has_final_:
            self.algorithm.final_step()
        else:
            self.algorithm.step()

        if "record_auxiliary" in self.monitor.__class__.__dict__:
            self.monitor.record_auxiliary(self.algorithm.record_step())

    def init_step(self):
        self._step(init=True, final=False)

    def final_step(self):
        self._step(init=False, final=True)

    def step(self):
        self._step()
