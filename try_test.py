# from algorithms import TensorHypE, TensorNSGA3, MORandom, TensorMOEAD
import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from evox.workflows import StdWorkflow, EvalMonitor
from evox.algorithms import PSO, NSGA2
# from problems import MoRobtrol, Obs_Normalizer, MoBraxProblem
from problems import MoBraxProblem, Obs_Normalizer
from jax import random
import jax
import jax.numpy as jnp
import time
# from flax import linen as nn
import torch
import torch.nn as nn
from evox.utils import ParamsAndVector

import json
# from tqdm import tqdm
# from evox.utils import TreeAndVector
# from evox.operators import non_dominated_sort
# from tensorboardX import SummaryWriter
# from evox.metrics import HV

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)
        
# Make sure that the model is on the same device, better to be on the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# Reset the random seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Initialize the MLP model
model = SimpleMLP().to(device)

adapter = ParamsAndVector(dummy_model=model)

# Set the population size
POP_SIZE = 8
OBJs = 2

model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)
lower_bound = torch.full_like(pop_center, -5)
upper_bound = torch.full_like(pop_center, 5)

algorithm = NSGA2(
    pop_size=POP_SIZE,
    n_objs=OBJs,
    lb=lower_bound,
    ub=upper_bound,
    device=device,
)
algorithm.setup()

obs_norm = {"clip_val": 5.0,
            "std_min": 1e-6,
            "std_max": 1e6,}

# Initialize the Brax problem
# obs_norm = Obs_Normalizer(observation_shape=8, useless=False)
problem = MoBraxProblem(
    policy=model,
    env_name="mo_swimmer",
    max_episode_length=1000,
    num_episodes=3,
    pop_size=POP_SIZE,
    device=device,
    backend="generalized",
    num_obj=OBJs,
    observation_shape=8,
    obs_norm=obs_norm,
    useless=False,
)

# set an monitor, and it can record the top 3 best fitnesses
monitor = EvalMonitor(
    # topk=3,
    device=device,
)
monitor.setup()

workflow = StdWorkflow(opt_direction="max")
workflow.setup(
    algorithm=algorithm,
    problem=problem,
    solution_transform=adapter,
    monitor=monitor,
    device=device,
)

# Set the maximum number of generations
max_generation = 50

# Run the workflow
for i in range(max_generation):
    if i % 10 == 0:
        print(f"Generation {i}")
    workflow.step()

monitor = workflow.get_submodule("monitor")
print(f"Solution history: {monitor.get_solution_history()}")
# best_params = adapter.to_params(monitor.get_best_solution())
print(f"Fitness history: {monitor.get_fitness_history()}")
