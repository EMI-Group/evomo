# Neuroevolution and MoRobtrol

MoRobtrol evaluates PyTorch policies on multiple control objectives using Brax (the default)
or Playground/MJX environments. Decision variables are network parameters, and outputs are
objective rewards for each policy. This chapter requires [optional dependencies](installation.md);
numerical optimization does not require a simulator.

## Parameter vectors and policy inputs

`evox.utils.ParamsAndVector` converts network parameter vectors to parameter dictionaries
with a population dimension. The algorithm searches parameter vectors, and the workflow
passes them to MoRobtrol through `solution_transform=adapter`. Rewards are maximized, so use
`opt_direction="max"`. The algorithm's internal `fit` contains negated rewards.

## Brax example

```{literalinclude} ../examples/morobtrol.py
:language: python
```

```sh
python -m pip install "evomo[neuroevolution]"
python docs/examples/morobtrol.py
```

This script uses short episodes and a few generations to demonstrate the interface; it is
not expected to produce a trained control policy. It uses CPU if either Torch or JAX lacks
a usable GPU. PyTorch CUDA visibility does not mean JAX uses the same device; check both.

MoRobtrol's `pop_size` must match the number of policies passed to each evaluation. This example
uses NSGA-II with a fixed population size. If switching to an algorithm that adjusts population
size, read the actual size before constructing the problem and verify that each evaluation
still passes the expected number of policies.

## Using Playground

After installing `evomo[playground]`, query task metadata and construct a network with matching
dimensions:

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

Use `MoRobtrol.available_playground_tasks()` to inspect tasks available in the installed
runtime. Native task assets may be downloaded on first use. Use the same `env_config`,
`observation_key`, and `objectives` when querying metadata and constructing the problem.
After changing the network, also recreate the parameter adapter, bounds, and algorithm.
Read `n_objs` from `problem.num_obj`.

## Randomness and observation normalization

| Parameter | Current behavior |
| --- | --- |
| `seed` | Initializes simulator randomness |
| `rotate_key=True` | Updates the random key on each evaluation |
| `rotate_key=False` | Reuses episode seeds; changing normalization state may still affect results |
| `num_episodes` | Episodes per candidate policy |
| `reduce_fn` | Aggregates rewards across episodes; defaults to `torch.mean` |
| `useless=True` | Historical name for disabling observation normalization, the default |
| `useless=False` | Enables shared running observation statistics and normalization |
| `obs_norm` | `[clip, min_variance, max_variance]`; defaults to `[5.0, 1e-6, 1e6]` |

`num_obj` and `observation_shape` are inferred from the environment by default. Explicit values
must match it. Rewards include the last valid terminal transition, and candidates share episode seeds.

Rewards from different physics engines or backends are not necessarily interchangeable.
Validate cross-engine comparisons separately. MoRobtrol's outer `vmap` supports batches of
policy parameters, not independent problem states or nested HPO repetitions. The combined
policy count must match `pop_size`.
