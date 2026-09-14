# Non-dominated sorting backends

The existing PyTorch implementation remains the default. An optional,
experimental Triton backend accelerates dominance construction and ranking on
CUDA. It retains PyTorch crowding distance and stable environmental selection;
the extended Triton crowding/sorting experiment is not part of this backend.

Select a backend once during setup, before compiling a function or algorithm:

```python
from evomo.operators.selection import get_non_dominate_backend

ops = get_non_dominate_backend("torch")  # Default; also works without Triton
# ops = get_non_dominate_backend("triton")

rank = ops.non_dominate_rank(fitness, cv=None)
survivors = ops.nd_environmental_selection(population, fitness, topk=100, cv=None)
```

Both modules expose `non_dominate_rank`, `dominate_relation`, `crowding_distance`,
and `nd_environmental_selection`, with the same argument and return conventions.
`non_dominate_rank` returns complete ranks for every individual; environmental
selection ranks through the complete cutoff front. Existing imports continue
to select PyTorch and do not load the Triton backend.

NSGA2 calls the default PyTorch environmental selection directly. Its optional
`selection_op` selects parents for mating and does not switch the environmental
selection backend. Resolving a backend above only selects standalone operators;
it does not change the operators used by NSGA2 or other algorithms.

## Requirements and limits

- Triton is optional and is not added to EvoMO's mandatory dependencies. Install
  a Triton distribution compatible with your PyTorch, CUDA and operating system
  before requesting it. PyTorch must expose `torch.library.triton_op` and
  `torch.library.wrap_triton`. Missing support produces an import error.
- The tested environment was PyTorch 2.11.0+cu128, Triton 3.6.0 and RTX 5060 Ti.
  CUDA float32/float64 objectives and a single `[N, M]` population are supported.
  CPU input falls back to PyTorch after the optional backend has been loaded.
- No Triton `vmap` rules are provided. Use PyTorch for batched/vmapped algorithms.
- Fullgraph compilation can execute the backend, but may recompile. This does
  not imply CUDA Graph support or a single fused GPU kernel.
- Historical size sweeps found small floating-point differences in crowding and
  algorithm states, including with the ranking-only backend. Equal final IGD
  does not guarantee exact intermediate equality. Use PyTorch when that is
  required. This integration does not resolve those numerical differences.
- Small populations can be slower with Triton. Choose explicitly according to
  your workload; no population-size heuristic switches backends automatically.
