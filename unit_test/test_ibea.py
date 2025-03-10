import time

import torch
from evox.algorithms import IBEA
from evox.metrics import igd
from evox.problems.numerical import DTLZ2
from evox.workflows import EvalMonitor, StdWorkflow

# Init the problem, algorithm and workflow.
prob = DTLZ2(m=3)
pf = prob.pf()
algo = IBEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)
workflow.init_step()

# Run the workflow for 100 steps
t = time.time()
for i in range(100):
    workflow.step()
    fit = workflow.algorithm.fit
    fit = fit[~torch.isnan(fit).any(dim=1)]
    if i % 10 == 0:
        print(igd(fit, pf))

