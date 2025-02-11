import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from typing import Optional, Callable

class PyMooFunction(Problem):
    '''
    Custom test function for use with PyMoo
    '''

    def __init__(self, func, n_var=2, n_obj=2, xl=0., xu=1.):
        self.count = 1
        self.func = func
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         xl=self.xl,
                         xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_ = torch.tensor(x, dtype=torch.double)
        self.count += 1
        out["F"] = self.func(x_).numpy()

def nsga2(problem: Problem, ngen: int=100, pop_size: int=100, seed: int=2436, callback: Optional[Callable]=None):
    """
    Perform NSGA-II optimization.

    Parameters:
        problem (pymoo.core.Problem): Optimization problem object.
        ngen (int): Number of generations.
        pop_size (int): Population size.
        seed (int): Random seed for optimization.
        callback (Optional[Callable]): Optional callback function.

    Returns:
        res (pymoo.core.result.Result): Optimization result.
    """
    algorithm = NSGA2(pop_size=pop_size)
    if callback:
        res = minimize(problem, algorithm, ('n_gen', ngen), savehistory=True, seed=seed, verbose=False, callback=callback)
    else:
        res = minimize(problem, algorithm, ('n_gen', ngen), savehistory=True, seed=seed, verbose=False)
    return res