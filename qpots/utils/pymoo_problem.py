import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

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

def nsga2(problem, ngen=100, pop_size=100, seed=2436, callback=None):
    """
    Perform NSGA-II optimization.

    Parameters:
        problem: Optimization problem object.
        ngen: Number of generations.
        pop_size: Population size.
        seed: Random seed for optimization.

    Returns:
        Optimization result.
    """
    algorithm = NSGA2(pop_size=pop_size)
    if callback:
        res = minimize(problem, algorithm, ('n_gen', ngen), savehistory=True, seed=seed, verbose=False, callback=callback)
    else:
        res = minimize(problem, algorithm, ('n_gen', ngen), savehistory=True, seed=seed, verbose=False)
    return res