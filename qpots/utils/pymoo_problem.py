import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from typing import Optional, Callable


class PyMooFunction(Problem):
    """
    Custom multi-objective test function for use with PyMoo.

    This class allows the integration of a PyTorch-based function into the PyMoo framework,
    enabling multi-objective optimization within the PyMoo environment.

    Parameters
    ----------
    func : Callable
        The function to be optimized. It should accept a PyTorch tensor and return a tensor.
    n_var : int, optional
        Number of input variables (default is 2).
    n_obj : int, optional
        Number of objective functions (default is 2).
    xl : float or array-like, optional
        Lower bound(s) for the input variables (default is 0.0).
    xu : float or array-like, optional
        Upper bound(s) for the input variables (default is 1.0).
    """

    def __init__(self, func: Callable, n_var: int = 2, n_obj: int = 2, xl=0.0, xu=1.0):
        self.count = 1
        self.func = func
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the function on a given set of input points.

        This method converts input data from NumPy arrays to PyTorch tensors, 
        computes function values, and stores the results.

        Parameters
        ----------
        x : numpy.ndarray
            The input variables as a NumPy array of shape `(n_samples, n_var)`.
        out : dict
            A dictionary where the function outputs are stored under the key `"F"`.
        *args, **kwargs
            Additional arguments for compatibility with PyMoo.

        Returns
        -------
        None
            Updates the `out` dictionary in-place with computed function values.
        """
        x_ = torch.tensor(x, dtype=torch.double)
        self.count += 1
        out["F"] = self.func(x_).numpy()


def nsga2(
    problem: Problem,
    ngen: int = 100,
    pop_size: int = 100,
    seed: int = 2436,
    callback: Optional[Callable] = None,
):
    """
    Perform NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimization.

    This function runs the NSGA-II algorithm on a given multi-objective optimization problem.

    Parameters
    ----------
    problem : pymoo.core.Problem
        The optimization problem to be solved.
    ngen : int, optional
        The number of generations to run the optimization (default is 100).
    pop_size : int, optional
        The size of the population for NSGA-II (default is 100).
    seed : int, optional
        Random seed for reproducibility (default is 2436).
    callback : Callable, optional
        An optional callback function to monitor the optimization process.

    Returns
    -------
    pymoo.core.result.Result
        The result of the optimization, containing the Pareto front and other relevant information.
    """
    algorithm = NSGA2(pop_size=pop_size)

    if callback:
        res = minimize(
            problem, algorithm, ("n_gen", ngen), savehistory=True, seed=seed, verbose=False, callback=callback
        )
    else:
        res = minimize(problem, algorithm, ("n_gen", ngen), savehistory=True, seed=seed, verbose=False)

    return res
