from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ1, DTLZ2, DTLZ3, DTLZ7, GMM, DH1, Penicillin,
    VehicleSafety, CarSideImpact, ConstrainedBraninCurrin,
    ZDT3, DiscBrake, MW7, OSY, WeldedBeam
)
from torch import Tensor
from typing import Callable, Optional


class Function:
    """
    Interface for multi-objective test functions.

    This class provides an abstraction over BoTorch test functions and allows for 
    user-defined objective functions. It supports retrieving function bounds and 
    constraints when available.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        dim: int = 2,
        nobj: int = 2,
        custom_func: Optional[Callable[[Tensor], Tensor]] = None,
        bounds: Optional[Tensor] = None,
        cons: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        Initialize a test function for multi-objective optimization.

        Parameters
        ----------
        name : str, optional
            Name of the predefined test function (case-insensitive). 
            If None, a custom function must be provided.
        dim : int
            Dimensionality of the input space. Defaults to 2.
        nobj : int
            Number of objectives for the test function. Defaults to 2.
        custom_func : Callable, optional
            A user-defined function that takes a tensor `X` as input and 
            returns an output tensor. If provided, `name` is ignored.
        bounds : Tensor, optional
            A tensor specifying the lower and upper bounds for the function.
            Required if using a custom function.
        cons : Callable, optional
            A constraint function that maps inputs to constraint values.

        Raises
        ------
        ValueError
            If a custom function is provided but `bounds` is not specified.
            If an unknown function name is provided.
        """
        self.name = name.lower() if name else None
        self.dim = dim
        self.nobj = nobj
        self.custom_func = custom_func
        self.bounds = bounds
        self.cons = cons

        if self.custom_func:
            if self.bounds is None:
                raise ValueError("Custom functions must specify bounds.")
        else:
            self._initialize_function()

    def _initialize_function(self):
        """
        Initialize a predefined BoTorch test function.

        This method sets up the corresponding function, bounds, and constraints 
        based on the selected function name.

        Raises
        ------
        ValueError
            If the specified function name is not recognized.
        """
        func_map = {
            "branincurrin": lambda: BraninCurrin(negate=True),
            "dtlz1": lambda: DTLZ1(self.dim, num_objectives=self.nobj, negate=True),
            "dtlz2": lambda: DTLZ2(self.dim, num_objectives=self.nobj, negate=True),
            "dtlz3": lambda: DTLZ3(self.dim, num_objectives=self.nobj),
            "dtlz7": lambda: DTLZ7(self.dim, num_objectives=self.nobj, negate=True),
            "dh1": lambda: DH1(self.dim, negate=True),
            "gmm": lambda: GMM(self.nobj, negate=True),
            "penicillin": lambda: Penicillin(negate=True),
            "vehicle": lambda: VehicleSafety(negate=True),
            "carside": lambda: CarSideImpact(negate=True),
            "zdt3": lambda: ZDT3(dim=self.dim, num_objectives=self.nobj, negate=True),
            "constrainedbc": lambda: ConstrainedBraninCurrin(negate=True),
            "discbrake": lambda: DiscBrake(),
            "mw7": lambda: MW7(dim=self.dim, negate=True),
            "osy": lambda: OSY(negate=True),
            "weldedbeam": lambda: WeldedBeam(negate=True),
        }

        if self.name not in func_map:
            raise ValueError(f"Unknown test function '{self.name}'. Check the available functions.")

        # Initialize function, bounds, and constraints
        self.func = func_map[self.name]()
        self.bounds = self.func.bounds.double()
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self.func.evaluate_slack

    def evaluate(self, X: Tensor) -> Tensor:
        """
        Evaluate the test function or custom function on input `X`.

        Parameters
        ----------
        X : Tensor
            A tensor of shape `(n, dim)`, where `n` is the number of points and `dim` is the input dimension.

        Returns
        -------
        Tensor
            A tensor of shape `(n, nobj)` containing the function outputs.
        """
        if self.custom_func:
            return self.custom_func(X)
        return self.func(X)

    def get_bounds(self) -> Tensor:
        """
        Retrieve the bounds for the function.

        Returns
        -------
        Tensor
            A tensor containing the lower and upper bounds for each input dimension.
        """
        return self.bounds

    def get_cons(self) -> Optional[Callable]:
        """
        Retrieve the constraint function for the test function.

        Returns
        -------
        Callable or None
            The constraint function if available; otherwise, None.
        """
        return self.cons
