from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ1, DTLZ2, DTLZ3, DTLZ7, GMM, DH1, Penicillin,
    VehicleSafety, CarSideImpact, ConstrainedBraninCurrin,
    ZDT3, DiscBrake, MW7, OSY, WeldedBeam
)
from torch import Tensor
from typing import Callable, Optional, Union

class Function:
    """
    Interface for multi-objective test functions.
    Provides an easy-to-use abstraction over BoTorch test functions and supports user-defined functions.
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
        Args:
            name (str, optional): Name of the predefined test function (case-insensitive).
            dim (int): Dimensionality of the input space.
            nobj (int): Number of objectives for the test function.
            custom_func (Callable, optional): User-defined function for optimization.
            bounds (Tensor, optional): Bounds for the custom function.
            cons (Callable, optional): Constraint function for the custom function.
        """
        self.name = name.lower() if name else None
        self.dim = dim
        self.nobj = nobj
        self.custom_func = custom_func
        self.bounds = bounds
        self.cons = cons

        if self.custom_func:
            # Validate custom function setup
            if not self.bounds:
                raise ValueError("Custom functions must specify bounds.")
        else:
            # Initialize predefined test functions
            self._initialize_function()

    def _initialize_function(self):
        # Mapping between function names and BoTorch test functions
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
            raise ValueError(f"Unknown test function '{self.name}'. Check the name or documentation.")

        # Initialize the function, bounds, and constraints
        self.func = func_map[self.name]()
        self.bounds = self.func.bounds.double()
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self.func.evaluate_slack

    def evaluate(self, X: Tensor) -> Tensor:
        """
        Evaluate the test function or custom function on input X.

        Args:
            X (Tensor): Input tensor of shape (n, dim).

        Returns:
            Tensor: Outputs from the function.
        """
        if self.custom_func:
            return self.custom_func(X)
        return self.func(X)

    def get_bounds(self) -> Tensor:
        """
        Retrieve the bounds for the function.

        Returns:
            Tensor: Lower and upper bounds as a tensor.
        """
        return self.bounds

    def get_cons(self) -> Optional[Callable]:
        """
        Retrieve the constraints for the function.

        Returns:
            Callable or None: Constraint function if available.
        """
        return self.cons
  