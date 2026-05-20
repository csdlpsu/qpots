from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ7, GMM, DH1, DH2, DH3, DH4, Penicillin,
    VehicleSafety, CarSideImpact, ConstrainedBraninCurrin,
    ZDT1,ZDT2,ZDT3, DiscBrake, MW7, OSY, WeldedBeam, C2DTLZ2,ToyRobust,BNH,SRN,CONSTR
)
#from examples.Fall_25_custom_functions import MultiFidelityCurrin, MultiFidelityForrester #Extra Multi-Fidelity Test functions 9/29
from botorch.test_functions.synthetic import Branin
from torch import Tensor
from typing import Callable, Optional
import torch
from qpots.config import get_device, get_dtype, to_runtime


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
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
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
        device : torch.device or str, optional
            Device used for generated bounds and evaluations. Defaults to the
            qPOTS runtime device.
        dtype : torch.dtype, optional
            Floating-point precision used for generated bounds and evaluations.
            Defaults to ``qpots.config.DEFAULT_DTYPE``.

        Raises
        ------
        ValueError
            If a custom function is provided but `bounds` is not specified.
            If an unknown function name is provided.
        """
        self.name = name.lower() if name else None
        self.dim = dim
        self.nobj = nobj
        self.device = get_device(device)
        self.dtype = get_dtype(dtype)
        self.custom_func = custom_func
        self.bounds = to_runtime(bounds, self.device, self.dtype) if torch.is_tensor(bounds) else bounds
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
            "dtlz1": lambda: DTLZ1(dim=self.dim, num_objectives=self.nobj, negate=True),
            "dtlz2": lambda: DTLZ2(dim=self.dim, num_objectives=self.nobj, negate=True),
            "c2dtlz2": lambda: C2DTLZ2(dim=self.dim, num_objectives=self.nobj, negate=True),
            "dtlz3": lambda: DTLZ3(dim=self.dim, num_objectives=self.nobj, negate=False),
            "dtlz4": lambda: DTLZ4(dim=self.dim, num_objectives=self.nobj, negate=False),
            "dtlz5": lambda: DTLZ5(dim=self.dim, num_objectives=self.nobj, negate=False),
            "dtlz7": lambda: DTLZ7(dim=self.dim, num_objectives=self.nobj, negate=True),
            "dh1": lambda: DH1(dim=self.dim, negate=True),
            "dh2": lambda: DH2(dim=self.dim, negate=True),
            "dh3": lambda: DH3(dim=self.dim, negate=True),
            "dh4": lambda: DH4(dim=self.dim, negate=True),
            "gmm": lambda: GMM(num_objectives=self.nobj, negate=True),
            "penicillin": lambda: Penicillin(negate=True),
            "vehicle": lambda: VehicleSafety(negate=True),
            "carside": lambda: CarSideImpact(negate=True),
            "zdt3": lambda: ZDT3(dim=self.dim, num_objectives=self.nobj, negate=True),
            "zdt2": lambda: ZDT2(dim=self.dim, num_objectives=self.nobj, negate=False),
            "zdt1": lambda: ZDT1(dim=self.dim, num_objectives=self.nobj, negate=False),
            "constrainedbc": lambda: ConstrainedBraninCurrin(negate=True),
            "discbrake": lambda: DiscBrake(negate=True),
            "mw7": lambda: MW7(dim=self.dim, negate=True),
            "osy": lambda: OSY(negate=True),
            "weldedbeam": lambda: WeldedBeam(negate=True),
            "branin": lambda: Branin(negate=True),
            "toyrobust": lambda: ToyRobust(negate=False),
            "srn": lambda: SRN(negate=False),
            "bnh": lambda: BNH(negate=False),
            "constr": lambda: CONSTR(negate=True),
            "mfcurrin": lambda: MultiFidelityCurrin(negate=False),
            "mfforrester": lambda: MultiFidelityForrester(negate=True),
        }

        if self.name not in func_map:
            raise ValueError(f"Unknown test function '{self.name}'. Check the available functions.")

        # Initialize function, bounds, and constraints
        self.func = func_map[self.name]()
        if hasattr(self.func, "to"):
            self.func = self.func.to(device=self.device, dtype=self.dtype)
        self.bounds = self.func.bounds.to(device=self.device, dtype=self.dtype)
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self._evaluate_constraints

    def _evaluate_constraints(self, X: Tensor) -> Tensor:
        """Evaluate BoTorch constraint slack values on the runtime device/dtype."""
        X = X.to(device=self.device, dtype=self.dtype)
        try:
            result = self.func.evaluate_slack(X)
        except ValueError as err:
            if "within the bounds" not in str(err) or not torch.is_tensor(self.bounds):
                raise
            X_eval = self.bounds[0] + X * (self.bounds[1] - self.bounds[0])
            result = self.func.evaluate_slack(X_eval)
        return result.to(device=self.device, dtype=self.dtype)

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
        X = X.to(device=self.device, dtype=self.dtype)
        if self.custom_func:
            return self.custom_func(X).to(device=self.device, dtype=self.dtype)
        try:
            result = self.func(X)
        except ValueError as err:
            if "within the bounds" not in str(err) or not hasattr(self.func, "evaluate_true"):
                raise
            if torch.is_tensor(self.bounds):
                X_eval = self.bounds[0] + X * (self.bounds[1] - self.bounds[0])
                result = self.func(X_eval)
            else:
                raise
        return result.to(device=self.device, dtype=self.dtype)

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
