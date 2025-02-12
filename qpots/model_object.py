import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize


class ModelObject:
    """
    A class representing multi-objective Gaussian Process (GP) models.

    This class constructs and fits independent Gaussian Process models for each objective 
    using Maximum Likelihood Estimation (MLE). The models are used in multi-objective 
    optimization problems where constraints can be included.
    """

    def __init__(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor, 
        bounds: torch.Tensor, 
        nobj: int, 
        ncons: int, 
        device: str, 
        noise_std: float = 1e-6
    ):
        """
        Initialize the multi-objective GP models.

        Parameters
        ----------
        train_x : torch.Tensor
            The input training data of shape `(n, d)`, where `n` is the number of samples 
            and `d` is the input dimension.
        train_y : torch.Tensor
            The output training data of shape `(n, k)`, where `k` is the number of objectives.
        bounds : torch.Tensor
            A tensor specifying the lower and upper bounds for the input space.
        nobj : int
            The number of objective functions.
        ncons : int
            The number of constraints in the problem.
        device : str
            The computation device, either `"cpu"` or `"cuda"`.
        noise_std : float, optional
            The standard deviation of noise added to the GP model. Defaults to `1e-6`.
        """
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.noise_std = noise_std
        self.bounds = bounds
        self.nobj = nobj
        self.ncons = ncons
        self.device = device
        self.models = []
        self.mlls = []

    def fit_gp(self):
        """
        Fit Gaussian Process (GP) models using Maximum Likelihood Estimation (MLE).

        This method fits `nobj` independent GP models, each corresponding to an objective function.
        The models are trained using exact marginal log likelihood.

        Returns
        -------
        None
        """
        num_outputs = self.train_y.shape[-1]
        print("Fitting GPs", flush=True)
        train_yvar = torch.ones_like(self.train_y[..., 0], dtype=torch.double).to(self.device).reshape(-1, 1) * self.noise_std ** 2

        # Fit a GP model for each objective
        for i in range(num_outputs):
            print(f"Fit: {i}", flush=True)
            model = SingleTaskGP(
                self.train_x,
                standardize(self.train_y[..., i]).reshape(-1, 1).double(),
                train_yvar
            ).to(self.train_x.device)

            self.models.append(model)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            self.mlls.append(mll)

            fit_gpytorch_mll(mll)

    def fit_gp_no_variance(self):
        """
        Fit Gaussian Process (GP) models without variance estimation.

        This method is similar to `fit_gp()`, but does not include variance in the GP model.
        It fits `nobj` independent GP models using Maximum Likelihood Estimation (MLE).

        Returns
        -------
        None
        """
        num_outputs = self.train_y.shape[-1]
        print("Fitting GPs", flush=True)

        # Fit a GP model for each objective without variance
        for i in range(num_outputs):
            print(f"Fit: {i}", flush=True)
            model = SingleTaskGP(
                self.train_x,
                standardize(self.train_y[..., i]).reshape(-1, 1).double(),
            ).to(self.train_x.device)

            self.models.append(model)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            self.mlls.append(mll)

            fit_gpytorch_mll(mll)
