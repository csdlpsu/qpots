import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize

class ModelObject:
    """
    A class to represent multi-objective GP models using independent MLE Gaussian Processes (GPs) 
    for each objective. Each GP is trained using maximum likelihood estimation (MLE) with fixed noise.
    """

    def __init__(self, train_x, train_y, bounds,
                 nobj,
                 ncons,
                 device, 
                 noise_std=1e-6):
        """
        Initialize the multi-objective GP models.

        Parameters:
            train_x (torch.Tensor): Training input data.
            train_y (torch.Tensor): Training output data (multi-objective).
            bounds (torch.Tensor): Bounds for the input data.
            nobj (int): Number of objective functions.
            ncons (int): Number of constraints in the problem.
            device (str): CUDA or CPU device for the arrays to operate on
            noise_std (float): Standard deviation of noise for the GP model. Default is 1e-2.
        """
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.noise_std = noise_std
        self.bounds = bounds
        self.nobj = nobj
        self.ncons= ncons
        self.device = device
        self.models = []
        self.mlls = []

    def fit_gp(self):
        """
        Fit Gaussian Process (GP) models for each objective using maximum likelihood estimation (MLE).
        The method creates K independent GP models, where K is the number of objectives, and fits them 
        using exact marginal log likelihood.

        Returns:
            None
        """
        num_outputs = self.train_y.shape[-1]
        print("fitting GPs", flush=True)
        train_yvar = torch.ones_like(self.train_y[..., 0], dtype=torch.double).to(self.device).reshape(-1, 1) * self.noise_std ** 2

        # Loop through each objective and fit a GP model
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
        Fit Gaussian Process (GP) models for each objective using maximum likelihood estimation (MLE).
        The method creates K independent GP models, where K is the number of objectives, and fits them 
        using exact marginal log likelihood. No variance in the SingleTaskGP.

        Returns:
            None
        """
        num_outputs = self.train_y.shape[-1]
        print("fitting GPs", flush=True)

        # Loop through each objective and fit a GP model
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