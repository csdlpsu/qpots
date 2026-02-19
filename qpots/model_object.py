import torch
from botorch.models import SingleTaskGP
from botorch.models import MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
from gpytorch.priors import GammaPrior


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
        ntrain: int,
        device: str, 
        noise_std: float = 1e-6,
        
        
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
        device : torch.device
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
        self.ntrain = ntrain
        self.device = device
        self.models = []
        self.mlls = []
        self.prev_state_dict = None
        
        

    def fit_gp(self, single_objective=False):
        """
        Fit Gaussian Process (GP) models using Maximum Likelihood Estimation (MLE).

        This method fits `nobj` independent GP models, each corresponding to an objective function.
        The models are trained using exact marginal log likelihood.

        Parameters
        ----------
        single_objective : bool
            If True, fit just one GP otherwise fit GP for each objective

        Returns
        -------
        None
        """
        num_outputs = self.train_y.shape[-1]
        print("Fitting GPs", flush=True)
        train_yvar = torch.ones_like(self.train_y[..., 0], dtype=torch.double).to(self.device).reshape(-1, 1) * self.noise_std ** 2

        # Fit a GP model for each objective
        
        if single_objective == True:
            print("fitting single objective")
            model = SingleTaskGP(
                self.train_x,
                standardize(self.train_y[..., 1]).reshape(-1, 1).double(),
            ).to(self.train_x.device)

            for i in range(2):
                self.models.append(model)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                self.mlls.append(mll)

                fit_gpytorch_mll(mll)
        else: 
            for i in range(num_outputs):
                self.ntrain=self.train_x.shape[0] #Setting number of training points
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
    
    def fit_multitask_gp(self):
        """
        Fit a MultiTask Gaussian Process (GP) model for objectives and constraints.

        This method constructs and fits a single `MultiTaskGP` model that jointly models
        all objectives and constraints. Missing (NaN) target values are ignored during
        training, and each input is augmented with a task index.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        print("Fitting MultiTaskGP", flush=True)
        
        num_inputs, dim = self.train_x.shape
        
        #Initial training data:
        x_init = self.train_x[:self.ntrain].unsqueeze(1).expand(-1, self.nobj+self.ncons, -1).reshape(-1, dim)
        #train_y_mt = self.standardize_ignore_nan(self.train_y)[:self.ntrain].reshape(-1,1)
        train_y_std=self.standardize_ignore_nan(self.train_y)
        #print("train_y_std:\n",train_y_std)
        train_y_mt = train_y_std[:self.ntrain].reshape(-1,1)
      
        
        task_ids_init = torch.arange(self.nobj+self.ncons).expand(self.ntrain,self.nobj+self.ncons).reshape(-1,1)
        train_x_mt = torch.cat([x_init,task_ids_init],dim=-1)
    
        
        #Additional training data:
        if num_inputs > self.ntrain:
            new_x=self.train_x[self.ntrain:]
            #new_y=self.standardize_ignore_nan(self.train_y)[self.ntrain:]
            new_y=train_y_std[self.ntrain:]
            nan_mask = ~torch.isnan(new_y)
            rows, tasks = nan_mask.nonzero(as_tuple=True) 
            
            if rows.numel() > 0: 
                new_x = new_x[rows]
                new_task_ids = tasks.unsqueeze(1)
                new_x_mt = torch.cat([new_x,new_task_ids],dim=-1)
                train_x_mt=torch.cat([train_x_mt,new_x_mt],dim=0)
                train_y_mt=torch.cat([train_y_mt,new_y[rows, tasks].reshape(-1,1)])
                
                #print("Past training")
                #print("train_x_mt:\n",train_x_mt)
                #print("train_y_mt:\n",train_y_mt)
                
        custom_kernel = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=self.train_x.shape[-1],
                        lengthscale_prior=GammaPrior(2.0, 2.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ) #New Matern 5/2 Kernel

        model = MultiTaskGP(
            train_x_mt,
            train_y_mt,
            task_feature=-1,
            outcome_transform=None, #Using None instead of standardize 2/18
            rank=1,#Added Rank=1 on 1/14
            covar_module=custom_kernel,
        ).to(self.train_x.device)
        
        self.models.append(model)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        self.mlls.append(mll)
        
        try:
            fit_gpytorch_mll(mll)
            self.prev_state_dict = model.state_dict()
        except ModelFittingError:
            print("WARNING: GP fitting failed. Restoring previous hyperparameters.")
            model.load_state_dict(self.prev_state_dict)
            


    def fit_gp_no_variance(self, single_objective=False):
        """
        Fit Gaussian Process (GP) models without variance estimation.

        This method is similar to `fit_gp()`, but does not include variance in the GP model.
        It fits `nobj` independent GP models using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        single_objective : bool
            If True, fit just one GP otherwise fit GP for each objective

        Returns
        -------
        None
        """
        num_outputs = self.train_y.shape[-1]
        print("Fitting GPs", flush=True)

        # Fit a GP model for each objective without variance
        if single_objective:
            model = SingleTaskGP(
                self.train_x,
                standardize(self.train_y[..., i]).reshape(-1, 1).double(),
            ).to(self.train_x.device)

            for i in range(2):
                self.models.append(model)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                self.mlls.append(mll)

                fit_gpytorch_mll(mll)
        else:
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

    def standardize_ignore_nan(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Standardize Y along dim=0 ignoring NaNs.
        NaNs remain in place.
        Parameters
        ----------
        Y : torch.Tensor
            Input tensor to be standardized.

        Returns
        -------
        torch.Tensor
            Standardized tensor with NaN values preserved.
        """
        mean = torch.nanmean(Y, dim=0, keepdim=True)

        diff = Y - mean
        diff_squared = diff ** 2
        diff_squared = torch.where(torch.isnan(diff_squared), torch.zeros_like(diff_squared), diff_squared)
        count = (~torch.isnan(Y)).sum(dim=0, keepdim=True)
        std = torch.sqrt(diff_squared.sum(dim=0, keepdim=True) / (count - 1))
        std = torch.where(std == 0, torch.ones_like(std), std) 

        Y_std = (Y - mean) / std
        return torch.where(torch.isnan(Y), torch.tensor(float('nan'), device=Y.device), Y_std)
    

