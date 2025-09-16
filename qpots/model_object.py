import torch
from botorch.models import SingleTaskGP
from botorch.models import MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize
from gpytorch.kernels import ScaleKernel, MaternKernel


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
        print("Fitting MultiTaskGP", flush=True)
        
        #9/3 - Updating the train_x and y adjustments for MTGP
        num_inputs, dim = self.train_x.shape
        
        #Initial training data :
        x_init = self.train_x[:self.ntrain].unsqueeze(1).expand(-1, self.nobj+self.ncons, -1).reshape(-1, dim)
        train_y_mt = self.standardize_ignore_nan(self.train_y)[:self.ntrain].reshape(-1,1)
        
        task_ids_init = torch.arange(self.nobj+self.ncons).expand(self.ntrain,self.nobj+self.ncons).reshape(-1,1)
        train_x_mt = torch.cat([x_init,task_ids_init],dim=-1)

        #Additional training data:
        if num_inputs > self.ntrain:
            new_x=self.train_x[self.ntrain:]
            #print("Fitting new_x post training",new_x)
            new_y=self.standardize_ignore_nan(self.train_y)[self.ntrain:]
            #print("Fitting new_y post training (standardized)",new_y)
            nan_mask = ~torch.isnan(new_y)
            #print("Fitting nan_mask ",nan_mask)
            rows, tasks = nan_mask.nonzero(as_tuple=True) 
            #print("Fitting rows and tasks",rows,tasks)
            if rows.numel() > 0: #Just in case there are no new values (might be able to delete because I am filtering empty rows eleswhere)
                new_x = new_x[rows]
                #print("Fitting new_x post new_x[rows]",new_x)
                new_task_ids = tasks.unsqueeze(1)
                #print("Chosen Task IDs in model_object fit_gp:\n",new_task_ids)
                new_x_mt = torch.cat([new_x,new_task_ids],dim=-1)
                train_x_mt=torch.cat([train_x_mt,new_x_mt],dim=0)

                train_y_mt=torch.cat([train_y_mt,new_y[rows, tasks].reshape(-1,1)])

        # 9/8 Testing for constraint handling
        """
        print(">> Entering fit_multitask_gp")
        print("train_x_mt shape:", train_x_mt.shape)
        print("train_y_mt shape:", train_y_mt.shape)
        print("train_x_mt example:",train_x_mt[:6])
        print("train_y example:", standardize(self.train_y)[:6])
        print("train_y_mt example:", train_y_mt[:6])
        """

        #Testing 8/25 Using Matern 5/2 Kernel 
        custom_kernel = ScaleKernel(MaternKernel(nu=2.5))

        model = MultiTaskGP(
            train_x_mt,
            train_y_mt,
            task_feature=-1,
            covar_module=custom_kernel
        ).to(self.train_x.device)
        
        self.models.append(model)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        self.mlls.append(mll)
        fit_gpytorch_mll(mll)
        print("Fitting successful")

        self.counter(train_x_mt)
        
    def counter(self,train_x_mt):
        objective_indices = train_x_mt[:, -1].long()
        #print("iteration added Task IDs in model_object:\n",train_x_mt[40:, -1])
        total_evals_per_objective = torch.zeros(self.nobj+self.ncons, dtype=torch.int64)
        
        for obj in range(self.nobj+self.ncons):
            total_evals_per_objective[obj] = (objective_indices == obj).sum()

        print("Total evaluations per objective:", total_evals_per_objective)
        #return total_evals_per_objective


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

    #Added 8/29 for partial information with train_y
    def standardize_ignore_nan(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Standardize Y along dim=0 ignoring NaNs.
        NaNs remain in place.
        Returns
        -------
        standardized Y tensor
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
    

