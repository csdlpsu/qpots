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
        self.device = device
        self.models = []
        self.mlls = []
        #Testing task_id storage 8/27
        self.task_ids = None
        self.ntrain = None

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
        num_outputs = self.train_y.shape[-1]
        num_inputs = self.train_x.shape[0] 
        print("Fitting MultiTaskGP", flush=True)
        
        #New section 8/27, attempting to deal with partial information, so it does not use every input (train_x) with every objective 
        if self.task_ids is None:
            self.ntrain=num_inputs
            self.task_ids=torch.arange(end=num_outputs).repeat_interleave(num_inputs).reshape(-1,1)
            train_x_mt=torch.cat([self.train_x.repeat(num_outputs,1),self.task_ids],dim=-1).double()
            train_y_mt=standardize(self.train_y).T.reshape(-1, 1).double()
            
        
        else: #Getting initial and new points, where new points are only on one task, initial are on all tasks (should work, tested in ROAR collab)
            
            #print(self.train_y)
            #Train_x
            init_train_x=torch.cat([self.train_x[:self.ntrain].repeat(num_outputs,1),self.task_ids[:2*self.ntrain]],dim=-1).double() #Jointly training on both tasks for the initial training data only
            new_train_x=torch.cat([self.train_x[self.ntrain:],self.task_ids[2*self.ntrain:]],dim=-1).double() #Getting the new train_x with its task_id
            train_x_mt=torch.cat([init_train_x,new_train_x]) #Stacking the init data and the new data

            #print("train_x:\n",self.train_x)
            #print("train_x_mt:\n",train_x_mt)

            #8/29 - Here I am assuming that the train_y is being entered with NaN values found in the tasks that are not being used
            #May need to question this assumption later, and have qPOTS deal with the NaN instead of the user, or maybe this will work with that too, but as of now it is successful
            #Relies on the standardize_ignore_nan in utils

            #Train_y
            standardized_train_y=self.standardize_ignore_nan(self.train_y) #standardizing first to sort it out
            train_y_mt=standardized_train_y[:self.ntrain].T.reshape(-1, 1).double() 
            #"""
            for val in self.train_y[self.ntrain:].reshape(-1,1):
                if not torch.isnan(val):
                    train_y_mt=torch.cat([train_y_mt,val.reshape(-1,1)])
            """
            # Append only non-NaN new entries
            mask = ~torch.isnan(self.train_y[self.ntrain:])  # shape [4, 2]
            new_train_y = self.train_y[self.ntrain:][mask].reshape(-1, 1).double()
            """
            #train_y_mt = torch.cat([train_y_mt, new_train_y])
            #print(train_y_mt)
            #print("train_x_mt shape:", train_x_mt.shape)   # should be [44, d+1]
            #print("train_y_mt shape:", train_y_mt.shape)   # should be [44, 1]

            #8/29 - Here I am assuming that the train_y is being entered with NaN values found in the tasks that are not being used
            #May need to question this assumption later, and have qPOTS deal with the NaN instead of the user, or maybe this will work with that too, but as of now it is successful
            #Relies on the standardize_ignore_nan in utils

            #Train_y
            standardized_train_y=self.standardize_ignore_nan(self.train_y) #standardizing first to sort it out
            train_y_mt=standardized_train_y[:self.ntrain].T.reshape(-1, 1).double() 
    
            for val in self.train_y[self.ntrain:].reshape(-1,1):
                if not torch.isnan(val):
                    train_y_mt=torch.cat([train_y_mt,val.reshape(-1,1)])
            
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
