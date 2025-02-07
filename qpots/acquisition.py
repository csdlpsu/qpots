import torch
from typing import Optional, Callable, Tuple
from torch import Tensor
from scipy.spatial.distance import cdist
from botorch.utils.transforms import normalize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition,
)
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from qpots.utils.utils import unstandardize, select_candidates
from qpots.utils.pymoo_problem import PyMooFunction, nsga2
from qpots.function import Function
from qpots.tsemo_runner import TSEMORunner
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


class Acquisition:
    """
    A class providing various acquisition functions and methods for multi-objective optimization.
    """

    def __init__(
        self,
        func: Function,
        gps: ModelListGP,
        cons: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
        q: int = 1,
        NUM_RESTARTS: int = 10,
        RAW_SAMPLES: int = 512
    ) -> None:
        """
        Initialize the multi-objective acquisition class.

        Parameters:
            func (Callable): The test function being optimized.
            gps (ModelListGP): Gaussian Process models.
            cons (Optional[Callable]): A vector-valued function of inequality constraints.
            device (torch.device): Device for computation (CPU/GPU).
            q (int, optional): Number of points to sample per iteration. Defaults to 1.
            NUM_RESTARTS (int, optional): Number of restarts for acquisition optimization.
            RAW_SAMPLES (int, optional): Number of raw samples for acquisition optimization.
        """
        self.func = func
        self.gps = gps
        self.cons = cons
        self.device = device
        self.q = q
        self.NUM_RESTARTS = NUM_RESTARTS
        self.RAW_SAMPLES = RAW_SAMPLES
        self.nobj = gps.nobj
        self.ncons = gps.ncons

    def nystrom_approx(
        self,
        x: Tensor,
        gps: ModelListGP,
        m: int,
        pareto_set: Optional[Tensor] = None,
        col_choice: str = "pareto",
        seed_iter: int = 1,
        reg: float = 1e-6,
    ) -> Tensor:
        """
        Perform Thompson sampling using the Nystrom approximation for GP models.

        Parameters:
            x (Tensor): Input points.
            gps (ModelListGP): A list of GP models.
            m (int): Number of landmarks for the Nystrom approximation.
            pareto_set (Optional[Tensor]): Pareto optimal points. Required if `col_choice` is 'pareto'.
            col_choice (str, optional): Column selection method ('pareto' or 'random'). Defaults to 'pareto'.
            seed_iter (int, optional): Seed iteration for randomness. Defaults to 1.
            reg (float, optional): Regularization constant for matrix inversion. Defaults to 1e-6.

        Returns:
            Tensor: Posterior samples from the GP models.
        """
        torch.manual_seed(1024 + seed_iter)
        samples_list = []

        for gp_model in gps.models:
            posterior = gp_model.posterior(normalize(x.to(self.device), gps.bounds.to(self.device)))
            mean = posterior.mean.to(self.device)
            covariance = posterior.mvn.covariance_matrix.to(self.device)

            if col_choice == "random":
                indices = torch.randperm(covariance.shape[-1])[:m]
            elif col_choice == "pareto":
                if pareto_set is None:
                    raise ValueError("Pareto set is required for 'pareto' column choice.")
                D = cdist(pareto_set.cpu(), x.cpu().numpy())
                cand_indices = D.argmin(axis=-1)
                indices = torch.unique(torch.tensor(cand_indices).argsort()[:m]).to(self.device)
                if len(indices) < m:
                    extra_indices = torch.randperm(covariance.shape[-1])[:m - len(indices)].to(self.device)
                    indices = torch.cat((indices, extra_indices)).to(self.device)
            else:
                raise NotImplementedError(f"Column choice '{col_choice}' is not implemented.")

            K_nm = covariance[:, indices]
            apprx_covariance = covariance[indices][:, indices]
            while True:
                try:
                    L_mm_cov = torch.linalg.cholesky(
                        apprx_covariance + reg * torch.eye(m).to(self.device)
                    ).to(self.device)
                    break
                except torch.linalg.LinAlgError:
                    reg *= 10

            z = torch.rand(m, dtype=torch.float64).to(self.device)
            sample = mean + K_nm @ (L_mm_cov @ z).reshape(-1, 1)
            samples_list.append(sample.detach())

        Ys_ = unstandardize(torch.cat(samples_list, -1), gps.train_y.to(self.device))

        if self.ncons > 0:
            ind_feasible = Ys_[..., -self.ncons :] <= 0
            Ys_[~ind_feasible.squeeze(), : self.nobj] = -1e12  # Arbitrary low value for infeasible points
            Ys = Ys_[..., : self.nobj]
        else:
            Ys = Ys_

        return -Ys

    def gp_posterior(self, x: Tensor, gps: ModelListGP, seed_iter: int = 1) -> Tensor:
        """
        Compute posterior samples for PyMoo optimization.

        Parameters:
            x (Tensor): Input points.
            gps (ModelListGP): Gaussian Process models.
            seed_iter (int, optional): Iteration index for seeding randomness. Defaults to 1.

        Returns:
            Tensor: Posterior samples.
        """
        torch.manual_seed(1024 + seed_iter)
        Ys_ = [
            model.posterior(normalize(x, gps.bounds)).sample().reshape(-1, 1) for model in gps.models
        ]
        Ys_ = unstandardize(torch.cat(Ys_, -1), gps.train_y.to(self.device))

        if self.ncons > 0:
            ind_feasible = (Ys_[..., -self.ncons :] >= 0).all(dim=-1)
            Ys_[~ind_feasible.squeeze(), : self.nobj] = -1e12
            Ys = Ys_[..., : self.nobj]
        else:
            Ys = Ys_

        return -Ys

    
    def qpots(
        self,
        bounds: Tensor,
        iteration: int,
        **kwargs,
    ) -> Tensor:
        """
        Perform Pareto Optimal Thompson Sampling (QPOTS).

        Parameters:
            bounds (Tensor): The bounds for the optimization problem.
            iteration (int): Current iteration index for seeding randomness.
            **kwargs: Additional arguments for customization.

        Returns:
            Tensor: Selected candidates.
        """
        def track_pareto(res):
            pareto_set[0] = res.opt.get("X")

        pareto_set = [None]

        if kwargs["nystrom"] == 1:
            gp_posterior_approx_ = lambda x: self.nystrom_approx(
                x.to(self.device),
                self.gps,
                int(0.2 * kwargs["iters"]),
                normalize(torch.tensor(pareto_set[0]).to(self.device), bounds).cpu().numpy()
                if pareto_set[0] is not None
                else torch.zeros_like(x),
                col_choice=kwargs["nychoice"],
            )
            pymoo_func_gp = PyMooFunction(
                gp_posterior_approx_,
                n_var=kwargs["dim"],
                n_obj=self.nobj,
                xl=bounds[0].detach().cpu().numpy(),
                xu=bounds[1].detach().cpu().numpy(),
            )
            res = nsga2(
                pymoo_func_gp,
                ngen=kwargs["ngen"],
                pop_size=100 * kwargs["dim"],
                seed=2430,
                callback=track_pareto,
            )
        else:
            gp_posterior_ = lambda x: self.gp_posterior(
                x.to(self.device), self.gps, seed_iter=iteration
            )
            pymoo_func_gp = PyMooFunction(
                gp_posterior_,
                n_var=kwargs["dim"],
                n_obj=self.nobj,
                xl=bounds[0].detach().cpu().numpy(),
                xu=bounds[1].detach().cpu().numpy(),
            )
            res = nsga2(
                pymoo_func_gp,
                ngen=kwargs["ngen"],
                pop_size=100 * kwargs["dim"],
                seed=2430,
            )
        
        selected_candidates = select_candidates(
            self.gps, res.X, self.device, q=kwargs["q"], seed=2043
        )
        
        return normalize(selected_candidates, bounds)

    def qlogei(self, ref_point: Tensor = torch.tensor([0.0, 0.0])) -> Tensor:
        """
        Optimize the qLogEI acquisition function and return new candidates.

        Parameters:
            ref_point (Tensor, optional): Reference point for hypervolume calculation. Defaults to [0.0, 0.0].

        Returns:
            Tensor: New candidate points based on qLogEI optimization.
        """
        standard_bounds = torch.zeros(2, self.func.dim, device=self.device)
        standard_bounds[1] = 1
        train_y = self.gps.train_y.to(self.device)
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point.to(self.device), Y=train_y[..., : self.nobj]
        )
        model = ModelListGP(*self.gps.models).to(self.device)
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.to(self.device),
            partitioning=partitioning,
            constraints=[lambda Z: Z[..., -self.gps.ncons]],
        )

        while True:
            try:
                new_x, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=standard_bounds,
                    q=self.q,
                    num_restarts=self.NUM_RESTARTS,
                    raw_samples=self.RAW_SAMPLES,
                    options={"batch_limit": 3, "maxiter": 1000},
                    sequential=True,
                )
                break
            except RuntimeError:
                pass
        return new_x.to(self.device)

    def parego(self) -> Tensor:
        """
        Perform qParEGO optimization using random weights for scalarization.

        Returns:
            Tensor: New candidate points based on qParEGO optimization.
        """
        standard_bounds = torch.tensor(
            [[0.0] * self.func.dim, [1.0] * self.func.dim],
            dtype=torch.double,
        ).to(self.device) # normalized bounds
        train_x = self.gps.train_x.to(self.device)
        train_y = self.gps.train_y.to(self.device)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        model = ModelListGP(*self.gps.models).to(self.device)
        pred = unstandardize(model.posterior(train_x).mean, train_y)

        acq_func_list = []
        for _ in range(self.q):
            weights = sample_simplex(self.func.nobj + self.gps.ncons).squeeze()
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
                constraints=[lambda z: z[..., -self.gps.ncons]],
            )
            acq_func_list.append(acq_func)

        new_x, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=standard_bounds,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 3, "maxiter": 1000},
        )
        return new_x.to(self.device)
    
    def qlogparego(self) -> Tensor:
        """
        Perform qParEGO optimization using qNParEGO from BoTorch.

        Returns:
            Tensor: New candidate points based on qParEGO optimization.
        """
        # Standardize bounds for optimization
        standard_bounds = torch.tensor(
            [[0.0] * self.func.dim, [1.0] * self.func.dim],
            dtype=torch.double,
        ).to(self.device) # normalized bounds


        train_x = self.gps.train_x.to(self.device)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        model = ModelListGP(*self.gps.models).to(self.device)

        # Use qNParEGO for acquisition function
        acq_func = qLogNParEGO(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            constraints=[lambda z: z[..., -self.gps.ncons]],
            prune_baseline=True, # This is recommended to improve performance
        )

        # Optimize the acquisition function
        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.q,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 3, "maxiter": 1000},
        )

        return new_x.to(self.device)

    def pesmo(self) -> Tensor:
        """
        Perform Predictive Entropy Search for Multi-Objective optimization (PESMO).

        Returns:
            Tensor: Candidate points generated by PESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack([torch.zeros(dim), torch.ones(dim)]).to(self.device)
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, _ = sample_optimal_points(
            model=model,
            bounds=bounds.double(),
            num_samples=11,
            num_points=3,
            optimizer=random_search_optimizer,
            optimizer_kwargs={"pop_size": 10000, "max_tries": 45},
        )
        acqf = qMultiObjectivePredictiveEntropySearch(model=model, pareto_sets=ps)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds.double(),
            q=self.q,
            num_restarts=5,
            raw_samples=512,
            options={"with_grad": False}
        )
        return candidates.to(self.device)

    def mesmo(self) -> Tensor:
        """
        Perform Multi-Objective Max-Value Entropy Search (MESMO).

        Returns:
            Tensor: Candidate points generated by MESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack([torch.zeros(dim), torch.ones(dim)]).to(self.device)
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, pf = sample_optimal_points(
            model=model,
            bounds=bounds.double(),
            num_samples=10,
            num_points=10,
            optimizer=random_search_optimizer,
            optimizer_kwargs={"pop_size": 10000, "max_tries": 45},
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)

        acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds.double(),
            q=self.q,
            num_restarts=8,
            raw_samples=512,
            sequential=True,
        )
        return candidates.to(self.device)

    def jesmo(self) -> Tensor:
        """
        Perform Joint Entropy Search for Multi-Objective optimization (JESMO).

        Returns:
            Tensor: Candidate points generated by JESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack([torch.zeros(dim), torch.ones(dim)]).to(self.device)
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, pf = sample_optimal_points(
            model=model,
            bounds=bounds.double(),
            num_samples=10,
            num_points=10,
            optimizer=random_search_optimizer,
            optimizer_kwargs={"pop_size": 10000, "max_tries": 45},
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)

        acqf = qLowerBoundMultiObjectiveJointEntropySearch(
            model=model,
            pareto_sets=ps,
            pareto_fronts=pf,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds.double(),
            q=self.q,
            num_restarts=8,
            raw_samples=512,
            sequential=True,
        )
        return candidates.to(self.device)

    def sobol(self) -> Tensor:
        """
        Generate random samples.

        Returns:
            Tensor: Randomly generated candidate points.
        """
        standard_bounds = torch.row_stack(
            [torch.zeros(self.func.dim), torch.ones(self.func.dim)]
        )
        return draw_sobol_samples(
            bounds=standard_bounds, n=1, q=self.q
        ).squeeze(1).to(self.device)
    
    def tsemo(self, save_dir, iters, ref_point, train_shape, rep=0):
        ts = TSEMORunner(self.func.name, 
                         self.gps.train_x, 
                         self.gps.train_y, 
                         lb=[0.0]*self.gps.train_x.shape[1], 
                         ub=[1.0]*self.gps.train_x.shape[1], 
                         iters=iters, 
                         batch_number=self.q)
        x, y, times = ts.tsemo_run(save_dir, rep)
        hv = ts.tsemo_hypervolume(y, ref_point, train_shape, iters)
        return x, y, times, hv

        
