from typing import Callable, Optional

import torch

# from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import MultiTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.transforms import normalize
from scipy.spatial.distance import cdist
from torch import Tensor

from qpots.config import RuntimeConfig, as_tensor, resolve_runtime, tensor_kwargs, to_runtime
from qpots.function import Function
from qpots.tsemo_runner import TSEMORunner
from qpots.utils.pymoo_problem import PyMooFunction, nsga2
from qpots.utils.utils import (
    select_candidates,
    select_candidates_total_correlation,
    unstandardize,
    unstandardize_ignore_nan,
)


class Acquisition:
    """
    A class providing various acquisition functions and methods for multi-objective optimization.
    """

    def __init__(
        self,
        func: Function,
        gps: ModelListGP,
        cons: Optional[Callable] = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        q: int = 1,
        NUM_RESTARTS: int = 10,
        RAW_SAMPLES: int = 512,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        """
        Initialize the multi-objective acquisition class.

        This class provides various acquisition functions for multi-objective Bayesian
        optimization. It supports Gaussian Process models and handles inequality constraints
        if provided.

        Parameters
        ----------
        func : Function
            The test function being optimized.
        gps : ModelListGP
            A list of Gaussian Process models used for Bayesian optimization.
        cons : Optional[Callable], optional
            A vector-valued function representing inequality constraints. If provided,
            the acquisition function will account for feasibility constraints.
        device : torch.device or str, optional
            The computational device to use. If omitted, qPOTS uses CUDA when
            available and falls back to CPU.
        dtype : torch.dtype, optional
            Floating-point precision. If omitted, qPOTS uses the dtype attached
            to ``gps`` when present, otherwise ``qpots.config.DEFAULT_DTYPE``.
        q : int, optional
            The number of candidate points to sample per iteration. Defaults to 1.
        NUM_RESTARTS : int, optional
            The number of restarts for optimizing the acquisition function.
            A higher value can improve optimization quality. Defaults to 10.
        RAW_SAMPLES : int, optional
            The number of raw samples used in acquisition optimization.
            Higher values increase exploration but may slow computation. Defaults to 512.
        """
        self.func = func
        self.gps = gps
        self.cons = cons
        inferred_device = getattr(gps, "device", None)
        if not isinstance(inferred_device, (str, torch.device)):
            inferred_device = None
        inferred_dtype = getattr(gps, "dtype", None)
        if not isinstance(inferred_dtype, torch.dtype):
            inferred_dtype = None
        resolved_runtime = resolve_runtime(
            runtime,
            device=device if device is not None else inferred_device,
            dtype=dtype if dtype is not None else inferred_dtype,
        )
        self.runtime = resolved_runtime
        self.device = resolved_runtime.device
        self.dtype = resolved_runtime.dtype
        self.tkwargs = tensor_kwargs(device=self.device, dtype=self.dtype)
        self.q = q
        self.NUM_RESTARTS = NUM_RESTARTS
        self.RAW_SAMPLES = RAW_SAMPLES
        self.nobj = gps.nobj
        self.ncons = gps.ncons

    def _nystrom_approx(
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
        Perform Thompson sampling using the Nyström approximation for Gaussian Process (GP) models.

        The Nyström approximation is a method for approximating large covariance matrices
        in Gaussian Process inference, improving computational efficiency while preserving
        accuracy. This function selects a subset of columns to approximate the full covariance
        matrix and then performs posterior sampling.

        Parameters
        ----------
        x : Tensor
            A tensor of input points for which posterior samples are computed.
        gps : ModelListGP
            A list of Gaussian Process models used to estimate the posterior distribution.
        m : int
            The number of landmarks (subset size) used in the Nyström approximation.
        pareto_set : Optional[Tensor], optional
            A tensor containing Pareto optimal points. Required if ``col_choice`` is 'pareto'.
            Defaults to None.
        col_choice : str, optional
            The column selection strategy for the Nyström approximation.
            Options:
            - 'pareto' (default): Select columns based on proximity to Pareto optimal points.
            - 'random': Select a random subset of columns.
        seed_iter : int, optional
            An iteration index for seeding randomness in sampling. Defaults to 1.
        reg : float, optional
            A small regularization constant added to the covariance matrix for numerical
            stability during matrix inversion. Defaults to 1e-6.

        Returns
        -------
        Tensor
            A tensor containing posterior samples from the GP models. Infeasible points
            are penalized if constraints exist.
        """
        torch.manual_seed(1024 + seed_iter)
        samples_list = []
        x = to_runtime(x, self.device, self.dtype)
        bounds = to_runtime(gps.bounds, self.device, self.dtype)

        for gp_model in gps.models:
            posterior = gp_model.posterior(normalize(x, bounds))
            mean = posterior.mean.to(**self.tkwargs)
            covariance = posterior.mvn.covariance_matrix.to(**self.tkwargs)

            if col_choice == "random":
                indices = torch.randperm(covariance.shape[-1], device=self.device)[:m]
            elif col_choice == "pareto":
                if pareto_set is None:
                    raise ValueError("Pareto set is required for 'pareto' column choice.")
                pareto_np = (
                    pareto_set.detach().cpu().numpy()
                    if isinstance(pareto_set, torch.Tensor)
                    else pareto_set
                )
                D = cdist(pareto_np, x.detach().cpu().numpy())
                cand_indices = D.argmin(axis=-1)
                indices = torch.unique(
                    torch.as_tensor(cand_indices, device=self.device).argsort()[:m]
                )
                if len(indices) < m:
                    extra_indices = torch.randperm(covariance.shape[-1], device=self.device)[
                        : m - len(indices)
                    ]
                    indices = torch.cat((indices, extra_indices))
            else:
                raise NotImplementedError(f"Column choice '{col_choice}' is not implemented.")

            K_nm = covariance[:, indices]
            apprx_covariance = covariance[indices][:, indices]
            while True:
                try:
                    L_mm_cov = torch.linalg.cholesky(
                        apprx_covariance + reg * torch.eye(m, **self.tkwargs)
                    )
                    break
                except torch.linalg.LinAlgError:
                    reg *= 10

            z = torch.rand(m, **self.tkwargs)
            sample = mean + K_nm @ (L_mm_cov @ z).reshape(-1, 1)
            samples_list.append(sample.detach())

        Ys_ = unstandardize(torch.cat(samples_list, -1), gps.train_y.to(self.device))

        if self.ncons > 0:
            ind_feasible = Ys_[..., -self.ncons :] <= 0
            Ys_[
                ~ind_feasible.squeeze(), : self.nobj
            ] = -1e12  # Arbitrary low value for infeasible points
            Ys = Ys_[..., : self.nobj]
        else:
            Ys = Ys_

        return -Ys

    def _gp_posterior(self, x: Tensor, gps: ModelListGP, seed_iter: int = 1) -> Tensor:
        """
        Compute posterior samples for PyMoo optimization.

        Parameters
        ----------
        x : Tensor
            A tensor of input points for which posterior samples are computed.
        gps : ModelListGP
            A list of Gaussian Process models used to estimate the posterior distribution.
        seed_iter : int, optional
            An iteration index for seeding randomness in sampling. Defaults to 1.

        Returns
        -------
        Tensor
            A tensor containing posterior samples from the GP models, with
            infeasible points penalized if constraints exist.
        """
        torch.manual_seed(1024 + seed_iter)
        x = to_runtime(x, self.device, self.dtype)
        bounds = to_runtime(gps.bounds, self.device, self.dtype)
        Ys_ = [
            model.posterior(normalize(x, bounds)).sample().reshape(-1, 1).to(**self.tkwargs)
            for model in gps.models
        ]
        Ys_ = unstandardize(torch.cat(Ys_, -1), gps.train_y.to(self.device))

        if self.ncons > 0:
            ind_feasible = (Ys_[..., -self.ncons :] >= 0).all(dim=-1)
            Ys_[~ind_feasible.squeeze(), : self.nobj] = -1e12  # Penalize infeasible points
            Ys = Ys_[..., : self.nobj]
        else:
            Ys = Ys_

        return -Ys

    def _mt_gp_posterior(self, x: Tensor, gps: MultiTaskGP, seed_iter: int = 1) -> Tensor:
        """
        Compute posterior samples for PyMoo optimization for MultiTaskGP.

        Parameters
        ----------
        x : Tensor
            A tensor of input points for which posterior samples are computed.
        gps : MultiTaskGP
            A multi-task Gaussian Process model used to estimate the posterior distribution.
        seed_iter : int, optional
            An iteration index for seeding randomness in sampling. Defaults to 1.

        Returns
        -------
        Tensor
            A tensor containing posterior samples from the GP models, with
            infeasible points penalized if constraints exist.
        """
        torch.manual_seed(1024 + seed_iter)

        x = to_runtime(x, self.device, self.dtype)
        model = gps.models[0]
        # MultiTaskGP.posterior returns all modeled tasks, so explicit task IDs
        # are not appended when drawing the sample path used by NSGA-II.
        standardized_samples = (
            model.posterior(normalize(x, to_runtime(gps.bounds, self.device, self.dtype)))
            .sample()
            .to(**self.tkwargs)
        )
        sampled_values = unstandardize_ignore_nan(standardized_samples, gps.train_y.to(self.device))

        if self.ncons > 0:
            feasible = (sampled_values[..., -self.ncons :] >= 0).all(dim=-1)
            sampled_values[~feasible.squeeze(), : self.nobj] = -1e12
            objective_samples = sampled_values[..., : self.nobj]
        else:
            objective_samples = sampled_values
        return -objective_samples

    def _make_qpots_posterior_path(
        self,
        bounds: Tensor,
        iteration: int,
        options: dict,
        pareto_state: list,
    ) -> Callable[[Tensor], Tensor]:
        """Create the sampled objective passed to the evolutionary optimizer."""
        if options["nystrom"] == 1:
            landmark_count = max(1, int(0.2 * options["iters"]))

            def posterior_path(x: Tensor) -> Tensor:
                pareto_points = (
                    normalize(as_tensor(pareto_state[0], **self.tkwargs), bounds)
                    if pareto_state[0] is not None
                    else torch.zeros_like(x)
                )
                return self._nystrom_approx(
                    x.to(self.device),
                    self.gps,
                    landmark_count,
                    pareto_points,
                    col_choice=options["nychoice"],
                    seed_iter=iteration,
                )

            return posterior_path

        if options.get("mt", 0) == 1:

            def posterior_path(x: Tensor) -> Tensor:
                return self._mt_gp_posterior(x.to(self.device), self.gps, seed_iter=iteration)
        else:

            def posterior_path(x: Tensor) -> Tensor:
                return self._gp_posterior(x.to(self.device), self.gps, seed_iter=iteration)

        return posterior_path

    def _optimize_qpots_posterior(self, bounds: Tensor, iteration: int, options: dict):
        """Optimize one posterior sample path and return its Pareto set."""
        pareto_state = [None]

        def track_pareto(result) -> None:
            pareto_state[0] = result.opt.get("X")

        posterior_path = self._make_qpots_posterior_path(bounds, iteration, options, pareto_state)
        pymoo_problem = PyMooFunction(
            posterior_path,
            n_var=options["dim"],
            n_obj=self.nobj,
            xl=bounds[0].detach().cpu().numpy(),
            xu=bounds[1].detach().cpu().numpy(),
            device=self.device,
            dtype=self.dtype,
        )
        callback = track_pareto if options["nystrom"] == 1 else None
        return nsga2(
            pymoo_problem,
            ngen=options["ngen"],
            pop_size=100 * options["dim"],
            seed=2430,
            callback=callback,
        )

    def _select_qpots_batch(
        self, pareto_points, bounds: Tensor, options: dict
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Select a diverse batch and, when requested, its oracle task IDs."""
        if options.get("partial_info", 0) == 1:
            candidates, task_ids = select_candidates_total_correlation(
                self.gps,
                pareto_points,
                self.device,
                q=options["q"],
                seed=2043,
                thresh=options.get("threshold"),
            )
            return normalize(candidates, bounds), task_ids
        candidates = select_candidates(
            self.gps, pareto_points, self.device, q=options["q"], seed=2043
        )
        return normalize(candidates, bounds)

    def qpots(
        self,
        bounds: Tensor,
        iteration: int,
        **kwargs,
    ) -> Tensor:
        """
        Perform Pareto Optimal Thompson Sampling (qPOTS).

        Parameters
        ----------
        bounds : Tensor
            A tensor representing the lower and upper bounds for the
            optimization problem.
        iteration : int
            The current iteration index, used for seeding randomness.
        **kwargs : dict
            Additional arguments for customization, including:

            - ``nystrom`` (int): Whether to use the Nystrom approximation (1 for yes, 0 for no).
            - ``iters`` (int): Number of iterations used in the Nystrom approximation.
            - ``nychoice`` (str): Column selection method for the Nystrom approximation.
            - ``dim`` (int): Dimensionality of the input space.
            - ``ngen`` (int): Number of generations for the NSGA-II optimization.
            - ``q`` (int): Number of candidates to select.
            - ``mt`` (int): Whether to use MultiTaskGP for posterior sampling (1 for yes, 0 for no).
            - ``partial_info`` (int): Whether to perform candidate selection using partial information (1 for yes, 0 for no).
            - ``threshold`` (float, optional): Total-correlation threshold used during partial-information selection.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            If ``partial_info == 0``, returns a normalized tensor of selected
            candidate points. If ``partial_info == 1``, returns
            ``(candidates, task_ids)``, where ``task_ids`` records which
            objectives or constraints should be evaluated at each candidate.
        """
        bounds = to_runtime(bounds, self.device, self.dtype)
        pareto_result = self._optimize_qpots_posterior(bounds, iteration, kwargs)
        return self._select_qpots_batch(pareto_result.X, bounds, kwargs)

    def qlogei(self, ref_point: Tensor | None = None) -> Tensor:
        """
        Optimize the qLogEI acquisition function and return new candidate points.

        Parameters
        ----------
        ref_point : Tensor, optional
            The reference point for hypervolume calculation, typically representing
            a baseline for performance. Defaults to ``[0.0, 0.0]``.

        Returns
        -------
        Tensor
            A tensor containing the new candidate points selected based on
            qLogEI optimization.
        """
        ref_point = (
            as_tensor([0.0, 0.0], **self.tkwargs)
            if ref_point is None
            else to_runtime(ref_point, self.device, self.dtype)
        )
        standard_bounds = torch.zeros(2, self.func.dim, **self.tkwargs)
        standard_bounds[1] = 1
        train_y = self.gps.train_y.to(**self.tkwargs)
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point, Y=train_y[..., : self.nobj]
        )
        model = ModelListGP(*self.gps.models).to(self.device)
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
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

        Returns
        -------
        Tensor
            A tensor containing the new candidate points selected based on qParEGO optimization.
        """
        standard_bounds = torch.tensor(
            [[0.0] * self.func.dim, [1.0] * self.func.dim],
            **self.tkwargs,
        )  # normalized bounds
        train_x = self.gps.train_x.to(**self.tkwargs)
        train_y = self.gps.train_y.to(**self.tkwargs)
        print(f"Inside parego: train_x shape: {train_x.shape}")
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        model = ModelListGP(*self.gps.models).to(self.device)
        pred = unstandardize(model.posterior(train_x).mean, train_y)

        acq_func_list = []
        for _ in range(self.q):
            weights = sample_simplex(self.func.nobj + self.gps.ncons, **self.tkwargs).squeeze()
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

        Returns
        -------
        Tensor
            A tensor containing the candidate points selected based on qParEGO optimization.
        """
        # Standardize bounds for optimization
        standard_bounds = torch.tensor(
            [[0.0] * self.func.dim, [1.0] * self.func.dim],
            **self.tkwargs,
        )  # normalized bounds

        train_x = self.gps.train_x.to(**self.tkwargs)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        model = ModelListGP(*self.gps.models).to(self.device)

        # Use qNParEGO for acquisition function
        acq_func = qLogNParEGO(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            constraints=[lambda z: z[..., -self.gps.ncons]],
            prune_baseline=True,  # This is recommended to improve performance
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

        Returns
        -------
        Tensor
            A tensor containing the candidate points selected by PESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack(
            [torch.zeros(dim, **self.tkwargs), torch.ones(dim, **self.tkwargs)]
        )
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, _ = sample_optimal_points(
            model=model,
            bounds=bounds,
            num_samples=11,
            num_points=3,
            optimizer=random_search_optimizer,
            optimizer_kwargs={"pop_size": 10000, "max_tries": 45},
        )
        acqf = qMultiObjectivePredictiveEntropySearch(model=model, pareto_sets=ps)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=self.q,
            num_restarts=5,
            raw_samples=512,
            options={"with_grad": False},
        )
        return candidates.to(self.device)

    def mesmo(self) -> Tensor:
        """
        Perform Multi-Objective Max-Value Entropy Search (MESMO).

        Returns
        -------
        Tensor
            A tensor containing the candidate points selected by MESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack(
            [torch.zeros(dim, **self.tkwargs), torch.ones(dim, **self.tkwargs)]
        )
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, pf = sample_optimal_points(
            model=model,
            bounds=bounds,
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
        try:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=self.q,
                num_restarts=8,
                raw_samples=512,
                sequential=True,
            )
        except RuntimeError as err:
            if "probability tensor contains" not in str(err):
                raise
            candidates = self.sobol()
        return candidates.to(self.device)

    def jesmo(self) -> Tensor:
        """
        Perform Joint Entropy Search for Multi-Objective optimization (JESMO).

        JESMO is an acquisition function that uses joint entropy search to efficiently
        explore the Pareto frontier in multi-objective Bayesian optimization.

        Returns
        -------
        Tensor
            A tensor containing the candidate points generated by JESMO.
        """
        dim = self.func.dim
        bounds = torch.row_stack(
            [torch.zeros(dim, **self.tkwargs), torch.ones(dim, **self.tkwargs)]
        )
        model = ModelListGP(*self.gps.models).to(self.device)

        ps, pf = sample_optimal_points(
            model=model,
            bounds=bounds,
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
        try:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=self.q,
                num_restarts=8,
                raw_samples=512,
                sequential=True,
            )
        except RuntimeError as err:
            if "probability tensor contains" not in str(err):
                raise
            candidates = self.sobol()
        return candidates.to(self.device)

    def sobol(self) -> Tensor:
        """
        Generate random Sobol sequence samples.

        Returns
        -------
        Tensor
            A tensor of randomly generated candidate points using the Sobol sequence.
        """
        standard_bounds = torch.row_stack(
            [torch.zeros(self.func.dim, **self.tkwargs), torch.ones(self.func.dim, **self.tkwargs)]
        )
        return draw_sobol_samples(bounds=standard_bounds, n=1, q=self.q).squeeze(0).to(self.device)

    def tsemo(self, save_dir: str, iters: int, ref_point: Tensor, train_shape: int, rep: int = 0):
        """
        Perform Thompson Sampling Efficient Multiobjective Optimization (TS-EMO).

        Parameters
        ----------
        save_dir : str
            The directory to save the TS-EMO results.
        iters : int
            How many iterations TS-EMO should run for.
        ref_point : Tensor
            The reference point for the hypervolume calculation.
        train_shape : int
            The shape for determining the size of bounds.
        rep : int, optional
            The repetition of the experiment. Defaults to 0.

        Returns
        -------
        x : np.ndarray
            The inputs of the function chosen by TS-EMO.
        y : np.ndarray
            The outputs of the function for each input.
        times : np.ndarray
            The time that each iteration takes.
        hv : list
            The list of the hypervolume at each iteration.
        pf : Tensor
            The Pareto frontier determined by TS-EMO.
        """
        ts = TSEMORunner(
            self.func.name,
            self.gps.train_x,
            self.gps.train_y,
            lb=[0.0] * self.gps.train_x.shape[1],
            ub=[1.0] * self.gps.train_x.shape[1],
            iters=iters,
            batch_number=self.q,
        )
        x, y, times = ts.tsemo_run(save_dir, rep)
        hv, pf = ts.tsemo_hypervolume(y, ref_point, train_shape, iters)
        return x, y, times, hv, pf
