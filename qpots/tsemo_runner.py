import torch
import numpy as np
import os
try:
    import matlab.engine
except ImportError:
    print("Failed to import matlab engine")
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning


class TSEMORunner:
    """
    Runs the TS-EMO algorithm iteratively using MATLAB.

    The `TSEMORunner` class interfaces with MATLAB's TS-EMO implementation, allowing 
    users to iteratively optimize a multi-objective function while updating results at each step.

    Notes
    -----
    - Requires a working MATLAB installation and the MATLAB Engine API for Python.
    - Paths to the TS-EMO MATLAB files must be correctly configured.
    """

    def __init__(
        self,
        func: str,
        x: list,
        y: list,
        lb: list,
        ub: list,
        iters: int,
        batch_number: int,
    ):
        """
        Initialize the TS-EMO runner.

        Parameters
        ----------
        func : str
            The name of the function to be optimized.
        x : list
            The initial input design points.
        y : list
            The initial function evaluations corresponding to `x`.
        lb : list
            The lower bounds of the input space.
        ub : list
            The upper bounds of the input space.
        iters : int
            The number of optimization iterations.
        batch_number : int
            The number of candidates to generate per iteration.

        Raises
        ------
        ImportError
            If MATLAB is not available or the MATLAB Engine API is not installed.
        """
        self._func = func
        self._x = x
        self._y = y
        self._lb = lb
        self._ub = ub
        self._iters = iters
        self._batch_number = batch_number
        self._eng = matlab.engine.start_matlab()

        # Get the directory of the current script (which is inside qpots)
        qpots_dir = os.path.dirname(os.path.abspath(__file__))
        ts_emo_dir = os.path.join(qpots_dir, "TS-EMO")

        # Define TS-EMO subdirectories
        ts_emo_paths = [
            ts_emo_dir,
            os.path.join(ts_emo_dir, "Test_functions"),
            os.path.join(ts_emo_dir, "Direct"),
            os.path.join(ts_emo_dir, "Mex_files/invchol"),
            os.path.join(ts_emo_dir, "Mex_files/hypervolume"),
            os.path.join(ts_emo_dir, "Mex_files/pareto front"),
            os.path.join(ts_emo_dir, "NGPM_v1.4"),
        ]

        # Add paths to MATLAB
        for path in ts_emo_paths:
            if os.path.exists(path):  # Ensure path exists before adding
                self._eng.addpath(path, nargout=0)
            else:
                print(f"Warning: The path {path} does not exist.")

    def tsemo_run(self, save_dir: str, rep: int):
        """
        Run the TS-EMO algorithm iteratively and save results after each iteration.

        Parameters
        ----------
        save_dir : str
            The directory where results should be saved.
        rep : int
            The repetition number used to differentiate saved files.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - `X`: The updated input design points.
            - `Y`: The updated objective function evaluations.
            - `times`: The runtime per iteration.

        Raises
        ------
        matlab.engine.EngineError
            If the MATLAB engine encounters an error while running TS-EMO.
        """
        # Load the initial X and Y state
        X = self._x
        Y = self._y

        try:
            [X_out, Y_out, times] = self._eng.TSEMO_run(
                self._func,
                matlab.double(X.tolist()),
                matlab.double(Y.tolist()),
                matlab.double(self._lb),
                matlab.double(self._ub),
                self._iters,
                self._batch_number,
                nargout=3,
            )
        except matlab.engine.EngineError as e:
            print(f"MATLAB engine encountered an error: {e}")
            raise

        # Convert MATLAB arrays to NumPy arrays
        X_np = np.array(X_out)
        Y_np = np.array(Y_out)
        times_np = np.array(times)

        # Save the updated X, Y, and times after each iteration
        np.save(f"{save_dir}/Y_{rep}.npy", Y_np.squeeze())
        np.save(f"{save_dir}/X_{rep}.npy", X_np)
        np.save(f"{save_dir}/times_{rep}.npy", times_np)

        return X_np, Y_np, times_np

    def tsemo_hypervolume(
        self, Y: torch.Tensor, ref_point: torch.Tensor, train_shape: int, iters: int
    ):
        """
        Compute the hypervolume and Pareto front for a given set of objective values.

        This function applies the Fast Nondominated Partitioning algorithm to evaluate 
        hypervolume improvement over multiple iterations.

        Parameters
        ----------
        Y : torch.Tensor
            A tensor of objective values, where each row represents a solution's evaluated objectives.
        ref_point : torch.Tensor
            A reference point for hypervolume calculation, typically set to be worse 
            than the worst observed objective values.
        train_shape : int
            The number of initial training points. Determines how many points 
            are included in the hypervolume calculation at each step.
        iters : int
            The number of iterations the optimization was run for.

        Returns
        -------
        Tuple[list, torch.Tensor]
            - `hv`: A list containing the hypervolume values computed at each iteration.
            - `pf`: A tensor representing the Pareto front (set of nondominated solutions).
        """
        hv = []
        pf = None
        for i in range(iters):
            # Compute the hypervolume for the current set of points (up to train_shape + i)
            bd1 = FastNondominatedPartitioning(
                ref_point=ref_point, Y=-1 * torch.tensor(Y[: train_shape + i, :])
            )
            hv.append(bd1.compute_hypervolume())
            pf = bd1.pareto_Y  # Store the current Pareto front

        return hv, pf
