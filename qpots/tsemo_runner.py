try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

if not MATLAB_AVAILABLE:
    print("MATLAB engine is not available. Some features may be disabled.")

import numpy as np
import os
import torch
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning

class TSEMORunner:
    """
    The 'mobo_TSEMO' class runs the TS-EMO algorithm iteratively in MATLAB, 
    updating and saving the results after each iteration.
    """

    def __init__(self, func: str, 
                 x: list, 
                 y: list, 
                 lb: list, 
                 ub: list, 
                 iters: int, 
                 batch_number: int):
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
            os.path.join(ts_emo_dir, "NGPM_v1.4")
        ]

        # Add paths to MATLAB
        for path in ts_emo_paths:
            if os.path.exists(path):  # Ensure path exists before adding
                self._eng.addpath(path, nargout=0)
            else:
                print(f"Warning: The path {path} does not exist.")

    def tsemo_run(self, save_dir, rep):
        """
        Run the TS-EMO algorithm iteratively and save results after each iteration.

        Parameters:
            save_dir (str): Directory to save results.
            rep (int): Repetition number to append to the file names.

        Returns:
            None
        """
        # Load the initial X and Y state (assume they are stored in self._x and self._y)
        X = self._x
        Y = self._y
        try:
            [X_out, Y_out, times] = self._eng.TSEMO_run(self._func, 
                                                    matlab.double(X.tolist()), 
                                                    matlab.double(Y.tolist()), 
                                                    matlab.double(self._lb), 
                                                    matlab.double(self._ub), 
                                                    self._iters,
                                                    self._batch_number, 
                                                    nargout=3)
        except matlab.engine.EngineError as e:
            print(f"MATLAB engine encountered an error: {e}")
            raise
        # Convert MATLAB arrays to NumPy arrays
        X_np = np.array(X_out)
        Y_np = np.array(Y_out)
        times_np = np.array(times)

        # Update X and Y with the new outputs from the MATLAB function
        X = X_np
        Y = Y_np

        # Save the updated X, Y, and times after each iteration
        np.save(f"{save_dir}/Y_{rep}.npy", Y_np.squeeze())
        np.save(f"{save_dir}/X_{rep}.npy", X_np)
        np.save(f"{save_dir}/times_{rep}.npy", times_np)

        return X, Y, times_np
    
    def tsemo_hypervolume(self, Y, ref_point, train_shape, iters):
        """
        Calculates the hypervolume and Pareto front of a set of objective values using the Fast Nondominated Partitioning algorithm.
        
        Parameters:
            Y (Tensor): A tensor of objective values, where each row corresponds to an evaluated solution.
            ref_point (Tensor): The reference point used for hypervolume calculation, typically worse than the worst objective values.
            train_shape (int): The number of initial training points, which determines how many points are included in the hypervolume calculation at each step.
            iters (int): Number of iterations that the optimization was ran for.
            
        Returns:
            hv (list): A tensor containing the hypervolume values computed at each iteration.
            pf (Tensor): The Pareto front of the objective values, which is the set of nondominated points.
        """
        hv = []
        pf = None
        for i in range(iters):
            # Compute the hypervolume for the current set of points (up to train_shape + i)
            bd1 = FastNondominatedPartitioning(ref_point=ref_point, Y=-1*torch.tensor(Y[:train_shape + i, :]))
            hv.append(bd1.compute_hypervolume())
            pf = bd1.pareto_Y  # Store the current Pareto front

        return hv, pf
