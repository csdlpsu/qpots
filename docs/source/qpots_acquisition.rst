qpots.acquisition 
=================

.. autoclass:: qpots.acquisition.Acquisition
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: tsemo  # Exclude MATLAB-dependent function

TS-EMO Acquisition (Requires MATLAB)
------------------------------------

.. note::
   The `tsemo` function requires a working installation of MATLAB and the MATLAB Engine API for Python.
   It will not run in environments where MATLAB is unavailable (e.g., Read the Docs).

**Function Signature**
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    def tsemo(self, save_dir: str, iters: int, ref_point: Tensor, train_shape: int, rep: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, Tensor]:

Description
^^^^^^^^^^^
Performs **Thompson Sampling Efficient Multiobjective Optimization (TS-EMO)** using MATLAB.

Parameters
^^^^^^^^^^
- **save_dir** (*str*):  
  The directory to save the TS-EMO results.
- **iters** (*int*):  
  The number of iterations TS-EMO should run for.
- **ref_point** (*torch.Tensor*):  
  The reference point for hypervolume calculation.
- **train_shape** (*int*):  
  Determines the size of the bounds.
- **rep** (*int, optional*):  
  The repetition of the experiment. Defaults to `0`.

Returns
^^^^^^^
- **x** (*np.ndarray*):  
  The inputs of the function chosen by TS-EMO.
- **y** (*np.ndarray*):  
  The outputs of the function for each input.
- **times** (*np.ndarray*):  
  The time taken for each iteration.
- **hv** (*list*):  
  The list of hypervolume values at each iteration.
- **pf** (*torch.Tensor*):  
  The Pareto frontier determined by TS-EMO.