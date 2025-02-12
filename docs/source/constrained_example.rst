Constrained Optimization Example
================================

This script demonstrates the optimization of a constrained problem using **Multi-Objective Bayesian Optimization**.

It leverages the **QPOTS** framework to optimize the **Disc Brake** problem, a multi-objective problem with constraints.

Overview
--------

- Uses **BoTorch** and **PyTorch** for Gaussian Process (GP) modeling.
- Implements **Pareto Optimal Thompson Sampling (QPOTS)** for optimization.
- Evaluates **hypervolume (HV)** to measure performance.
- Saves results (candidates, hypervolume, and timing) for post-analysis.

Script Details
--------------

.. literalinclude:: ../../qpots/examples/constrained_optimization.py
   :language: python
   :linenos:
   :caption: constrained_optimization.py

How It Works
------------

1. **Initialize the Problem**:
   - Defines the `discbrake` function with **4 decision variables**, **2 objectives**, and **4 constraints**.

2. **Train the Initial Gaussian Process Model**:
   - Generates **40 random training points**.
   - Normalizes and evaluates function outputs.
   - Fits **Gaussian Processes (GPs)** to model objectives and constraints.

3. **Iterative Optimization**:
   - Runs for **200 iterations**.
   - Uses **QPOTS** to suggest new points.
   - Updates the GP model with new data.
   - Computes **hypervolume (HV)** to track optimization progress.

4. **Result Storage**:
   - Saves:
     - `train_x.npy`: Candidate points.
     - `train_y.npy`: Objective and constraint evaluations.
     - `hv.npy`: Hypervolume values.
     - `times.npy`: Computation time per iteration.

Example Output
--------------

.. code-block:: console

    Iteration: 0, New candidate: tensor([...]), Time: 0.23s, HV: 0.758
    Iteration: 1, New candidate: tensor([...]), Time: 0.19s, HV: 0.782
    ...
    Iteration: 199, New candidate: tensor([...]), Time: 0.30s, HV: 0.915

Usage
-----

Run the script using:

.. code-block:: sh

   python constrained_optimization.py

Ensure that dependencies such as **BoTorch**, **PyTorch**, and **PyMoo** are installed.

