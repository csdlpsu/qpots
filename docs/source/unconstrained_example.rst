Unconstrained Example
=====================

This example demonstrates how to use **qPOTS** for multi-objective optimization on the **BraninCurrin** function from BoTorch. 
It is a non-HPC implementation designed for local execution.

Overview
--------

- Optimizes a **2-dimensional, 2-objective problem**.
- Uses **Pareto Optimal Thompson Sampling (QPOTS)** for acquisition.
- Evaluates **hypervolume (HV)** at each step.
- Saves training data, hypervolume values, and computational times for analysis.

Script Details
--------------

.. literalinclude:: ../../examples/unconstrained_branin_currin.py
   :language: python
   :linenos:
   :caption: botorch_qpots_example.py

.. How It Works
.. ------------

.. 1. **Problem Setup**:
..    - Uses **BraninCurrin** as the test function.
..    - Initializes **20 random training points** in a **2D space**.
..    - Evaluates objectives on a normalized space.

.. 2. **Training Gaussian Process (GP) Models**:
..    - Constructs independent **GP models** for each objective.
..    - Trains the GPs using **maximum likelihood estimation (MLE)**.

.. 3. **Iterative Optimization (50 Iterations)**:
..    - Runs **50 iterations** of optimization.
..    - Uses **QPOTS** to sample new points.
..    - Updates GP models with new data.
..    - Computes **hypervolume (HV)** to track improvement.

.. 4. **Results & Storage**:
..    - Saves:
..     `train_x.npy`: Candidate points.
..     `train_y.npy`: Objective evaluations.
..     `hv.npy`: Hypervolume values.
..     `times.npy`: Computation time per iteration.

Example Output
--------------

.. code-block:: console

    Iteration: 0, New candidate: tensor([...]), Time: 0.12s, HV: 4478.89
    Iteration: 1, New candidate: tensor([...]), Time: 0.14s, HV: 4480.92
    ...
    Iteration: 49, New candidate: tensor([...]), Time: 0.20s, HV: 4997.88

Usage
-----

To run the script locally, use:

.. code-block:: sh

   python unconstrained_branin_currin.py

Ensure dependencies such as **BoTorch**, **PyTorch**, and **qPOTS** are installed.

