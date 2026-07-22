<p align="center">
  <img src="https://raw.githubusercontent.com/csdlpsu/qpots/main/assets/qpots-logo.png" alt="qPOTS logo" width="150">
</p>

<p align="center">
  <strong>Batch Pareto Optimal Thompson Sampling for multiobjective Bayesian optimization</strong>
</p>

<p align="center">
  <a href="https://proceedings.mlr.press/v258/renganathan25a.html"><img alt="AISTATS 2025 paper" src="https://img.shields.io/badge/Paper-AISTATS%202025-2563eb"></a>
  <a href="https://pypi.org/project/qpots/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/qpots?label=PyPI&cacheSeconds=60"></a>
  <a href="https://qpots.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/readthedocs/qpots"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPL--3.0-green"></a>
  <a href="https://github.com/csdlpsu/qpots/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/csdlpsu/qpots/actions/workflows/ci.yml/badge.svg"></a>
</p>

$q\texttt{POTS}$ is a Python package for sample-efficient multiobjective Bayesian optimization. It implements **Pareto Optimal Thompson Sampling**, a batch acquisition strategy that selects candidates according to their probability of being Pareto optimal under Gaussian-process posterior samples.

The project is maintained by the Computational Complex Engineered Systems Design Laboratory ([CSDL](https://sites.psu.edu/csdl/)) at Penn State.

## Why $q\texttt{POTS}$?

Multiobjective optimization often requires many expensive function evaluations before a useful Pareto front emerges. $q\texttt{POTS}$ combines Gaussian-process surrogate modeling with evolutionary optimization over posterior samples, giving users a practical way to propose informative batches without directly optimizing a difficult analytical acquisition function.

Use $q\texttt{POTS}$ when you need to:

- optimize two or more competing objectives with limited evaluation budget;
- propose one or more candidates per Bayesian optimization iteration;
- handle BoTorch benchmark functions or your own custom objectives;
- compare $q\texttt{POTS}$ against common multiobjective acquisition strategies; or
- run TS-EMO baselines when MATLAB Engine is available.

## Installation

Install the latest release from PyPI:

```bash
pip install qpots
```

To install from source:

```bash
git clone https://github.com/csdlpsu/qpots
cd qpots
pip install .
```

$q\texttt{POTS}$ requires Python 3.11 or newer and is continuously tested on Python 3.11, 3.12, and 3.13. The core $q\texttt{POTS}$ implementation uses Python dependencies installed by `pip`, including BoTorch, PyTorch, GPyTorch, NumPy, SciPy, scikit-learn, and pymoo.

### Optional MATLAB Engine

The MATLAB Engine is only needed if you plan to use the TS-EMO baseline included with this repository. $q\texttt{POTS}$ itself and the BoTorch-based acquisition functions do not require MATLAB.

Install MATLAB Engine with the version that matches your local MATLAB installation. For example, MATLAB R2023b uses:

```bash
pip install matlabengine==23.2.1
```

See MathWorks' [MATLAB Engine for Python installation guide](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for release-specific instructions.

## Quick Start

The example below runs $q\texttt{POTS}$ on the two-objective Branin-Currin benchmark.

```python
from qpots import Function, QPOTSConfig, QPOTSRunner

problem = Function("branincurrin", dim=2, nobj=2)
config = QPOTSConfig(n_initial=20, iterations=50, batch_size=1)
result = QPOTSRunner(problem, config).run()

print(result.train_x)
print(result.train_y)
```

## Runtime Precision And Device

$q\texttt{POTS}$ uses CUDA when PyTorch detects a GPU and otherwise falls back to CPU. Configure newly-created objects without editing the installed package:

```python
import torch
from qpots import RuntimeConfig, set_default_runtime

set_default_runtime(RuntimeConfig(device="cpu", dtype=torch.float64))
```

Per-object `runtime`, `device`, and `dtype` arguments override this default.

For complete scripts, see:

- [Unconstrained Branin](examples/unconstrained_branin.py)
- [Constrained optimization](examples/constrained_example.py)
- [Decoupled optimization (OSY)](examples/decoupled_osy_example.py)
- [Custom objective functions](examples/custom_function_example.py)
- [Multiple acquisition functions](examples/multiple_acquisitions_example.py)
- [HPC-style runs](examples/hpc_example.py)

## qPOTS-Decoupled

Use **qPOTS-Decoupled** when objectives or constraints can be measured
separately. A multitask Gaussian process shares information between outputs,
while a total-correlation gate and mutual-information rule select which output
tasks to query at each candidate. Earlier project material called this mode
``qPOTS-DOE`` (decoupled oracle evaluations); the new name avoids confusion
with the established abbreviation for design of experiments.

See the dedicated [qPOTS-Decoupled guide](https://qpots.readthedocs.io/en/latest/qpots_doe.html)
and the [decoupled OSY example](examples/decoupled_osy_example.py).

## Documentation

The hosted documentation includes installation notes, API references, and worked examples:

[qpots.readthedocs.io](https://qpots.readthedocs.io/en/latest/)

## Public API and compatibility

The recommended public interfaces are importable directly from ``qpots``. The
existing module imports used by qPOTS 2.0 scripts remain supported. See the
[API compatibility policy](https://qpots.readthedocs.io/en/latest/api_compatibility.html)
for the supported surface, migration guidance, and deprecation policy.

## Main Reference

The main reference for this repository is the AISTATS 2025 paper:

> Ashwin Renganathan and Kade Carlson. $q\texttt{POTS}$: Efficient Batch Multiobjective Bayesian Optimization via Pareto Optimal Thompson Sampling. Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, PMLR 258:4051-4059, 2025.

```bibtex
@inproceedings{renganathan2025qpots,
  title={qPOTS: Efficient Batch Multiobjective Bayesian Optimization via Pareto Optimal Thompson Sampling},
  author={Renganathan, Ashwin and Carlson, Kade},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4051--4059},
  year={2025},
  organization={PMLR}
}
```

Additional links:

- [AISTATS/PMLR paper page](https://proceedings.mlr.press/v258/renganathan25a.html)
- [PDF](https://raw.githubusercontent.com/mlresearch/v258/main/assets/renganathan25a/renganathan25a.pdf)
- [arXiv preprint](https://arxiv.org/abs/2310.15788)

## Development

Clone the repository and install the package in editable mode with the test dependencies:

```bash
git clone https://github.com/csdlpsu/qpots
cd qpots
python -m pip install -e ".[test,docs]"
```

Run the test suite with:

```bash
pytest
```

The package source lives in [`qpots/`](qpots/), examples live in [`examples/`](examples/), tests live in [`tests/`](tests/), and Sphinx documentation lives in [`docs/`](docs/).

## Contributing And Support

Contributions, bug reports, documentation improvements, and research-use questions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

Use the [GitHub issue tracker](https://github.com/csdlpsu/qpots/issues) to report bugs, request features, ask usage questions, or seek support. For bug reports, include your qPOTS version, Python version, operating system, relevant dependency versions, CPU/GPU context, and a minimal reproducible example when possible.

qPOTS is maintained by the Computational Complex Engineered Systems Design Laboratory at Penn State. Maintainer response times may vary with academic schedules, but issues with reproducible examples and clear research context are easiest to triage.

## License

This project is distributed under the terms of the [GNU General Public License v3.0](LICENSE).
