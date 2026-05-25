---
title: 'qPOTS: Batch Pareto optimal Thompson sampling for multiobjective Bayesian optimization'
tags:
  - Python
  - Bayesian optimization
  - multiobjective optimization
  - Gaussian process
  - Thompson sampling
  - decoupled evaluations
authors:
  - name: Kade E. Carlson
    affiliation: "1"
  - name: Ashwin Renganathan
    orcid: 0000-0001-6948-6932
    affiliation: "1, 2"
  - name: Peter E. Bachman
    affiliation: "1"
affiliations:
  - name: Department of Aerospace Engineering, The Pennsylvania State University, United States
    index: 1
    ror: "04p491231"
  - name: Institute for Computational and Data Sciences, The Pennsylvania State University, United States
    index: 2
    ror: "04p491231"
date: 22 May 2026
bibliography: paper.bib
---

# Summary

`qPOTS` is a Python package for researchers who need to optimize expensive black-box systems with more than one competing objective. In such problems, each experiment, simulation, or hardware test can be costly, and the best answer is rarely a single design but instead a set of trade-off designs that form a Pareto front. Examples include aircraft design, process optimization, materials discovery, and machine-learning model selection. `qPOTS` helps users decide which designs to evaluate next by combining Gaussian-process surrogate models, Thompson sampling, and evolutionary search over sampled Pareto fronts.

The package implements Pareto optimal Thompson sampling (POTS) for batch multiobjective Bayesian optimization [@thompson1933likelihood; @renganathan2025qpots]. It provides tools for unconstrained and constrained benchmark functions, Gaussian-process model fitting and prediction, batch candidate selection via a maximin diversity rule, expected hypervolume tracking, and comparison with standard multiobjective acquisition functions. The package also includes `qPOTS-DOE`, a major multitask extension for decoupled oracle evaluations: when objectives or constraints can be measured separately, the software uses a multitask Gaussian process (MTGP) to transfer information across correlated tasks and then selects only the subset of oracles that are most informative for the current candidate [@bonilla2007multi; @alvarez2012kernels; @renganathan2026qpotsdoe].

`qPOTS` is released under the GNU General Public License v3.0 and is built on widely used scientific Python libraries including PyTorch, GPyTorch, BoTorch, and pymoo [@paszke2019pytorch; @gardner2018gpytorch; @balandat2020botorch; @blank2020pymoo].

# Statement of need

Multiobjective optimization is central to scientific and engineering design because improving one objective often degrades another. Classical evolutionary algorithms such as NSGA-II can approximate Pareto fronts accurately but require many objective evaluations [@deb2002fast]. That cost is prohibitive when each evaluation requires a high-fidelity computational simulation, a physical experiment, or a hardware test that may take hours or days. Bayesian optimization addresses this by fitting probabilistic surrogate models to existing data and using them to select the next most informative experiments [@jones1998efficient; @rasmussen2006gaussian; @shahriari2016taking].

Multiobjective Bayesian optimization is substantially harder than its single-objective counterpart. Acquisition functions must reason about a set-valued Pareto frontier rather than a scalar optimum, making them nonconvex, nondifferentiable, or expensive to optimize in batch settings. `qPOTS` addresses this by replacing direct acquisition-function maximization with posterior-sample optimization: it draws sample paths from the current surrogate model, solves a comparatively inexpensive multiobjective problem on those samples using evolutionary search, and selects a diverse batch of candidates from the resulting sampled Pareto set [@renganathan2025qpots]. This avoids the need for differentiable analytical acquisitions and naturally yields batches without additional approximation.

A further practical challenge arises when the objectives and constraints of a problem cannot be observed simultaneously. In aerodynamic design, for instance, drag and lift coefficients may require separate solver runs; in chemical processes, different properties may require different laboratory analyses. Existing decoupled Bayesian optimization methods typically assume independent objective models or rely on known scalar oracle costs. `qPOTS-DOE` fills this gap by fitting a single multitask Gaussian process over all objectives and constraints jointly, using inter-task posterior correlation to decide whether decoupled evaluations are beneficial, and selecting the most informative oracle subset via a Gaussian mutual-information criterion [@renganathan2026qpotsdoe]. The method comes with theoretical guarantees: asymptotic posterior concentration holds even under decoupled observations (Theorem 4.1 in @renganathan2026qpotsdoe), and the mutual-information subset rule selects the oracles that maximize expected posterior uncertainty reduction in one step (Theorem 4.2 in @renganathan2026qpotsdoe).

The target audience is researchers and engineers who need reproducible, sample-efficient multiobjective optimization workflows in Python. `qPOTS` is designed for method developers benchmarking against standard acquisition strategies, and for application researchers optimizing expensive custom black-box functions under limited evaluation budgets, including scenarios where not all objectives need to be queried at every iteration.

# State of the field

BoTorch provides a general platform for Bayesian optimization and implements strong multiobjective acquisitions such as qEHVI, qNEHVI, and their logarithmic variants [@daulton2020differentiable; @ament2023unexpected; @balandat2020botorch]. Other influential methods include entropy-search approaches (PESMO, MESMO, JESMO) and lookahead methods such as the hypervolume knowledge gradient [@garrido2019pesmoc; @belakaria2019max; @tu2022joint; @daulton2023hypervolume]. These are important baselines, and several are exposed in `qPOTS` workflows for direct comparison.

`qPOTS` occupies a different niche. Rather than contributing a single acquisition function to an existing library, it packages the full research workflow around Pareto optimal Thompson sampling: benchmark functions, constrained and unconstrained examples, batch candidate selection, hypervolume evaluation, baseline comparisons, tests, documentation, and reproducible scripts. A focused package was preferable to a narrow BoTorch primitive because the scholarly contribution spans the sample-path Pareto optimization workflow, the maximin diversity-based batch selection rule, the benchmark suite, and the newer decoupled multitask workflow. At the same time, `qPOTS` deliberately builds on BoTorch and GPyTorch rather than reimplementing Gaussian-process infrastructure.

The `qPOTS-DOE` extension addresses a specific gap in decoupled multiobjective optimization. Existing decoupled approaches are often cost-weighted and frequently assume independent objective or constraint models. `qPOTS-DOE` instead uses a multitask Gaussian process with an intrinsic coregionalization kernel, a total-correlation gate $\mathrm{TC}(\mathbf{x}) = -\tfrac{1}{2} \log |\mathbf{R}(\mathbf{x})|$ computed from the posterior correlation matrix $\mathbf{R}(\mathbf{x})$, and a Gaussian mutual-information subset rule to decide which oracles to evaluate at a candidate point [@renganathan2026qpotsdoe]. This enables meaningful decoupling even when scalar oracle costs are uniform, provided the tasks exhibit useful posterior correlation.

# Software design

The package is organized into four main components. The `Function` interface wraps built-in benchmarks and user-defined objectives behind a common API. `ModelObject` manages Gaussian-process fitting and prediction, supporting both independent per-objective GPs (the original `qPOTS` mode) and a single joint multitask GP for `qPOTS-DOE`. `Acquisition` contains the candidate-generation logic for `qPOTS` and all comparison methods. Utility modules compute hypervolume indicators, filter candidates by diversity and dominance, manage data transformations, and support experiment scripts.

This design makes several trade-offs explicit. First, `qPOTS` uses posterior-sample optimization rather than closed-form acquisition maximization. This introduces stochastic evolutionary-optimization overhead per iteration, but avoids the need for differentiable analytical acquisitions and naturally supports batch candidate generation without additional approximations. Second, the package delegates multiobjective search over sampled posterior paths to NSGA-II via pymoo, keeping the Python interface simple while using a mature and well-tested evolutionary optimizer. Third, precision and device handling are centralized in `config.py`, so users can run CPU-only examples or move tensor computations to CUDA hardware with a single setting change.

The `qPOTS-DOE` multitask extension adds a two-stage decision layer on top of the original candidate selection. In the standard coupled mode, all objectives and constraints are observed at every chosen design. In decoupled mode, the MTGP posterior correlation matrix is used to compute a scalar total-correlation score at the proposed candidate; if this score exceeds a threshold $\tau$, the algorithm selects only an informative subset of oracles by maximizing the Gaussian mutual information $I(\mathbf{Y}_{\mathcal{S}}(\mathbf{x}); \mathbf{Y}_{\bar{\mathcal{S}}}(\mathbf{x}) \mid \mathcal{D})$ between selected and unselected tasks. The MTGP then uses posterior mean imputation to fill unobserved task values and continues the optimization loop. This architecture preserves the original `qPOTS` sample-path candidate generation while adding principled oracle selection.

![Illustration of `qPOTS` on a constrained two-objective problem. Shown are the initial training data, the Pareto front approximation, and the batch of candidates proposed by the acquisition strategy. \label{fig:constrained}](assets/qpots_constrained_illustration.png){ width=85% }

![The `qPOTS-DOE` multitask update uses posterior task correlation to decide when decoupled evaluations are informative. The Branin-Currin example illustrates task-specific observations, total-correlation evolution, and uncertainty reduction from multitask information sharing [@renganathan2026qpotsdoe]. \label{fig:doe}](assets/qpots_doe_total_correlation.png){ width=100% }

# Research impact statement

`qPOTS` is the reference implementation for the AISTATS 2025 paper introducing Pareto optimal Thompson sampling for batch multiobjective Bayesian optimization [@renganathan2025qpots]. The method was evaluated on eight synthetic and real-world benchmarks against qEHVI, qNEHVI, qLogNEHVI, ParEGO, PESMO, MESMO, JESMO, TS-EMO, and Sobol baselines, demonstrating competitive or superior hypervolume improvements with substantially lower computational cost per iteration compared to Monte Carlo acquisition methods. The package has also supported an applied aerodynamic design study of the NASA Common Research Model, where `qPOTS` was deployed for 24-dimensional, two-objective aerodynamic design optimization and released with open-source reproducibility materials [@carlson2026multiobjective].

The `qPOTS-DOE` extension further expands the implementation to correlated objective and constraint tasks with partial observations. In benchmark experiments, `qPOTS-DOE` is compared against coupled `qPOTS`, qLogNEHVI, HVKG, PESMO, MESMO, JESMO, and Sobol baselines [@renganathan2026qpotsdoe]. The method achieves the highest hypervolume on six of eight synthetic benchmarks and on all four real-world benchmarks tested, while reducing the total number of scalar oracle evaluations by approximately 30% on constrained problems (e.g., 692 versus 976 evaluations on the OSY benchmark) without sacrificing Pareto front quality. The repository provides tests, worked examples, hosted documentation at ReadTheDocs, and PyPI and source installation paths for users who want to reproduce the published workflows or adapt them to new expensive black-box optimization problems.

# AI usage disclosure

OpenAI ChatGPT and Anthropic Claude were used to assist with drafting and formatting this JOSS manuscript and with preparing supporting figure assets from author-provided papers. Generative AI was not used to design the `qPOTS` or `qPOTS-DOE` algorithms, to implement the software, or to generate any benchmark results. The authors are responsible for all technical claims in this manuscript; claims should be verified against the source code, hosted documentation, the AISTATS `qPOTS` paper [@renganathan2025qpots], and the `qPOTS-DOE` manuscript [@renganathan2026qpotsdoe].

# Acknowledgements

The authors acknowledge the Computational Complex Engineered Systems Design Laboratory at Penn State for supporting development of this software. The authors also acknowledge the Penn State Institute for Computational and Data Sciences for access to computational research infrastructure through the Roar Core Facility. The authors have no specific external grant funding to declare for this software publication.

# References
