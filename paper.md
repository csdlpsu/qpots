---
title: 'qPOTS: A Python Package for Sample-efficient Batch Constrained Multiobjective Bayesian Optimization'
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

`qPOTS` is a Python package for researchers and engineers who need to optimize expensive black-box systems with more than one competing objective. In such problems, each experiment, simulation, or hardware test can be costly, and the best answer is rarely a single design but instead a set of trade-off designs that form a Pareto front. Examples include aircraft design, process optimization, materials discovery, and machine learning model selection. `qPOTS` helps users decide which designs to evaluate next by combining Gaussian-process surrogate models, Thompson sampling, and evolutionary search over sampled Pareto fronts.

The package implements batch Pareto optimal Thompson sampling (qPOTS) for multiobjective Bayesian optimization [@thompson1933likelihood; @renganathan2025qpots]. It provides tools for unconstrained and constrained benchmark functions, Gaussian-process model fitting and prediction, batch candidate selection via a maximin diversity rule, expected hypervolume tracking, and comparison with standard multiobjective Bayesian optimization acquisition functions. The package also includes `qPOTS-Decoupled`, a multitask mode for decoupled oracle evaluations: when objectives or constraints can be measured separately, the software can transfer information across correlated tasks and select a subset of outputs to query [@bonilla2007multi; @alvarez2012kernels]. Earlier project material called this mode `qPOTS-DOE`; the new name avoids confusion with the established abbreviation for design of experiments.

`qPOTS` is released under the GNU General Public License v3.0 and is built on widely used scientific Python libraries including PyTorch, GPyTorch, BoTorch, and pymoo [@paszke2019pytorch; @gardner2018gpytorch; @balandat2020botorch; @blank2020pymoo].

# Statement of need

Multiobjective optimization is central to scientific and engineering design because improving one objective often degrades another. Classical evolutionary algorithms such as NSGA-II can approximate Pareto fronts accurately but require many objective evaluations [@deb2002fast]. That cost is prohibitive when each evaluation requires a high-fidelity computational simulation, a physical experiment, or a hardware test that may take hours or days. Bayesian optimization addresses this by fitting probabilistic surrogate models to existing data and using them to select the next most informative experiments [@jones1998efficient; @rasmussen2006gaussian; @shahriari2016taking].

Multiobjective Bayesian optimization is substantially harder than its single-objective counterpart. Acquisition functions must reason about a set-valued Pareto frontier rather than a scalar optimum, making them nonconvex, nondifferentiable, or expensive to optimize in batch settings. `qPOTS` addresses this by replacing direct acquisition-function maximization with posterior-sample optimization: it draws sample paths from the current surrogate model, solves a comparatively inexpensive multiobjective problem on those samples using evolutionary search, and selects a diverse batch of candidates from the resulting sampled Pareto set [@renganathan2025qpots]. This avoids the need for differentiable analytical acquisitions and naturally yields batches without additional approximation.

A further opportunity arises when the objectives and constraints of a problem can be observed independently. In aerodynamic design, for instance, drag and lift coefficients may be evaluated at multiple operating conditions that require separate solver runs; in chemical processes, different properties may require different laboratory analyses. Existing decoupled Bayesian optimization methods typically assume independent objective models or rely on known scalar oracle costs. `qPOTS-Decoupled` supports this setting by fitting a single multitask Gaussian process over all objectives and constraints jointly, using inter-task posterior correlation to decide whether decoupled evaluations are beneficial, and selecting an informative oracle subset via a Gaussian mutual-information criterion.

The target audience is researchers and engineers who need reproducible, sample-efficient multiobjective optimization workflows in Python. `qPOTS` is designed for method developers benchmarking against standard acquisition strategies, and for application researchers optimizing expensive custom black-box functions under limited evaluation budgets, including scenarios where not all objectives need to be queried at every iteration.

# State of the field

BoTorch provides a general platform for Bayesian optimization and implements strong multiobjective acquisitions such as qEHVI, qNEHVI, and their logarithmic variants [@daulton2020differentiable; @ament2023unexpected; @balandat2020botorch]. Other influential methods include entropy-search approaches (PESMO, MESMO, JESMO) and lookahead methods such as the hypervolume knowledge gradient [@garrido2019pesmoc; @belakaria2019max; @tu2022joint; @daulton2023hypervolume]. These are important baselines, and several are exposed in `qPOTS` workflows for direct comparison.

`qPOTS` complements rather than replaces BoTorch. BoTorch supplies Gaussian-process models, posterior sampling, and baseline acquisition functions, while `qPOTS` coordinates a cross-library workflow that is not represented by one BoTorch acquisition primitive: it optimizes sampled multiobjective paths with NSGA-II, selects a diverse batch from the sampled Pareto set, evaluates constrained or partially observed outputs, and records the resulting optimization history. Packaging that orchestration separately lets `qPOTS` retain direct interoperability with BoTorch and pymoo while providing a stable research workflow, benchmark registry, and high-level runner. Users can still access `ModelObject` and `Acquisition` directly when they need lower-level control.

The decoupled mode extends this software workflow to partially observed, correlated outputs. Its task-correlation gate and oracle-subset rule are described in the companion methods manuscript [@renganathan2026qpotsdoe]. This JOSS paper focuses on the implementation and user interfaces rather than presenting the method's derivation or making new benchmark-performance claims.

# Software design

The public API separates configuration, evaluation, modeling, candidate selection, and orchestration. `RuntimeConfig` controls tensor precision and device placement without requiring source edits, and typed `QPOTSConfig` objects validate optimization settings. The extensible `Function` interface handles input conversion and output-shape validation while delegating objective evaluation to a callable or subclass hook; `EvaluationResult` represents objectives and optional constraints explicitly, and a registry supplies the built-in benchmarks. `ModelObject` is responsible for independent or multitask Gaussian-process fitting, while `Acquisition` is responsible for candidate and optional task selection.

`QPOTSRunner` composes those components into complete `run()` and incremental `step()` workflows. It owns the optimization loop, works in physical design coordinates at the user boundary, and exposes normalized coordinates in result objects where the surrogate workflow needs them. Model and acquisition factories, runtime configuration, and iteration callbacks are injected through the runner rather than fixed inside it. This keeps each component independently replaceable and allows new benchmark functions, models, acquisitions, and reporting callbacks to be added without modifying the runner's control flow.

The design makes two computational trade-offs explicit. Posterior-sample optimization adds stochastic evolutionary-search overhead, but avoids requiring a differentiable analytical acquisition and naturally generates a sampled Pareto set. The package therefore delegates sampled-path search to NSGA-II through pymoo and retains BoTorch and GPyTorch for probabilistic modeling instead of reimplementing either dependency. Coupled and decoupled workflows share this candidate-generation path; decoupled runs add multitask modeling and optional task selection while representing skipped outputs as missing observations.

![Illustration of `qPOTS` on a constrained two-objective problem. Shown are the initial training data, the Pareto front approximation, and the batch of candidates proposed by the acquisition strategy. \label{fig:constrained}](assets/qpots_constrained_illustration.png){ width=85% }

![The `qPOTS-DOE` multitask update uses posterior task correlation to decide when decoupled evaluations are informative. The Branin-Currin example illustrates task-specific observations, total-correlation evolution, and uncertainty reduction from multitask information sharing. \label{fig:doe}](assets/qpots_doe_total_correlation.png){ width=100% }

# Research impact statement

`qPOTS` is the reference implementation for the AISTATS 2025 paper introducing Pareto optimal Thompson sampling for batch multiobjective Bayesian optimization [@renganathan2025qpots]. The method was evaluated on eight synthetic and real-world benchmarks against qEHVI, qNEHVI, qLogNEHVI, ParEGO, PESMO, MESMO, JESMO, TS-EMO, and Sobol baselines, demonstrating competitive or superior hypervolume improvements with substantially lower computational cost per iteration compared to Monte Carlo acquisition methods. The package has also supported an applied aerodynamic design study of the NASA Common Research Model, where `qPOTS` was deployed for 24-dimensional, two-objective aerodynamic design optimization and released with open-source reproducibility materials [@carlson2026multiobjective].

The `qPOTS-DOE` extension further expands the implementation to correlated objective and constraint tasks with partial observations. In benchmark experiments, `qPOTS-DOE` is compared against coupled `qPOTS`, qLogNEHVI, HVKG, PESMO, MESMO, JESMO, and Sobol baselines. The method achieves the highest hypervolume on six of eight synthetic benchmarks and on all four real-world benchmarks tested, while reducing the total number of scalar oracle evaluations by approximately 30% on constrained problems (e.g., 692 versus 976 evaluations on the OSY benchmark) without sacrificing Pareto front quality. The repository provides tests, worked examples, hosted documentation at ReadTheDocs, and PyPI and source installation paths for users who want to reproduce the published workflows or adapt them to new expensive black-box optimization problems.

The broader research community has also begun to use `qPOTS` as a reference method and benchmark baseline. Recent work has compared against or built on Pareto optimal Thompson sampling in applications including generative molecular design, sustainable process systems, automated class-imbalance learning, diffusion-based Pareto-front refinement, and many-objective Bayesian optimization [@muthyala2025generative; @kudva2026multi; @wang2025automated; @hotegni2025spread; @jiang2026we]. These independent uses indicate that the code is serving not only as a reproducibility artifact for the original paper, but also as shared infrastructure for evaluating new multiobjective Bayesian optimization methods.

# AI usage disclosure

Anthropic Claude was used to assist with refactoring the code and structuring the Readme.md before submission. Generative AI was not used to design the `qPOTS` or `qPOTS-DOE` algorithms, to implement the software, or to generate any benchmark results. The authors are responsible for all technical claims in this manuscript; claims should be verified against the source code, hosted documentation, the AISTATS `qPOTS` paper [@renganathan2025qpots], and the `qPOTS-DOE` manuscript (soon to be posted on arXiv).

# Acknowledgements

The authors acknowledge the Computational Complex Engineered Systems Design Laboratory at Penn State for supporting development of this software. The authors also acknowledge the Penn State Institute for Computational and Data Sciences for access to computational research infrastructure through the Roar Core Facility. The authors have no specific external grant funding to declare for this software publication.

# References
