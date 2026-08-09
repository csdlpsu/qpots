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
date: 9 August 2026
bibliography: paper.bib
---

# Summary

An aircraft engineer may need to reduce both drag and structural weight, even though improving one can worsen the other and each high-fidelity simulation may take hours. Similar trade-offs occur in process optimization, materials discovery, and machine-learning system design. The useful result is therefore usually a set of alternatives, called a Pareto front, rather than one optimum. `qPOTS` is a Python package that helps researchers and engineers decide which expensive designs to evaluate next.

The package implements batch Pareto optimal Thompson sampling (qPOTS) for multiobjective Bayesian optimization [@thompson1933likelihood; @renganathan2025qpots]. It provides tools for unconstrained and constrained benchmark functions, Gaussian-process model fitting and prediction, batch candidate selection via a maximin diversity rule, expected hypervolume tracking, and comparison with standard multiobjective Bayesian optimization acquisition functions. The package also includes `qPOTS-Decoupled`, a multitask mode for decoupled oracle evaluations: when objectives or constraints can be measured separately, the software can transfer information across correlated tasks and select a subset of outputs to query [@bonilla2007multi; @alvarez2012kernels]. Earlier project material called this mode `qPOTS-DOE`; the new name avoids confusion with the established abbreviation for design of experiments.

`qPOTS` is released under the GNU General Public License v3.0 and is built on widely used scientific Python libraries including PyTorch, GPyTorch, BoTorch, and pymoo [@paszke2019pytorch; @gardner2018gpytorch; @balandat2020botorch; @blank2020pymoo].

# Statement of need

Multiobjective optimization is central to scientific and engineering design because improving one objective often degrades another. Classical evolutionary algorithms such as NSGA-II can approximate Pareto fronts accurately but require many objective evaluations [@deb2002fast]. That cost is prohibitive when each evaluation requires a high-fidelity computational simulation, a physical experiment, or a hardware test that may take hours or days. Bayesian optimization addresses this by fitting probabilistic surrogate models to existing data and using them to select the next most informative experiments [@jones1998efficient; @rasmussen2006gaussian; @shahriari2016taking].

Multiobjective Bayesian optimization is substantially harder than its single-objective counterpart. Acquisition functions must reason about a set-valued Pareto frontier rather than a scalar optimum, making them nonconvex, nondifferentiable, or expensive to optimize in batch settings. `qPOTS` addresses this by replacing direct acquisition-function maximization with posterior-sample optimization: it draws sample paths from the current surrogate model, solves a comparatively inexpensive multiobjective problem on those samples using evolutionary search, and selects a diverse batch of candidates from the resulting sampled Pareto set [@renganathan2025qpots]. This avoids the need for differentiable analytical acquisitions and naturally yields batches without additional approximation.

A further opportunity arises when the objectives and constraints can be observed separately. In aerodynamic design, coefficients at different operating conditions may require separate solver runs; in chemical processes, properties may require different laboratory analyses. Decoupled methods already include predictive-entropy approaches such as PESMOC and lookahead approaches such as hypervolume knowledge gradient, which can allocate partial information [@garrido2019pesmoc; @daulton2023hypervolume]. `qPOTS-Decoupled` instead combines the qPOTS posterior-sample workflow with a multitask Gaussian process and an output-selection rule based on posterior task dependence. A multitask model can reduce uncertainty for an unqueried output when correlations are supported by the data. It also costs more to fit and can transfer misleading information when outputs are weakly related; independent models remain the safer choice in that case.

The target audience is researchers and engineers who need reproducible, sample-efficient multiobjective optimization workflows in Python. `qPOTS` is designed for method developers benchmarking against standard acquisition strategies, and for application researchers optimizing expensive custom black-box functions under limited evaluation budgets, including scenarios where not all objectives need to be queried at every iteration.

# State of the field

BoTorch provides a PyTorch platform for Bayesian optimization and implements multiobjective acquisitions such as qEHVI, qNEHVI, and their logarithmic variants [@daulton2020differentiable; @ament2023unexpected; @balandat2020botorch]. Trieste provides a modular TensorFlow-based Bayesian-optimization loop with multiobjective and constrained workflows [@picheny2023trieste]. Method-specific alternatives include entropy-search acquisitions (PESMOC, MESMO, and JESMO), hypervolume knowledge gradient for partial information, and TS-EMO, which also combines Gaussian-process sampling with evolutionary multiobjective search [@garrido2019pesmoc; @belakaria2019max; @tu2022joint; @daulton2023hypervolume; @bradford2018efficient].

`qPOTS` complements these libraries rather than replacing them. BoTorch and GPyTorch supply probabilistic models and baseline acquisitions, while pymoo supplies constrained evolutionary search [@blank2020pymoo]. qPOTS contributes an opinionated reference workflow for the qPOTS method: optimize sampled multioutput paths with NSGA-II, select a diverse batch from the sampled Pareto set, evaluate coupled or partially observed outputs, and retain typed iteration history. A separate package keeps that method-specific orchestration reproducible without requiring it to become a general BoTorch primitive. Users can still access `ModelObject` and `Acquisition` when they need lower-level control.

The decoupled mode extends this software workflow to partially observed, correlated outputs. Its task-correlation gate and oracle-subset rule are described in the companion methods manuscript [@renganathan2026qpotsdoe]. This JOSS paper focuses on the implementation and user interfaces rather than presenting the method's derivation or making new benchmark-performance claims.

# Software design

The public API separates evaluation, modeling, candidate selection, and orchestration so each boundary can be tested and replaced independently. `Function` and `EvaluationResult` validate the contract with a benchmark, simulator, or experiment; `ModelObject` owns independent or multitask Gaussian-process fitting; and `Acquisition` owns candidate and optional task selection. `RuntimeConfig` centralizes precision and device placement, while `QPOTSConfig` validates the settings passed between these components.

`QPOTSRunner` composes the components into complete `run()` and incremental `step()` workflows. Injected model and acquisition factories make surrogate experiments, comparator implementations, and isolated tests possible without forking the loop. This flexibility adds interface surface compared with constructing fixed classes inside `run()`, but the fixed design was rejected because it would couple the reference algorithm to one model and make research comparisons harder to reproduce. Callbacks expose progress without placing plotting or storage policy inside the optimizer.

The package delegates sampled-path search to pymoo because it provides a maintained Python interface to constrained multiobjective evolutionary algorithms and exposes the resulting population needed by qPOTS. Libraries such as jMetalPy, pygmo, and DEAP could supply evolutionary components, but switching would add an adapter and another behavioral dependency without replacing the BoTorch-facing parts of the workflow. The runner is an in-process state machine because every iteration exchanges tensors and fitted models; file-oriented workflow systems can still invoke it, but do not replace this per-iteration control flow. Figure \ref{fig:constrained} connects these choices in a constrained run.

![Illustration of `qPOTS` on a constrained two-objective problem. Shown are the initial training data, the Pareto front approximation, and the batch of candidates proposed by the acquisition strategy. \label{fig:constrained}](assets/qpots_constrained_illustration.png){ width=85% }

Figure \ref{fig:doe} illustrates the additional information available to a decoupled run: task-specific observations produce a multitask posterior whose cross-task correlation can guide output selection.

![The `qPOTS-Decoupled` multitask update uses posterior task correlation to decide when decoupled evaluations are informative. The Branin-Currin example illustrates task-specific observations, total-correlation evolution, and uncertainty reduction from multitask information sharing. \label{fig:doe}](assets/qpots_doe_total_correlation.png){ width=100% }

# Research impact statement

`qPOTS` is the reference implementation for the AISTATS 2025 paper introducing Pareto optimal Thompson sampling for batch multiobjective Bayesian optimization [@renganathan2025qpots]. That paper evaluates the method on synthetic and real-world benchmarks against analytical, entropy-search, Thompson-sampling, and random baselines. The software has also supported a 24-dimensional, two-objective aerodynamic design study of the NASA Common Research Model with open reproducibility materials [@carlson2026multiobjective]. These are demonstrated uses of the method and its research code lineage; they do not by themselves establish broad adoption of the current packaged release.

The current release turns that research implementation into a tested Python package with a stable runner, documented custom-function boundary, constrained and partially observed workflows, and PyPI and source installation paths. The companion manuscript describes the decoupled algorithm and remains under review [@renganathan2026qpotsdoe]; its results are not evidence of external software adoption and are not repeated as new claims here.

Independent methodological evaluation is beginning to emerge: the SPREAD preprint includes qPOTS in a numerical comparison [@hotegni2025spread]. Publications in molecular design, sustainable process systems, class-imbalance learning, and many-objective optimization cite the qPOTS method [@muthyala2025generative; @kudva2026multi; @wang2025automated; @jiang2026we]. Because several predate the packaged repository or do not report using it, we treat them as evidence of awareness of the method, not adoption of this software release.

# AI usage disclosure

Anthropic Claude assisted with code refactoring and README structure before submission. OpenAI Codex assisted during review revisions with drafting tests, examples, documentation, and manuscript edits, and with locating bibliographic metadata. Generative AI was not used to design the qPOTS or qPOTS-Decoupled algorithms or to generate benchmark results. The authors reviewed every change, reran the software tests and documented examples, checked citations against publication records, and remain responsible for the manuscript and software.

# Acknowledgements

The authors acknowledge the Penn State Institute for Computational and Data Sciences for access to computational research infrastructure through the Roar Core Facility.

# References
