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
  - name: Ashwin Renganathan
    orcid: 0000-0001-6948-6932
    affiliation: "1, 2"
  - name: Peter E. Bachman
    affiliation: "1"
  - name: Kade E. Carlson
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

Designing engineered systems involves optimizing the system for some performance objective under several design constraints. In practice, though, there is typically more than one objective to meet and the objectives may conflict with each other. For instance, an aircraft designer may need to reduce the aerodynamic drag on the aircraft while ensuring passenger safety.
Aerodynamic efficiency typically leads to slender aircraft cross-sections which can, however, compromise the safety of the aircraft under gusty conditions. Similar tradeoffs occur in chemical process optimization, materials discovery, and machine learning hyperparameter optimization. Optimizing simultaneously for more than one objective is called multiobjective optimization, and the solution to such problems is no longer a single point but a set of points that offer a pragmatic compromise between the objectives.

`qPOTS` is a Python package that solves constrained multiobjective optimization when there is no derivative information and every query of the objectives and constraints is expensive, thus necessitating "sample efficiency" (i.e., solving the problem with as few evaluations of objectives and constraints as possible).  The package implements batch Pareto optimal Thompson sampling (qPOTS), a Bayesian optimization (BO) method [@frazier2018tutorial; @garnett2023bayesian; @renganathan2025qpots]. The method uses Thompson sampling [@thompson1933likelihood], and the prefix $q$ denotes the number of design points selected in each batch. The package also includes `qPOTS-Decoupled`, which supports decoupled objective and constraint evaluations, that is, when objectives and constraints can be evaluated independently by computer models using separate resources. In this setting, the algorithm improves sample efficiency by evaluating only a subset of objectives and constraints during each design iteration [@renganathan2026qpotsdoe]. The cited manuscript calls this method qPOTS-DOE, where DOE stands for decoupled oracle evaluations.

`qPOTS` is released under the GNU General Public License v3.0 and is built on widely used scientific Python libraries including `PyTorch`, `GPyTorch`, `BoTorch`, and `Pymoo` [@paszke2019pytorch; @gardner2018gpytorch; @balandat2020botorch; @blank2020pymoo].

# Statement of need

The solution to a multiobjective optimization problem is called the "Pareto set", and its image in the objective space is called the "Pareto frontier".  Computing the Pareto set/frontier is nontrivial. When derivative information is available and the objectives and constraints are smooth and not expensive to evaluate, one may use gradient-based optimization techniques over several scalarized versions of the objectives [@fliege2016constrained; @fliege2009newton]. However, in real-world problems, good quality derivatives are rarely available, objective/constraint smoothness is not guaranteed, and the Pareto frontier may have a complex topology (disconnected, nonconvex, etc.). In such situations, multiobjective Bayesian optimization (MOBO) serves as a suitable alternative. MOBO learns global surrogate models for the objectives and constraints, and leverages Bayesian decision theory to generate a sequence of design points (iterates) that balance exploration and exploitation of the design space. This involves constructing an "acquisition" function that quantifies the utility of a candidate iterate, and solving an "inner" optimization problem to determine the iterates. This inner optimization depends only on the surrogate model and is independent of the expensive objectives and constraints.

MOBO is an active area of research; there are several existing works that innovate on the Bayesian decision-theoretic acquisition function [@belakaria2019max; @tu2022joint; @daulton2020differentiable], or use scalarization to fit into (single objective) BO [@knowles2006parego], or modify existing approaches to promote diversity in the iterates [@konakovic2020diversity], among others. However, existing methods have one or more of the following limitations: (1) they require nontrivial inner acquisition function optimization due to nonconvexity and/or stochasticity, which ultimately affects the accuracy of the predicted Pareto set/frontier; (2) they do not support batch sampling (generating a batch of iterates); or (3) they cannot handle nonlinear constraints. See @renganathan2025qpots for further details. `qPOTS` was introduced to address all of these limitations, with improved sample efficiency and solution accuracy being our overarching goal.

Evolutionary algorithms [@deb2002fast] are quite popular for derivative-free multiobjective optimization. However, they rely on generating large populations of the objectives that must be mutated and crossed-over multiple times. Although this leads to accurate solutions, it can penalize sample efficiency. `qPOTS` combines the sample efficiency of BO with the accuracy of evolutionary algorithms [@renganathan2025qpots]. By fitting Gaussian process (GP) surrogates [@rasmussen2006gaussian] for the objectives and constraints, `qPOTS` replaces the conventional acquisition function optimization in MOBO with a cheap inner multiobjective optimization over the GP posterior sample paths, leveraging evolutionary algorithms. Nonlinear constraints are handled as part of the Gaussian process Thompson sampling [@renganathan2025qpots].

Thompson sampling approaches similar to `qPOTS` already exist for MOBO, including `TS-EMO` [@bradford2018efficient] and `TS-TCH` [@paria2020flexible].
The main difference between `qPOTS` and `TS-EMO` is that `TS-EMO` generates GP posterior Thompson samples, but still has to compute the maximum hypervolume improvement over them to choose the candidate — this inner optimization problem can be difficult to solve [@renganathan2025qpots]. Additionally, `TS-EMO` is unable to handle constraints. `TS-TCH` scalarizes the Thompson samples to leverage single-objective acquisition functions. This inherits the limitations of scalarized approaches in computing disconnected and nonconvex Pareto frontiers.

A further opportunity arises when the objectives and constraints can be evaluated asynchronously using independent compute resources. For example, in aerodynamic design, objectives may be aircraft performance at different operating conditions that may be evaluated separately. In such situations, if the objectives and constraints are correlated with each other, MOBO can proceed with only a subset of objective and constraint evaluations at each iterate. The `qPOTS` package includes `qPOTS-Decoupled` [@renganathan2026qpotsdoe] to address this, which combines the `qPOTS` posterior sampling workflow with a multitask Gaussian process (MTGP) [@bonilla2007multi] that jointly learns all objectives and constraints. MTGPs learn the correlations across the objectives and constraints, which can be exploited to determine which iterates warrant a decoupled evaluation. Decoupled MOBO methods such as PESMOC and HVKG can already progress with partial information [@garrido2019pesmoc; @daulton2023hypervolume]. However, existing methods depend on differences in evaluation cost across the objectives and constraints to determine decoupling. In contrast, `qPOTS-Decoupled` can decouple evaluations under uniform evaluation costs; differences in evaluation cost will further improve the sample efficiency of `qPOTS-Decoupled`.

`qPOTS` builds on `GPyTorch` [@gardner2018gpytorch] and existing GP and MTGP wrappers in `BoTorch` [@balandat2020botorch] for surrogate modeling, `Pymoo` [@blank2020pymoo] for evolutionary algorithms, and fully supports GPU acceleration. Even though `qPOTS` builds on these existing libraries, there is a compelling case for a standalone package for the following reasons. First, `qPOTS` is anchored on Pareto optimal Thompson sampling — this makes it compatible with probabilistic models beyond GPs such as Bayesian neural networks and variational autoencoders. Future versions of the package could therefore support these alternative surrogate models. Second, generative models may be integrated into the framework to learn the distribution of the Pareto set in Thompson sampling. Third, `qPOTS-Decoupled` can be potentially applied to other problems such as optimal experiment design and sensor placement, where optimal selection of a subset from a correlated set of tasks is of interest. Therefore, a standalone package will promote wider usage, contributions, and extensions that might not be possible when integrated into another package.

The target audience is researchers and engineers who need a sample-efficient derivative-free multiobjective optimization software package in Python. `qPOTS` is designed for method developers benchmarking against standard acquisition strategies, and for application researchers optimizing expensive custom black-box functions under limited evaluation budgets.

# State of the field

While multiobjective optimization problems can be solved via a variety of approaches including gradient-based approaches and evolutionary algorithms, we only review libraries relevant to MOBO here.
`BoTorch` provides a PyTorch-based platform for Bayesian optimization (including MOBO) and already implements MOBO acquisition functions such as qNEHVI, qNParEGO, and their logarithmic variants [@daulton2020differentiable; @ament2023unexpected; @balandat2020botorch], entropy-search acquisitions (PESMOC, MESMO, and JESMO) [@garrido2019pesmoc; @belakaria2019max; @tu2022joint], and hypervolume knowledge gradient [@daulton2023hypervolume]. `Trieste` provides a modular TensorFlow-based Bayesian-optimization loop with multiobjective and constrained workflows [@picheny2023trieste]. `TS-EMO` [@bradford2018efficient] is another Thompson sampling based MOBO algorithm (similar to `qPOTS`) that is based on MATLAB. `qPOTS` differs methodologically from these libraries, as explained in the previous section and in more detail in @renganathan2025qpots, and has demonstrated better empirical performance in the reported comparisons [@renganathan2025qpots].

# Software design

`qPOTS` coordinates the modeling, search, and evaluation steps of its optimization algorithms. Each iteration must keep design points in physical coordinates consistent with the normalized tensors used by the surrogate models and the candidate populations used by `Pymoo`. In decoupled runs, it must also track which objectives and constraints are evaluated at each point. The `QPOTSRunner` class manages this sequence: fit the surrogate model, search its posterior sample paths, select a batch of points and outputs to evaluate, evaluate them, and update the observations.

The runner exposes model and acquisition objects and accepts callbacks that are invoked during optimization. These interfaces allow users to compare surrogate models, change candidate selection policies, and connect external simulators without copying the optimization loop. This flexibility introduces additional layers of software and more interfaces for users to understand. Default settings, configuration validation, results with explicit types, and a fixed sequence of optimization steps help manage that complexity. Users can call `run()` to execute an optimization run or `step()` to advance one iteration under the control of an external system.

Coupled and decoupled modes can both use MTGPs and share the same representation of the optimization state. Task identifiers record which outputs are requested, and unevaluated outputs are represented as missing observations. This avoids maintaining separate implementations, but requires stricter validation and a surrogate model that can handle missing data. The package therefore coordinates more than a `BoTorch` acquisition function: it manages posterior path construction, Pareto set search, batch and output selection, and updates from partial observations across its dependencies.

`BoTorch` and `GPyTorch` provide models, posterior sampling, and baseline acquisition functions that operate on tensors [@balandat2020botorch; @gardner2018gpytorch]. Reimplementing these components would add numerical code to maintain without contributing functionality specific to `qPOTS`. `Pymoo` supports bounded optimization problems with vectorized evaluations, seeded execution, nondominated solution sets, and callbacks, making it suitable for the inner multiobjective optimization [@blank2020pymoo]. Alternatives include jMetalPy, pagmo/pygmo, and DEAP [@benitezhidalgo2019jmetalpy; @biscani2020pagmo; @fortin2012deap]. Using `Pymoo` adds a dependency and requires conversion between its CPU-based NumPy arrays and the PyTorch tensors used by the surrogate models. The package keeps these conversions in an adapter so that replacing the optimizer would require fewer changes elsewhere. We did not use a file-based workflow system such as Snakemake [@koster2012snakemake] because each new design, and sometimes the choice of outputs to evaluate, depends on a newly fitted posterior model held in memory. `QPOTSRunner` does not provide file-based checkpointing, scheduling, or provenance tracking; external systems can provide these capabilities by calling `step()` and using callbacks.

# Research impact statement

`qPOTS` is the reference implementation for the AISTATS 2025 paper introducing Pareto optimal Thompson sampling for batch multiobjective Bayesian optimization [@renganathan2025qpots]. That paper evaluates the method on synthetic and real-world benchmarks against analytical, entropy-search, Thompson-sampling, and random baselines. The software has also supported an aerodynamic design study of the NASA Common Research Model with open reproducibility materials [@carlson2026multiobjective]. These studies demonstrate research applications of the method and the code on which this package is based.

Independent evaluation is beginning to emerge: SPREAD includes `qPOTS` in a numerical comparison of MOBO methods [@hotegni2025spread]. Other publications cite qPOTS while discussing generative molecular design, sustainable process systems, class-imbalance learning, and many-objective optimization [@muthyala2025generative; @kudva2026multi, Ch. 3; @wang2025automated; @jiang2026we]. We treat those citations as evidence both of awareness of `qPOTS` and of its use as a state-of-the-art benchmark in the MOBO community.

# AI usage disclosure

Anthropic Claude assisted with code refactoring and README structure before submission. OpenAI Codex assisted during review revisions with drafting tests, examples, and documentation, and with manuscript language and citation corrections. Generative AI was not used to design the `qPOTS` or `qPOTS-Decoupled` algorithms or to generate benchmark results. The authors reviewed every change, reran the software tests and documented examples, checked citations against publication records, and remain responsible for the manuscript and software.

# Acknowledgements

The authors acknowledge the Penn State Institute for Computational and Data Sciences for access to computational research infrastructure through the Roar Core Facility.

# References
