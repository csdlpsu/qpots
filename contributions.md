# Author Contributions

Author contributions are reported using the [CRediT (Contributor Roles Taxonomy)](https://credit.niso.org/) framework.

## Kade E. Carlson

- **Conceptualization**: Co-developed the `qPOTS-DOE` extension, including the decoupled evaluation framework and oracle-subset selection strategy.
- **Software**: Primary developer of the `qPOTS` package codebase, including the acquisition module, model-fitting infrastructure, benchmark functions, test suite, and CI/CD configuration. Implemented the multitask Gaussian process integration and partial-observation utilities for `qPOTS-DOE`.
- **Validation**: Designed and executed the numerical benchmark experiments for both `qPOTS` (AISTATS 2025) and `qPOTS-DOE`, including comparisons against qEHVI, qLogNEHVI, HVKG, PESMO, MESMO, JESMO, TS-EMO, and Sobol baselines.
- **Investigation**: Conducted the applied aerodynamic design optimization study of the NASA Common Research Model using `qPOTS`.
- **Data Curation**: Prepared benchmark function implementations, ground-truth Pareto fronts, and reproducibility scripts for released experiments.
- **Writing – Original Draft**: Contributed to drafting the AISTATS 2025 paper, the `qPOTS-DOE` manuscript, and this JOSS paper.
- **Writing – Review & Editing**: Reviewed and edited all manuscripts.
- **Visualization**: Produced figures for the AISTATS paper, the `qPOTS-DOE` manuscript, and the JOSS paper.

## Ashwin Renganathan

- **Conceptualization**: Originated and led the design of the Pareto optimal Thompson sampling methodology for `qPOTS` and the decoupled multitask framework for `qPOTS-DOE`, including the total-correlation gate and the Gaussian mutual-information oracle-subset rule.
- **Methodology**: Developed the theoretical foundations of `qPOTS` and `qPOTS-DOE`, including the asymptotic consistency result (Theorem 4.1) and the mutual-information optimality result (Theorem 4.2) for decoupled evaluations.
- **Software**: Contributed to the design of the package architecture, the sample-path optimization workflow, and the multitask acquisition logic.
- **Supervision**: Supervised the development of the `qPOTS` package, the benchmark studies, and the applied aerodynamic design study.
- **Writing – Original Draft**: Drafted the AISTATS 2025 paper, the `qPOTS-DOE` manuscript, and this JOSS paper.
- **Writing – Review & Editing**: Reviewed and edited all manuscripts.
- **Project Administration**: Coordinated the research timeline, repository organization, and JOSS submission.
- **Funding Acquisition**: Secured computational resources through the Penn State Institute for Computational and Data Sciences.

## Peter E. Bachman

- **Conceptualization**: Contributed to the design and scope of the `qPOTS-DOE` decoupled evaluation extension.
- **Software**: Contributed to the implementation of the `qPOTS-DOE` extension, including components of the oracle-subset selection and multitask model integration.
- **Validation**: Participated in benchmark evaluation and analysis for the `qPOTS-DOE` manuscript.
- **Writing – Original Draft**: Contributed to drafting the `qPOTS-DOE` manuscript and this JOSS paper.
- **Writing – Review & Editing**: Reviewed and edited manuscripts.
