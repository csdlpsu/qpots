# Draft response to JOSS reviewer 2 (issue #6)

> Local author-review draft. Do not commit, push, or post to GitHub yet.

Thank you again for the detailed review and for providing executable reproductions.
We addressed the blocking correctness report first, then revised the user
documentation, examples, software packaging, and manuscript as detailed below.

## Blocking items

### 1. Constrained multitask optimization

Confirmed and fixed in the issue #5 changeset. Constraint feasibility and
objective penalties now share one broadcasting-safe implementation across the
independent, multitask, Nyström, and total-correlation posterior paths. A row is
feasible only when every constraint is nonnegative, and only objective columns
receive the acquisition penalty. The reviewer-sized constrained MTGP and
Nyström runs now complete with output shape `(14, 6)`. The separate local
`REVIEW_RESPONSE_ISSUE_5.md` contains the point-by-point technical response.

We also audited impact rather than inferring it from the fix. The historical
constrained experiment drivers recorded two series: a posterior-mean diagnostic
(`*_hv.npy`) and ground-truth performance from complete benchmark evaluations
(`*_true_hv.npy`). Candidate selection did not consume the reporting helper
fixed in issue #5. The reported loop output and `true_hv` series use
`compute_true_hypervolume`, which filters constraints independently of the
defective posterior-mean helper. Before making an unconditional no-impact claim
for any published figure, we will additionally match that figure's plotting
input to the archived `*_true_hv.npy` artifacts. If a figure used the diagnostic
series, we will regenerate it and report the difference instead.


## Paper comments
We did an almost complete re-write of the paper. We initially left out some key details since they were already documented in our AISTATS paper, but we decided to bring it back in per the reviewer comments. We agree that the paper.md now reads more self-contained.

### Summary

re-written in plain language.

### Statement of need

- substantially re-written
- we also call out similar methods (TS-EMO, TS-TCH) and articulate their differences from qPOTS.

### State of the field

- substantially re-written
- we now frame the discussion around other MOBO libraries and establish the need for a standalone qPOTS

### Software design

- substantially re-written
- removed items that are not directly relevant to section

### Research impact

- re-written around community usage and evidence of awareness of our library to the community.
- on citing the 2023 version of qPOTS- this was potentially because the first version of the qPOTS paper was posted to arXiv in 2023. The peer-reviewed (AISTATS) version was published in 2025 and the software package released a little later. However, the AISTATS version had LaTeX typesetting for "qPOTS' in the title -- this somehow created a separate BibTeX entry on Google Scholar as opposed to merging with the 2023 one. Therefore, other researchers ended up citing the 2023 paper.
- However, we don't think this needs to be clarified in the paper because we link the main AISTATS version on the repository, and more recent citations to the paper correctly point to the 2025 version. Finally, we also believe that in the future, users might directly cite the JOSS version when they want to cite the package specifically.

### AI disclosure

Updated the disclosure to include OpenAI Codex assistance during review
revisions, described the human verification performed.

## Software and documentation comments

### CPU-only tests

Added a session-scoped test fixture that forces the qPOTS default runtime to
CPU/float64 and restores the original configuration after the suite. The
installation guide now states explicitly that CI is CPU-only and does not claim
a CUDA CI job.

### Constrained tutorial figure

The tutorial documents that posterior sampling and evolutionary search are
stochastic across hardware and dependency versions, supplies the complete
seeded reproduction command, and supplies a reduced CPU command. It filters
infeasible rows before computing the Pareto front.

### qPOTS-Decoupled publication links and Figure 2

We decided to remove this figure. Once the qPOTS-Decoupled paper is public, it will be linked in the Readme.md so people can look at that. Currently, we felt it was not adding much value to the paper.md.

### Broken constrained examples

The Welded Beam and OSY programs now include reviewer-sized `--quick` modes.
Both were rerun against the corrected checkout. Documentation states the output
ordering, all-constraints feasibility rule, and meaning of `NaN` for unqueried
tasks.

### Custom engineering workflow and optimization quality

The former abstract custom function was replaced with a constrained cantilever
beam design. It demonstrates minimization through negated objectives,
nonnegative stress slack, physical bounds, `EvaluationResult`, a full runner
configuration, feasible Pareto filtering, and the boundary for an external
simulator. Its figure shows the feasible front and hypervolume evolution.

## Third-party licensing

We agreed that the existing bundled licenses did not establish a sufficiently
clear redistribution basis for all TS-EMO components. The complete vendored
MATLAB tree, including DIRECT and hypervolume binaries/sources, has therefore
been removed from qPOTS distributions. `TSEMORunner` remains as an optional
interoperability layer, validates a separately obtained checkout supplied with
`tsemo_path`, and does not copy or redistribute it. Packaging tests assert the
legacy tree is absent, and `THIRD_PARTY_NOTICES.md` records the policy.

## Validation summary

- Constrained MTGP quick reproduction: passes, output shape `(14, 6)`.
- Constrained decoupled OSY quick reproduction: passes and reports selected
  scalar tasks.
- Custom cantilever quick workflow: passes and generates its front/HV figure.
- Full pytest suite: 206 tests pass.
- Ruff lint and formatting checks pass.
- Sphinx builds 22 pages with warnings treated as errors.
- A clean wheel and source distribution build passes `twine check` and the
  distribution verifier; an isolated install confirms that qPOTS imports and
  that no vendored TS-EMO tree is present.
