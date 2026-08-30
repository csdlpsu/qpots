# Draft response to issue #5

> Local draft for author review. Do not post to GitHub yet.

Thank you for the detailed report and reproductions. We confirmed all four defects and agree
with the diagnosis that the constraint-handling branches lacked direct regression coverage.
We have addressed each point as follows.

## 1. Constrained multitask runs crash

Confirmed. The multitask posterior sample has a leading sample dimension, and squeezing the
feasibility mask removed that dimension before indexing the sampled values.

We replaced the local indexing block with a shared, broadcasting-safe constraint helper. It
retains arbitrary leading sample or batch dimensions, requires all constraint outputs to be
nonnegative, and applies the acquisition penalty only to the objective columns.

The exact constrained welded-beam multitask reproduction now completes. With 12 initial
points, one iteration, and a batch size of two, the current runner returns `train_y` with shape
`(14, 6)`.

## 2. Nyström constraint handling uses the wrong sign and shape

Confirmed. The Nyström path used `<= 0` and did not reduce feasibility across all constraint
columns.

This path now uses the same shared helper as the other posterior-sampling paths. Feasibility is
defined consistently as every constraint being `>= 0`, and the mask has one entry per sampled
point. A deterministic regression with two constraints verifies the sign, reduction, and exact
objective penalty. The constrained welded-beam Nyström reproduction also completes with
`train_y` shape `(14, 6)`.

This defect did not affect the previously generated multitask comparison runs: their recorded
configurations set `nystrom=0`, so candidate generation never entered `_nystrom_approx`. The
defect was exposed by the reviewer's new constrained Nyström reproduction, not by the
historical experiment path.

## 3. Constrained posterior-mean hypervolume includes infeasible points

Confirmed. The old implementation wrote the penalty into the constraint columns and then
discarded those columns, leaving the infeasible objective values unchanged.

The reporting function now filters infeasible rows before extracting objective columns and
computing the nondominated set. Filtering, rather than applying a fixed objective penalty, also
makes the function correct for both maximization and minimization. Regression tests cover both
directions and the case where no feasible points exist, which returns zero hypervolume.

The historical constrained drivers kept posterior-mean hypervolume and ground-truth benchmark
hypervolume in separate files. Candidate generation did not consume this reporting helper, and
the per-iteration metric printed as `HV` was computed from complete benchmark outputs by
`compute_true_hypervolume`, which filters feasible rows independently. We can therefore justify
no impact on candidate locations or the saved `*_true_hv.npy` series. Before extending that
statement to any published plot, we will verify that the plot consumed the `*_true_hv.npy`
artifact; if it used the diagnostic `*_hv.npy` series, it must instead be regenerated and the
numerical change disclosed.

## 4. `posterior_mean_fill` fails outside task column 0

Confirmed. A posterior queried for one explicit task has a final output dimension of size one,
so indexing that dimension with the original task number fails for every task after task 0.

The assignment now removes only the singleton output dimension with `squeeze(-1)` and assigns
the resulting vector to the requested original column. The multitask fixture now deliberately
places missing values in task column 1, and the tests verify that all missing values are filled
without changing observed values.

## Shared cause and regression coverage

We centralized the feasibility convention and acquisition penalty in an internal constraint
utility. The independent-GP, multitask-GP, Nyström, and total-correlation posterior paths now use
the same implementation. The penalty helper returns a copy, preserves constraint values, and
supports both two-dimensional point sets and tensors with leading sample dimensions.

Validation completed:

- 12 focused constraint regressions pass.
- The constrained multitask reviewer reproduction completes.
- The constrained Nyström reviewer reproduction completes.
- The full test suite passes: 197 tests, including nine new regression cases.
- Ruff lint and formatting checks pass.
