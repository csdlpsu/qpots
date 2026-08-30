# Changelog

## Unreleased

## v2.2.0

- Fixes constraint handling for multitask and Nyström posterior sampling,
  constrained posterior-mean hypervolume reporting, and multitask missing-value
  imputation.
- Adds reproducible constrained, decoupled, and custom engineering examples,
  including feasible-front and hypervolume visualizations.
- Adds Python 3.14 to the supported test matrix and runs the documented
  constrained workflows against the built wheel outside the source checkout.
- Stabilizes total-correlation calculations without producing negative values
  for independent output tasks.
- Pins the test runtime to CPU and replaces the bundled TS-EMO sources with an
  optional external-checkout integration through ``tsemo_path``.

## v2.1.0

- Adds injectable runtime configuration, extensible function evaluation, and a typed high-level optimization runner.
- Declares optional example dependencies and improves acquisition-code readability and style checks.
- Consolidates dependency metadata in `pyproject.toml` and modernizes package license metadata.
- Tests Python 3.11, 3.12, and 3.13, including a monthly compatibility run.
- Automates validated PyPI publishing and GitHub Releases from semantic version tags.
- Uses an absolute logo URL so the project image renders on PyPI.
- Adds an introduction and statement of need, complete installation and
  citation guidance, guided optimization tutorials, a constrained visualization,
  and dedicated qPOTS-Decoupled documentation.
- Defines and tests the supported top-level API, ships a ``py.typed`` marker,
  and documents compatibility and deprecation guarantees for qPOTS 2.x users.
- Adds reproducible acceptance checks for supported Python versions, examples,
  documentation, distributions, installed-wheel imports, and release artifacts.

## v2.0.1

- Adds JOSS-readiness updates, including contributor guidance, README support
  policy, refreshed CI and ReadTheDocs installation paths, repository hygiene
  for local-only files, and updated dependency floors/pins.

## v2.0.0

- Adds the major multitask update, including expanded multitask Gaussian-process utilities, partial-information candidate selection helpers, and updated API documentation for the new workflow.
