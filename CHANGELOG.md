# Changelog

## Unreleased

## v2.2.1

- Incorporates the final JOSS editorial revisions to the manuscript and bibliography.
- Aligns the archival citation title with the paper and records the paper's author
  order and all three authors' ORCID identifiers.
- Clarifies qPOTS-Decoupled figure reproduction instructions.
- Preserves the reviewed optimization implementation from v2.2.0.

## v2.2.0

- Fixes constrained multitask posterior sampling and related constraint-handling
  edge cases.
- Adds executable examples, clearer documentation, Python 3.14 coverage, and
  installed-wheel acceptance tests.
- Improves decoupled-workflow numerical stability and moves optional TS-EMO
  interoperability to an external checkout.

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
