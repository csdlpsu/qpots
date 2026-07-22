# qPOTS Test and Acceptance Plan

This plan defines the evidence required to accept a pull request or publish a
qPOTS release. The GitHub Actions `test` and `acceptance` jobs are the source of
record; a local run is useful preparation but does not replace the supported
Python-version matrix.

## Pull Request Acceptance Gate

A change is accepted only when every applicable gate below passes.

| Area | Automated evidence | Acceptance criterion |
| --- | --- | --- |
| Supported Python | `python -m pytest -q tests/` on Python 3.11, 3.12, and 3.13 | All tests pass on every supported version. |
| Core algorithms | Acquisition, model, runner, function, and utility tests | Constrained, unconstrained, multitask, Nyström, partial-information, and TS-EMO wrapper tests pass. |
| Public interfaces | `tests/test_public_api.py` and `tests/test_compatibility.py` | Supported top-level exports and qPOTS 2.0 call patterns remain available. |
| Runtime and evaluation | `tests/test_config.py`, `tests/test_function.py`, and `tests/test_runner.py` | Runtime precedence, evaluation shapes, bounds, constraints, seeds, callbacks, fit counts, and physical/normalized coordinates remain correct. |
| Style | `ruff check .` and `ruff format --check .` | Maintained Python sources, selected examples, tests, and tools pass without changes. Bundled legacy TS-EMO code remains excluded. |
| Examples | `python -m compileall -q examples` | Every checked-in Python example is syntactically valid on the acceptance Python version. |
| Documentation | `sphinx -W -b html` | The complete documentation builds with warnings treated as errors. |
| Packaging | `python -m build`, `twine check`, and `tools/verify_distribution.py` | Exactly one valid wheel and source archive are produced with correct metadata, extras, typing marker, and dependency declarations. |
| Installed artifact | Wheel reinstall followed by an import outside the checkout | The installed wheel reports a version and exposes `QPOTSRunner` without relying on the source tree. |
| Paper | Draft PDF workflow | The JOSS paper builds through the Open Journals draft action. |

The `acceptance` job runs only after all Python matrix jobs pass. It uploads the
built documentation and distributions for seven days so reviewers and
maintainers can inspect the exact accepted artifacts.

## Release Acceptance Gate

A semantic-version tag may publish only after the release workflow:

1. verifies that the tag, `pyproject.toml` version, and changelog section agree;
2. reruns the complete tests, style checks, and example compilation;
3. rebuilds the documentation with warnings as errors;
4. builds and validates the wheel and source distribution;
5. uploads immutable artifacts before the PyPI publication job starts.

PyPI publication uses trusted publishing and is followed by creation of a
GitHub Release from the same artifacts. A failure in any earlier step prevents
both publication actions.

## Optional Integration Boundaries

MATLAB is not required for core acceptance. TS-EMO process and wrapper behavior
is tested with mocks; a real MATLAB Engine run is environment-specific and is a
manual integration check when that optional baseline changes. MPI is similarly
an optional HPC integration: ordinary examples and core imports must not depend
on `mpi4py`.

GPU hardware is not required for pull-request acceptance. Runtime tests verify
device selection and explicit CPU configuration, while CUDA execution remains
conditional on an available compatible PyTorch/CUDA environment.

## Reviewer-Change Coverage

The JOSS review changes are covered cumulatively:

- Software-quality behavior is exercised by the configuration, function,
  runner, model, acquisition, and readability suites.
- Repository and packaging behavior is exercised by packaging and release-note
  tests plus the distribution and release gates.
- Documentation changes are protected by the warning-free Sphinx build and
  example compilation.
- Public-interface compatibility is protected by explicit export and legacy
  call-pattern regression tests.
- Paper formatting and citations are checked by the Open Journals draft build;
  scientific content remains subject to author and reviewer inspection.

Reviewer checkboxes remain under reviewer control. Passing these gates provides
the test evidence for an author response but does not mark review items as
accepted on the reviewer's behalf.
