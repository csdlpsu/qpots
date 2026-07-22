# Contributing to qPOTS

Thank you for your interest in contributing to qPOTS. Contributions that improve correctness, documentation, examples, tests, packaging, or support for research workflows are welcome.

qPOTS is research software for multiobjective Bayesian optimization. Please keep changes focused, reproducible, and aligned with the package's existing BoTorch/PyTorch-based design.

## Ways To Contribute

- Report bugs or installation problems.
- Request features that would support a clear research workflow.
- Improve documentation, examples, or docstrings.
- Add tests for existing behavior or newly supported use cases.
- Fix bugs, packaging issues, or compatibility problems.
- Improve optional TS-EMO integration without making MATLAB required for core qPOTS use.

## Asking Questions And Reporting Issues

Use the GitHub issue tracker for bug reports, feature requests, documentation problems, and support questions:

https://github.com/csdlpsu/qpots/issues

Before opening a new issue, please check whether a related issue already exists. When reporting a bug, include:

- the qPOTS version or commit hash;
- your operating system and Python version;
- the relevant package versions, especially `torch`, `botorch`, `gpytorch`, `numpy`, `scipy`, `pymoo`, and `scikit-learn`;
- whether you are using CPU or GPU;
- a minimal code example or example script that reproduces the problem;
- the full error message or traceback;
- whether the issue involves the optional MATLAB Engine / TS-EMO path.

For feature requests, please describe the research use case, the expected behavior, and any relevant references or related software.

## Development Setup

Clone the repository and install qPOTS in editable mode with test and documentation dependencies:

```bash
git clone https://github.com/csdlpsu/qpots
cd qpots
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

qPOTS requires Python 3.11 or newer and is continuously tested on Python 3.11, 3.12, and 3.13. The core qPOTS package does not require MATLAB. MATLAB Engine is only needed for users who want to run the bundled TS-EMO baseline.

## Running Tests

Run the test suite before opening a pull request:

```bash
python -m pytest tests/
```

The tests should pass without MATLAB installed. Tests for TS-EMO wrapper behavior use mocks so the core CI workflow remains available to Python-only contributors.

## Building Documentation

Build the Sphinx documentation locally with:

```bash
python -m sphinx -b html docs/source docs/build/html
```

When adding or changing public APIs, please update docstrings and, when needed, the corresponding files in `docs/source/`.

## Pull Request Guidelines

Please open pull requests against the `main` branch. A good pull request should:

- describe the motivation and scope of the change;
- link related issues, papers, or examples when relevant;
- include tests for behavior changes;
- update documentation for user-facing changes;
- keep unrelated formatting, generated files, and refactors out of the diff;
- preserve compatibility with the supported Python versions unless the change explicitly updates support policy.

Small, focused pull requests are easier to review than broad changes that mix bug fixes, refactoring, documentation, and new features.

## Maintainer Release Process

Releases use semantic `vX.Y.Z` tags. Before tagging a release:

1. Update the version in `pyproject.toml` and `docs/source/conf.py`.
2. Rename the `Unreleased` changelog section to the same `vX.Y.Z` tag and add a new empty `Unreleased` section above it.
3. Merge the change into `main` and confirm that CI passes.
4. Create and push the matching tag, for example `git tag v2.1.0` followed by `git push origin v2.1.0`.

The release workflow verifies that the tag, package version, and changelog agree; runs the test and style suites; builds and checks the distributions; publishes to PyPI through the `pypi` environment's trusted-publisher configuration; and creates a GitHub Release with the matching changelog notes and distribution files. No long-lived PyPI token is stored in the repository.

## Code Style

Follow the style already used in the repository. Prefer clear names, explicit tensor/device/dtype handling, and concise comments around non-obvious numerical or optimization logic. Public functions and classes should include docstrings that explain parameters, return values, and important assumptions.

## Maintainer Response Expectations

This project is maintained by an academic research group. We aim to review issues and pull requests when maintainer time is available, but response times may vary around teaching, conference, and research deadlines. Reports that include a minimal reproducible example and complete environment information are usually easiest to address.

## License

By contributing to qPOTS, you agree that your contributions will be distributed under the same license as the project, the GNU General Public License v3.0.
