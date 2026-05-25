# qPOTS JOSS submission notes

This draft follows the current JOSS paper format: YAML metadata, `Summary`, `Statement of need`, `State of the field`, `Software design`, `Research impact statement`, `AI usage disclosure`, `Acknowledgements`, and `References`.

## Files in this bundle

- `paper.md`: JOSS manuscript draft.
- `paper.bib`: curated BibTeX file with references used in the manuscript.
- `figures/qpots_doe_total_correlation.png`: figure referenced in the manuscript.
- `figures/qpots_constrained_illustration.png`: optional figure extracted from the qPOTS paper.
- `CITATION.cff`: draft citation metadata for the repository.

## Review criteria coverage

- Statement of need: identifies the problem, target audience, and relation to Bayesian optimization and multiobjective optimization.
- State of the field: compares qPOTS against BoTorch, qEHVI/qLogNEHVI, entropy-search methods, and hypervolume knowledge gradient; includes a build-vs-contribute justification.
- Software design: explains the package architecture and the trade-offs behind sample-path optimization, evolutionary search, batch selection, and multitask decoupling.
- Research impact: cites the AISTATS qPOTS paper, the NASA Common Research Model application, and the qPOTS-DOE benchmark update.
- AI usage disclosure: included as a required section.
- Acknowledgements: included, but should be edited with grant numbers or a clear statement that no specific external funding supported the work.

## Items to verify before submission

1. Confirm author order, affiliations, and whether Peter E. Bachman should be a JOSS paper author.
2. Add ORCIDs for Kade E. Carlson and Peter E. Bachman if available.
3. Confirm the qPOTS-DOE BibTeX entry author list and year. The attached manuscript is anonymous, so this draft infers the likely author list from the package metadata.
4. Confirm whether `version: 2.0.1` in `CITATION.cff` is the version to archive for JOSS.
5. Publish or tag a release and archive it with Zenodo before final JOSS acceptance; update repository release metadata as needed.
6. The PyPI page found during preparation lists version 1.0.8, while the GitHub `pyproject.toml` lists 2.0.1. Update PyPI or avoid claiming a PyPI release of 2.0.1 until it is published.
7. Verify figure permissions and whether the qPOTS-DOE figure should be included or replaced with a new software workflow diagram.
8. Replace the AI usage disclosure with the authors' final, accurate disclosure if any AI tools were or were not used for software development, documentation, or paper preparation.
9. Compile locally with the JOSS Docker command from the JOSS documentation:

```bash
docker run --rm \
  --volume $PWD:/data \
  --user $(id -u):$(id -g) \
  --env JOURNAL=joss \
  openjournals/inara
```

10. Run the package tests and ensure the repository has visible installation instructions, example usage, documentation, and a clear OSI-approved license.
