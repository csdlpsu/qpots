from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).parents[1]


def test_example_dependencies_are_optional():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    core = metadata["project"]["dependencies"]
    extras = metadata["project"]["optional-dependencies"]

    assert not any(
        dependency.startswith(("pandas", "matplotlib", "mpi4py"))
        for dependency in core
    )
    assert set(extras["examples"]) == {"matplotlib", "pandas"}
    assert extras["hpc"] == ["mpi4py"]


def test_core_package_imports_without_optional_example_modules():
    import qpots

    assert qpots.QPOTSRunner is not None
