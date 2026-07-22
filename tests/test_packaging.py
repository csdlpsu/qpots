import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_example_dependencies_are_optional():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    core = metadata["project"]["dependencies"]
    extras = metadata["project"]["optional-dependencies"]

    assert not any(dependency.startswith(("pandas", "matplotlib", "mpi4py")) for dependency in core)
    assert set(extras["examples"]) == {"matplotlib", "pandas"}
    assert extras["hpc"] == ["mpi4py"]


def test_pyproject_is_the_only_dependency_manifest():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert metadata["project"]["license"] == "GPL-3.0-only"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert not (PROJECT_ROOT / "requirements.txt").exists()
    assert not (PROJECT_ROOT / "tests" / "requirements.txt").exists()
    assert all("<3" not in dependency for dependency in metadata["project"]["dependencies"])


def test_supported_python_versions_are_declared():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]

    assert project["requires-python"] == ">=3.11"
    assert {
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    }.issubset(project["classifiers"])


def test_core_package_imports_without_optional_example_modules():
    import qpots

    assert qpots.QPOTSRunner is not None
