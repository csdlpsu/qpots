"""Verify qPOTS wheel and source-distribution acceptance requirements."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED_EXTRAS = {"docs", "examples", "hpc", "test"}
REQUIRED_METADATA = {
    "Name": "qpots",
    "Requires-Python": ">=3.11",
    "License-Expression": "GPL-3.0-only",
}


def _parse_metadata(content: bytes, source: str):
    metadata = BytesParser().parsebytes(content)
    for field, expected in REQUIRED_METADATA.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{source}: metadata field {field!r} must be {expected!r}; "
                f"got {metadata.get(field)!r}"
            )
    extras = set(metadata.get_all("Provides-Extra", []))
    if extras != REQUIRED_EXTRAS:
        raise ValueError(
            f"{source}: expected extras {sorted(REQUIRED_EXTRAS)}, got {sorted(extras)}"
        )
    return metadata


def _reject_dependency_manifests(names: list[str], source: str) -> None:
    forbidden = [name for name in names if PurePosixPath(name).name == "requirements.txt"]
    if forbidden:
        raise ValueError(f"{source}: obsolete dependency manifests found: {forbidden}")


def verify_wheel(path: Path) -> None:
    """Validate wheel metadata and required package files."""
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"Expected a wheel file, got {path}")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _reject_dependency_manifests(names, path.name)
        if "qpots/py.typed" not in names:
            raise ValueError(f"{path.name}: qpots/py.typed is missing")

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError(f"{path.name}: expected exactly one METADATA and WHEEL file")
        _parse_metadata(archive.read(metadata_names[0]), path.name)
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_names[0]))
        if wheel_metadata.get("Root-Is-Purelib") != "true":
            raise ValueError(f"{path.name}: wheel must be platform-independent pure Python")


def verify_sdist(path: Path) -> None:
    """Validate source-distribution metadata and required source files."""
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise ValueError(f"Expected a .tar.gz source distribution, got {path}")

    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _reject_dependency_manifests(names, path.name)
        typed_names = [name for name in names if name.endswith("/qpots/py.typed")]
        metadata_names = [
            name
            for name in names
            if PurePosixPath(name).name == "PKG-INFO" and len(PurePosixPath(name).parts) == 2
        ]
        verifier_names = [name for name in names if name.endswith("/tools/verify_distribution.py")]
        acceptance_names = [name for name in names if name.endswith("/ACCEPTANCE.md")]
        if len(typed_names) != 1:
            raise ValueError(f"{path.name}: qpots/py.typed is missing")
        if len(metadata_names) != 1:
            raise ValueError(f"{path.name}: expected exactly one PKG-INFO file")
        if len(verifier_names) != 1:
            raise ValueError(f"{path.name}: distribution verifier is missing")
        if len(acceptance_names) != 1:
            raise ValueError(f"{path.name}: acceptance plan is missing")
        metadata_file = archive.extractfile(metadata_names[0])
        if metadata_file is None:
            raise ValueError(f"{path.name}: could not read PKG-INFO")
        _parse_metadata(metadata_file.read(), path.name)


def verify_dist_directory(dist_dir: Path) -> tuple[Path, Path]:
    """Validate the single wheel and source archive in ``dist_dir``."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"{dist_dir}: expected one wheel and one source distribution; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} source archive(s)"
        )
    verify_wheel(wheels[0])
    verify_sdist(sdists[0])
    return wheels[0], sdists[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="Directory containing one wheel and one sdist")
    args = parser.parse_args()
    wheel, sdist = verify_dist_directory(args.dist_dir)
    print(f"Accepted {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
