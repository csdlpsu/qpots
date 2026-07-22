import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.verify_distribution import verify_dist_directory

METADATA = b"""Metadata-Version: 2.4
Name: qpots
Version: 2.1.0
Requires-Python: >=3.11
License-Expression: GPL-3.0-only
Provides-Extra: docs
Provides-Extra: examples
Provides-Extra: hpc
Provides-Extra: test

"""
WHEEL_METADATA = b"""Wheel-Version: 1.0
Generator: test
Root-Is-Purelib: true
Tag: py3-none-any

"""


def _write_wheel(path: Path, *, include_typed: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("qpots/__init__.py", "")
        if include_typed:
            archive.writestr("qpots/py.typed", "")
        archive.writestr("qpots-2.1.0.dist-info/METADATA", METADATA)
        archive.writestr("qpots-2.1.0.dist-info/WHEEL", WHEEL_METADATA)


def _write_sdist(path: Path, *, include_requirements: bool = False) -> None:
    members = {
        "qpots-2.1.0/PKG-INFO": METADATA,
        "qpots-2.1.0/qpots/py.typed": b"",
        "qpots-2.1.0/tools/verify_distribution.py": b"",
    }
    if include_requirements:
        members["qpots-2.1.0/requirements.txt"] = b"torch"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_distribution_pair_is_accepted(tmp_path):
    wheel = tmp_path / "qpots-2.1.0-py3-none-any.whl"
    sdist = tmp_path / "qpots-2.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    assert verify_dist_directory(tmp_path) == (wheel, sdist)


def test_wheel_must_include_typing_marker(tmp_path):
    _write_wheel(tmp_path / "qpots-2.1.0-py3-none-any.whl", include_typed=False)
    _write_sdist(tmp_path / "qpots-2.1.0.tar.gz")

    with pytest.raises(ValueError, match="py.typed is missing"):
        verify_dist_directory(tmp_path)


def test_sdist_rejects_obsolete_dependency_manifest(tmp_path):
    _write_wheel(tmp_path / "qpots-2.1.0-py3-none-any.whl")
    _write_sdist(tmp_path / "qpots-2.1.0.tar.gz", include_requirements=True)

    with pytest.raises(ValueError, match="obsolete dependency manifests"):
        verify_dist_directory(tmp_path)


def test_distribution_directory_requires_exactly_one_pair(tmp_path):
    with pytest.raises(ValueError, match="expected one wheel and one source distribution"):
        verify_dist_directory(tmp_path)
