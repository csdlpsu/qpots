"""Validate release metadata and extract notes for a tagged release."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

TAG_PATTERN = re.compile(r"v(?P<version>\d+\.\d+\.\d+)")
HEADING_PATTERN = re.compile(r"^## (?P<title>[^\n]+)$", re.MULTILINE)


def validate_tag_version(tag: str, pyproject_text: str) -> str:
    """Return the version for a valid tag matching the project metadata."""
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"Release tag must use the vX.Y.Z form; received {tag!r}")

    version = match.group("version")
    project_version = tomllib.loads(pyproject_text)["project"]["version"]
    if version != project_version:
        raise ValueError(f"Tag {tag!r} does not match project version {project_version!r}")
    return version


def extract_release_notes(changelog_text: str, tag: str) -> str:
    """Extract the non-empty changelog section headed by ``tag``."""
    headings = list(HEADING_PATTERN.finditer(changelog_text))
    for index, heading in enumerate(headings):
        if heading.group("title").strip() != tag:
            continue
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog_text)
        notes = changelog_text[start:end].strip()
        if not notes:
            raise ValueError(f"Changelog section for {tag!r} is empty")
        return notes + "\n"
    raise ValueError(f"CHANGELOG.md has no section headed '## {tag}'")


def main() -> None:
    """Validate release files and write the selected changelog section."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pyproject", required=True, type=Path)
    parser.add_argument("--changelog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    validate_tag_version(args.tag, args.pyproject.read_text())
    notes = extract_release_notes(args.changelog.read_text(), args.tag)
    args.output.write_text(notes)


if __name__ == "__main__":
    main()
